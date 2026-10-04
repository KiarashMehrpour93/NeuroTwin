from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List
import numpy as np


@dataclass
class NeuroProfile:
    subject_id: int
    samples: int
    feature_mean: List[float]
    feature_std: List[float]
    stability: float


class AdaptiveNeuroTwin:
    """
    Research prototype for maintaining a personalized computational
    EEG profile.

    This is NOT a diagnostic or treatment system.
    """

    def __init__(self):
        self.profiles: Dict[int, NeuroProfile] = {}

    @staticmethod
    def _stability(feature_std: np.ndarray) -> float:
        variation = float(np.mean(feature_std))
        return float(1.0 / (1.0 + variation))

    def build_profile(
        self,
        subject_id: int,
        features: np.ndarray,
    ) -> NeuroProfile:

        features = np.asarray(features, dtype=np.float64)

        if features.ndim != 2:
            raise ValueError(
                "features must have shape (samples, features)"
            )

        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)

        profile = NeuroProfile(
            subject_id=int(subject_id),
            samples=int(len(features)),
            feature_mean=mean.tolist(),
            feature_std=std.tolist(),
            stability=self._stability(std),
        )

        self.profiles[subject_id] = profile
        return profile

    def update_profile(
        self,
        subject_id: int,
        new_features: np.ndarray,
    ) -> NeuroProfile:

        new_features = np.asarray(
            new_features,
            dtype=np.float64,
        )

        if subject_id not in self.profiles:
            return self.build_profile(
                subject_id,
                new_features,
            )

        old = self.profiles[subject_id]

        old_n = old.samples
        new_n = len(new_features)

        old_mean = np.asarray(old.feature_mean)
        old_std = np.asarray(old.feature_std)

        new_mean = np.mean(new_features, axis=0)
        new_std = np.std(new_features, axis=0)

        total_n = old_n + new_n

        combined_mean = (
            old_mean * old_n + new_mean * new_n
        ) / total_n

        combined_variance = (
            old_n
            * (old_std**2 + (old_mean - combined_mean) ** 2)
            + new_n
            * (new_std**2 + (new_mean - combined_mean) ** 2)
        ) / total_n

        combined_std = np.sqrt(
            np.maximum(combined_variance, 0)
        )

        updated = NeuroProfile(
            subject_id=int(subject_id),
            samples=int(total_n),
            feature_mean=combined_mean.tolist(),
            feature_std=combined_std.tolist(),
            stability=self._stability(combined_std),
        )

        self.profiles[subject_id] = updated

        return updated

    def similarity(
        self,
        subject_id: int,
        features: np.ndarray,
    ) -> float:

        if subject_id not in self.profiles:
            raise ValueError(
                f"No profile exists for subject {subject_id}"
            )

        profile = self.profiles[subject_id]

        mean = np.asarray(profile.feature_mean)
        std = np.asarray(profile.feature_std)

        features = np.asarray(
            features,
            dtype=np.float64,
        )

        current_mean = np.mean(features, axis=0)

        normalized_distance = np.mean(
            np.abs(current_mean - mean)
            / (std + 1e-8)
        )

        similarity = 1.0 / (1.0 + normalized_distance)

        return float(
            np.clip(similarity, 0.0, 1.0)
        )

    def export_profile(self, subject_id: int) -> dict:
        if subject_id not in self.profiles:
            raise ValueError(
                f"No profile exists for subject {subject_id}"
            )

        return asdict(self.profiles[subject_id])