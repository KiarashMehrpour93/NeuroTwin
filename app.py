from pathlib import Path
import io
import json
import tempfile
import numpy as np
import pandas as pd
import streamlit as st

from scipy.signal import butter, sosfiltfilt, hilbert

import networkx as nx
import matplotlib.pyplot as plt

from src.preprocessing import preprocess_subject
from src.features import extract_features
from src.neurotwin import NeuroTwin
from src.adaptive_model import AdaptiveNeuroTwin


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="NeuroTwin 3.1",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

CONNECTIVITY_DIR = RESULTS_DIR / "connectivity"
GENERALIZATION_DIR = RESULTS_DIR / "generalization"

CONNECTIVITY_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GENERALIZATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# HELPERS
# =========================================================

def load_json(path):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def safe_float(value, default=0.0):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def bounded(value):
    return float(
        np.clip(
            value,
            0.0,
            1.0
        )
    )


def cosine_raw(a, b):

    a = np.asarray(
        a,
        dtype=np.float64
    ).reshape(-1)

    b = np.asarray(
        b,
        dtype=np.float64
    ).reshape(-1)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na < 1e-12 or nb < 1e-12:
        return 0.0

    return float(
        np.clip(
            np.dot(a, b) /
            (na * nb),
            -1.0,
            1.0
        )
    )


def normalize_cosine(a, b):

    value = cosine_raw(
        a,
        b
    )

    return float(
        np.clip(
            (value + 1.0) / 2.0,
            0.0,
            1.0
        )
    )


# =========================================================
# SCIENTIFICALLY CORRECT BASELINE SCALING
# =========================================================

def feature_distance(
    baseline,
    simulated
):

    baseline = np.asarray(
        baseline,
        dtype=np.float64
    ).reshape(-1)

    simulated = np.asarray(
        simulated,
        dtype=np.float64
    ).reshape(-1)

    if len(baseline) == 0:
        return 0.0

    center = np.median(
        baseline
    )

    mad = np.median(
        np.abs(
            baseline - center
        )
    )

    if mad > 1e-12:

        scale = (
            1.4826 *
            mad
        )

    else:

        scale = np.std(
            baseline
        )

    scale = max(
        float(scale),
        1e-8
    )

    base_scaled = (
        baseline - center
    ) / scale

    sim_scaled = (
        simulated - center
    ) / scale

    distance = np.linalg.norm(
        sim_scaled -
        base_scaled
    )

    denominator = np.sqrt(
        len(baseline)
    )

    return float(
        distance /
        max(
            denominator,
            1e-12
        )
    )


def similarity_v30(
    baseline,
    simulated
):

    cosine_component = normalize_cosine(
        baseline,
        simulated
    )

    distance = feature_distance(
        baseline,
        simulated
    )

    distance_component = float(
        np.exp(
            -distance
        )
    )

    score = (
        0.45 *
        cosine_component
        +
        0.55 *
        distance_component
    )

    return bounded(
        score
    )


def relative_shift(
    baseline,
    simulated
):

    baseline = np.asarray(
        baseline,
        dtype=np.float64
    )

    simulated = np.asarray(
        simulated,
        dtype=np.float64
    )

    denominator = np.linalg.norm(
        baseline
    )

    if denominator < 1e-12:
        return 0.0

    return float(
        np.linalg.norm(
            simulated -
            baseline
        ) /
        denominator
    )


def stability_score(
    baseline,
    simulated
):

    shift = relative_shift(
        baseline,
        simulated
    )

    return float(
        np.exp(
            -shift * 3.0
        )
    )


def pattern_preservation(
    baseline,
    simulated
):

    baseline = np.asarray(
        baseline,
        dtype=np.float64
    )

    simulated = np.asarray(
        simulated,
        dtype=np.float64
    )

    baseline_centered = (
        baseline -
        np.mean(baseline)
    )

    simulated_centered = (
        simulated -
        np.mean(simulated)
    )

    return normalize_cosine(
        baseline_centered,
        simulated_centered
    )


def consistency_score(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    if len(values) < 2:
        return 0.0

    mean = abs(
        np.mean(values)
    )

    if mean < 1e-12:
        return 0.0

    cv = (
        np.std(values) /
        mean
    )

    return bounded(
        1.0 /
        (
            1.0 + cv
        )
    )


def uncertainty_score(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    if len(values) < 2:
        return 1.0

    mean = abs(
        np.mean(values)
    )

    if mean < 1e-12:
        return 1.0

    uncertainty = (
        np.std(values) /
        mean
    )

    return float(
        np.clip(
            uncertainty,
            0.0,
            1.0
        )
    )


def confidence_interval(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    if len(values) < 2:
        return 0.0

    std = np.std(
        values,
        ddof=1
    )

    return float(
        1.96 *
        std /
        np.sqrt(
            len(values)
        )
    )


# =========================================================
# SUBJECT DATA CACHE
# =========================================================

@st.cache_data(
    show_spinner=False
)
def load_subject_data_cached(
    subject_id
):

    epochs = preprocess_subject(
        int(subject_id)
    )

    X = extract_features(
        epochs
    )

    X = np.asarray(
        X,
        dtype=np.float64
    )

    return epochs, X


@st.cache_data(
    show_spinner=False
)
def load_subject_features_cached(
    subject_id
):

    epochs = preprocess_subject(
        int(subject_id)
    )

    X = extract_features(
        epochs
    )

    return np.asarray(
        X,
        dtype=np.float64
    )


# =========================================================
# CONNECTIVITY ENGINE
# =========================================================

class FunctionalConnectivityEngine:

    """
    NeuroTwin 3.0 Functional Connectivity Engine.

    Computes:
        - PLV
        - Band coherence
        - Combined connectivity
        - Network metrics
        - Brain graph
    """

    def __init__(
        self,
        epochs,
        low_freq=8.0,
        high_freq=30.0
    ):

        self.epochs = epochs

        self.low_freq = float(
            low_freq
        )

        self.high_freq = float(
            high_freq
        )

        self.sfreq = float(
            epochs.info["sfreq"]
        )

        self.data = np.asarray(
            epochs.get_data(
                copy=True
            ),
            dtype=np.float64
        )

        self.n_epochs = (
            self.data.shape[0]
        )

        self.n_channels = (
            self.data.shape[1]
        )

        self.channel_names = list(
            epochs.ch_names
        )

    # -----------------------------------------------------
    # BANDPASS
    # -----------------------------------------------------

    def bandpass(
        self,
        data
    ):

        nyquist = (
            self.sfreq /
            2.0
        )

        low = max(
            self.low_freq /
            nyquist,
            1e-5
        )

        high = min(
            self.high_freq /
            nyquist,
            0.999
        )

        sos = butter(
            4,
            [
                low,
                high
            ],
            btype="bandpass",
            output="sos"
        )

        return sosfiltfilt(
            sos,
            data,
            axis=-1
        )

    # -----------------------------------------------------
    # PLV
    # -----------------------------------------------------

    def compute_plv(self):

        filtered = self.bandpass(
            self.data
        )

        analytic = hilbert(
            filtered,
            axis=-1
        )

        phase = np.angle(
            analytic
        )

        plv = np.zeros(
            (
                self.n_channels,
                self.n_channels
            ),
            dtype=np.float64
        )

        for i in range(
            self.n_channels
        ):

            plv[i, i] = 1.0

            for j in range(
                i + 1,
                self.n_channels
            ):

                phase_difference = (
                    phase[:, i, :] -
                    phase[:, j, :]
                )

                value = np.abs(
                    np.mean(
                        np.exp(
                            1j *
                            phase_difference
                        )
                    )
                )

                plv[i, j] = value
                plv[j, i] = value

        return np.clip(
            plv,
            0.0,
            1.0
        )

    # -----------------------------------------------------
    # BAND COHERENCE
    # -----------------------------------------------------

    def compute_coherence(self):

        data = self.bandpass(
            self.data
        )

        n_samples = (
            data.shape[-1]
        )

        nperseg = min(
            256,
            n_samples
        )

        if nperseg < 8:
            return np.eye(
                self.n_channels
            )

        freqs = np.fft.rfftfreq(
            nperseg,
            d=1.0 /
            self.sfreq
        )

        band_mask = (
            (freqs >= self.low_freq) &
            (freqs <= self.high_freq)
        )

        if not np.any(
            band_mask
        ):
            return np.eye(
                self.n_channels
            )

        windows = []

        for epoch in data:

            if (
                epoch.shape[-1]
                < nperseg
            ):
                continue

            step = max(
                nperseg // 2,
                1
            )

            for start in range(
                0,
                epoch.shape[-1] -
                nperseg +
                1,
                step
            ):

                segment = epoch[
                    :,
                    start:
                    start +
                    nperseg
                ]

                window = np.hanning(
                    nperseg
                )

                segment = (
                    segment *
                    window
                )

                spectrum = np.fft.rfft(
                    segment,
                    axis=-1
                )

                windows.append(
                    spectrum
                )

        if not windows:

            return np.eye(
                self.n_channels
            )

        F = np.asarray(
            windows
        )

        F = F[
            :,
            :,
            band_mask
        ]

        coherence = np.zeros(
            (
                self.n_channels,
                self.n_channels
            ),
            dtype=np.float64
        )

        for i in range(
            self.n_channels
        ):

            coherence[i, i] = 1.0

            Xi = F[
                :,
                i,
                :
            ]

            Sxx = np.mean(
                np.abs(Xi) ** 2,
                axis=0
            )

            for j in range(
                i + 1,
                self.n_channels
            ):

                Xj = F[
                    :,
                    j,
                    :
                ]

                Syy = np.mean(
                    np.abs(Xj) ** 2,
                    axis=0
                )

                Sxy = np.mean(
                    Xi *
                    np.conj(Xj),
                    axis=0
                )

                numerator = np.abs(
                    Sxy
                ) ** 2

                denominator = (
                    Sxx *
                    Syy
                    +
                    1e-12
                )

                coh = (
                    numerator /
                    denominator
                )

                value = np.mean(
                    np.clip(
                        coh,
                        0.0,
                        1.0
                    )
                )

                coherence[i, j] = value
                coherence[j, i] = value

        return np.clip(
            coherence,
            0.0,
            1.0
        )

    # -----------------------------------------------------
    # COMBINED CONNECTIVITY
    # -----------------------------------------------------

    def compute_all(self):

        plv = self.compute_plv()

        coherence = (
            self.compute_coherence()
        )

        combined = (
            0.5 * plv +
            0.5 * coherence
        )

        np.fill_diagonal(
            combined,
            1.0
        )

        return {
            "plv": plv,
            "coherence": coherence,
            "combined": np.clip(
                combined,
                0.0,
                1.0
            )
        }


# =========================================================
# BRAIN NETWORK ENGINE
# =========================================================

class BrainNetworkEngine:

    def __init__(
        self,
        matrix,
        channel_names
    ):

        self.matrix = np.asarray(
            matrix,
            dtype=np.float64
        )

        self.channel_names = list(
            channel_names
        )

        self.matrix = np.nan_to_num(
            self.matrix,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        np.fill_diagonal(
            self.matrix,
            0.0
        )

    # -----------------------------------------------------
    # GRAPH
    # -----------------------------------------------------

    def graph(
        self,
        threshold=0.35
    ):

        G = nx.Graph()

        for i, name in enumerate(
            self.channel_names
        ):

            G.add_node(
                i,
                name=name
            )

        for i in range(
            len(self.channel_names)
        ):

            for j in range(
                i + 1,
                len(self.channel_names)
            ):

                weight = float(
                    self.matrix[i, j]
                )

                if weight >= threshold:

                    G.add_edge(
                        i,
                        j,
                        weight=weight
                    )

        return G

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    def metrics(
        self,
        threshold=0.35
    ):

        G = self.graph(
            threshold
        )

        n = max(
            G.number_of_nodes(),
            1
        )

        degrees = dict(
            G.degree()
        )

        strength = {}

        for node in G.nodes:

            strength[node] = float(
                sum(
                    data.get(
                        "weight",
                        0.0
                    )
                    for _, _, data
                    in G.edges(
                        node,
                        data=True
                    )
                )
            )

        clustering = nx.clustering(
            G,
            weight="weight"
        )

        try:

            betweenness = nx.betweenness_centrality(
                G,
                weight="weight",
                normalized=True
            )

        except Exception:

            betweenness = {
                node: 0.0
                for node in G.nodes
            }

        mean_degree = (
            np.mean(
                list(
                    degrees.values()
                )
            )
            if degrees
            else 0.0
        )

        mean_strength = (
            np.mean(
                list(
                    strength.values()
                )
            )
            if strength
            else 0.0
        )

        mean_clustering = (
            np.mean(
                list(
                    clustering.values()
                )
            )
            if clustering
            else 0.0
        )

        mean_betweenness = (
            np.mean(
                list(
                    betweenness.values()
                )
            )
            if betweenness
            else 0.0
        )

        density = nx.density(
            G
        )

        return {
            "nodes": n,
            "edges": G.number_of_edges(),
            "density": float(
                density
            ),
            "mean_degree": float(
                mean_degree
            ),
            "mean_strength": float(
                mean_strength
            ),
            "mean_clustering": float(
                mean_clustering
            ),
            "mean_betweenness": float(
                mean_betweenness
            ),
            "degrees": degrees,
            "strength": strength,
            "clustering": clustering,
            "betweenness": betweenness
        }


# =========================================================
# NEURAL DYNAMICS
# =========================================================

class NeuralDynamicsEngine:

    """
    Simple computational network dynamics.

    This is NOT a biological whole-brain model.
    It is a network-level mathematical simulation.
    """

    def __init__(
        self,
        connectivity
    ):

        W = np.asarray(
            connectivity,
            dtype=np.float64
        )

        W = np.nan_to_num(
            W,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        np.fill_diagonal(
            W,
            0.0
        )

        max_value = np.max(
            np.abs(W)
        )

        if max_value > 1e-12:

            W = W / max_value

        self.W = W

    def simulate(
        self,
        steps=100,
        coupling=0.08,
        decay=0.12,
        seed=42
    ):

        rng = np.random.default_rng(
            seed
        )

        n = self.W.shape[0]

        state = rng.normal(
            0.0,
            0.2,
            n
        )

        history = [
            state.copy()
        ]

        W_norm = self.W.copy()

        row_sum = np.sum(
            W_norm,
            axis=1,
            keepdims=True
        )

        row_sum[row_sum < 1e-12] = 1.0

        W_norm = (
            W_norm /
            row_sum
        )

        for _ in range(
            steps
        ):

            network_input = (
                W_norm @ state
            )

            noise = rng.normal(
                0.0,
                0.01,
                n
            )

            delta = (
                -decay *
                state
                +
                coupling *
                network_input
                +
                noise
            )

            state = np.tanh(
                state +
                delta
            )

            history.append(
                state.copy()
            )

        history = np.asarray(
            history,
            dtype=np.float64
        )

        return history


# =========================================================
# DIGITAL BRAIN TWIN
# =========================================================

class DigitalBrainTwin:

    def __init__(
        self,
        subject,
        feature_vector,
        connectivity,
        channel_names
    ):

        self.subject = int(
            subject
        )

        self.feature_vector = np.asarray(
            feature_vector,
            dtype=np.float64
        )

        self.connectivity = np.asarray(
            connectivity,
            dtype=np.float64
        )

        self.channel_names = list(
            channel_names
        )

        self.network = BrainNetworkEngine(
            self.connectivity,
            self.channel_names
        )

        self.state_history = []

    def network_metrics(
        self,
        threshold=0.35
    ):

        return self.network.metrics(
            threshold
        )

    def simulate(
        self,
        steps=100,
        coupling=0.08,
        decay=0.12,
        seed=42
    ):

        engine = NeuralDynamicsEngine(
            self.connectivity
        )

        history = engine.simulate(
            steps=steps,
            coupling=coupling,
            decay=decay,
            seed=seed
        )

        self.state_history = history

        return history


# =========================================================
# V3.0 RESEARCH ENGINE
# =========================================================

class NeuroTwinV30Engine:

    def __init__(
        self,
        baseline,
        intensity=0.5,
        repetitions=50,
        stages=5,
        seed=42,
        robustness_runs=10
    ):

        self.baseline = np.asarray(
            baseline,
            dtype=np.float64
        ).reshape(-1)

        self.intensity = float(
            np.clip(
                intensity,
                0.0,
                1.0
            )
        )

        self.repetitions = int(
            max(
                5,
                repetitions
            )
        )

        self.stages = int(
            max(
                1,
                stages
            )
        )

        self.seed = int(
            seed
        )

        self.robustness_runs = int(
            np.clip(
                robustness_runs,
                1,
                10
            )
        )

        self.robustness_seeds = list(
            range(
                1,
                self.robustness_runs + 1
            )
        )

        self.strategies = [
            "Motor Strategy A",
            "Motor Strategy B",
            "Motor Strategy C"
        ]

    # -----------------------------------------------------
    # GENERATE
    # -----------------------------------------------------

    def generate_vector(
        self,
        baseline,
        strategy,
        intensity,
        repetition,
        stage,
        seed_offset=0
    ):

        baseline = np.asarray(
            baseline,
            dtype=np.float64
        ).reshape(-1)

        n = len(
            baseline
        )

        if n == 0:
            return baseline.copy()

        rng = np.random.default_rng(
            self.seed
            +
            seed_offset * 1000003
            +
            repetition * 7919
            +
            stage * 104729
        )

        x = np.linspace(
            0.0,
            1.0,
            n
        )

        centered = (
            baseline -
            np.mean(baseline)
        )

        scale = np.std(
            centered
        )

        if scale < 1e-12:
            scale = 1.0

        normalized = (
            centered /
            scale
        )

        phase = (
            repetition * 0.17
            +
            stage * 0.43
            +
            seed_offset * 0.31
        )

        if strategy == "Motor Strategy A":

            pattern = (
                np.sin(
                    2 *
                    np.pi *
                    x +
                    phase
                )
                +
                0.35 *
                np.sin(
                    6 *
                    np.pi *
                    x +
                    phase
                )
            )

            pattern /= (
                np.std(pattern)
                +
                1e-8
            )

            delta = (
                pattern *
                scale *
                0.35 *
                intensity
            )

        elif strategy == "Motor Strategy B":

            pattern = (
                0.65 *
                np.cos(
                    4 *
                    np.pi *
                    x +
                    phase
                )
                +
                0.35 *
                np.sin(
                    10 *
                    np.pi *
                    x +
                    phase * 0.5
                )
            )

            pattern /= (
                np.std(pattern)
                +
                1e-8
            )

            delta = (
                pattern *
                scale *
                0.30 *
                intensity
            )

        else:

            random_pattern = rng.normal(
                0.0,
                1.0,
                n
            )

            adaptive_weight = (
                0.4
                +
                0.6 *
                np.abs(
                    np.tanh(
                        normalized
                    )
                )
            )

            delta = (
                random_pattern
                *
                adaptive_weight
                *
                scale
                *
                0.25
                *
                intensity
            )

        profile_component = (
            np.tanh(
                normalized
            )
            *
            scale
            *
            0.05
            *
            intensity
        )

        return (
            baseline +
            delta +
            profile_component
        )

    # -----------------------------------------------------
    # EVALUATE
    # -----------------------------------------------------

    def evaluate_strategy(
        self,
        baseline,
        strategy,
        intensity,
        stage,
        seed_offset=0
    ):

        similarities = []
        stabilities = []
        shifts = []
        patterns = []
        responses = []
        profiles = []

        for repetition in range(
            self.repetitions
        ):

            simulated = self.generate_vector(
                baseline,
                strategy,
                intensity,
                repetition,
                stage,
                seed_offset
            )

            similarity = similarity_v30(
                baseline,
                simulated
            )

            stability = stability_score(
                baseline,
                simulated
            )

            shift = relative_shift(
                baseline,
                simulated
            )

            preservation = pattern_preservation(
                baseline,
                simulated
            )

            fd = feature_distance(
                baseline,
                simulated
            )

            distance_component = bounded(
                np.exp(
                    -fd
                )
            )

            compatibility = (
                0.30 *
                similarity
                +
                0.25 *
                stability
                +
                0.25 *
                preservation
                +
                0.20 *
                distance_component
            )

            similarities.append(
                similarity
            )

            stabilities.append(
                stability
            )

            shifts.append(
                shift
            )

            patterns.append(
                preservation
            )

            responses.append(
                compatibility
            )

            profiles.append(
                simulated
            )

        similarities = np.asarray(
            similarities
        )

        stabilities = np.asarray(
            stabilities
        )

        shifts = np.asarray(
            shifts
        )

        patterns = np.asarray(
            patterns
        )

        responses = np.asarray(
            responses
        )

        consistency = consistency_score(
            responses
        )

        uncertainty = uncertainty_score(
            responses
        )

        mean_response = float(
            np.mean(
                responses
            )
        )

        response_std = float(
            np.std(
                responses,
                ddof=1
            )
            if len(responses) > 1
            else 0.0
        )

        ci95 = confidence_interval(
            responses
        )

        adjusted_response = (
            mean_response
            *
            (
                0.75 +
                0.25 *
                consistency
            )
            *
            (
                1.0 -
                0.20 *
                uncertainty
            )
        )

        adaptation_potential = bounded(
            0.40 *
            mean_response
            +
            0.30 *
            consistency
            +
            0.30 *
            (
                1.0 -
                uncertainty
            )
        )

        return {
            "Stage": stage,

            "Strategy": strategy,

            "Similarity":
                float(
                    np.mean(
                        similarities
                    )
                ),

            "Similarity Std":
                float(
                    np.std(
                        similarities,
                        ddof=1
                    )
                    if len(similarities) > 1
                    else 0.0
                ),

            "Stability":
                float(
                    np.mean(
                        stabilities
                    )
                ),

            "Stability Std":
                float(
                    np.std(
                        stabilities,
                        ddof=1
                    )
                    if len(stabilities) > 1
                    else 0.0
                ),

            "Feature Shift":
                float(
                    np.mean(
                        shifts
                    )
                ),

            "Feature Shift Std":
                float(
                    np.std(
                        shifts,
                        ddof=1
                    )
                    if len(shifts) > 1
                    else 0.0
                ),

            "Pattern Preservation":
                float(
                    np.mean(
                        patterns
                    )
                ),

            "Response Index":
                mean_response,

            "Response Std":
                response_std,

            "CI95":
                ci95,

            "Consistency":
                consistency,

            "Uncertainty":
                uncertainty,

            "Adaptation Potential":
                adaptation_potential,

            "Adjusted Response":
                float(
                    adjusted_response
                ),

            "Profiles":
                profiles
        }

    # -----------------------------------------------------
    # ROBUSTNESS
    # -----------------------------------------------------

    def robustness_evaluation(
        self,
        baseline,
        intensity,
        stage
    ):

        detailed_rows = []

        original_seed = self.seed

        for experiment_seed in self.robustness_seeds:

            self.seed = int(
                experiment_seed
            )

            for strategy in self.strategies:

                result = self.evaluate_strategy(
                    baseline,
                    strategy,
                    intensity,
                    stage,
                    0
                )

                detailed_rows.append(
                    {
                        "Seed":
                            experiment_seed,

                        "Strategy":
                            strategy,

                        "Response":
                            result["Response Index"],

                        "Adjusted Response":
                            result["Adjusted Response"],

                        "Similarity":
                            result["Similarity"],

                        "Stability":
                            result["Stability"],

                        "Feature Shift":
                            result["Feature Shift"],

                        "Pattern Preservation":
                            result["Pattern Preservation"],

                        "Consistency":
                            result["Consistency"],

                        "Uncertainty":
                            result["Uncertainty"],

                        "CI95":
                            result["CI95"],

                        "Adaptation Potential":
                            result["Adaptation Potential"]
                    }
                )

        self.seed = original_seed

        detailed_df = pd.DataFrame(
            detailed_rows
        )

        summary_rows = []

        for strategy in self.strategies:

            subset = detailed_df[
                detailed_df["Strategy"] ==
                strategy
            ]

            adjusted = subset[
                "Adjusted Response"
            ].to_numpy()

            response = subset[
                "Response"
            ].to_numpy()

            mean_adjusted = float(
                np.mean(
                    adjusted
                )
            )

            std_adjusted = float(
                np.std(
                    adjusted,
                    ddof=1
                )
                if len(adjusted) > 1
                else 0.0
            )

            ci95 = (
                1.96 *
                std_adjusted /
                np.sqrt(
                    len(adjusted)
                )
                if len(adjusted) > 1
                else 0.0
            )

            summary_rows.append(
                {
                    "Strategy":
                        strategy,

                    "Seeds":
                        len(adjusted),

                    "Mean Adjusted Response":
                        mean_adjusted,

                    "Adjusted Response Std":
                        std_adjusted,

                    "Minimum":
                        float(
                            np.min(
                                adjusted
                            )
                        ),

                    "Maximum":
                        float(
                            np.max(
                                adjusted
                            )
                        ),

                    "CI95":
                        float(
                            ci95
                        ),

                    "Mean Response":
                        float(
                            np.mean(
                                response
                            )
                        ),

                    "Response Std":
                        float(
                            np.std(
                                response,
                                ddof=1
                            )
                            if len(response) > 1
                            else 0.0
                        ),

                    "Seed Consistency":
                        consistency_score(
                            adjusted
                        ),

                    "Mean Similarity":
                        float(
                            subset[
                                "Similarity"
                            ].mean()
                        ),

                    "Mean Stability":
                        float(
                            subset[
                                "Stability"
                            ].mean()
                        ),

                    "Mean Feature Shift":
                        float(
                            subset[
                                "Feature Shift"
                            ].mean()
                        ),

                    "Mean Uncertainty":
                        float(
                            subset[
                                "Uncertainty"
                            ].mean()
                        )
                }
            )

        return {
            "detailed":
                detailed_df,

            "summary":
                pd.DataFrame(
                    summary_rows
                )
        }

    # -----------------------------------------------------
    # STAGE
    # -----------------------------------------------------

    def run_stage(
        self,
        baseline,
        stage,
        intensity
    ):

        return [
            self.evaluate_strategy(
                baseline,
                strategy,
                intensity,
                stage
            )
            for strategy in self.strategies
        ]

    # -----------------------------------------------------
    # SELECT
    # -----------------------------------------------------

    def select(
        self,
        results
    ):

        return max(
            results,
            key=lambda x:
            x["Adjusted Response"]
        )

    # -----------------------------------------------------
    # UPDATE
    # -----------------------------------------------------

    def virtual_update(
        self,
        baseline,
        selected_result,
        intensity
    ):

        profiles = selected_result[
            "Profiles"
        ]

        selected_vector = np.mean(
            np.vstack(
                profiles
            ),
            axis=0
        )

        learning_rate = (
            0.20 *
            intensity
        )

        return (
            (
                1.0 -
                learning_rate
            )
            *
            baseline
            +
            learning_rate
            *
            selected_vector
        )

    # -----------------------------------------------------
    # MULTI STAGE
    # -----------------------------------------------------

    def run_multistage(self):

        current = (
            self.baseline.copy()
        )

        history = []

        for stage in range(
            1,
            self.stages + 1
        ):

            results = self.run_stage(
                current,
                stage,
                self.intensity
            )

            selected = self.select(
                results
            )

            updated = self.virtual_update(
                current,
                selected,
                self.intensity
            )

            history.append(
                {
                    "stage":
                        stage,

                    "results":
                        results,

                    "selected":
                        selected,

                    "baseline_before":
                        current.copy(),

                    "baseline_after":
                        updated.copy()
                }
            )

            current = updated

        return history

    # -----------------------------------------------------
    # SENSITIVITY
    # -----------------------------------------------------

    def sensitivity_analysis(
        self,
        baseline
    ):

        intensities = np.round(
            np.arange(
                0.1,
                1.01,
                0.1
            ),
            2
        )

        rows = []

        for value in intensities:

            for strategy in self.strategies:

                result = self.evaluate_strategy(
                    baseline,
                    strategy,
                    float(value),
                    stage=999
                )

                rows.append(
                    {
                        "Intensity":
                            float(value),

                        "Strategy":
                            strategy,

                        "Response":
                            result[
                                "Adjusted Response"
                            ],

                        "Similarity":
                            result[
                                "Similarity"
                            ],

                        "Stability":
                            result[
                                "Stability"
                            ],

                        "Feature Shift":
                            result[
                                "Feature Shift"
                            ],

                        "Pattern Preservation":
                            result[
                                "Pattern Preservation"
                            ],

                        "Consistency":
                            result[
                                "Consistency"
                            ],

                        "Uncertainty":
                            result[
                                "Uncertainty"
                            ],

                        "CI95":
                            result[
                                "CI95"
                            ]
                    }
                )

        return pd.DataFrame(
            rows
        )


# =========================================================
# GENERALIZATION
# =========================================================

def run_cross_subject_generalization(
    subjects,
    repetitions,
    seeds,
    intensity
):

    rows = []

    for subject_id in subjects:

        try:

            X = load_subject_features_cached(
                subject_id
            )

            baseline = np.mean(
                X,
                axis=0
            )

            engine = NeuroTwinV30Engine(
                baseline=baseline,
                intensity=intensity,
                repetitions=repetitions,
                stages=1,
                seed=42,
                robustness_runs=seeds
            )

            robustness = engine.robustness_evaluation(
                baseline,
                intensity,
                stage=1
            )

            detailed = robustness[
                "detailed"
            ]

            for strategy in engine.strategies:

                subset = detailed[
                    detailed["Strategy"] ==
                    strategy
                ]

                rows.append(
                    {
                        "Subject":
                            subject_id,

                        "Strategy":
                            strategy,

                        "Adjusted Response":
                            subset[
                                "Adjusted Response"
                            ].mean(),

                        "Response":
                            subset[
                                "Response"
                            ].mean(),

                        "Similarity":
                            subset[
                                "Similarity"
                            ].mean(),

                        "Stability":
                            subset[
                                "Stability"
                            ].mean(),

                        "Feature Shift":
                            subset[
                                "Feature Shift"
                            ].mean(),

                        "Pattern Preservation":
                            subset[
                                "Pattern Preservation"
                            ].mean(),

                        "Consistency":
                            subset[
                                "Consistency"
                            ].mean(),

                        "Uncertainty":
                            subset[
                                "Uncertainty"
                            ].mean()
                    }
                )

        except Exception as e:

            rows.append(
                {
                    "Subject":
                        subject_id,

                    "Strategy":
                        "ERROR",

                    "Adjusted Response":
                        np.nan,

                    "Response":
                        np.nan,

                    "Similarity":
                        np.nan,

                    "Stability":
                        np.nan,

                    "Feature Shift":
                        np.nan,

                    "Pattern Preservation":
                        np.nan,

                    "Consistency":
                        np.nan,

                    "Uncertainty":
                        np.nan,

                    "Error":
                        str(e)
                }
            )

    detailed_df = pd.DataFrame(
        rows
    )

    valid = detailed_df[
        detailed_df["Strategy"] !=
        "ERROR"
    ].copy()

    if valid.empty:

        return {
            "detailed":
                detailed_df,

            "summary":
                pd.DataFrame()
        }

    summary = (
        valid
        .groupby(
            "Strategy"
        )
        .agg(
            {
                "Adjusted Response":
                    ["mean", "std"],

                "Response":
                    ["mean", "std"],

                "Similarity":
                    "mean",

                "Stability":
                    "mean",

                "Feature Shift":
                    "mean",

                "Pattern Preservation":
                    "mean",

                "Consistency":
                    "mean",

                "Uncertainty":
                    "mean"
            }
        )
        .reset_index()
    )

    summary.columns = [
        "Strategy",
        "Adjusted Mean",
        "Adjusted Std",
        "Response Mean",
        "Response Std",
        "Similarity Mean",
        "Stability Mean",
        "Feature Shift Mean",
        "Pattern Preservation Mean",
        "Consistency Mean",
        "Uncertainty Mean"
    ]

    return {
        "detailed":
            detailed_df,

        "summary":
            summary
    }


# =========================================================
# METRICS
# =========================================================

metrics_v1 = load_json(
    RESULTS_DIR /
    "metrics.json"
)

metrics_v2 = load_json(
    RESULTS_DIR /
    "v2_metrics.json"
)


# =========================================================
# SESSION STATE
# =========================================================

state_defaults = {

    "profile":
        None,

    "adaptive_twin":
        None,

    "subject_features":
        None,

    "epochs":
        None,

    "v30_results":
        None,

    "sensitivity_results":
        None,

    "robustness_results":
        None,

    "engine":
        None,

    "experiment_subject":
        None,

    "adaptive_updated":
        False,

    "connectivity_results":
        None,

    "brain_network_metrics":
        None,

    "digital_twin":
        None,

    "neural_simulation":
        None,

    "generalization_results":
        None,

    "mri_summary":
        None,

    "mri_features":
        None,

    "mri_volume":
        None,

    "mri_filename":
        None,

    "whole_brain_simulation":
        None,

    "clinical_validation":
        None
}


for key, value in state_defaults.items():

    if key not in st.session_state:

        st.session_state[
            key
        ] = value


# =========================================================
# HEADER
# =========================================================

st.title(
    "🧠 NeuroTwin 3.0"
)

st.caption(
    "Personalized Multimodal Computational "
    "Brain Digital Twin"
)

st.info(
    "Research prototype: EEG-based computational "
    "modeling, functional connectivity, brain-network "
    "analysis and simulation only. Not intended for "
    "diagnosis, treatment, or clinical decision-making."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "Research Controls"
)

subject = st.sidebar.selectbox(
    "Research Subject",
    range(1, 11),
    format_func=lambda x:
        f"Subject {x:02d}"
)

st.sidebar.divider()

st.sidebar.subheader(
    "Dataset"
)

st.sidebar.write(
    "PhysioNet EEGMMIDB"
)

st.sidebar.write(
    "Subjects: 01–10"
)

st.sidebar.write(
    "Runs: R04 / R08 / R12"
)

st.sidebar.divider()

st.sidebar.subheader(
    "NeuroTwin 3.0 Engine"
)

stages = st.sidebar.slider(
    "Adaptation Stages",
    1,
    7,
    5
)

repetitions = st.sidebar.slider(
    "Repetitions / Strategy",
    10,
    100,
    50,
    step=10
)

robustness_runs = st.sidebar.slider(
    "Experiment Seeds",
    1,
    10,
    10
)

seed = st.sidebar.number_input(
    "Main Experiment Seed",
    0,
    999999,
    42
)

intensity = st.sidebar.slider(
    "Simulation Intensity",
    0.1,
    1.0,
    0.5,
    0.05
)

st.sidebar.divider()

st.sidebar.subheader(
    "Connectivity"
)

connectivity_threshold = st.sidebar.slider(
    "Network Edge Threshold",
    0.05,
    0.90,
    0.35,
    0.05
)

neural_steps = st.sidebar.slider(
    "Neural Simulation Steps",
    20,
    300,
    100,
    10
)

neural_coupling = st.sidebar.slider(
    "Network Coupling",
    0.01,
    0.50,
    0.08,
    0.01
)

st.sidebar.divider()

st.sidebar.caption(
    "NeuroTwin 3.0"
)

st.sidebar.caption(
    "EEG → Connectivity → Brain Network "
    "→ Digital Twin → Simulation"
)


# =========================================================
# TOP METRICS
# =========================================================

baseline_accuracy = safe_float(
    metrics_v1.get(
        "accuracy",
        0
    )
)

v2_accuracy = safe_float(
    metrics_v2.get(
        "mean_accuracy",
        0
    )
)

v2_f1 = safe_float(
    metrics_v2.get(
        "mean_f1",
        0
    )
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Baseline Accuracy",
    f"{baseline_accuracy:.3f}"
)

c2.metric(
    "CSP + LDA",
    f"{v2_accuracy:.3f}"
)

c3.metric(
    "Mean F1",
    f"{v2_f1:.3f}"
)

c4.metric(
    "Subjects",
    "10"
)


# =========================================================
# PERFORMANCE
# =========================================================

st.divider()

st.subheader(
    "📊 Model Performance"
)

performance_df = pd.DataFrame(
    {
        "Model": [
            "Baseline",
            "CSP + LDA"
        ],

        "Accuracy": [
            baseline_accuracy,
            v2_accuracy
        ]
    }
)

st.bar_chart(
    performance_df.set_index(
        "Model"
    )
)

st.caption(
    "Dataset evaluation metrics only; "
    "not clinical performance."
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

st.subheader(
    "Confusion Matrix — CSP + LDA"
)

cm = metrics_v2.get(
    "confusion_matrix"
)

if (
    isinstance(cm, list)
    and len(cm) == 2
):

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual 0",
            "Actual 1"
        ],
        columns=[
            "Predicted 0",
            "Predicted 1"
        ]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

else:

    st.info(
        "Confusion matrix unavailable."
    )


# =========================================================
# PERSONALIZED PROFILE
# =========================================================

st.divider()

st.subheader(
    "🧬 Personalized NeuroTwin"
)

if st.button(
    "🧠 Build Personalized NeuroTwin 3.0 Profile",
    use_container_width=True
):

    try:

        with st.spinner(
            f"Processing Subject {subject:02d}..."
        ):

            epochs, X = load_subject_data_cached(
                subject
            )

            twin = NeuroTwin()

            profile = twin.build_profile(
                X
            )

            adaptive_twin = AdaptiveNeuroTwin()

            adaptive_twin.build_profile(
                subject,
                X
            )

            st.session_state.profile = profile
            st.session_state.adaptive_twin = adaptive_twin
            st.session_state.subject_features = X
            st.session_state.epochs = epochs

            st.session_state.v30_results = None
            st.session_state.sensitivity_results = None
            st.session_state.robustness_results = None
            st.session_state.connectivity_results = None
            st.session_state.brain_network_metrics = None
            st.session_state.digital_twin = None
            st.session_state.neural_simulation = None
            st.session_state.generalization_results = None
            st.session_state.whole_brain_simulation = None
            st.session_state.clinical_validation = None
            st.session_state.engine = None
            st.session_state.experiment_subject = subject
            st.session_state.adaptive_updated = False

        st.success(
            f"Personalized profile created for "
            f"Subject {subject:02d}."
        )

    except Exception as e:

        st.error(
            f"Profile creation failed: {e}"
        )


# =========================================================
# PROFILE
# =========================================================

profile = st.session_state.profile
X = st.session_state.subject_features
adaptive_twin = st.session_state.adaptive_twin


if profile is not None:

    st.subheader(
        f"Subject {subject:02d} Computational Profile"
    )

    p1, p2, p3, p4 = st.columns(4)

    p1.metric(
        "EEG Samples",
        profile["samples"]
    )

    p2.metric(
        "Features",
        len(
            profile[
                "feature_mean"
            ]
        )
    )

    p3.metric(
        "Stability",
        f"{profile['stability']:.3f}"
    )

    adaptive_profile = None

    if adaptive_twin is not None:

        adaptive_profile = (
            adaptive_twin.profiles.get(
                subject
            )
        )

    p4.metric(
        "Adaptive Samples",
        adaptive_profile.samples
        if adaptive_profile is not None
        else profile["samples"]
    )

    feature_mean = np.asarray(
        profile[
            "feature_mean"
        ],
        dtype=np.float64
    )

    feature_df = pd.DataFrame(
        {
            "Feature": [
                f"F{i + 1}"
                for i in range(
                    len(feature_mean)
                )
            ],

            "Mean":
                feature_mean
        }
    )

    st.line_chart(
        feature_df.set_index(
            "Feature"
        )
    )


# =========================================================
# FUNCTIONAL CONNECTIVITY
# =========================================================

if profile is not None:

    st.divider()

    st.subheader(
        "🔗 Functional Connectivity"
    )

    st.write(
        """
NeuroTwin 3.0 estimates functional connectivity
from the preprocessed EEG using Phase Locking Value
(PLV) and frequency-band coherence.
"""
    )

    if st.button(
        "🔗 Compute PLV + Coherence Brain Network",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Computing functional connectivity..."
            ):

                connectivity_engine = (
                    FunctionalConnectivityEngine(
                        st.session_state.epochs
                    )
                )

                connectivity = (
                    connectivity_engine.compute_all()
                )

                combined = connectivity[
                    "combined"
                ]

                network_engine = BrainNetworkEngine(
                    combined,
                    connectivity_engine.channel_names
                )

                network_metrics = (
                    network_engine.metrics(
                        connectivity_threshold
                    )
                )

                digital_twin = DigitalBrainTwin(
                    subject=subject,
                    feature_vector=np.mean(
                        X,
                        axis=0
                    ),
                    connectivity=combined,
                    channel_names=
                        connectivity_engine.channel_names
                )

                st.session_state.connectivity_results = {
                    **connectivity,
                    "channel_names":
                        connectivity_engine.channel_names
                }

                st.session_state.brain_network_metrics = (
                    network_metrics
                )

                st.session_state.digital_twin = (
                    digital_twin
                )

            st.success(
                "Functional connectivity and personalized "
                "brain network created."
            )

        except Exception as e:

            st.error(
                f"Connectivity analysis failed: {e}"
            )


# =========================================================
# CONNECTIVITY RESULTS
# =========================================================

connectivity_results = (
    st.session_state.connectivity_results
)

if connectivity_results is not None:

    st.markdown(
        "### 🧠 Connectivity Overview"
    )

    plv = connectivity_results[
        "plv"
    ]

    coherence = connectivity_results[
        "coherence"
    ]

    combined = connectivity_results[
        "combined"
    ]

    channel_names = connectivity_results[
        "channel_names"
    ]

    a, b, c, d = st.columns(4)

    a.metric(
        "Channels",
        len(channel_names)
    )

    b.metric(
        "Mean PLV",
        f"{np.mean(plv):.4f}"
    )

    c.metric(
        "Mean Coherence",
        f"{np.mean(coherence):.4f}"
    )

    d.metric(
        "Mean Connectivity",
        f"{np.mean(combined):.4f}"
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "PLV",
            "Coherence",
            "Combined"
        ]
    )

    with tab1:

        st.dataframe(
            pd.DataFrame(
                plv,
                index=channel_names,
                columns=channel_names
            ).round(4),
            use_container_width=True
        )

    with tab2:

        st.dataframe(
            pd.DataFrame(
                coherence,
                index=channel_names,
                columns=channel_names
            ).round(4),
            use_container_width=True
        )

    with tab3:

        st.dataframe(
            pd.DataFrame(
                combined,
                index=channel_names,
                columns=channel_names
            ).round(4),
            use_container_width=True
        )


# =========================================================
# NETWORK METRICS
# =========================================================

network_metrics = (
    st.session_state.brain_network_metrics
)

if network_metrics is not None:

    st.divider()

    st.subheader(
        "🌐 Personalized Brain Network"
    )

    n1, n2, n3, n4, n5 = st.columns(5)

    n1.metric(
        "Nodes",
        network_metrics["nodes"]
    )

    n2.metric(
        "Edges",
        network_metrics["edges"]
    )

    n3.metric(
        "Density",
        f"{network_metrics['density']:.4f}"
    )

    n4.metric(
        "Mean Strength",
        f"{network_metrics['mean_strength']:.4f}"
    )

    n5.metric(
        "Clustering",
        f"{network_metrics['mean_clustering']:.4f}"
    )

    node_rows = []

    for i, name in enumerate(
        connectivity_results[
            "channel_names"
        ]
    ):

        node_rows.append(
            {
                "Channel":
                    name,

                "Degree":
                    network_metrics[
                        "degrees"
                    ].get(
                        i,
                        0
                    ),

                "Strength":
                    network_metrics[
                        "strength"
                    ].get(
                        i,
                        0.0
                    ),

                "Clustering":
                    network_metrics[
                        "clustering"
                    ].get(
                        i,
                        0.0
                    ),

                "Betweenness":
                    network_metrics[
                        "betweenness"
                    ].get(
                        i,
                        0.0
                    )
            }
        )

    node_df = pd.DataFrame(
        node_rows
    )

    st.dataframe(
        node_df.sort_values(
            "Strength",
            ascending=False
        ).round(5),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# BRAIN GRAPH
# =========================================================

if (
    connectivity_results is not None
    and network_metrics is not None
):

    st.subheader(
        "🕸️ Brain Connectivity Graph"
    )

    network_engine = BrainNetworkEngine(
        connectivity_results[
            "combined"
        ],
        connectivity_results[
            "channel_names"
        ]
    )

    graph = network_engine.graph(
        connectivity_threshold
    )

    if graph.number_of_edges() > 0:

        fig, ax = plt.subplots(
            figsize=(12, 8)
        )

        pos = nx.spring_layout(
            graph,
            seed=42,
            weight="weight"
        )

        node_sizes = [
            300 +
            1500 *
            network_metrics[
                "strength"
            ].get(
                node,
                0.0
            )
            for node in graph.nodes
        ]

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=node_sizes,
            ax=ax
        )

        nx.draw_networkx_edges(
            graph,
            pos,
            width=[
                0.5 +
                3.0 *
                data.get(
                    "weight",
                    0.0
                )
                for _, _, data
                in graph.edges(
                    data=True
                )
            ],
            alpha=0.45,
            ax=ax
        )

        labels = {
            node:
                connectivity_results[
                    "channel_names"
                ][node]
            for node in graph.nodes
        }

        nx.draw_networkx_labels(
            graph,
            pos,
            labels=labels,
            font_size=8,
            ax=ax
        )

        ax.set_title(
            "Subject-Specific EEG Functional Connectivity Network"
        )

        ax.axis(
            "off"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(
            fig
        )

    else:

        st.warning(
            "No edges passed the selected threshold. "
            "Try lowering the network threshold."
        )


# =========================================================
# NEURAL DYNAMICS
# =========================================================

if (
    st.session_state.digital_twin
    is not None
):

    st.divider()

    st.subheader(
        "⚙️ Neural Network Dynamics Simulation"
    )

    st.write(
        """
This module simulates mathematical network dynamics
on the subject-specific connectivity matrix. It is
a computational network model, not a validated
biophysical whole-brain model.
"""
    )

    if st.button(
        "⚙️ Run Neural Dynamics Simulation",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Simulating network dynamics..."
            ):

                history = (
                    st.session_state.digital_twin.simulate(
                        steps=neural_steps,
                        coupling=neural_coupling,
                        decay=0.12,
                        seed=int(seed)
                    )
                )

                st.session_state.neural_simulation = (
                    history
                )

            st.success(
                "Neural network simulation completed."
            )

        except Exception as e:

            st.error(
                f"Neural simulation failed: {e}"
            )


simulation = (
    st.session_state.neural_simulation
)

if simulation is not None:

    st.markdown(
        "### Dynamic Brain State"
    )

    sim_mean = np.mean(
        simulation,
        axis=1
    )

    sim_std = np.std(
        simulation,
        axis=1
    )

    dynamic_df = pd.DataFrame(
        {
            "Mean Network State":
                sim_mean,

            "State Variability":
                sim_std
        }
    )

    st.line_chart(
        dynamic_df
    )

    s1, s2, s3 = st.columns(3)

    s1.metric(
        "Simulation Steps",
        len(simulation)
    )

    s2.metric(
        "Initial Mean State",
        f"{sim_mean[0]:.4f}"
    )

    s3.metric(
        "Final Mean State",
        f"{sim_mean[-1]:.4f}"
    )


# =========================================================
# MRI ENGINE
# =========================================================

class MRIBrainEngine:
    """
    Structural MRI ingestion and feature extraction.

    This module intentionally does not claim anatomical connectivity from a
    structural MRI volume alone. True white-matter structural connectivity
    requires diffusion MRI/tractography or an appropriate atlas/connectome.
    """

    @staticmethod
    def load_uploaded(uploaded_file):
        name = str(getattr(uploaded_file, "name", "mri"))
        suffix = name.lower()
        raw = uploaded_file.getvalue()

        if suffix.endswith(".npy"):
            volume = np.load(io.BytesIO(raw), allow_pickle=False)
            affine = np.eye(4)
            voxel_sizes = (1.0, 1.0, 1.0)
            source_type = "NumPy volume"

        elif suffix.endswith(".npz"):
            archive = np.load(io.BytesIO(raw), allow_pickle=False)
            keys = list(archive.keys())
            if not keys:
                raise ValueError("The NPZ file does not contain an array.")
            volume = archive[keys[0]]
            affine = np.eye(4)
            voxel_sizes = (1.0, 1.0, 1.0)
            source_type = f"NumPy archive ({keys[0]})"

        elif suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
            try:
                import nibabel as nib
            except ImportError as exc:
                raise ImportError(
                    "NIfTI MRI support requires nibabel. Install it with "
                    "pip install nibabel and restart Streamlit."
                ) from exc

            with tempfile.NamedTemporaryFile(
                suffix=".nii.gz" if suffix.endswith(".nii.gz") else ".nii",
                delete=False
            ) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name

            try:
                image = nib.load(tmp_path)
                volume = np.asarray(image.get_fdata(dtype=np.float32))
                affine = np.asarray(image.affine, dtype=np.float64)
                zooms = tuple(float(x) for x in image.header.get_zooms()[:3])
                voxel_sizes = zooms if len(zooms) == 3 else (1.0, 1.0, 1.0)
                source_type = "NIfTI MRI"
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass

        else:
            raise ValueError(
                "Unsupported MRI format. Use .nii, .nii.gz, .npy or .npz."
            )

        volume = np.asarray(volume, dtype=np.float32)
        volume = np.squeeze(volume)
        if volume.ndim != 3:
            raise ValueError(
                f"MRI volume must be 3-D after squeezing; received shape {volume.shape}."
            )

        return {
            "volume": volume,
            "affine": affine,
            "voxel_sizes": voxel_sizes,
            "source_type": source_type,
            "filename": name,
        }

    @staticmethod
    def summarize(volume, voxel_sizes=(1.0, 1.0, 1.0)):
        x = np.asarray(volume, dtype=np.float32)
        finite = np.isfinite(x)
        values = x[finite]

        if values.size == 0:
            raise ValueError("MRI volume contains no finite voxel values.")

        nonzero = np.abs(values) > 1e-8
        nz = values[nonzero]
        if nz.size == 0:
            nz = values

        p05, p50, p95 = np.percentile(nz, [5, 50, 95])
        dynamic = max(float(p95 - p05), 1e-8)
        normalized = np.clip((x - p05) / dynamic, 0.0, 1.0)
        brain_mask = np.isfinite(x) & (normalized > 0.05)

        voxel_volume = float(np.prod(voxel_sizes))
        brain_volume_ml = float(brain_mask.sum() * voxel_volume / 1000.0)

        coords = np.argwhere(brain_mask)
        if len(coords):
            center_of_mass = coords.mean(axis=0).tolist()
        else:
            center_of_mass = [float(v) for v in np.array(x.shape) / 2.0]

        return {
            "shape": tuple(int(v) for v in x.shape),
            "voxel_sizes_mm": tuple(float(v) for v in voxel_sizes),
            "voxel_volume_mm3": voxel_volume,
            "finite_voxels": int(values.size),
            "nonzero_voxels": int(nonzero.sum()),
            "brain_mask_voxels": int(brain_mask.sum()),
            "brain_volume_ml_proxy": brain_volume_ml,
            "intensity_mean": float(np.mean(nz)),
            "intensity_std": float(np.std(nz)),
            "intensity_p05": float(p05),
            "intensity_median": float(p50),
            "intensity_p95": float(p95),
            "center_of_mass_voxel": center_of_mass,
            "normalized_volume": normalized,
            "brain_mask": brain_mask,
        }

    @staticmethod
    def feature_vector(summary):
        return np.asarray([
            summary["intensity_mean"],
            summary["intensity_std"],
            summary["intensity_p05"],
            summary["intensity_median"],
            summary["intensity_p95"],
            summary["brain_volume_ml_proxy"],
            *summary["center_of_mass_voxel"],
        ], dtype=np.float64)


# =========================================================
# WHOLE-BRAIN COMPUTATIONAL MODEL
# =========================================================

class WholeBrainModelEngine:
    """
    Network-level whole-brain computational prototype.

    Uses a Hopf/oscillator-style coupled network. When MRI is supplied, its
    global structural features modulate node parameters; this is NOT an
    anatomical connectome and is not a validated biophysical model.
    """

    @staticmethod
    def _prepare_matrix(matrix):
        w = np.asarray(matrix, dtype=np.float64)
        if w.ndim != 2 or w.shape[0] != w.shape[1]:
            raise ValueError("Whole-brain connectivity must be a square matrix.")
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = 0.5 * (w + w.T)
        np.fill_diagonal(w, 0.0)
        mx = float(np.max(np.abs(w))) if w.size else 0.0
        return w / mx if mx > 0 else w

    @staticmethod
    def simulate(connectivity, steps=200, coupling=0.08, seed=42,
                 mri_features=None, dt=0.05):
        w = WholeBrainModelEngine._prepare_matrix(connectivity)
        n = w.shape[0]
        steps = int(max(2, steps))
        rng = np.random.default_rng(int(seed))

        if mri_features is None:
            node_gain = np.ones(n, dtype=np.float64)
        else:
            f = np.asarray(mri_features, dtype=np.float64).reshape(-1)
            f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
            if len(f) == 0 or np.std(f) < 1e-12:
                node_gain = np.ones(n, dtype=np.float64)
            else:
                f = (f - np.mean(f)) / (np.std(f) + 1e-8)
                node_gain = 1.0 + 0.10 * np.tanh(np.interp(
                    np.linspace(0, len(f) - 1, n),
                    np.arange(len(f)),
                    f
                ))

        x = rng.normal(0.0, 0.05, size=n)
        y = rng.normal(0.0, 0.05, size=n)
        history = np.zeros((steps, n), dtype=np.float64)
        omega = np.linspace(0.8, 1.2, n)

        for t in range(steps):
            radius2 = x * x + y * y
            coupling_x = w @ x - x * np.sum(w, axis=1)
            coupling_y = w @ y - y * np.sum(w, axis=1)
            drive = node_gain - 1.0

            dx = (node_gain - radius2) * x - omega * y
            dy = (node_gain - radius2) * y + omega * x
            dx += float(coupling) * coupling_x + 0.05 * drive
            dy += float(coupling) * coupling_y

            x = x + float(dt) * dx
            y = y + float(dt) * dy
            history[t] = np.tanh(np.sqrt(x * x + y * y))

        return history


# =========================================================
# CLINICAL VALIDATION ENGINE
# =========================================================

class ClinicalValidationEngine:
    """Metrics for an externally supplied clinical validation table."""

    @staticmethod
    def _binary_confusion(y_true, y_pred):
        labels = list(pd.unique(pd.concat([
            pd.Series(y_true), pd.Series(y_pred)
        ], ignore_index=True)))
        if len(labels) != 2:
            return None
        positive = labels[-1]
        negative = labels[0]
        yt = np.asarray(y_true) == positive
        yp = np.asarray(y_pred) == positive
        tp = int(np.sum(yt & yp))
        tn = int(np.sum(~yt & ~yp))
        fp = int(np.sum(~yt & yp))
        fn = int(np.sum(yt & ~yp))
        return tp, tn, fp, fn, positive, negative

    @staticmethod
    def _safe_div(a, b):
        return float(a / b) if b else 0.0

    @classmethod
    def evaluate(cls, df, y_true_col, y_pred_col, y_prob_col=None,
                 bootstrap=1000, seed=42):
        data = df[[y_true_col, y_pred_col]].copy()
        if y_prob_col and y_prob_col in df.columns:
            data["__prob__"] = pd.to_numeric(df[y_prob_col], errors="coerce")

        data = data.dropna(subset=[y_true_col, y_pred_col])
        if len(data) < 2:
            raise ValueError("At least two complete validation rows are required.")

        y_true = data[y_true_col].to_numpy()
        y_pred = data[y_pred_col].to_numpy()
        accuracy = float(np.mean(y_true == y_pred))

        classes = list(pd.unique(pd.concat([
            pd.Series(y_true), pd.Series(y_pred)
        ], ignore_index=True)))
        per_class_f1 = []
        per_class_recall = []
        for label in classes:
            yt = y_true == label
            yp = y_pred == label
            tp = np.sum(yt & yp)
            fp = np.sum(~yt & yp)
            fn = np.sum(yt & ~yp)
            precision = cls._safe_div(tp, tp + fp)
            recall = cls._safe_div(tp, tp + fn)
            f1 = cls._safe_div(2 * precision * recall, precision + recall)
            per_class_recall.append(recall)
            per_class_f1.append(f1)

        result = {
            "n": int(len(data)),
            "classes": classes,
            "accuracy": accuracy,
            "balanced_accuracy": float(np.mean(per_class_recall)),
            "macro_f1": float(np.mean(per_class_f1)),
            "confusion_matrix": pd.crosstab(
                pd.Series(y_true, name="True"),
                pd.Series(y_pred, name="Predicted"),
                dropna=False
            ),
        }

        binary = cls._binary_confusion(y_true, y_pred)
        if binary is not None:
            tp, tn, fp, fn, positive, negative = binary
            sensitivity = cls._safe_div(tp, tp + fn)
            specificity = cls._safe_div(tn, tn + fp)
            precision = cls._safe_div(tp, tp + fp)
            npv = cls._safe_div(tn, tn + fn)
            f1 = cls._safe_div(2 * precision * sensitivity, precision + sensitivity)
            result.update({
                "positive_class": positive,
                "negative_class": negative,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "precision": precision,
                "npv": npv,
                "f1": f1,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            })

            if "__prob__" in data.columns:
                prob = data["__prob__"].to_numpy(dtype=float)
                mask = np.isfinite(prob)
                if np.sum(mask) == len(prob):
                    order = np.argsort(prob)
                    ranks = np.empty_like(order, dtype=float)
                    ranks[order] = np.arange(1, len(prob) + 1)
                    positive_mask = y_true == positive
                    n_pos = int(np.sum(positive_mask))
                    n_neg = int(len(prob) - n_pos)
                    if n_pos > 0 and n_neg > 0:
                        auc = (np.sum(ranks[positive_mask]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
                        result["roc_auc"] = float(auc)

        rng = np.random.default_rng(int(seed))
        boot = []
        for _ in range(int(max(0, bootstrap))):
            idx = rng.integers(0, len(data), size=len(data))
            boot.append(float(np.mean(y_true[idx] == y_pred[idx])))
        result["accuracy_ci95"] = (
            tuple(np.percentile(boot, [2.5, 97.5]).tolist())
            if boot else None
        )
        return result


# =========================================================
# MRI / WHOLE-BRAIN / CLINICAL VALIDATION
# =========================================================

st.divider()
st.subheader("🧲 MRI Integration")
st.write(
    "Upload a structural MRI volume for subject-level MRI feature extraction. "
    "The module accepts NIfTI (.nii/.nii.gz) and NumPy (.npy/.npz) volumes."
)

mri_file = st.file_uploader(
    "Structural MRI",
    type=["nii", "nii.gz", "npy", "npz"],
    key="neurotwin_mri_upload"
)

if mri_file is not None:
    if st.button(
        "🧲 Process MRI",
        use_container_width=True
    ):
        try:
            with st.spinner("Loading and extracting MRI features..."):
                mri_loaded = MRIBrainEngine.load_uploaded(mri_file)
                mri_summary = MRIBrainEngine.summarize(
                    mri_loaded["volume"],
                    mri_loaded["voxel_sizes"]
                )
                st.session_state.mri_volume = mri_loaded["volume"]
                st.session_state.mri_summary = mri_summary
                st.session_state.mri_features = MRIBrainEngine.feature_vector(
                    mri_summary
                )
                st.session_state.mri_filename = mri_loaded["filename"]
            st.success("MRI feature extraction completed.")
        except Exception as e:
            st.error(f"MRI processing failed: {e}")

mri_summary = st.session_state.mri_summary

if mri_summary is not None:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MRI Shape", " × ".join(map(str, mri_summary["shape"])))
    m2.metric("Voxel Size", " × ".join(f"{v:.2f}" for v in mri_summary["voxel_sizes_mm"]))
    m3.metric("MRI Volume Proxy", f"{mri_summary['brain_volume_ml_proxy']:.1f} mL")
    m4.metric("Intensity Std", f"{mri_summary['intensity_std']:.4f}")

    st.dataframe(
        pd.DataFrame([
            ["File", st.session_state.mri_filename],
            ["Source", "Structural MRI volume"],
            ["Finite voxels", mri_summary["finite_voxels"]],
            ["Non-zero voxels", mri_summary["nonzero_voxels"]],
            ["P05 intensity", mri_summary["intensity_p05"]],
            ["Median intensity", mri_summary["intensity_median"]],
            ["P95 intensity", mri_summary["intensity_p95"]],
            ["Center of mass (voxel)", str(mri_summary["center_of_mass_voxel"])],
        ], columns=["MRI Parameter", "Value"]),
        use_container_width=True,
        hide_index=True
    )

    volume = st.session_state.mri_volume
    z = volume.shape[2] // 2
    y = volume.shape[1] // 2
    x = volume.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(volume[:, :, z].T, cmap="gray", origin="lower")
    axes[0].set_title("Axial")
    axes[1].imshow(volume[:, y, :].T, cmap="gray", origin="lower")
    axes[1].set_title("Coronal")
    axes[2].imshow(volume[x, :, :].T, cmap="gray", origin="lower")
    axes[2].set_title("Sagittal")
    for ax in axes:
        ax.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# =========================================================
# WHOLE-BRAIN MODEL
# =========================================================

st.divider()
st.subheader("🌐 Whole-Brain Computational Model")
st.write(
    "This module runs a network-level coupled oscillator model on the "
    "subject-specific connectivity matrix. If MRI features are available, "
    "they modulate model parameters as an MRI-informed computational prior. "
    "This is not a validated anatomical connectome or biophysical model."
)

whole_brain_ready = (
    st.session_state.connectivity_results is not None
)

if not whole_brain_ready:
    st.info("Build Functional Connectivity first to enable the Whole-Brain Model.")
else:
    if st.button(
        "🌐 Run Whole-Brain Model",
        use_container_width=True
    ):
        try:
            matrix = st.session_state.connectivity_results["combined"]
            mri_features = st.session_state.mri_features
            with st.spinner("Running whole-brain computational simulation..."):
                wb_history = WholeBrainModelEngine.simulate(
                    matrix,
                    steps=int(neural_steps),
                    coupling=float(neural_coupling),
                    seed=int(seed),
                    mri_features=mri_features
                )
                st.session_state.whole_brain_simulation = wb_history
            st.success("Whole-brain computational simulation completed.")
        except Exception as e:
            st.error(f"Whole-brain model failed: {e}")

whole_brain_simulation = st.session_state.whole_brain_simulation
if whole_brain_simulation is not None:
    wb_mean = np.mean(whole_brain_simulation, axis=1)
    wb_std = np.std(whole_brain_simulation, axis=1)
    wb_df = pd.DataFrame({
        "Whole-Brain Mean State": wb_mean,
        "Whole-Brain State Variability": wb_std
    })
    st.line_chart(wb_df)
    w1, w2, w3 = st.columns(3)
    w1.metric("Model Nodes", whole_brain_simulation.shape[1])
    w2.metric("Initial State", f"{wb_mean[0]:.4f}")
    w3.metric("Final State", f"{wb_mean[-1]:.4f}")


# =========================================================
# CLINICAL VALIDATION
# =========================================================

st.divider()
st.subheader("🧪 Clinical Validation")
st.write(
    "Upload an independent validation CSV containing observed clinical labels "
    "and model predictions. The dashboard reports statistical validation "
    "metrics; uploading data does not make the system clinically validated."
)

clinical_file = st.file_uploader(
    "Clinical validation CSV",
    type=["csv"],
    key="neurotwin_clinical_validation_upload"
)

if clinical_file is not None:
    try:
        clinical_df = pd.read_csv(clinical_file)
        columns = list(clinical_df.columns)
        if len(columns) >= 2:
            cc1, cc2, cc3 = st.columns(3)
            y_true_col = cc1.selectbox(
                "Observed clinical label",
                columns,
                key="clinical_y_true"
            )
            y_pred_col = cc2.selectbox(
                "Model prediction",
                columns,
                index=1 if len(columns) > 1 else 0,
                key="clinical_y_pred"
            )
            probability_options = ["None"] + columns
            y_prob_choice = cc3.selectbox(
                "Optional positive-class probability",
                probability_options,
                key="clinical_y_prob"
            )

            st.dataframe(
                clinical_df.head(10),
                use_container_width=True,
                hide_index=True
            )

            if st.button(
                "🧪 Run Clinical Validation",
                use_container_width=True
            ):
                try:
                    result = ClinicalValidationEngine.evaluate(
                        clinical_df,
                        y_true_col=y_true_col,
                        y_pred_col=y_pred_col,
                        y_prob_col=None if y_prob_choice == "None" else y_prob_choice,
                        bootstrap=1000,
                        seed=int(seed)
                    )
                    st.session_state.clinical_validation = result
                    st.success(
                        "Clinical validation metrics calculated from the supplied dataset."
                    )
                except Exception as e:
                    st.error(f"Clinical validation failed: {e}")
        else:
            st.warning("The validation CSV needs at least two columns.")
    except Exception as e:
        st.error(f"Could not read the clinical validation CSV: {e}")

clinical_validation = st.session_state.clinical_validation
if clinical_validation is not None:
    cv1, cv2, cv3, cv4 = st.columns(4)
    cv1.metric("Validation N", clinical_validation["n"])
    cv2.metric("Accuracy", f"{clinical_validation['accuracy']:.3f}")
    cv3.metric("Balanced Accuracy", f"{clinical_validation['balanced_accuracy']:.3f}")
    cv4.metric("Macro F1", f"{clinical_validation['macro_f1']:.3f}")

    binary_keys = [
        "sensitivity", "specificity", "precision", "npv", "f1", "roc_auc"
    ]
    binary_present = [k for k in binary_keys if k in clinical_validation]
    if binary_present:
        metric_cols = st.columns(len(binary_present))
        for col, key in zip(metric_cols, binary_present):
            col.metric(key.replace("_", " ").title(), f"{clinical_validation[key]:.3f}")

    ci = clinical_validation.get("accuracy_ci95")
    if ci is not None:
        st.caption(f"Bootstrap 95% CI for accuracy: {ci[0]:.3f}–{ci[1]:.3f}")

    st.markdown("### Clinical Validation Confusion Matrix")
    st.dataframe(
        clinical_validation["confusion_matrix"],
        use_container_width=True
    )

# =========================================================
# V3.0 ADAPTIVE ENGINE
# =========================================================

st.divider()

st.subheader(
    "🚀 NeuroTwin 3.0 Computational Adaptation Engine"
)

st.write(
    """
The v3.0 simulation engine combines the previous
multi-stage computational model with the personalized
brain-state representation.
"""
)

if profile is None:

    st.info(
        "Build the Personalized NeuroTwin Profile first."
    )

else:

    st.warning(
        "Strategy A/B/C are mathematical simulation "
        "strategies. Their scores must not be interpreted "
        "as clinical treatment rankings."
    )

    if st.button(
        "🚀 Run Full NeuroTwin 3.0 Engine",
        type="primary",
        use_container_width=True
    ):

        baseline_vector = np.mean(
            np.asarray(
                X,
                dtype=np.float64
            ),
            axis=0
        )

        engine = NeuroTwinV30Engine(
            baseline=baseline_vector,
            intensity=float(
                intensity
            ),
            repetitions=int(
                repetitions
            ),
            stages=int(
                stages
            ),
            seed=int(
                seed
            ),
            robustness_runs=int(
                robustness_runs
            )
        )

        with st.spinner(
            "Running multi-stage, sensitivity and "
            "multi-seed computational analysis..."
        ):

            multistage = (
                engine.run_multistage()
            )

            sensitivity = (
                engine.sensitivity_analysis(
                    baseline_vector
                )
            )

            robustness = (
                engine.robustness_evaluation(
                    baseline_vector,
                    float(
                        intensity
                    ),
                    stage=1
                )
            )

        st.session_state.engine = engine
        st.session_state.v30_results = multistage
        st.session_state.sensitivity_results = sensitivity
        st.session_state.robustness_results = robustness
        st.session_state.adaptive_updated = False

        st.success(
            "NeuroTwin 3.0 computational engine completed."
        )


# =========================================================
# V3 RESULTS
# =========================================================

results = (
    st.session_state.v30_results
)

if results:

    st.divider()

    st.subheader(
        "📈 Multi-Stage Computational Results"
    )

    summary_rows = []

    for record in results:

        selected = record[
            "selected"
        ]

        summary_rows.append(
            {
                "Stage":
                    record["stage"],

                "Selected Strategy":
                    selected["Strategy"],

                "Similarity":
                    selected["Similarity"],

                "Stability":
                    selected["Stability"],

                "Feature Shift":
                    selected["Feature Shift"],

                "Pattern Preservation":
                    selected[
                        "Pattern Preservation"
                    ],

                "Compatibility":
                    selected[
                        "Response Index"
                    ],

                "Consistency":
                    selected[
                        "Consistency"
                    ],

                "Uncertainty":
                    selected[
                        "Uncertainty"
                    ],

                "Adaptation Potential":
                    selected[
                        "Adaptation Potential"
                    ],

                "Adjusted Response":
                    selected[
                        "Adjusted Response"
                    ]
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    st.dataframe(
        summary_df.round(5),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# STAGE DETAILS
# =========================================================

if results:

    st.subheader(
        "🔬 Stage-by-Stage Analysis"
    )

    for record in results:

        stage = record[
            "stage"
        ]

        selected = record[
            "selected"
        ]

        with st.expander(
            f"Stage {stage} — "
            f"{selected['Strategy']}",
            expanded=(
                stage == 1
            )
        ):

            rows = []

            for r in record[
                "results"
            ]:

                rows.append(
                    {
                        "Strategy":
                            r["Strategy"],

                        "Similarity":
                            r["Similarity"],

                        "Stability":
                            r["Stability"],

                        "Feature Shift":
                            r["Feature Shift"],

                        "Pattern Preservation":
                            r[
                                "Pattern Preservation"
                            ],

                        "Compatibility":
                            r[
                                "Response Index"
                            ],

                        "Response Std":
                            r[
                                "Response Std"
                            ],

                        "CI95":
                            r[
                                "CI95"
                            ],

                        "Consistency":
                            r[
                                "Consistency"
                            ],

                        "Uncertainty":
                            r[
                                "Uncertainty"
                            ],

                        "Adaptation Potential":
                            r[
                                "Adaptation Potential"
                            ],

                        "Adjusted":
                            r[
                                "Adjusted Response"
                            ]
                    }
                )

            stage_df = pd.DataFrame(
                rows
            )

            st.dataframe(
                stage_df.round(5),
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                stage_df[
                    [
                        "Strategy",
                        "Similarity",
                        "Stability",
                        "Compatibility",
                        "Adjusted"
                    ]
                ].set_index(
                    "Strategy"
                )
            )


# =========================================================
# MULTI-SEED
# =========================================================

robustness = (
    st.session_state.robustness_results
)

if robustness is not None:

    st.divider()

    st.subheader(
        "🧪 Exact Multi-Seed Robustness"
    )

    detailed = robustness[
        "detailed"
    ]

    summary = robustness[
        "summary"
    ]

    st.dataframe(
        summary.round(6),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        "### Seed-by-Seed Response"
    )

    seed_chart = (
        detailed
        .pivot(
            index="Seed",
            columns="Strategy",
            values="Adjusted Response"
        )
    )

    st.line_chart(
        seed_chart
    )

    st.markdown(
        "### Detailed Seed Results"
    )

    st.dataframe(
        detailed.round(6),
        use_container_width=True,
        hide_index=True
    )

    robustness_csv = (
        detailed
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )

    st.download_button(
        "⬇️ Download v3.0 Seed Robustness CSV",
        data=robustness_csv,
        file_name=(
            f"NeuroTwin_v3_0_Robustness_"
            f"Subject_{subject:02d}.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )


# =========================================================
# TRAJECTORY
# =========================================================

if results:

    st.divider()

    st.subheader(
        "🔄 Computational Adaptation Trajectory"
    )

    trajectory_rows = []

    for record in results:

        before = record[
            "baseline_before"
        ]

        after = record[
            "baseline_after"
        ]

        trajectory_rows.append(
            {
                "Stage":
                    record["stage"],

                "Selected":
                    record[
                        "selected"
                    ]["Strategy"],

                "State Shift":
                    relative_shift(
                        before,
                        after
                    ),

                "State Similarity":
                    normalize_cosine(
                        before,
                        after
                    ),

                "Pattern Preservation":
                    pattern_preservation(
                        before,
                        after
                    )
            }
        )

    trajectory_df = pd.DataFrame(
        trajectory_rows
    )

    st.dataframe(
        trajectory_df.round(5),
        use_container_width=True,
        hide_index=True
    )

    st.line_chart(
        trajectory_df.set_index(
            "Stage"
        )
    )


# =========================================================
# FINAL STATE
# =========================================================

if results:

    final_record = results[
        -1
    ]

    final_selected = (
        final_record[
            "selected"
        ]
    )

    st.subheader(
        "🏁 Final Computational State"
    )

    f1, f2, f3, f4 = st.columns(4)

    f1.metric(
        "Final Stage",
        final_record[
            "stage"
        ]
    )

    f2.metric(
        "Model-Selected Strategy",
        final_selected[
            "Strategy"
        ]
    )

    f3.metric(
        "Adjusted Response",
        f"{final_selected['Adjusted Response']:.5f}"
    )

    f4.metric(
        "Adaptation Potential",
        f"{final_selected['Adaptation Potential']:.5f}"
    )

    st.info(
        """
The displayed strategy is selected by the mathematical
computational model under the current simulation settings.

This result does not establish clinical superiority,
treatment effectiveness, rehabilitation effectiveness,
or patient outcome prediction.
"""
    )


# =========================================================
# SENSITIVITY
# =========================================================

sensitivity = (
    st.session_state.sensitivity_results
)

if sensitivity is not None:

    st.divider()

    st.subheader(
        "📊 Sensitivity Analysis"
    )

    response_pivot = sensitivity.pivot(
        index="Intensity",
        columns="Strategy",
        values="Response"
    )

    st.line_chart(
        response_pivot
    )

    st.dataframe(
        sensitivity.round(5),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# GLOBAL STRATEGY COMPARISON
# =========================================================

if results:

    st.divider()

    st.subheader(
        "🥊 Global Computational Strategy Comparison"
    )

    all_rows = []

    for record in results:

        for r in record[
            "results"
        ]:

            all_rows.append(
                {
                    "Stage":
                        record["stage"],

                    "Strategy":
                        r["Strategy"],

                    "Compatibility":
                        r["Adjusted Response"],

                    "Similarity":
                        r["Similarity"],

                    "Stability":
                        r["Stability"],

                    "Feature Shift":
                        r["Feature Shift"],

                    "Pattern Preservation":
                        r[
                            "Pattern Preservation"
                        ],

                    "Consistency":
                        r[
                            "Consistency"
                        ],

                    "Uncertainty":
                        r[
                            "Uncertainty"
                        ]
                }
            )

    global_df = pd.DataFrame(
        all_rows
    )

    global_summary = (
        global_df
        .groupby(
            "Strategy"
        )
        .mean(
            numeric_only=True
        )
        .reset_index()
    )

    st.dataframe(
        global_summary.round(5),
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        global_summary.set_index(
            "Strategy"
        )[
            [
                "Compatibility",
                "Similarity",
                "Stability",
                "Consistency"
            ]
        ]
    )


# =========================================================
# INITIAL VS FINAL
# =========================================================

if results:

    st.divider()

    st.subheader(
        "🧬 Initial vs Final Computational Profile"
    )

    initial = np.asarray(
        results[
            0
        ][
            "baseline_before"
        ],
        dtype=np.float64
    )

    final = np.asarray(
        results[
            -1
        ][
            "baseline_after"
        ],
        dtype=np.float64
    )

    comparison = pd.DataFrame(
        {
            "Initial":
                initial,

            "Final":
                final
        }
    )

    st.line_chart(
        comparison
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Feature Shift",
        f"{relative_shift(initial, final):.5f}"
    )

    c2.metric(
        "Initial → Final Similarity",
        f"{normalize_cosine(initial, final):.5f}"
    )

    c3.metric(
        "Pattern Preservation",
        f"{pattern_preservation(initial, final):.5f}"
    )


# =========================================================
# ADAPTIVE PROFILE
# =========================================================

if (
    results
    and adaptive_twin is not None
    and not st.session_state.adaptive_updated
):

    st.divider()

    st.subheader(
        "🔄 Adaptive NeuroTwin Profile"
    )

    final_vector = np.asarray(
        results[
            -1
        ][
            "baseline_after"
        ],
        dtype=np.float64
    )

    try:

        before_profile = (
            adaptive_twin.profiles.get(
                subject
            )
        )

        previous_samples = (
            before_profile.samples
            if before_profile is not None
            else profile["samples"]
        )

        adaptive_twin.update_profile(
            subject,
            final_vector.reshape(
                1,
                -1
            )
        )

        updated_profile = (
            adaptive_twin.profiles.get(
                subject
            )
        )

        if updated_profile is not None:

            u1, u2, u3 = st.columns(3)

            u1.metric(
                "Previous Samples",
                previous_samples
            )

            u2.metric(
                "Updated Samples",
                updated_profile.samples
            )

            u3.metric(
                "Updated Stability",
                f"{updated_profile.stability:.4f}"
            )

            st.caption(
                "The added observation is computationally "
                "generated and is not a real post-intervention "
                "EEG measurement."
            )

            st.session_state.adaptive_updated = True

    except Exception as e:

        st.warning(
            f"Adaptive profile update failed: {e}"
        )


# =========================================================
# 10-SUBJECT GENERALIZATION
# =========================================================

st.divider()

st.subheader(
    "🌍 10-Subject Cross-Subject Generalization"
)

st.write(
    """
This experiment evaluates whether the computational
patterns observed in the NeuroTwin model remain
reasonably consistent across Subjects 01–10.

This is cross-subject computational generalization,
not clinical generalization.
"""
)

generalization_repetitions = st.slider(
    "Generalization Repetitions / Strategy",
    5,
    50,
    20,
    5,
    key="generalization_repetitions"
)

if st.button(
    "🌍 Run 10-Subject Generalization",
    use_container_width=True
):

    try:

        with st.spinner(
            "Running cross-subject computational generalization..."
        ):

            generalization = (
                run_cross_subject_generalization(
                    subjects=list(
                        range(
                            1,
                            11
                        )
                    ),
                    repetitions=int(
                        generalization_repetitions
                    ),
                    seeds=int(
                        robustness_runs
                    ),
                    intensity=float(
                        intensity
                    )
                )
            )

            st.session_state.generalization_results = (
                generalization
            )

            detailed = generalization[
                "detailed"
            ]

            if not detailed.empty:

                detailed.to_csv(
                    GENERALIZATION_DIR /
                    "neurotwin_v3_0_10_subject_generalization.csv",
                    index=False
                )

        st.success(
            "10-subject generalization completed."
        )

    except Exception as e:

        st.error(
            f"Generalization failed: {e}"
        )


generalization = (
    st.session_state.generalization_results
)

if generalization is not None:

    detailed = generalization[
        "detailed"
    ]

    summary = generalization[
        "summary"
    ]

    st.markdown(
        "### Subject-Level Results"
    )

    st.dataframe(
        detailed.round(5),
        use_container_width=True,
        hide_index=True
    )

    if not summary.empty:

        st.markdown(
            "### Cross-Subject Aggregate"
        )

        st.dataframe(
            summary.round(5),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            "### Adjusted Response Across Subjects"
        )

        chart = (
            detailed[
                detailed["Strategy"] !=
                "ERROR"
            ]
            .pivot(
                index="Subject",
                columns="Strategy",
                values="Adjusted Response"
            )
        )

        st.line_chart(
            chart
        )

        st.markdown(
            "### Cross-Subject Similarity"
        )

        similarity_chart = (
            detailed[
                detailed["Strategy"] !=
                "ERROR"
            ]
            .pivot(
                index="Subject",
                columns="Strategy",
                values="Similarity"
            )
        )

        st.line_chart(
            similarity_chart
        )

        generalization_csv = (
            detailed
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        )

        st.download_button(
            "⬇️ Download 10-Subject Generalization CSV",
            data=generalization_csv,
            file_name=(
                "NeuroTwin_v3_0_10_Subject_Generalization.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )


# =========================================================
# RESEARCH EXPORT
# =========================================================

if results:

    st.divider()

    st.subheader(
        "📥 Research Export"
    )

    export_rows = []

    for record in results:

        for r in record[
            "results"
        ]:

            export_rows.append(
                {
                    "Version":
                        "3.0",

                    "Subject":
                        subject,

                    "Stage":
                        r["Stage"],

                    "Strategy":
                        r["Strategy"],

                    "Similarity":
                        r["Similarity"],

                    "Similarity_Std":
                        r["Similarity Std"],

                    "Stability":
                        r["Stability"],

                    "Stability_Std":
                        r["Stability Std"],

                    "Feature_Shift":
                        r["Feature Shift"],

                    "Feature_Shift_Std":
                        r["Feature Shift Std"],

                    "Pattern_Preservation":
                        r[
                            "Pattern Preservation"
                        ],

                    "Compatibility_Index":
                        r[
                            "Response Index"
                        ],

                    "Response_Std":
                        r[
                            "Response Std"
                        ],

                    "CI95":
                        r["CI95"],

                    "Consistency":
                        r[
                            "Consistency"
                        ],

                    "Uncertainty":
                        r[
                            "Uncertainty"
                        ],

                    "Adaptation_Potential":
                        r[
                            "Adaptation Potential"
                        ],

                    "Adjusted_Response":
                        r[
                            "Adjusted Response"
                        ],

                    "Intensity":
                        intensity,

                    "Repetitions":
                        repetitions,

                    "Experiment_Seeds":
                        robustness_runs,

                    "Main_Seed":
                        seed
                }
            )

    export_df = pd.DataFrame(
        export_rows
    )

    csv_data = (
        export_df
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )

    st.download_button(
        "⬇️ Download NeuroTwin 3.0 Research CSV",
        data=csv_data,
        file_name=(
            f"NeuroTwin_v3_0_Subject_"
            f"{subject:02d}.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )


# =========================================================
# RESEARCH STATUS
# =========================================================

st.divider()

st.subheader(
    "🔬 NeuroTwin 3.0 Research Status"
)

connectivity_status = (
    "✓ Complete"
    if connectivity_results is not None
    else "○ Ready"
)

network_status = (
    "✓ Complete"
    if network_metrics is not None
    else "○ Ready"
)

dynamics_status = (
    "✓ Complete"
    if simulation is not None
    else "○ Ready"
)

generalization_status = (
    "✓ Complete"
    if generalization is not None
    else "○ Ready"
)

status = pd.DataFrame(
    {
        "Component": [

            "Public EEG Dataset",

            "EEG Preprocessing",

            "Feature Extraction",

            "Baseline Model",

            "CSP + LDA Evaluation",

            "Subject-wise Cross Validation",

            "Personalized NeuroTwin",

            "Adaptive Profile",

            "Feature-Aware Simulation",

            "Repeated Simulation",

            "Multi-Seed Robustness",

            "Sensitivity Analysis",

            "Multi-Stage Engine",

            "Scientific Feature Distance",

            "Confidence Interval",

            "Uncertainty Analysis",

            "Adaptive State Update",

            "Functional Connectivity",

            "PLV",

            "Band Coherence",

            "Connectivity Matrix",

            "Brain Network Graph",

            "Network Metrics",

            "Neural Dynamics",

            "Digital Brain Twin",

            "10-Subject Generalization",

            "Structural MRI",

            "Whole-Brain Computational Model",

            "Clinical Validation"
        ],

        "Status": [

            "✓ Complete",

            "✓ Complete",

            "✓ Complete",

            "✓ Complete",

            "✓ Complete",

            "✓ Complete",

            "✓ Complete",

            "✓ Connected",

            "✓ Connected",

            "✓ Connected",

            "✓ Connected",

            "✓ Connected",

            "✓ Connected",

            "✓ Connected",

            "✓ Connected",

            "✓ Connected",

            "✓ Prototype",

            connectivity_status,

            connectivity_status,

            connectivity_status,

            connectivity_status,

            network_status,

            network_status,

            dynamics_status,

            "✓ Connected",

            generalization_status,

            "✓ Complete" if mri_summary is not None else "○ Ready",

            "✓ Complete" if whole_brain_simulation is not None else "○ Ready",

            "✓ Complete" if clinical_validation is not None else "○ Awaiting validation data"
        ]
    }
)

st.dataframe(
    status,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# COMPUTATIONAL INTERPRETATION
# =========================================================

st.divider()

st.subheader(
    "🧠 NeuroTwin 3.0 Computational Interpretation"
)

st.write(
    """
NeuroTwin 3.0 extends the previous EEG feature-based
prototype into a computational brain-network framework.

The current architecture combines:

• EEG preprocessing

• Spectral and spatial EEG features

• Subject-specific feature representation

• Functional connectivity

• Phase Locking Value (PLV)

• Frequency-band coherence

• Connectivity matrices

• Graph-based brain-network representation

• Degree, strength, clustering and betweenness

• Mathematical neural-network dynamics

• Digital Brain Twin computational state

• Multi-stage adaptive simulation

• Multi-seed robustness

• Sensitivity analysis

• Uncertainty estimation

• Cross-subject computational generalization

• Structural MRI feature extraction

• MRI-informed whole-brain computational modeling

• External clinical validation metrics
"""
)

st.warning(
    """
Important scientific limitation:

NeuroTwin 3.0 is a computational research prototype.

The functional connectivity estimates describe
relationships in the recorded EEG signals. They do not
prove anatomical connections between brain regions.

The neural dynamics module is a mathematical network
simulation and is NOT a validated biophysical whole-brain
model.

Strategy A, B and C are mathematical simulation
strategies, not validated rehabilitation interventions.

Higher computational scores must not be interpreted as
medical superiority.

The current system does not diagnose disease, predict
individual clinical recovery, prescribe treatment, or
establish rehabilitation effectiveness.

Clinical translation would require appropriate clinical
datasets, real intervention and post-intervention data,
independent validation, prospective study design,
appropriate statistical testing, and clinical oversight.

Structural MRI alone does not establish white-matter structural connectivity.
The Whole-Brain Model is therefore a network-level computational model unless
an appropriate subject-specific connectome is supplied.

The Clinical Validation module only computes validation statistics from the
uploaded labels and predictions. It does not establish clinical validity,
clinical utility, or treatment effectiveness by itself.
"""
)


# =========================================================
# METADATA
# =========================================================

with st.expander(
    "📋 Experiment Metadata"
):

    metadata = {

        "Version":
            "NeuroTwin 3.0",

        "Subject":
            f"{subject:02d}",

        "Dataset":
            "PhysioNet EEGMMIDB",

        "Stages":
            stages,

        "Repetitions / Strategy":
            repetitions,

        "Experiment Seeds":
            f"1–{robustness_runs}",

        "Simulation Intensity":
            intensity,

        "Main Seed":
            seed,

        "Connectivity":
            "PLV + 8–30 Hz Band Coherence",

        "Network Threshold":
            connectivity_threshold,

        "Neural Simulation Steps":
            neural_steps,

        "Neural Coupling":
            neural_coupling,

        "Features":
            (
                len(
                    profile[
                        "feature_mean"
                    ]
                )
                if profile is not None
                else 0
            ),

        "Connectivity Channels":
            (
                len(
                    connectivity_results[
                        "channel_names"
                    ]
                )
                if connectivity_results is not None
                else 0
            ),

        "MRI File":
            st.session_state.mri_filename or "None",

        "MRI Integration":
            "Structural MRI features" if mri_summary is not None else "Not loaded",

        "Whole-Brain Model":
            "MRI-informed network simulation" if whole_brain_simulation is not None else "Not run",

        "Clinical Validation":
            "External validation metrics" if clinical_validation is not None else "Not supplied"
    }

    metadata_df = pd.DataFrame(
        list(
            metadata.items()
        ),
        columns=[
            "Parameter",
            "Value"
        ]
    )

    st.dataframe(
        metadata_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "NeuroTwin 3.1 — Personalized Computational "
    "Brain Digital Twin Research Prototype"
)

st.caption(
    "EEG → MRI → Functional Connectivity → Brain Network "
    "→ Whole-Brain Model → Digital Twin → Clinical Validation"
)

st.caption(
    "Research and computational simulation only. "
    "Not a medical diagnostic, treatment, or "
    "clinical decision system."
)
