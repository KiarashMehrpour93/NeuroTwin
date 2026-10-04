import numpy as np

class NeuroTwin:
    def __init__(self):
        self.profile = {}
    def build_profile(self, X):
        self.profile = {'feature_mean': np.mean(X, axis=0), 'feature_std': np.std(X, axis=0), 'stability': self._stability(X), 'samples': len(X)}
        return self.profile
    def _stability(self, X):
        variation = np.mean(np.std(X, axis=0))
        return float(1 / (1 + variation))
