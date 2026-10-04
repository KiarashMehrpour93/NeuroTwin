import numpy as np
from mne.decoding import CSP


class CSPFeatureExtractor:
    """
    CSP feature extractor for two-class EEG motor imagery.
    """

    def __init__(self, n_components=6):
        self.n_components = n_components
        self.csp = CSP(
            n_components=n_components,
            reg="ledoit_wolf",
            log=True,
            norm_trace=False,
        )

    def fit_transform(self, epochs, y):
        X = epochs.get_data(copy=True)

        # CSP expects:
        # (n_epochs, n_channels, n_times)
        X = np.asarray(X, dtype=np.float64)

        return self.csp.fit_transform(X, y)

    def transform(self, epochs):
        X = epochs.get_data(copy=True)
        X = np.asarray(X, dtype=np.float64)

        return self.csp.transform(X)