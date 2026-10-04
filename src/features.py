import numpy as np
from scipy.signal import welch

def extract_features(epochs):
    data = epochs.get_data()
    sfreq = epochs.info['sfreq']
    bands = {'theta':(4,8), 'alpha':(8,13), 'mu':(8,13), 'beta':(13,30)}
    out = []
    for epoch in data:
        row = []
        for channel in epoch:
            freqs, psd = welch(channel, fs=sfreq, nperseg=min(256, len(channel)))
            for low, high in bands.values():
                mask = (freqs >= low) & (freqs <= high)
                row.append(np.log(np.mean(psd[mask]) + 1e-12))
        out.append(row)
    return np.asarray(out)

def get_labels(epochs):
    labels = epochs.events[:, -1]
    unique = np.unique(labels)
    if len(unique) != 2:
        raise ValueError(f'Expected 2 classes, found {len(unique)}.')
    mapping = {unique[0]:0, unique[1]:1}
    return np.array([mapping[x] for x in labels])
