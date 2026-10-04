from pathlib import Path
import mne

DATA_DIR = Path('data/raw')

def load_subject(subject: int):
    files = mne.datasets.eegbci.load_data(subjects=subject, runs=[4,8,12], path=str(DATA_DIR), update_path=False, verbose=False)
    raws = []
    for file in files:
        raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
        mne.datasets.eegbci.standardize(raw)
        raw.set_eeg_reference('average', projection=False)
        raw.filter(8.0, 30.0, fir_design='firwin', verbose=False)
        raws.append(raw)
    return raws

def create_epochs(raw_list):
    all_epochs = []
    for raw in raw_list:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        selected = {k: event_id[k] for k in ('T1','T2') if k in event_id}
        if not selected:
            continue
        all_epochs.append(mne.Epochs(raw, events, event_id=selected, tmin=1.0, tmax=4.0, baseline=None, preload=True, reject_by_annotation=True, verbose=False))
    if not all_epochs:
        raise RuntimeError('No valid T1/T2 epochs found.')
    return mne.concatenate_epochs(all_epochs)

def preprocess_subject(subject: int):
    return create_epochs(load_subject(subject))
