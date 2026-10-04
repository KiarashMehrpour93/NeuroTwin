from pathlib import Path
import mne

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = list(range(1, 11))
RUNS = [4, 8, 12]

for subject in SUBJECTS:
    print(f"Downloading subject {subject}...")

    files = mne.datasets.eegbci.load_data(
        subjects=subject,
        runs=RUNS,
        path=str(DATA_DIR),
        update_path=False,
        verbose=True
    )

    print(f"Downloaded {len(files)} files.")

print("Download completed.")