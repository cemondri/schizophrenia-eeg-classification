"""
EEG-based schizophrenia classification (capstone project).

This project brings together the machine learning workflow on a resting-state
EEG dataset: feature extraction, patient-level cross-validation (to avoid
leakage), model comparison (random forest vs gradient boosting), and honest
evaluation of class imbalance.

The goal is NOT a state-of-the-art classifier, but a demonstration of a
careful, leakage-aware methodology on a small clinical dataset.

Dataset: EEG in Schizophrenia (Olejarczyk and Jernajczyk)
https://doi.org/10.18150/repod.0107441
14 schizophrenia patients + 14 healthy controls, 15 min eyes-closed resting
state, 19 channels (10-20 montage), 250 Hz. Not included in this repo.
"""

import numpy as np
import mne
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, GroupKFold
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
# Folder containing the .edf files (s01..s14 = patients, h01..h14 = controls)
DATA_DIR = 'data'

# Standard EEG frequency bands (Hz)
bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 40),
}

EPOCH_SECONDS = 4.0


def process_file(file_name):
    """Read one EDF file, extract band-power features per epoch, and label.

    The label is set at the person level: every epoch from a patient file
    (name starting with 's') is labeled 1, every epoch from a control file
    ('h') is labeled 0.

    Returns:
        X: feature matrix, shape (n_epochs, 95)  -- 19 channels x 5 bands
        y: labels, shape (n_epochs,)  -- all the same value for one file
    """
    path = f'{DATA_DIR}/{file_name}.edf'
    label = 1 if file_name[0] == 's' else 0

    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.filter(1.0, 40.0, verbose=False)
    epochs = mne.make_fixed_length_epochs(
        raw, duration=EPOCH_SECONDS, preload=True, verbose=False
    )

    # Band power per channel per band
    band_features = []
    for name, (fmin, fmax) in bands.items():
        psd = epochs.compute_psd(fmin=fmin, fmax=fmax, verbose=False)
        power = psd.get_data().mean(axis=2)   # average over frequency -> (epochs, channels)
        band_features.append(power)
    X = np.log(np.concatenate(band_features, axis=1))   # (epochs, 95)

    # NOTE: inter-channel synchronization (mean absolute correlation across all
    # channel pairs) was also tested here, but it did not improve, and slightly
    # hurt, patient-independent accuracy. A single averaged value is likely too
    # coarse to capture the regional / frequency-specific connectivity changes
    # associated with schizophrenia. Left out of the final feature set.

    y = np.full(len(epochs), label)
    return X, y


def build_dataset():
    """Process all 28 files and assemble the full dataset with patient groups."""
    file_names = []
    for n in range(1, 15):
        file_names.append(f's{n:02d}')   # patients
        file_names.append(f'h{n:02d}')   # controls

    X_list, y_list, group_list = [], [], []
    for file_name in file_names:
        X_f, y_f = process_file(file_name)
        X_list.append(X_f)
        y_list.append(y_f)
        # tag every epoch with its person id, for patient-level splitting
        group_list.append(np.full(len(y_f), file_name))
        print(f"{file_name}: {X_f.shape[0]} epochs")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    groups = np.concatenate(group_list)
    return X, y, groups


def main():
    print("Building dataset...")
    X, y, groups = build_dataset()
    print(f"\nTotal: {X.shape[0]} epochs, {X.shape[1]} features")
    print(f"Schizophrenia: {np.sum(y == 1)} / Control: {np.sum(y == 0)}")
    print(f"Subjects: {len(np.unique(groups))}\n")

    gkf = GroupKFold(n_splits=5)

    # --- Leakage demonstration: epoch-level (wrong) vs person-level (correct) ---
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    leaky = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    clean = cross_val_score(rf, X, y, groups=groups, cv=gkf, scoring='accuracy')
    print("--- Leakage check ---")
    print(f"Epoch-level split (leaky):   {leaky.mean():.3f}")
    print(f"Person-level split (honest): {clean.mean():.3f}\n")

    # --- Model comparison (person-level) ---
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    rf_acc = cross_val_score(rf, X, y, groups=groups, cv=gkf, scoring='accuracy')
    gb_acc = cross_val_score(gb, X, y, groups=groups, cv=gkf, scoring='accuracy')
    print("--- Model comparison (person-level) ---")
    print(f"Random forest:     {rf_acc.mean():.3f}")
    print(f"Gradient boosting: {gb_acc.mean():.3f}\n")

    # --- Detailed evaluation of the better model ---
    gb_pred = cross_val_predict(gb, X, y, groups=groups, cv=gkf)
    print("--- Gradient boosting: confusion matrix and report (person-level) ---")
    print(confusion_matrix(y, gb_pred))
    print(classification_report(y, gb_pred, target_names=['control', 'schizophrenia']))


if __name__ == '__main__':
    main()
