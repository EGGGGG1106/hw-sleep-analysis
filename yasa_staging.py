"""Exercise B: YASA automated sleep staging vs manual annotation comparison."""

import argparse
from pathlib import Path

import mne
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yasa
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
)


STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 0,
    "Movement time": 0,
}

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def load_manual_hypnogram(hypno_path: str, n_epochs: int, epoch_dur: float) -> np.ndarray:
    """Load manual hypnogram annotations and return integer array."""
    annots = mne.read_annotations(hypno_path)
    stages = np.zeros(n_epochs, dtype=int)

    for ann in annots:
        val = STAGE_MAP.get(ann["description"])
        if val is None:
            continue
        start = int(ann["onset"] / epoch_dur)
        dur = max(1, int(ann["duration"] / epoch_dur))
        for i in range(start, min(start + dur, n_epochs)):
            stages[i] = val

    return stages


def find_channels(raw: mne.io.Raw) -> dict:
    """Find EEG and EOG channels by name patterns."""
    ch_names = raw.ch_names
    result = {}

    for name in ch_names:
        ln = name.lower()
        if "eeg" in ln or "fpz" in ln or "pz" in ln or "cz" in ln or "c4" in ln or "c3" in ln:
            if "eeg" not in result:
                result["eeg"] = name
        if "eog" in ln or "eog" in ln:
            if "eog" not in result:
                result["eog"] = name
        if "emg" in ln:
            if "emg" not in result:
                result["emg"] = name

    return result


def run_yasa_staging(raw: mne.io.Raw, channels: dict) -> np.ndarray:
    """Run YASA automatic sleep staging."""
    eeg_name = channels.get("eeg")
    eog_name = channels.get("eog")
    emg_name = channels.get("emg")

    sls = yasa.SleepStaging(
        raw,
        eeg_name=eeg_name,
        eog_name=eog_name,
        emg_name=emg_name,
    )

    hypno = sls.predict()

    label_to_int = {"W": 0, "WAKE": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "REM": 4}
    if hasattr(hypno, "hypno"):
        labels = hypno.hypno.values
    elif hasattr(hypno, "values"):
        labels = hypno.values
    else:
        labels = list(hypno)
    return np.array([label_to_int[s] for s in labels])


def plot_comparison_hypnogram(manual: np.ndarray, predicted: np.ndarray,
                              epoch_dur: float, save_path: str):
    """Plot manual vs predicted hypnograms side by side."""
    t_hours = np.arange(len(manual)) * epoch_dur / 3600.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)

    for ax, data, title, color in [
        (axes[0], manual, "Manual Annotation", "#2563eb"),
        (axes[1], predicted, "YASA Prediction", "#dc2626"),
    ]:
        ax.step(t_hours[:len(data)], data[:len(data)], where="post", linewidth=1.2, color=color)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(STAGE_NAMES)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlim(t_hours[0], t_hours[-1])

    axes[1].set_xlabel("Time (hours)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(manual: np.ndarray, predicted: np.ndarray, save_path: str):
    """Plot confusion matrix."""
    cm = confusion_matrix(manual, predicted, labels=[0, 1, 2, 3, 4])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=STAGE_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix: Manual vs YASA")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="YASA sleep staging vs manual annotation")
    parser.add_argument("--psg", required=True, help="Path to PSG EDF file")
    parser.add_argument("--hypnogram", required=True, help="Path to Hypnogram EDF file")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PSG data...")
    raw = mne.io.read_raw_edf(args.psg, preload=True, verbose=False)

    epoch_dur = 30.0
    n_epochs = int(raw.times[-1] / epoch_dur)

    print("Loading manual hypnogram...")
    manual = load_manual_hypnogram(args.hypnogram, n_epochs, epoch_dur)

    print("Finding EEG/EOG/EMG channels...")
    channels = find_channels(raw)
    print(f"  EEG: {channels.get('eeg', 'N/A')}")
    print(f"  EOG: {channels.get('eog', 'N/A')}")
    print(f"  EMG: {channels.get('emg', 'N/A')}")

    print("Running YASA automatic sleep staging...")
    predicted = run_yasa_staging(raw, channels)

    n = min(len(manual), len(predicted))
    manual = manual[:n]
    predicted = predicted[:n]

    acc = accuracy_score(manual, predicted)
    kappa = cohen_kappa_score(manual, predicted)
    report = classification_report(manual, predicted, labels=[0, 1, 2, 3, 4],
                                   target_names=STAGE_NAMES, zero_division=0)

    print(f"\n{'='*50}")
    print(f"  Accuracy       : {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Cohen's Kappa  : {kappa:.4f}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(report)

    print("Plotting comparison hypnograms...")
    plot_comparison_hypnogram(manual, predicted, epoch_dur,
                              str(output_dir / "comparison_hypnogram.png"))

    print("Plotting confusion matrix...")
    plot_confusion_matrix(manual, predicted, str(output_dir / "confusion_matrix.png"))

    results_text = f"""YASA Sleep Staging Results
==========================
Accuracy      : {acc:.4f} ({acc*100:.1f}%)
Cohen's Kappa : {kappa:.4f}

Classification Report:
{report}
"""
    (output_dir / "yasa_results.txt").write_text(results_text, encoding="utf-8")

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
