"""Exercise A: Generate a sleep report from PSG and Hypnogram EDF files."""

import sys
import argparse
from pathlib import Path

import mne
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "R",
    "Sleep stage ?": "W",
    "Movement time": "W",
}

STAGE_ORDER = ["W", "N1", "N2", "N3", "R"]
STAGE_NUMERIC = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}


def load_hypnogram(hypno_path: str, psg_duration: float) -> tuple[np.ndarray, float]:
    """Load hypnogram from EDF and return per-epoch stage array + epoch duration."""
    annots = mne.read_annotations(hypno_path)
    epoch_dur = 30.0
    n_epochs = int(psg_duration / epoch_dur)
    stages = np.full(n_epochs, "W", dtype="U3")

    for ann in annots:
        desc = ann["description"]
        stage = STAGE_MAP.get(desc)
        if stage is None:
            continue
        start_epoch = int(ann["onset"] / epoch_dur)
        n_ann_epochs = max(1, int(ann["duration"] / epoch_dur))
        for i in range(start_epoch, min(start_epoch + n_ann_epochs, n_epochs)):
            stages[i] = stage

    return stages, epoch_dur


def compute_sleep_stats(stages: np.ndarray, epoch_dur: float) -> dict:
    """Compute standard sleep statistics."""
    n_epochs = len(stages)
    total_recording_time = n_epochs * epoch_dur / 60.0

    counts = {s: np.sum(stages == s) for s in STAGE_ORDER}
    durations_min = {s: counts[s] * epoch_dur / 60.0 for s in STAGE_ORDER}

    total_sleep_epochs = sum(counts[s] for s in ["N1", "N2", "N3", "R"])
    total_sleep_time = total_sleep_epochs * epoch_dur / 60.0

    first_sleep = None
    for i, s in enumerate(stages):
        if s != "W":
            first_sleep = i
            break

    last_sleep = None
    for i in range(len(stages) - 1, -1, -1):
        if stages[i] != "W":
            last_sleep = i
            break

    if first_sleep is not None and last_sleep is not None:
        spt = (last_sleep - first_sleep + 1) * epoch_dur / 60.0
        sol = first_sleep * epoch_dur / 60.0
    else:
        spt = 0
        sol = total_recording_time

    sleep_efficiency = (total_sleep_time / spt * 100) if spt > 0 else 0.0

    rem_latency = None
    if first_sleep is not None:
        for i in range(first_sleep, len(stages)):
            if stages[i] == "R":
                rem_latency = (i - first_sleep) * epoch_dur / 60.0
                break

    awakenings = 0
    if first_sleep is not None and last_sleep is not None:
        for i in range(first_sleep + 1, last_sleep + 1):
            if stages[i] == "W" and stages[i - 1] != "W":
                awakenings += 1

    percentages = {}
    for s in STAGE_ORDER:
        pct = (durations_min[s] / total_sleep_time * 100) if total_sleep_time > 0 else 0
        percentages[s] = pct

    return {
        "total_recording_time": total_recording_time,
        "total_sleep_time": total_sleep_time,
        "sleep_period_time": spt,
        "sleep_onset_latency": sol,
        "sleep_efficiency": sleep_efficiency,
        "rem_latency": rem_latency,
        "awakenings": awakenings,
        "durations_min": durations_min,
        "percentages": percentages,
    }


def plot_hypnogram(stages: np.ndarray, epoch_dur: float, save_path: str):
    """Plot and save the hypnogram."""
    numeric = np.array([STAGE_NUMERIC[s] for s in stages])
    t_hours = np.arange(len(stages)) * epoch_dur / 3600.0

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.step(t_hours, numeric, where="post", linewidth=1.2, color="#2563eb")
    ax.set_yticks(list(STAGE_NUMERIC.values()))
    ax.set_yticklabels(list(STAGE_NUMERIC.keys()))
    ax.invert_yaxis()
    ax.set_xlabel("Time (hours)")
    ax.set_title("Hypnogram")
    ax.set_xlim(t_hours[0], t_hours[-1])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_stage_distribution(stats: dict, save_path: str):
    """Bar chart of sleep stage durations and percentages."""
    stages = STAGE_ORDER
    durations = [stats["durations_min"][s] for s in stages]
    pcts = [stats["percentages"][s] for s in stages]
    colors = ["#94a3b8", "#60a5fa", "#3b82f6", "#1d4ed8", "#f59e0b"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(stages, durations, color=colors, edgecolor="white")
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Duration (min)")
    ax.set_title("Sleep Stage Distribution")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_eeg_psd(raw: mne.io.Raw, save_path: str):
    """Plot EEG power spectral density."""
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    if len(eeg_picks) == 0:
        return False

    fig, ax = plt.subplots(figsize=(10, 4))
    spectrum = raw.compute_psd(method="welch", fmin=0.5, fmax=40, picks=eeg_picks, n_fft=2048)
    psds, freqs = spectrum.get_data(return_freqs=True)
    psds_db = 10 * np.log10(psds * 1e12 + 1e-30)

    ch_names = [raw.ch_names[i] for i in eeg_picks]
    for i, name in enumerate(ch_names):
        ax.plot(freqs, psds_db[i], label=name, linewidth=1)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB/Hz, µV²)")
    ax.set_title("EEG Power Spectral Density")
    ax.legend(fontsize=8)
    ax.set_xlim(0.5, 40)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return True


def generate_html_report(stats: dict, output_dir: Path):
    """Generate the final HTML report."""
    rem_lat_str = f"{stats['rem_latency']:.1f} min" if stats["rem_latency"] is not None else "N/A"

    stage_rows = ""
    for s in STAGE_ORDER:
        stage_rows += (
            f"<tr><td>{s}</td>"
            f"<td>{stats['durations_min'][s]:.1f}</td>"
            f"<td>{stats['percentages'][s]:.1f}%</td></tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="utf-8">
<title>Sleep Report</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #f8fafc; color: #1e293b; }}
h1 {{ color: #1e40af; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; }}
h2 {{ color: #1e3a5f; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th, td {{ border: 1px solid #cbd5e1; padding: 10px 14px; text-align: center; }}
th {{ background: #1e40af; color: white; }}
tr:nth-child(even) {{ background: #f1f5f9; }}
img {{ max-width: 100%; border: 1px solid #e2e8f0; border-radius: 6px; margin: 12px 0; }}
.metric {{ display: inline-block; background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 24px; margin: 8px; text-align: center; min-width: 160px; }}
.metric .value {{ font-size: 1.6em; font-weight: bold; color: #1e40af; }}
.metric .label {{ font-size: 0.85em; color: #64748b; margin-top: 4px; }}
</style>
</head>
<body>
<h1>Sleep Report</h1>

<h2>Summary Metrics</h2>
<div>
  <div class="metric"><div class="value">{stats['total_recording_time']:.0f} min</div><div class="label">Total Recording Time</div></div>
  <div class="metric"><div class="value">{stats['total_sleep_time']:.0f} min</div><div class="label">Total Sleep Time</div></div>
  <div class="metric"><div class="value">{stats['sleep_efficiency']:.1f}%</div><div class="label">Sleep Efficiency</div></div>
  <div class="metric"><div class="value">{stats['sleep_onset_latency']:.1f} min</div><div class="label">Sleep Onset Latency</div></div>
  <div class="metric"><div class="value">{rem_lat_str}</div><div class="label">REM Latency</div></div>
  <div class="metric"><div class="value">{stats['awakenings']}</div><div class="label">Awakenings</div></div>
</div>

<h2>Hypnogram</h2>
<img src="hypnogram.png" alt="Hypnogram">

<h2>Sleep Stage Distribution</h2>
<table>
<tr><th>Stage</th><th>Duration (min)</th><th>Percentage</th></tr>
{stage_rows}
</table>
<img src="stage_distribution.png" alt="Sleep Stage Distribution">

<h2>EEG Power Spectral Density</h2>
<img src="eeg_psd.png" alt="EEG PSD">

</body>
</html>"""

    (output_dir / "sleep_report.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate sleep report from PSG + Hypnogram EDF")
    parser.add_argument("--psg", required=True, help="Path to PSG EDF file")
    parser.add_argument("--hypnogram", required=True, help="Path to Hypnogram EDF file")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PSG data...")
    raw = mne.io.read_raw_edf(args.psg, preload=True, verbose=False)
    psg_duration = raw.times[-1]

    print("Loading hypnogram...")
    stages, epoch_dur = load_hypnogram(args.hypnogram, psg_duration)

    print("Computing sleep statistics...")
    stats = compute_sleep_stats(stages, epoch_dur)

    print(f"  Total Recording Time : {stats['total_recording_time']:.1f} min")
    print(f"  Total Sleep Time     : {stats['total_sleep_time']:.1f} min")
    print(f"  Sleep Efficiency     : {stats['sleep_efficiency']:.1f}%")
    print(f"  Sleep Onset Latency  : {stats['sleep_onset_latency']:.1f} min")
    rem = stats['rem_latency']
    print(f"  REM Latency          : {rem:.1f} min" if rem else "  REM Latency          : N/A")
    print(f"  Awakenings           : {stats['awakenings']}")

    print("Plotting hypnogram...")
    plot_hypnogram(stages, epoch_dur, str(output_dir / "hypnogram.png"))

    print("Plotting stage distribution...")
    plot_stage_distribution(stats, str(output_dir / "stage_distribution.png"))

    print("Plotting EEG PSD...")
    plot_eeg_psd(raw, str(output_dir / "eeg_psd.png"))

    print("Generating HTML report...")
    generate_html_report(stats, output_dir)

    print(f"\nDone! Report saved to {output_dir / 'sleep_report.html'}")


if __name__ == "__main__":
    main()
