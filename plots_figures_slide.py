import os
import math
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from scipy.signal import welch, correlate, spectrogram, resample_poly

# ----------------------------
# CONFIG: put your filenames here
# ----------------------------
DRONE_WAV = "B_S2_D1_067-bebop_000_.wav"
NONDRONE_WAV = "1-101336-A-302.wav"

OUT_DIR = "figs"
TARGET_SR = 44100

# Plot ranges (for slides)
FMAX = 8000
ACF_MAX_LAG_MS = 60

# Welch PSD params
PSD_NPERSEG = 2048
PSD_NOVERLAP = 1024

# Spectrogram params
SPEC_NPERSEG = 1024
SPEC_NOVERLAP = 512
SPEC_NFFT = 1024

# MFCC params
N_MFCC = 13
MFCC_NFFT = 512
MFCC_WIN_MS = 25
MFCC_HOP_MS = 10
MFCC_N_MELS = 40


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def savefig(path: str):
    # Tight export for beamer slides
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("Saved:", path)

def load_audio(path: str, target_sr: int = TARGET_SR):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)  # stereo -> mono

    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        x = resample_poly(x, up, down)
        sr = target_sr

    x = x.astype(np.float64)
    x = x / (np.max(np.abs(x)) + 1e-12)  # normalize
    return x, sr

def normalized_acf(x: np.ndarray, max_lag_samples: int):
    acf = correlate(x, x, mode="full")
    acf = acf[len(acf) // 2:]          # keep non-negative lags
    acf = acf / (acf[0] + 1e-12)       # normalize
    return acf[:max_lag_samples]


# ----------------------------
# Figure generators
# ----------------------------
def plot_welch_psd(x, sr, out_path):
    f, Pxx = welch(
        x, fs=sr, window="hann",
        nperseg=PSD_NPERSEG, noverlap=PSD_NOVERLAP,
        scaling="density"
    )
    Pxx_db = 10 * np.log10(Pxx + 1e-20)

    plt.figure(figsize=(9, 4))
    plt.plot(f, Pxx_db)
    plt.xlim(0, FMAX)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.title("Welch PSD — Drone")
    plt.axvspan(100, 400, alpha=0.2, label="Fundamental (100–400 Hz)")
    plt.axvspan(5000, 8000, alpha=0.12, label="Harmonics (5–8 kHz)")
    plt.grid(True)
    plt.legend(loc="best")
    savefig(out_path)

def plot_acf_compare(x_drone, x_non, sr, out_path):
    max_lag_samples = int(sr * ACF_MAX_LAG_MS / 1000.0)
    acf_d = normalized_acf(x_drone, max_lag_samples)
    acf_n = normalized_acf(x_non, max_lag_samples)
    lags_ms = np.arange(max_lag_samples) * 1000.0 / sr

    plt.figure(figsize=(9, 4))
    plt.plot(lags_ms, acf_d, label="Drone (periodic)")
    plt.plot(lags_ms, acf_n, label="Non-drone (aperiodic)")
    plt.xlabel("Lag (ms)")
    plt.ylabel("Normalized ACF")
    plt.title("ACF Comparison (Drone vs Non-drone)")
    plt.grid(True)
    plt.legend(loc="best")
    savefig(out_path)

def plot_spectrogram(x, sr, title, out_path):
    f, t, S = spectrogram(
        x, fs=sr, window="hann",
        nperseg=SPEC_NPERSEG, noverlap=SPEC_NOVERLAP,
        nfft=SPEC_NFFT, scaling="density", mode="magnitude"
    )
    S_db = 20 * np.log10(S + 1e-12)

    plt.figure(figsize=(9, 4.5))
    plt.pcolormesh(t, f, S_db, shading="auto")
    plt.ylim(0, FMAX)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.colorbar(label="Magnitude (dB)")
    savefig(out_path)

def mfcc_matrix(x, sr):
    """
    Uses librosa if available. If not, uses a scipy-only MFCC approximation.
    Output shape: (N_MFCC, frames)
    """
    hop = int(MFCC_HOP_MS * sr / 1000.0)
    win = int(MFCC_WIN_MS * sr / 1000.0)

    try:
        import librosa
        mfcc = librosa.feature.mfcc(
            y=x, sr=sr, n_mfcc=N_MFCC,
            n_fft=MFCC_NFFT, hop_length=hop, win_length=win,
            n_mels=MFCC_N_MELS, fmax=FMAX
        )
        return mfcc
    except Exception:
        # SciPy-only fallback (approx.)
        from scipy.fft import rfft
        from scipy.signal.windows import hamming
        from scipy.fftpack import dct

        # framing
        if len(x) < win:
            x = np.pad(x, (0, win - len(x)))
        n_frames = 1 + (len(x) - win) // hop
        w = hamming(win, sym=False)

        frames = np.stack([x[i * hop:i * hop + win] * w for i in range(n_frames)], axis=1)
        X = np.abs(rfft(frames, n=MFCC_NFFT, axis=0)) ** 2

        def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
        def mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        mels = np.linspace(hz_to_mel(0), hz_to_mel(FMAX), MFCC_N_MELS + 2)
        hz = mel_to_hz(mels)
        bins = np.floor((MFCC_NFFT + 1) * hz / sr).astype(int)

        fb = np.zeros((MFCC_N_MELS, X.shape[0]))
        for m in range(1, MFCC_N_MELS + 1):
            a, b, c = bins[m - 1], bins[m], bins[m + 1]
            a = max(a, 0)
            c = min(c, X.shape[0] - 1)
            for k in range(a, b):
                fb[m - 1, k] = (k - a) / max(b - a, 1)
            for k in range(b, c):
                fb[m - 1, k] = (c - k) / max(c - b, 1)

        melE = fb @ X
        log_melE = np.log(melE + 1e-12)
        mfcc = dct(log_melE, type=2, axis=0, norm="ortho")[:N_MFCC, :]
        return mfcc

def plot_mfcc_heatmap(x, sr, title, out_path):
    mfcc = mfcc_matrix(x, sr)

    plt.figure(figsize=(9, 3.6))
    plt.imshow(mfcc, aspect="auto", origin="lower")
    plt.colorbar(label="MFCC value")
    plt.xlabel("Frame index")
    plt.ylabel("MFCC coefficient")
    plt.title(title)
    savefig(out_path)

def plot_accuracy_bars(out_path):
    # Replace with your report's numbers if different
    models = ["SVM(ACF)", "SVM(PSD)", "SVM(MFCC)", "RF", "CNN", "LSTM", "CRNN"]
    binary = np.array([86.7, 85.6, 89.2, 87.5, 96.2, 92.3, 95.4])
    multi  = np.array([79.8, 78.4, 82.3, 80.1, 91.2, 87.6, 90.1])

    x = np.arange(len(models))
    w = 0.38

    plt.figure(figsize=(10, 4))
    plt.bar(x - w / 2, binary, width=w, label="Binary")
    plt.bar(x + w / 2, multi,  width=w, label="Multi-class")
    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.ylim(60, 100)
    plt.title("Model Performance Comparison")
    plt.grid(True, axis="y")
    plt.legend(loc="best")
    savefig(out_path)

def plot_accuracy_vs_snr(out_path):
    # Replace with your report's numbers if different
    snr = np.array([20, 10, 5, 0])
    svm  = np.array([89.2, 82.1, 71.3, 58.4])
    cnn  = np.array([96.2, 91.5, 84.2, 72.8])
    lstm = np.array([92.3, 86.7, 78.5, 65.1])
    crnn = np.array([95.4, 89.8, 81.9, 69.3])

    plt.figure(figsize=(8, 4.5))
    plt.plot(snr, svm,  marker="o", label="SVM(MFCC)")
    plt.plot(snr, cnn,  marker="o", label="CNN")
    plt.plot(snr, lstm, marker="o", label="LSTM")
    plt.plot(snr, crnn, marker="o", label="CRNN")
    plt.gca().invert_xaxis()
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs SNR")
    plt.grid(True)
    plt.legend(loc="best")
    savefig(out_path)


# ----------------------------
# MAIN
# ----------------------------
def main():
    ensure_dir(OUT_DIR)

    x_drone, sr = load_audio(DRONE_WAV, TARGET_SR)
    x_non, sr2 = load_audio(NONDRONE_WAV, sr)
    assert sr == sr2

    plot_welch_psd(x_drone, sr, os.path.join(OUT_DIR, "fig_psd_drone.png"))
    plot_acf_compare(x_drone, x_non, sr, os.path.join(OUT_DIR, "fig_acf_compare.png"))

    plot_spectrogram(x_drone, sr, "Spectrogram (STFT) — Drone", os.path.join(OUT_DIR, "fig_spectrogram_drone.png"))
    plot_spectrogram(x_non, sr, "Spectrogram (STFT) — Non-drone", os.path.join(OUT_DIR, "fig_spectrogram_nondrone.png"))

    plot_mfcc_heatmap(x_drone, sr, "MFCC (13) — Drone", os.path.join(OUT_DIR, "fig_mfcc_drone.png"))
    plot_mfcc_heatmap(x_non, sr, "MFCC (13) — Non-drone", os.path.join(OUT_DIR, "fig_mfcc_nondrone.png"))

    plot_accuracy_bars(os.path.join(OUT_DIR, "fig_accuracy_bars.png"))
    plot_accuracy_vs_snr(os.path.join(OUT_DIR, "fig_accuracy_vs_snr.png"))

    print("\nDone. Upload the PNGs in ./figs/ to Overleaf (same folder name).")

if __name__ == "__main__":
    main()
