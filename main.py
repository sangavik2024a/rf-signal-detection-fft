import numpy as np
import tkinter as tk
from tkinter import ttk

from scipy.signal import welch
from scipy.stats import chi2

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def compute_roc(labels, scores):
    thresholds = np.sort(np.unique(scores))[::-1]
    tpr_list = []
    fpr_list = []

    P = np.sum(labels == 1)
    Nn = np.sum(labels == 0)

    for th in thresholds:
        pred = (scores >= th).astype(int)

        TP = np.sum((pred == 1) & (labels == 1))
        FP = np.sum((pred == 1) & (labels == 0))

        tpr_list.append(TP / P)
        fpr_list.append(FP / Nn)

    return np.array(fpr_list), np.array(tpr_list)


class RFDetectionGUI:

    def __init__(self, root):
        self.root = root
        root.title("RF Detection (Final Clean Version)")
        root.geometry("1200x800")

        self.create_controls()

        self.fig = Figure(figsize=(10, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_plots()

    def create_controls(self):

        frame = ttk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        def field(label, default, col):
            ttk.Label(frame, text=label).grid(row=0, column=col*2)
            var = tk.StringVar(value=str(default))
            ttk.Entry(frame, textvariable=var, width=10)\
                .grid(row=0, column=col*2+1)
            return var

        self.N_var = field("Samples (N)", 512, 0)
        self.snr_var = field("SNR (dB)", -5, 1)
        self.pfa_var = field("Pfa", 0.01, 2)

        ttk.Button(frame,
                   text="Run",
                   command=self.update_plots).grid(row=0, column=10)

    def update_plots(self):

        N = int(self.N_var.get())
        snr_db = float(self.snr_var.get())
        pfa = float(self.pfa_var.get())

        snr_linear = 10 ** (snr_db / 10)
        threshold = chi2.ppf(1 - pfa, df=N)

        energy_h0 = []
        energy_h1 = []

        for _ in range(300):
            noise = np.random.randn(N)
            signal = np.sqrt(snr_linear) * np.random.randn(N)
            interference = 0.7*np.sin(2*np.pi*0.05*np.arange(N))

            energy_h0.append(np.sum(noise**2))
            energy_h1.append(np.sum((noise+signal+interference)**2))

        energy_h0 = np.array(energy_h0)
        energy_h1 = np.array(energy_h1)

        labels = np.concatenate([np.zeros(300), np.ones(300)])
        scores = np.concatenate([energy_h0, energy_h1])

        self.fig.clear()
        axs = self.fig.subplots(2, 2)

        # ================= PSD =================
        ax1 = axs[0, 0]

        signal = np.random.randn(N)
        interference = 0.7*np.sin(2*np.pi*0.05*np.arange(N))

        f, P_signal = welch(signal)
        f, P_interf = welch(signal + interference)

        ax1.semilogy(f, P_signal, label="Signal only", linewidth=2)
        ax1.semilogy(f, P_interf, label="Signal + Interference", linewidth=2)

        peak_idx = np.argmax(P_interf)
        peak_freq = f[peak_idx]

        ax1.axvline(peak_freq, linestyle='--')
        ax1.text(peak_freq, max(P_interf), " Interference Peak", fontsize=9)

        ax1.set_title("PSD: Interference shows as sharp peak")
        ax1.set_xlabel("Frequency (Normalized)")
        ax1.set_ylabel("Power/Frequency")
        ax1.legend()
        ax1.grid(True)

        # ================= ENERGY =================
        ax2 = axs[0, 1]

        ax2.hist(energy_h0/N, bins=40, alpha=0.6, label="Noise")
        ax2.hist(energy_h1/N, bins=40, alpha=0.6, label="Signal")

        ax2.axvline(threshold/N, linestyle='--', label="Threshold")

        ax2.set_title("Energy Detection")
        ax2.set_xlabel("Normalized Energy")
        ax2.set_ylabel("Probability Density")
        ax2.legend()
        ax2.grid(True)

        # ================= ROC =================
        ax3 = axs[1, 0]

        fpr, tpr = compute_roc(labels, scores)

        ax3.plot(fpr, tpr, linewidth=2)
        ax3.plot([0,1],[0,1],'--')

        ax3.set_title("ROC Curve")
        ax3.set_xlabel("False Alarm Probability")
        ax3.set_ylabel("Detection Probability")
        ax3.grid(True)

        axs[1,1].axis('off')

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = RFDetectionGUI(root)
    root.mainloop()
