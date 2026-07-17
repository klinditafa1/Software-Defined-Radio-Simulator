# Runs the modulation/channel/FEC sweeps and dumps plots + a CSV into
# --output-dir (default results/). Run with -h for all the flags.
#
# python3 run_ber_analysis.py --quick        fast smoke test
# python3 run_ber_analysis.py --channels awgn --max-db 20

import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sdr_simulator import (
    MOD_BITS_PER_SYMBOL, MOD_NAME,
    modulate, awgn,
    simulate_link, simulate_ofdm_link,
    theoretical_ber_awgn, theoretical_ber_rayleigh_qpsk,
    FEC_CODE_RATE,
)

MOD_COLORS = {4: "#1f77b4", 16: "#d62728", 64: "#2ca02c"}
FEC_LABEL = {"none": "Uncoded", "hamming74": "Hamming(7,4)", "conv": "Conv. K=3, R=1/2 (Viterbi)"}
FEC_COLORS = {"none": "#7f7f7f", "hamming74": "#ff7f0e", "conv": "#9467bd"}


def parse_args():
    p = argparse.ArgumentParser(
        description="SDR link simulator: modulation + channel + FEC + BER analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--modulations", type=int, nargs="+", default=[4, 16, 64], choices=[4, 16, 64],
                    help="Modulation orders to test (4=QPSK, 16=16-QAM, 64=64-QAM).")
    p.add_argument("--channels", nargs="+", default=["awgn", "rayleigh", "ofdm"],
                    choices=["awgn", "rayleigh", "ofdm"],
                    help="Which channel sweeps to run.")

    p.add_argument("--min-db", type=float, default=0.0, help="Min Eb/N0 (dB) for the AWGN sweep.")
    p.add_argument("--max-db", type=float, default=14.0, help="Max Eb/N0 (dB) for the AWGN sweep.")
    p.add_argument("--step-db", type=float, default=1.0, help="Eb/N0 step (dB) for the AWGN sweep.")

    p.add_argument("--rayleigh-max-db", type=float, default=30.0, help="Max Eb/N0 (dB) for Rayleigh.")
    p.add_argument("--rayleigh-step-db", type=float, default=2.0, help="Eb/N0 step (dB) for Rayleigh.")

    p.add_argument("--ofdm-max-db", type=float, default=30.0, help="Max Eb/N0 (dB) for OFDM.")
    p.add_argument("--ofdm-step-db", type=float, default=2.0, help="Eb/N0 step (dB) for OFDM.")
    p.add_argument("--n-subcarriers", type=int, default=64, help="OFDM subcarrier count.")
    p.add_argument("--cp-len", type=int, default=16, help="OFDM cyclic prefix length (samples).")
    p.add_argument("--n-taps", type=int, default=4, help="Multipath channel tap count for OFDM.")

    p.add_argument("--bits-per-point", type=int, default=400_000,
                    help="Information bits simulated per Eb/N0 point (auto-increased at high SNR).")

    p.add_argument("--fec", nargs="+", default=["hamming74", "conv"], choices=["hamming74", "conv"],
                    help="FEC schemes to compare against uncoded transmission "
                         "(dedicated AWGN coding-gain sweep). 'none'/uncoded is always included.")
    p.add_argument("--fec-modulation", type=int, default=4, choices=[4, 16, 64],
                    help="Modulation used for the FEC comparison sweep.")
    p.add_argument("--fec-max-db", type=float, default=10.0, help="Max Eb/N0 (dB) for the FEC sweep.")
    p.add_argument("--fec-step-db", type=float, default=1.0, help="Eb/N0 step (dB) for the FEC sweep.")
    p.add_argument("--fec-bits-per-point", type=int, default=200_000,
                    help="Info bits per point for the FEC sweep (Viterbi decoding is the slow part).")
    p.add_argument("--skip-fec", action="store_true", help="Skip the FEC comparison sweep entirely.")

    p.add_argument("--output-dir", default="results", help="Directory to write plots/CSV into.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--quick", action="store_true",
                    help="Fast preset: fewer Eb/N0 points and fewer bits, for a smoke test.")

    args = p.parse_args()

    if args.quick:
        args.step_db = max(args.step_db, 3.0)
        args.rayleigh_step_db = max(args.rayleigh_step_db, 6.0)
        args.ofdm_step_db = max(args.ofdm_step_db, 6.0)
        args.bits_per_point = min(args.bits_per_point, 60_000)
        args.fec_step_db = max(args.fec_step_db, 3.0)
        args.fec_bits_per_point = min(args.fec_bits_per_point, 40_000)

    return args


def run_sweep(modulations, ebn0_range, channel_type, bits_per_point, ofdm=False,
              n_subcarriers=64, cp_len=16, n_taps=4):
    """Return dict: results[M] = list of BER values matching ebn0_range."""
    results = {M: [] for M in modulations}
    for M in modulations:
        for ebn0 in ebn0_range:
            n_bits = bits_per_point
            if ofdm:
                ber = simulate_ofdm_link(M, ebn0, n_bits, n_subcarriers=n_subcarriers,
                                          cp_len=cp_len, n_taps=n_taps)
            else:
                ber = simulate_link(M, ebn0, n_bits, channel_type)
            # Re-run with more bits if too few errors landed for a stable estimate at high SNR
            if ber * n_bits < 100 and ebn0 < ebn0_range[-1]:
                extra_bits = max(bits_per_point * 5, 2_000_000)
                if ofdm:
                    ber = simulate_ofdm_link(M, ebn0, extra_bits, n_subcarriers=n_subcarriers,
                                              cp_len=cp_len, n_taps=n_taps)
                else:
                    ber = simulate_link(M, ebn0, extra_bits, channel_type)
            results[M].append(ber)
            print(f"  {MOD_NAME[M]:8s} Eb/N0={ebn0:5.1f} dB -> BER = {ber:.3e}")
    return results


def run_fec_sweep(M, ebn0_range, fec_schemes, bits_per_point):
    """Return dict: results[fec_name] = list of BER values matching ebn0_range (AWGN channel)."""
    schemes = ["none"] + [f for f in fec_schemes if f != "none"]
    results = {f: [] for f in schemes}
    for fec in schemes:
        for ebn0 in ebn0_range:
            n_bits = bits_per_point
            ber = simulate_link(M, ebn0, n_bits, "awgn", fec=fec)
            if ber * n_bits < 30 and ebn0 < ebn0_range[-1]:
                ber = simulate_link(M, ebn0, bits_per_point * 4, "awgn", fec=fec)
            results[fec].append(ber)
            print(f"  {FEC_LABEL[fec]:28s} Eb/N0={ebn0:5.1f} dB -> BER = {ber:.3e}")
    return results


def plot_ber(ebn0_range, results, modulations, title, path, theoretical_fn=None, theo_label=""):
    plt.figure(figsize=(7, 5.5))
    for M in modulations:
        ber = np.clip(np.array(results[M]), 1e-7, None)
        plt.semilogy(ebn0_range, ber, "o-", color=MOD_COLORS[M], label=f"{MOD_NAME[M]} (simulated)")
    if theoretical_fn is not None:
        theo = np.clip(theoretical_fn(ebn0_range), 1e-7, None)
        plt.semilogy(ebn0_range, theo, "k--", label=theo_label)
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title(title)
    plt.grid(True, which="both", linestyle=":", alpha=0.7)
    plt.legend()
    plt.ylim(1e-6, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot: {path}")


def plot_fec_comparison(ebn0_range, results, M, path):
    plt.figure(figsize=(7, 5.5))
    for fec, ber_list in results.items():
        ber = np.clip(np.array(ber_list), 1e-7, None)
        plt.semilogy(ebn0_range, ber, "o-", color=FEC_COLORS[fec], label=FEC_LABEL[fec])
    plt.xlabel("Eb/N0 (dB)  (per INFORMATION bit)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title(f"Coding Gain — {MOD_NAME[M]} over AWGN\nUncoded vs Hamming(7,4) vs Convolutional (Viterbi)")
    plt.grid(True, which="both", linestyle=":", alpha=0.7)
    plt.legend()
    plt.ylim(1e-6, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot: {path}")


def plot_constellations(modulations, path):
    fig, axes = plt.subplots(1, len(modulations), figsize=(5 * len(modulations), 5))
    if len(modulations) == 1:
        axes = [axes]
    for ax, M in zip(axes, modulations):
        n_bits = 4000 * MOD_BITS_PER_SYMBOL[M]
        bits = np.random.randint(0, 2, n_bits)
        tx_syms, _ = modulate(bits, M)
        rx_syms = awgn(tx_syms, EbN0_dB=12, bits_per_symbol=MOD_BITS_PER_SYMBOL[M])
        ax.scatter(rx_syms.real, rx_syms.imag, s=4, alpha=0.35, color="tab:blue", label="RX (noisy)")
        ax.scatter(tx_syms.real[:1000], tx_syms.imag[:1000], s=25, color="black", marker="x",
                   label="ideal TX points" if M == modulations[0] else None)
        ax.set_title(f"{MOD_NAME[M]} constellation @ Eb/N0=12dB (AWGN)")
        ax.set_xlabel("In-phase (I)")
        ax.set_ylabel("Quadrature (Q)")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot: {path}")


def save_csv(all_results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sweep", "series", "EbN0_dB", "BER"])
        for sweep_name, (ebn0_range, results) in all_results.items():
            for series, ber_list in results.items():
                label = MOD_NAME.get(series, FEC_LABEL.get(series, str(series)))
                for ebn0, ber in zip(ebn0_range, ber_list):
                    w.writerow([sweep_name, label, ebn0, ber])
    print(f"Saved CSV: {path}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    if "awgn" in args.channels:
        print("\n=== AWGN channel sweep ===")
        ebn0_range = np.arange(args.min_db, args.max_db + 1e-9, args.step_db)
        res = run_sweep(args.modulations, ebn0_range, "awgn", args.bits_per_point)
        theo = (lambda x: theoretical_ber_awgn(4, x)) if 4 in args.modulations else None
        plot_ber(ebn0_range, res, args.modulations, "BER vs Eb/N0 — AWGN Channel",
                  os.path.join(args.output_dir, "ber_awgn.png"),
                  theoretical_fn=theo, theo_label="QPSK theoretical")
        all_results["AWGN"] = (ebn0_range, res)

    if "rayleigh" in args.channels:
        print("\n=== Flat Rayleigh fading channel sweep ===")
        ebn0_range = np.arange(args.min_db, args.rayleigh_max_db + 1e-9, args.rayleigh_step_db)
        res = run_sweep(args.modulations, ebn0_range, "rayleigh", args.bits_per_point)
        theo = theoretical_ber_rayleigh_qpsk if 4 in args.modulations else None
        plot_ber(ebn0_range, res, args.modulations, "BER vs Eb/N0 — Flat Rayleigh Fading",
                  os.path.join(args.output_dir, "ber_rayleigh.png"),
                  theoretical_fn=theo, theo_label="QPSK theoretical (Rayleigh)")
        all_results["Rayleigh"] = (ebn0_range, res)

    if "ofdm" in args.channels:
        print("\n=== OFDM over multipath Rayleigh channel sweep ===")
        ebn0_range = np.arange(args.min_db, args.ofdm_max_db + 1e-9, args.ofdm_step_db)
        res = run_sweep(args.modulations, ebn0_range, "rayleigh", args.bits_per_point, ofdm=True,
                         n_subcarriers=args.n_subcarriers, cp_len=args.cp_len, n_taps=args.n_taps)
        plot_ber(ebn0_range, res, args.modulations, "BER vs Eb/N0 — OFDM over Multipath Rayleigh Channel",
                  os.path.join(args.output_dir, "ber_ofdm_multipath.png"))
        all_results["OFDM_Multipath"] = (ebn0_range, res)

    if not args.skip_fec:
        print(f"\n=== FEC coding-gain sweep ({MOD_NAME[args.fec_modulation]}, AWGN) ===")
        ebn0_range = np.arange(args.min_db, args.fec_max_db + 1e-9, args.fec_step_db)
        res = run_fec_sweep(args.fec_modulation, ebn0_range, args.fec, args.fec_bits_per_point)
        plot_fec_comparison(ebn0_range, res, args.fec_modulation,
                             os.path.join(args.output_dir, "ber_fec_comparison.png"))
        all_results["FEC_Comparison"] = (ebn0_range, res)

    print("\n=== Constellation diagrams ===")
    plot_constellations(args.modulations, os.path.join(args.output_dir, "constellations.png"))

    save_csv(all_results, os.path.join(args.output_dir, "ber_results.csv"))

    print(f"\nAll done. See the '{args.output_dir}/' folder for plots and CSV data.")


if __name__ == "__main__":
    main()
