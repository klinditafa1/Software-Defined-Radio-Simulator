# SDR Simulator

Software-defined radio link simulated end to end in Python, no SDR hardware
or MATLAB toolbox needed.

- Modulation: QPSK, 16-QAM, 64-QAM, OFDM
- Channels: AWGN, flat Rayleigh fading, multipath Rayleigh (for OFDM)
- FEC: Hamming(7,4), rate-1/2 convolutional code with Viterbi decoding
- Receiver: minimum-distance demod + BER counting
- CLI for running whatever sweep you want without touching the code

![AWGN BER](docs/images/ber_awgn.png)
![Coding gain](docs/images/ber_fec_comparison.png)
![Rayleigh BER](docs/images/ber_rayleigh.png)
![OFDM multipath BER](docs/images/ber_ofdm_multipath.png)

Constellation scatter plots (QPSK/16-QAM/64-QAM at 12dB): `docs/images/constellations.png`

## Setup

```bash
pip install -r requirements.txt
```

## Running it

```bash
python3 run_ber_analysis.py
```

Takes about a minute or two, writes plots + a CSV into `results/`. Options:

```bash
python3 run_ber_analysis.py --quick                  # fast smoke test
python3 run_ber_analysis.py --max-db 20 --step-db 2   # wider AWGN sweep
python3 run_ber_analysis.py --modulations 4 16        # drop 64-QAM
python3 run_ber_analysis.py --channels awgn           # skip Rayleigh/OFDM
python3 run_ber_analysis.py --fec conv --fec-modulation 16
python3 run_ber_analysis.py --skip-fec
python3 run_ber_analysis.py --bits-per-point 1000000  # smoother curves, slower
```

Full flag list: `python3 run_ber_analysis.py --help`

## Tests

```bash
pytest tests/ -v
```

Checks modulation round-trips with zero noise, BER goes down as SNR goes up,
64-QAM is worse than QPSK at the same Eb/N0, Rayleigh is worse than AWGN,
both FEC schemes actually correct errors and beat uncoded above the coding
gain threshold, OFDM converges at high SNR. Also runs in CI on every push.

## Using it as a library

```python
from sdr_simulator import simulate_link, simulate_ofdm_link

ber = simulate_link(M=4, EbN0_dB=8, n_bits=1_000_000, channel_type="awgn")
ber = simulate_link(M=16, EbN0_dB=15, n_bits=1_000_000, channel_type="rayleigh")
ber = simulate_ofdm_link(M=64, EbN0_dB=20, n_bits=1_000_000)
```

Lower-level pieces are exposed too if you want to look at intermediate
signals:

```python
from sdr_simulator import modulate, demodulate, awgn

tx_syms, n_pad = modulate(bits, M=16)
rx_syms = awgn(tx_syms, EbN0_dB=10, bits_per_symbol=4)
rx_bits = demodulate(rx_syms, M=16)
```

## Notes on the implementation

QPSK/16-QAM/64-QAM all go through the same Gray-coded square-QAM builder
(`build_qam_table`) — QPSK is just the M=4 case, no separate code path.

Rayleigh fading assumes perfect CSI at the receiver (divide by the known
channel gain) so the plots isolate fading loss from estimation error, which
is the usual simplification unless you're specifically studying channel
estimation.

OFDM channel is an L-tap complex Gaussian multipath model, convolved with
the TX waveform in the time domain, then equalized per-subcarrier with a
one-tap zero-forcing equalizer after the FFT — this is the whole point of
OFDM, turning a frequency-selective channel into a bunch of independent
flat-fading subchannels.

For FEC, Eb/N0 in the plots is per information bit, not per coded bit, so a
rate-1/2 code is genuinely spending half the energy per channel symbol
that uncoded transmission gets. That's why the coded curves sit *above*
uncoded at low SNR — the code hasn't earned back its rate penalty yet.
Above roughly 5-6dB for these two codes it crosses over and coding wins.
