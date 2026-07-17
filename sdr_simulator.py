"""
SDR link simulator - no hardware needed.

Digital modulators (QPSK/16-QAM/64-QAM built off one shared Gray-coded
square-QAM engine), OFDM with cyclic prefix, AWGN/Rayleigh channels,
two FEC schemes, and a minimum-distance receiver + BER counting.
"""

import numpy as np


# Generic Gray-coded square-QAM constellation

def build_qam_table(M):
    """
    Gray-coded square-QAM constellation for M in {4, 16, 64, ...}.
    Returns (symbols, bits) - symbols normalized to unit avg energy,
    bits is (M, log2(M)) with first half mapping to I, second half to Q.
    """
    k = int(np.log2(M))
    if 2 ** k != M or k % 2 != 0:
        raise ValueError("M must be a power of 4 (4, 16, 64, 256, ...)")

    kb = k // 2                      # bits per axis (I or Q)
    m = 2 ** kb                      # PAM levels per axis
    levels = np.arange(-(m - 1), m, 2, dtype=float)   # e.g. [-3,-1,1,3] for m=4

    # Standard binary-reflected Gray code, position -> gray value
    gray = [i ^ (i >> 1) for i in range(m)]
    # natural binary value b -> PAM level (so Gray-coded bits differ by 1 bit
    # between adjacent levels, minimizing bit errors for nearest-neighbor mistakes)
    nat_to_level = np.zeros(m)
    for pos, g in enumerate(gray):
        nat_to_level[g] = levels[pos]

    table_syms = []
    table_bits = []
    for I in range(m):
        for Q in range(m):
            bits_I = [int(b) for b in format(I, '0{}b'.format(kb))]
            bits_Q = [int(b) for b in format(Q, '0{}b'.format(kb))]
            table_bits.append(bits_I + bits_Q)
            table_syms.append(nat_to_level[I] + 1j * nat_to_level[Q])

    table_syms = np.array(table_syms, dtype=complex)
    table_bits = np.array(table_bits, dtype=int)

    # Normalize to unit average symbol energy
    avg_energy = np.mean(np.abs(table_syms) ** 2)
    table_syms /= np.sqrt(avg_energy)

    return table_syms, table_bits


# Cache tables so we don't rebuild them every call
_TABLE_CACHE = {}


def get_table(M):
    if M not in _TABLE_CACHE:
        _TABLE_CACHE[M] = build_qam_table(M)
    return _TABLE_CACHE[M]


# Modulator / Demodulator

def modulate(bits, M):
    """
    Map a 1-D array of bits (0/1 ints) to complex symbols for a given M
    (4=QPSK, 16=16-QAM, 64=64-QAM).

    Bits are zero-padded at the end if not a multiple of log2(M).
    Returns: symbols (complex ndarray), n_pad (int, number of pad bits added)
    """
    table_syms, table_bits = get_table(M)
    k = int(np.log2(M))

    bits = np.asarray(bits, dtype=int)
    n_pad = (-len(bits)) % k
    if n_pad:
        bits = np.concatenate([bits, np.zeros(n_pad, dtype=int)])

    groups = bits.reshape(-1, k)

    # Build a lookup: bit-pattern (as int index via binary weights) -> symbol
    weights = 1 << np.arange(k - 1, -1, -1)
    bit_table_idx = table_bits @ weights          # index for each row of table_bits
    group_idx = groups @ weights                  # index for each incoming bit group

    # Build direct lookup array of size 2**k: lookup[bit_pattern_int] = symbol
    lookup = np.zeros(2 ** k, dtype=complex)
    lookup[bit_table_idx] = table_syms

    symbols = lookup[group_idx]
    return symbols, n_pad


def demodulate(rx_symbols, M):
    """
    Hard-decision minimum-distance demodulation.
    Returns a flat bit array (multiple of log2(M) long).
    """
    table_syms, table_bits = get_table(M)
    # distance from every rx symbol to every constellation point
    dists = np.abs(rx_symbols[:, None] - table_syms[None, :]) ** 2
    idx = np.argmin(dists, axis=1)
    bits = table_bits[idx].flatten()
    return bits


MOD_BITS_PER_SYMBOL = {4: 2, 16: 4, 64: 6}
MOD_NAME = {4: "QPSK", 16: "16-QAM", 64: "64-QAM"}


# Channels

def awgn(signal, EbN0_dB, bits_per_symbol, samples_per_symbol=1):
    """Add complex AWGN so the signal hits the requested Eb/N0."""
    EbN0_lin = 10 ** (EbN0_dB / 10.0)
    EsN0_lin = EbN0_lin * bits_per_symbol
    N0 = 1.0 / (EsN0_lin * samples_per_symbol)
    noise_std = np.sqrt(N0 / 2.0)
    noise = noise_std * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise


def rayleigh_flat_fading(symbols, EbN0_dB, bits_per_symbol):
    """One random complex gain per symbol, then AWGN, then perfect-CSI equalization."""
    h = (np.random.randn(*symbols.shape) + 1j * np.random.randn(*symbols.shape)) / np.sqrt(2)
    rx = h * symbols
    rx = awgn(rx, EbN0_dB, bits_per_symbol)
    equalized = rx / h            # perfect channel knowledge (coherent detection)
    return equalized, h


def multipath_rayleigh_channel(tx_time, EbN0_dB, bits_per_symbol, n_taps=4):
    """L-tap complex Gaussian channel + AWGN. Needs cp_len >= n_taps-1 to avoid ISI."""
    h = (np.random.randn(n_taps) + 1j * np.random.randn(n_taps)) / np.sqrt(2 * n_taps)
    rx = np.convolve(tx_time, h)[:len(tx_time)]
    rx = awgn(rx, EbN0_dB, bits_per_symbol)
    return rx, h


# OFDM

def ofdm_modulate(bits, M, n_subcarriers=64, cp_len=16):
    """bits -> QAM symbols -> subcarrier blocks -> IFFT -> add CP."""
    table_syms, _ = get_table(M)
    symbols, n_pad_bits = modulate(bits, M)

    n_pad_syms = (-len(symbols)) % n_subcarriers
    if n_pad_syms:
        filler = table_syms[np.random.randint(0, M, n_pad_syms)]
        symbols = np.concatenate([symbols, filler])

    freq_blocks = symbols.reshape(-1, n_subcarriers)
    time_blocks = np.fft.ifft(freq_blocks, axis=1) * np.sqrt(n_subcarriers)
    cp = time_blocks[:, -cp_len:]
    tx_blocks = np.concatenate([cp, time_blocks], axis=1)
    tx_time = tx_blocks.flatten()

    return tx_time, freq_blocks.shape[0], n_pad_syms, n_pad_bits


def ofdm_demodulate(rx_time, n_ofdm_symbols, n_subcarriers=64, cp_len=16, h_taps=None):
    """Strip CP -> FFT -> optional per-subcarrier zero-forcing equalization."""
    rx_blocks = rx_time.reshape(n_ofdm_symbols, n_subcarriers + cp_len)
    rx_blocks = rx_blocks[:, cp_len:]
    freq = np.fft.fft(rx_blocks, axis=1) / np.sqrt(n_subcarriers)

    if h_taps is not None:
        H = np.fft.fft(h_taps, n_subcarriers)   # channel frequency response
        freq = freq / H[None, :]                # one-tap zero-forcing equalizer

    return freq.flatten()


# End-to-end link simulation helpers

def simulate_link(M, EbN0_dB, n_bits, channel_type="awgn", fec="none"):
    """
    Single-carrier end-to-end sim for one Eb/N0 point.
    Eb/N0 is per information bit, so a rate-R code gets less energy per
    coded bit (Ec = R*Eb) in exchange for error correction.
    """
    bits_per_symbol = MOD_BITS_PER_SYMBOL[M]
    code_rate = FEC_CODE_RATE[fec]

    tx_info_bits = np.random.randint(0, 2, n_bits)
    coded_bits = fec_encode(tx_info_bits, fec)

    tx_syms, n_pad = modulate(coded_bits, M)
    eff_bits_per_symbol = bits_per_symbol * code_rate   # Ec = R * Eb

    if channel_type == "awgn":
        rx_syms = awgn(tx_syms, EbN0_dB, eff_bits_per_symbol)
    elif channel_type == "rayleigh":
        rx_syms, _ = rayleigh_flat_fading(tx_syms, EbN0_dB, eff_bits_per_symbol)
    else:
        raise ValueError("channel_type must be 'awgn' or 'rayleigh'")

    rx_coded_bits = demodulate(rx_syms, M)
    rx_coded_bits = rx_coded_bits[:len(coded_bits)]     # drop modulation padding
    rx_info_bits = fec_decode(rx_coded_bits, fec)
    rx_info_bits = rx_info_bits[:n_bits]                # drop FEC block padding

    errors = np.sum(rx_info_bits != tx_info_bits)
    return errors / n_bits


def simulate_ofdm_link(M, EbN0_dB, n_bits, n_subcarriers=64, cp_len=16, n_taps=4,
                        n_channel_realizations=15, fec="none"):
    """
    OFDM link over a multipath Rayleigh channel. A single channel draw can
    land in a deep fade and make the BER noisy, so we average over several
    independent channel realizations per Eb/N0 point.
    """
    bits_per_symbol = MOD_BITS_PER_SYMBOL[M]
    code_rate = FEC_CODE_RATE[fec]
    eff_bits_per_symbol = bits_per_symbol * code_rate
    bits_per_run = max(n_bits // n_channel_realizations, n_subcarriers * bits_per_symbol)

    total_errors = 0
    total_bits = 0
    for _ in range(n_channel_realizations):
        tx_info_bits = np.random.randint(0, 2, bits_per_run)
        coded_bits = fec_encode(tx_info_bits, fec)

        tx_time, n_sym, n_pad_syms, n_pad_bits = ofdm_modulate(
            coded_bits, M, n_subcarriers=n_subcarriers, cp_len=cp_len
        )

        rx_time, h = multipath_rayleigh_channel(tx_time, EbN0_dB, eff_bits_per_symbol, n_taps=n_taps)

        rx_freq = ofdm_demodulate(rx_time, n_sym, n_subcarriers=n_subcarriers,
                                   cp_len=cp_len, h_taps=h)

        rx_coded_bits = demodulate(rx_freq, M)
        rx_coded_bits = rx_coded_bits[:len(coded_bits)]
        rx_info_bits = fec_decode(rx_coded_bits, fec)
        rx_info_bits = rx_info_bits[:len(tx_info_bits)]

        total_errors += np.sum(rx_info_bits != tx_info_bits)
        total_bits += len(tx_info_bits)

    return total_errors / total_bits


# Forward Error Correction (FEC)
# Hamming(7,4) - block code, rate 4/7, corrects 1 bit error per 7-bit word.
# Convolutional K=3 rate 1/2 (generators 7,5 octal) - decoded with Viterbi.

FEC_CODE_RATE = {"none": 1.0, "hamming74": 4 / 7, "conv": 1 / 2}

# ---- Hamming(7,4) block code ----------------------------------------

# Systematic generator matrix G = [I4 | P]; codeword = [d0 d1 d2 d3 p0 p1 p2]
_HAMMING_G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
])
# Parity check matrix H = [P^T | I3] (consistent with G above)
_HAMMING_H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1],
])
# Map each possible non-zero syndrome to the bit position it flags
_HAMMING_SYNDROME_TO_POS = {}
for _col in range(7):
    _HAMMING_SYNDROME_TO_POS[tuple(_HAMMING_H[:, _col])] = _col


def hamming74_encode(bits):
    """Encode a bit array with Hamming(7,4). Pads with zeros to a multiple of 4."""
    bits = np.asarray(bits, dtype=int)
    n_pad = (-len(bits)) % 4
    if n_pad:
        bits = np.concatenate([bits, np.zeros(n_pad, dtype=int)])
    blocks = bits.reshape(-1, 4)
    coded = (blocks @ _HAMMING_G) % 2
    return coded.flatten()


def hamming74_decode(coded_bits):
    """Compute syndrome per 7-bit block, flip the bit it points to, return data bits."""
    coded_bits = np.asarray(coded_bits, dtype=int)
    n_pad = (-len(coded_bits)) % 7
    if n_pad:
        coded_bits = np.concatenate([coded_bits, np.zeros(n_pad, dtype=int)])
    blocks = coded_bits.reshape(-1, 7).copy()
    syndromes = (blocks @ _HAMMING_H.T) % 2

    for i in range(blocks.shape[0]):
        syn = tuple(syndromes[i])
        if syn != (0, 0, 0):
            pos = _HAMMING_SYNDROME_TO_POS.get(syn)
            if pos is not None:
                blocks[i, pos] ^= 1     # flip the bit the syndrome points to

    return blocks[:, :4].flatten()


# ---- Rate-1/2 convolutional code (K=3, generators 7/5 octal) + Viterbi ----

_CONV_G1 = np.array([1, 1, 1])   # octal 7
_CONV_G2 = np.array([1, 0, 1])   # octal 5
_CONV_K = 3                       # constraint length
_CONV_MEM = _CONV_K - 1           # 2 memory bits -> 4 trellis states


def conv_encode(bits, g1=_CONV_G1, g2=_CONV_G2):
    """Rate-1/2 conv encoder, output interleaved [c1_0,c2_0,c1_1,c2_1,...]. Flushes trellis with K-1 zero bits."""
    bits = np.asarray(bits, dtype=int)
    K = len(g1)
    mem = K - 1
    extended = np.concatenate([np.zeros(mem, dtype=int), bits, np.zeros(mem, dtype=int)])
    windows = np.lib.stride_tricks.sliding_window_view(extended, K)[:len(bits) + mem]
    out1 = (windows @ g1) % 2
    out2 = (windows @ g2) % 2
    coded = np.empty(2 * len(out1), dtype=int)
    coded[0::2] = out1
    coded[1::2] = out2
    return coded


def _build_conv_trellis(g1=_CONV_G1, g2=_CONV_G2):
    """Precompute (next_state, out1, out2) for every (state, input bit)."""
    K = len(g1)
    mem = K - 1
    n_states = 2 ** mem
    trans = {}
    for s in range(n_states):
        mem_bits = [(s >> i) & 1 for i in range(mem)][::-1]     # register, oldest..newest
        for bit in (0, 1):
            window = np.array([bit] + mem_bits)                # matches conv_encode's window order
            out1 = int(np.sum(window * g1) % 2)
            out2 = int(np.sum(window * g2) % 2)
            next_mem_bits = ([bit] + mem_bits)[:mem]
            next_state = 0
            for i, b in enumerate(next_mem_bits[::-1]):
                next_state |= (b << i)
            trans[(s, bit)] = (next_state, out1, out2)
    return trans, n_states, mem


_CONV_TRANS, _CONV_N_STATES, _ = _build_conv_trellis()


def viterbi_decode(rx_bits, g1=_CONV_G1, g2=_CONV_G2):
    """Hard-decision Viterbi decode, traces back from the flushed all-zero end state."""
    trans, n_states, mem = _CONV_TRANS, _CONV_N_STATES, _CONV_MEM
    rx_bits = np.asarray(rx_bits, dtype=int)
    n_steps = len(rx_bits) // 2

    INF = np.inf
    pm = np.full(n_states, INF)
    pm[0] = 0
    prev_state_tbl = np.zeros((n_steps, n_states), dtype=int)
    prev_bit_tbl = np.zeros((n_steps, n_states), dtype=int)

    for t in range(n_steps):
        r1, r2 = rx_bits[2 * t], rx_bits[2 * t + 1]
        new_pm = np.full(n_states, INF)
        new_ps = np.zeros(n_states, dtype=int)
        new_pb = np.zeros(n_states, dtype=int)
        for s in range(n_states):
            if pm[s] == INF:
                continue
            for bit in (0, 1):
                ns, o1, o2 = trans[(s, bit)]
                branch_metric = (o1 != r1) + (o2 != r2)
                metric = pm[s] + branch_metric
                if metric < new_pm[ns]:
                    new_pm[ns] = metric
                    new_ps[ns] = s
                    new_pb[ns] = bit
        pm = new_pm
        prev_state_tbl[t] = new_ps
        prev_bit_tbl[t] = new_pb

    # Traceback from the all-zero state (trellis was flushed to state 0)
    state = 0
    decoded = np.zeros(n_steps, dtype=int)
    for t in range(n_steps - 1, -1, -1):
        decoded[t] = prev_bit_tbl[t, state]
        state = prev_state_tbl[t, state]

    return decoded[:-mem] if mem > 0 else decoded


def fec_encode(bits, fec):
    if fec == "none":
        return bits
    elif fec == "hamming74":
        return hamming74_encode(bits)
    elif fec == "conv":
        return conv_encode(bits)
    raise ValueError(f"Unknown fec scheme: {fec}")


def fec_decode(coded_bits, fec):
    if fec == "none":
        return coded_bits
    elif fec == "hamming74":
        return hamming74_decode(coded_bits)
    elif fec == "conv":
        return viterbi_decode(coded_bits)
    raise ValueError(f"Unknown fec scheme: {fec}")


# Theoretical BER (for sanity-check plots)

def q_function(x):
    from scipy.special import erfc
    return 0.5 * erfc(x / np.sqrt(2))


def theoretical_ber_awgn(M, EbN0_dB):
    """Theoretical BER for Gray-coded square M-QAM over AWGN."""
    EbN0 = 10 ** (EbN0_dB / 10.0)
    k = np.log2(M)
    if M == 4:
        return q_function(np.sqrt(2 * EbN0))
    # General square M-QAM approximation (Gray-coded), valid for M=16,64,...
    m = np.sqrt(M)
    term = (2 * (1 - 1 / m) / np.log2(M)) * q_function(np.sqrt(3 * k * EbN0 / (M - 1)))
    return term


def theoretical_ber_rayleigh_qpsk(EbN0_dB):
    """Theoretical BER for QPSK over flat Rayleigh fading (Gray-coded)."""
    EbN0 = 10 ** (EbN0_dB / 10.0)
    return 0.5 * (1 - np.sqrt(EbN0 / (1 + EbN0)))
