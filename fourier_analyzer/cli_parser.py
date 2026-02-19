from filters import (
    butterworth_bandreject,
    butterworth_bandpass,
    butterworth_highpass,
    butterworth_lowpass,
    gaussian_bandreject,
    gaussian_bandpass,
    gaussian_highpass,
    gaussian_lowpass,
    high_frequency_emphasis,
    homomorphic_filter,
    ideal_bandreject,
    ideal_bandpass,
    ideal_highpass,
    ideal_lowpass,
    laplacian_filter,
    notch_pass_butterworth,
    notch_pass_ideal,
    notch_reject_butterworth,
    notch_reject_ideal,
)


def parse_command(command, shape):
    tokens = command.strip().split()
    if not tokens:
        return None

    cmd = tokens[0].lower()

    # Low/High pass
    if cmd in ("ideal_lp", "ilp"):
        return ideal_lowpass(shape, float(tokens[1]))
    if cmd in ("ideal_hp", "ihp"):
        return ideal_highpass(shape, float(tokens[1]))
    if cmd in ("gaussian_lp", "glp"):
        return gaussian_lowpass(shape, float(tokens[1]))
    if cmd in ("gaussian_hp", "ghp"):
        return gaussian_highpass(shape, float(tokens[1]))
    if cmd in ("butter_lp", "blp"):
        return butterworth_lowpass(shape, float(tokens[1]), int(tokens[2]))
    if cmd in ("butter_hp", "bhp"):
        return butterworth_highpass(shape, float(tokens[1]), int(tokens[2]))

    # Band reject/pass
    if cmd in ("ideal_br", "ibr"):
        return ideal_bandreject(shape, float(tokens[1]), float(tokens[2]))
    if cmd in ("ideal_bp", "ibp"):
        return ideal_bandpass(shape, float(tokens[1]), float(tokens[2]))
    if cmd in ("gaussian_br", "gbr"):
        return gaussian_bandreject(shape, float(tokens[1]), float(tokens[2]))
    if cmd in ("gaussian_bp", "gbp"):
        return gaussian_bandpass(shape, float(tokens[1]), float(tokens[2]))
    if cmd in ("butter_br", "bbr"):
        return butterworth_bandreject(shape, float(tokens[1]), float(tokens[2]), int(tokens[3]))
    if cmd in ("butter_bp", "bbp"):
        return butterworth_bandpass(shape, float(tokens[1]), float(tokens[2]), int(tokens[3]))

    # Notch
    if cmd in ("notch_ideal_reject", "nir"):
        return notch_reject_ideal(shape, float(tokens[1]), float(tokens[2]), float(tokens[3]))
    if cmd in ("notch_ideal_pass", "nip"):
        return notch_pass_ideal(shape, float(tokens[1]), float(tokens[2]), float(tokens[3]))
    if cmd in ("notch_butter_reject", "nbr"):
        return notch_reject_butterworth(
            shape,
            float(tokens[1]),
            float(tokens[2]),
            float(tokens[3]),
            int(tokens[4]),
        )
    if cmd in ("notch_butter_pass", "nbp"):
        return notch_pass_butterworth(
            shape,
            float(tokens[1]),
            float(tokens[2]),
            float(tokens[3]),
            int(tokens[4]),
        )

    # Frequency-domain enhancement
    if cmd == "laplacian":
        return laplacian_filter(shape)
    if cmd in ("hfe", "high_emphasis"):
        family = tokens[4] if len(tokens) > 4 else "gaussian"
        return high_frequency_emphasis(
            shape,
            float(tokens[1]),
            float(tokens[2]),
            float(tokens[3]),
            family=family,
        )
    if cmd in ("homo", "homomorphic"):
        gamma_h = float(tokens[2]) if len(tokens) > 2 else 2.0
        gamma_l = float(tokens[3]) if len(tokens) > 3 else 0.5
        c = float(tokens[4]) if len(tokens) > 4 else 1.0
        return homomorphic_filter(shape, float(tokens[1]), gamma_h, gamma_l, c)

    return None
