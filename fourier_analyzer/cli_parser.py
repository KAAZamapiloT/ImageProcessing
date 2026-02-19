from filters import *


def parse_command(command, shape):
    tokens = command.strip().split()
    if not tokens:
        return None

    cmd = tokens[0]

    if cmd == "ideal_lp":
        return ideal_lowpass(shape, float(tokens[1]))

    if cmd == "ideal_hp":
        return ideal_highpass(shape, float(tokens[1]))

    if cmd == "gaussian_lp":
        return gaussian_lowpass(shape, float(tokens[1]))

    if cmd == "gaussian_hp":
        return gaussian_highpass(shape, float(tokens[1]))

    if cmd == "butter_lp":
        return butterworth_lowpass(shape, float(tokens[1]), int(tokens[2]))

    if cmd == "butter_hp":
        return butterworth_highpass(shape, float(tokens[1]), int(tokens[2]))

    if cmd == "bandpass":
        return bandpass(shape, float(tokens[1]), float(tokens[2]))

    if cmd == "bandreject":
        return bandreject(shape, float(tokens[1]), float(tokens[2]))

    if cmd == "notch":
        return notch_filter(shape, int(tokens[1]), int(tokens[2]), float(tokens[3]))

    if cmd == "homo":
        return homomorphic_filter(shape, float(tokens[1]))

    return None
