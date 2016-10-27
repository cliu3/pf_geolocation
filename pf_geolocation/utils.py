import numpy as np
from scipy.optimize import curve_fit


def fit(x, data):
    def my_sin(x, freq, amplitude, phase, offset):
        return amplitude * np.sin(x * freq + phase) + offset

    p = 12.420601
    guess_freq = 2 * np.pi / (p / 24)
    guess_amplitude = 3 * np.std(data) / (2**0.5)
    guess_phase = 0
    guess_offset = np.mean(data)

    p0 = [guess_freq, guess_amplitude,
          guess_phase, guess_offset]

    fit = curve_fit(my_sin, x, data, p0=p0)

    data_fit = my_sin(x, *fit[0])

    return data_fit


def predict(times, amps, phases):
    """
    Predict tidal elevation using harmonics of first 8 most significant 
    tidal constituents.

    """

    from pytides.tide import Tide
    import pytides.constituent as cons
    from datetime import datetime

    constituents = [cons._M2, cons._N2, cons._S2,
                    cons._O1, cons._K1, cons._K2, cons._P1, cons._Q1]

    model = np.zeros(len(constituents), dtype=Tide.dtype)
    model['constituent'] = constituents
    model['amplitude'] = amps
    model['phase'] = phases

    tide = Tide(model=model, radians=False)

    pred = tide.at(times)
    return pred
