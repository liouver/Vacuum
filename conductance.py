'''
The conductance for orifices: C = 36.24 * sqrt(T / M) * A
For long round tube: 37.94 * sqrt(T / M) * d**3 / l
unit: m**3 / s
'''

import numpy as np

pi = np.pi
T = 297
M_H = 2
M_air = 28.97
d = 0.004763  # m, diameter
L = 0.25  # m, length


def cal_conductance(types):
    if types == 1:
        C = 36.24 * np.sqrt(T / M_H) * pi * (d / 2)**2 * 10**3
    elif types == 2:
        C = 36.24 * np.sqrt(T / M_air) * pi * (d / 2)**2 * 10**3
    elif types == 3:
        C = 37.94 * np.sqrt(T / M_H) * d**3 / L * 10**3
    elif types == 4:
        C = 37.94 * np.sqrt(T / M_air) * d**3 / L * 10**3
    print(C)
    return C


cal_conductance(1)
