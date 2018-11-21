'''Create on Nov 2017
@ author: Wei
calculate the hydrogen distribution in the stainless steel
and the outgassing rate on the vaccum
C(t=0, x) =C0, C(t,x=0)=C(t,x=L)=S
asume S = 0, C0 = 1'''

import numpy as np
import matplotlib.pyplot as plt


pi = np.pi
E_D = 60.3 * 10**(3)  # J/mol the bingding energy of absorption atomic hydrogen
D_0 = 0.0122  # cm**2/s  diffussion constant
R = 8.314  # J/mol*K  universal gas constant
# T = np.linspace(285, 1225, 941)  # K heat temperature
L = 0.3  # cm thickness of stainless steel


def f1(k, t):
    ds = np.exp(-(2 * k + 1)**2 * pi**2 * t)
    return ds


def f2(k, t, x):
    ds = (1 / (2 * k + 1)) * \
        np.exp(-(2 * k + 1)**2 * pi**2 * t) * \
        np.sin(((2 * k + 1) / L) * pi * x)
    return ds


def sigma_func(f, n, args=()):
    '''Sigma Function
    sum of f(k, x ...) from k = 0 to k = n-1'''
    sum = 0
    size = len(args)
    for i in range(n):
        if size == 0:
            sum += f(i)
        elif size == 1:
            sum += f(i, args[0])
        elif size == 2:
            sum += f(i, args[0], args[1])
        elif size == 3:
            sum += f(i, args[0], args[1])
        else:
            raise ValueError('args number is incorrect, need rewrite function')
    return sum


def main():
    t = np.arange(0, 28801, 1)
    x = np.arange(0, L + 0.01, 0.01).reshape(int(L / 0.01 + 1), 1)
    T = np.array([373, 523, 673, 873, 1073, 1223]).reshape(6, 1)
    D = D_0 * np.exp(-E_D / (R * T))
    t_ad = t * D * L**(-2)
    args1 = (t_ad,)
    q0 = 4 * D / L * sigma_func(f1, 100, args1)
    Ti = 5
    args2 = (t_ad[Ti, :], x)
    c0 = 4 / pi * sigma_func(f2, 100, args2)
    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.set_ylim([0, 10**(0)])
    plt.figure()
    plt.loglog(t, q0[0, :], t, q0[1, :], t, q0[2, :],
               t, q0[3, :], t, q0[4, :], t, q0[5, :])
    plt.xlabel('Time (s)')
    plt.ylabel('Outgassing rate ($C_0$)')
    plt.legend([r'100$^\circ$', r'250$^\circ$', r'400$^\circ$', r'600$^\circ$',
                r'800$^\circ$', r'950$^\circ$'], loc='best',
               fontsize=12, frameon=False)
    plt.title('Thickness = %.1f cm' % L)

    plt.figure()
    plt.plot(x, c0[:, 1], x, c0[:, 3600], x, c0[:, 7200], x, c0[:, 14400],
             x, c0[:, 21600], x, c0[:, 28800])
    plt.xlabel('depth (cm)')
    plt.ylabel('Hydrogen concentration ($C_0$)')
    plt.legend(['1 s', '1 h', '2 h', '4 h', '6 h', '8 h'],
               loc='center', fontsize=12, frameon=False)
    plt.title(r'Temperature = %.0f$^\circ$' % (T[Ti] - 273))
    plt.show()


if __name__ == '__main__':
    main()
