''' Create on Dec 2017 @ author: Wei
calculate the hydrogen distribution in the stainless steel
and the outgassing rate on the vaccum
slove the diffusion eqaution: du(x,t)/dt = D * d2u(x,t)/dx2 = g(x, t)
u(t=0,x)=C0, u(t,x=0)=0, u(t,x=L)=S'''
# PV = nRT (pa, m**3, mol, J/mol*K, K)
# Ref: J. Vac. Sci. Technol. A 16, 188 (1998)
# Ref: J. Vac. Sci. Technol. A 13, 545 (1995)

import numpy as np
import matplotlib.pyplot as plt

E_D = 54 * 10**(3)  # J/mol the bingding energy of absorption atomic hydrogen
D_0 = 0.0089  # cm**2/s  diffussion constant
R = 8.314  # J/mol*K  universal gas constant

L = 0.3  # cm thickness of stainless steel
N = 301  # number of subdivisions
x = np.linspace(0, L, N)
h = x[1] - x[0]  # discretisation stepsize in x - direction
dt = 0.005  # step size or time
# K = 4 * 10**(-23)  # cm**4/(atom * s)


def compute_u_time(u, D, h, dt, K, S):
    """ given a u (x , t ) in array , compute g (x , t )= D * d ^2 u / dx ^2
    using central differences with spacing h ,
    and return g (x , t ).
    Then compute the solution for u after t, u = u + g * dt"""
    # # # boundary condtion
    u[0] = (-D + np.sqrt(D**2 + 4 * D * K * h * u[1])) / (2 * K * h)
    u[-1] = (-D + np.sqrt(D**2 + 4 * D * K * h * (u[-2] - S))) / (2 * K * h)\
        + S
    # # # within stainless steel
    d2u_dx2 = np.zeros(u. shape, np . float)
    for i in range(1, len(u) - 1):
        d2u_dx2[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / h ** 2
    for i in range(1, len(u) - 1):
        u[i] = u[i] + D * d2u_dx2[i] * dt
    return u


def compute_concentration(D, time, C0, K, S):
    ''' Given the time to be calculated, compute the outgassing rate q(t),
     and the hydrogen concentration u(x ,t) '''
    u = C0 * np.ones(np.size(x))
    steps = 1 / dt
    u_line = np.arange(0, time).reshape(time, 1)
    u_data = u * u_line
    for i in range(time):
        for j in range(int(steps)):
            u = compute_u_time(u, D, h, dt, K, S)
        u_data[i, :] = u
    return u_data


def compute_outgass_T(D, time, C0, K, S, T):
    q_data = T * np.linspace(0, time, time).reshape(time, 1)
    for i in range(len(T)):
        u = compute_concentration(D[i], time, C0, K[i], S[i])
        q_data[:, i] = D[i] * (u[:, 1] - u[:, 0]) / h  # atom/(s*cm**2)
        q_data[:, i] = q_data[:, i] * R * T[i] / (133.322 * 6.022 * 10**20)
    return q_data


def save_data(q_data, u_data):
    q_data = q_data * 10**10
    len1 = np.size(q_data[:, 0])
    N = int(20 * np.log10(len1))
    file1 = open('outgassing.txt', 'w+')
    var = -1
    file1.write('%.1f ' % 0)
    for j in range(np.size(q_data[0, :])):
        file1.write('%.2f ' % q_data[0, j])
    file1.write('\n')
    for i in range(N + 1):
        i = int(10**(i / 20))
        if (i != var):
            file1.write('%.1f ' % i)
            for j in range(np.size(q_data[0, :])):
                file1.write('%.4f ' % q_data[i, j])
            file1.write('\n')
        var = i
    file1.close()
    len2 = np.size(u_data[:, 0])
    file2 = open('concentration.txt', 'w+')
    file2.write('%.1f ' % 0)
    for j in range(np.size(u_data[0, :])):
        file2.write('%.1f ' % u_data[0, j])
    file2.write('\n')
    for a in range(int(np.log10(len2))):
        for b in range(9):
            i = (b + 1) * 10**a
            file2.write('%.1f ' % i)
            for j in range(np.size(u_data[0, :])):
                file2.write('%.1f ' % u_data[i, j])
            file2.write('\n')
    len2 = len2 - 1
    file2.write('%.1f' % len2)
    for j in range(np.size(u_data[0, :])):
        file2.write('%.1f ' % u_data[len2, j])
    file2.write('\n')
    file2.close()


def plot_outgassing(t, q_data):
    plt.figure()
    plt.loglog(t, q_data[:, 0], t, q_data[:, 1], t, q_data[:, 2],
               t, q_data[:, 3], t, q_data[:, 4], t, q_data[:, 5])
    plt.xlabel('Time (s)')
    plt.ylabel('Outgassing rate ($torr \cdot L \cdot s^{-1} \cdot cm^{-2}$)')
    plt.legend(['100$^\circ$', '250$^\circ$', '400$^\circ$', '600$^\circ$',
                '800$^\circ$', '950$^\circ$', '100$^\circ$'],
               loc='best', fontsize=12, frameon=False)
    plt.title('Thickness = 0.3 cm')
    plt.savefig('outgassing_time', format='pdf')


def plot_concentration(x, u_data):
    plt.figure()
    plt.plot(x, u_data[1, :], x, u_data[10, :], x, u_data[100, :], x, u_data[
             500, :], x, u_data[1000, :], x, u_data[5000, :])
    plt.xlim([0, 0.3])
    plt.xlabel('depth (cm)')
    plt.ylabel('Hydrogen concentration ($atom \cdot cm^{-3}$)')
    plt.legend(['1 s', '10 s', '100 s', '500 s', '1000 s', '5000 s'],
               loc='best', fontsize=12, frameon=False)
    plt.title('Temperature = 950$^\circ$')
    plt.savefig('H_concentration1', format='pdf')
    plt.figure()
    plt.plot(x, u_data[5000, :], x, u_data[10000, :], x, u_data[40000, :],
             x, u_data[80000, :], x, u_data[100000, :])
    plt.xlim([0, 0.3])
    plt.xlabel('depth (cm)')
    plt.ylabel('Hydrogen concentration ($atom \cdot cm^{-3}$)')
    plt.legend(['1 s', '10 s', '100 s', '500 s', '1000 s', '5000 s'],
               loc='best', fontsize=12, frameon=False)
    plt.title('Temperature = 950$^\circ$')
    plt.savefig('H_concentration2', format='pdf')


def main():
    time = 1 * 10**5 + 1
    t = np.arange(0, time)
    T = np.array([373, 523, 673, 873, 1073, 1223])
    #  diffusion constant, Ref: J. Nuclear Materials 128, 622 (1984)
    #  Ref: Vacuum 69 (2003) 501–512, IJNE 32, 100 (2007)
    D = D_0 * np.exp(-E_D / (R * T))
    #  recombination constant, Ref: J. Nuclear Materials 128, 622 (1984)
    K = 1.47127 * 10**(-20) * np.exp(-5996.137 / T)
    #  solubility, Ref: book-'Vacuum technology'; IJNE 32, 100 (2007)
    S = 1.336 * np.sqrt(760) * np.exp(-0.918 * 10**3 / T) \
        * np.sqrt(3.8 * 10**(-4))
    S = S * 133.322 * 6.022 * 10**17 / (273 * 8.314)
    #  Initial concentration, Ref: J. Vac. Sci. Technol. A 13, 545 (1995)
    # C0 = 0.3 torr * L @ 273 K / cm**3
    C0 = 0.3 * 133.322 * 6.022 * 10**20 / (273 * R)  # atom/cm**3

    u_data = compute_concentration(D[5], time, C0, K[5], S[5])  # atom/cm**3
    q_data = compute_outgass_T(D, time, C0, K, S, T)
#     u_data = u_data * R * T[5] / (133.322 * 6.022 * 10**20)  # torr*L/cm**3
#    save_data(q_data, u_data)
    plot_outgassing(t, q_data)
    plot_concentration(x, u_data)
    plt.show()


if __name__ == '__main__':
    main()
