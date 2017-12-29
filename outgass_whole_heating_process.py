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
time = 45 * 10**2 + 1
t1 = 1440
t2 = 2880
t3 = 4320
T0 = 297
T1 = 1223


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
        d = D[i]
        k = K[i]
        s = S[i]
        for j in range(int(steps)):
            u = compute_u_time(u, d, h, dt, k, s)
        u_data[i, :] = u
    return u_data


def compute_outgass_T(u, D, time, T):
    # q_data = np.linspace(0, time, time)
    q_data = D * (u[:, 1] - u[:, 0]) / h  # atom/(s*cm**2)
    q_data = q_data * R * T / (133.322 * 6.022 * 10**20)
    return q_data


def set_temperature(time):
    T = np.zeros(time)
    for i in np.arange(0, t1):
        T[i] = T0 + (T1 - T0) * i / t1
    for i in np.arange(t1, t2):
        T[i] = T1
    for i in np.arange(t2, t3):
        T[i] = T1 - (T1 - T0) * (i - t2) / (t3 - t2)
    for i in np.arange(t3, time):
        T[i] = T0
    return T


def plot_outgassing(t, q_data):
    plt.figure()
    plt.semilogy(t, q_data)
    plt.xlabel('Time (s)')
    plt.ylabel('Outgassing rate ($torr \cdot L \cdot s^{-1} \cdot cm^{-2}$)')
    plt.title('Thickness = 0.3 cm')
    plt.grid(True, which='major', axis='both', ls='-')
    plt.axvline(x=t1, lw=1, c='r', ls='--')
    plt.text((t1 / 6), 0.01 * q_data[t1], '297 K to 1223 K')
    plt.axvline(x=t2, lw=1, c='r', ls='--')
    plt.text((t1 + (t2 - t1) / 3), 0.01 * q_data[t1], '1223 K')
    plt.text((t2 + (t3 - t2) / 3), 0.01 * q_data[t1], '1223 K to 297 K')
    figname = 'outgassing_time' + str(len(t)) + '.pdf'
    plt.savefig(figname, format='pdf')


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
    plt.savefig('H_concentration1_1223K.pdf', format='pdf')
    plt.figure()
    plt.plot(x, u_data[5000, :], x, u_data[10000, :], x, u_data[40000, :],
             x, u_data[80000, :], x, u_data[100000, :])
    plt.xlim([0, 0.3])
    plt.xlabel('depth (cm)')
    plt.ylabel('Hydrogen concentration ($atom \cdot cm^{-3}$)')
    plt.legend(['1 s', '10 s', '100 s', '500 s', '1000 s', '5000 s'],
               loc='best', fontsize=12, frameon=False)
    plt.title('Temperature = 950$^\circ$')
    plt.savefig('H_concentration2_1223K.pdf', format='pdf')


def save_data(q_data, u_data):
    q_data = q_data * 10**10
    len1 = np.size(q_data)
    N = int(100 * np.log10(len1))
    file1_name = 'outgassing' + str(len1) + '.txt'
    file1 = open(file1_name, 'w+')
    var = -1
    file1.write('%.1f ' % 0)
    file1.write('%.4f ' % q_data[0])
    file1.write('\n')
    for i in range(N + 1):
        i = int(10**(i / 100))
        if (i != var):
            file1.write('%.1f ' % i)
            file1.write('%.4f ' % q_data[i])
            file1.write('\n')
        var = i
    file1.close()

# save the concetration in the steel at t = 0,1,2...9,10,20,30...90,100,200...
    len2 = np.size(u_data[:, 0])
    file2_name = 'concentration' + str(len2) + '.txt'
    file2 = open(file2_name, 'w+')
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


def main():
    t = np.arange(0, time)
    T = set_temperature(time)

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

    u_data = compute_concentration(D, time, C0, K, S)  # atom/cm**3
    q_data = compute_outgass_T(u_data, D, time, T)
    save_data(q_data, u_data)
#    plot_concentration(x, u_data)
    plot_outgassing(t, q_data)
    plt.show()


if __name__ == '__main__':
    main()
