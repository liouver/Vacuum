''' Create on Nov 2017
@ author: Wei
calculate the hydrogen distribution in the stainless steel
and the outgassing rate on the vaccum
slove the diffusion eqaution: du(x,t)/dt = D * d2u(x,t)/dx2 = g(x, t)
u(t=0,x)=C0, u(t,x=0)=S0, u(t,x=L)=S1'''
# PV = nRT (pa, m**3, mol, J/mol*K, K)

import numpy as np
import matplotlib.pyplot as plt

E_D = 60.3 * 10**(3)  # J/mol the bingding energy of absorption atomic hydrogen
D_0 = 0.0122  # cm**2/s  diffussion constant
R = 8.314  # J/mol*K  universal gas constant
L = 0.3  # cm thickness of stainless steel

N = 301  # number of subdivisions
x = np.linspace(0, L, N)
h = x[1] - x[0]  # discretisation stepsize in x - direction
# C0 = 1
# S0 = 0
# S1 = 0
# S1 = 1.336 * np.exp(-0.918 * 10**3 / T) # H/R = 0.918 * 10**3
dt = 0.01  # step size or time


def compute_g(u, D, h):
    """ given a u (x , t ) in array , compute g (x , t )= D * d ^2 u / dx ^2
    using central differences with spacing h ,
    and return g (x , t ). """
    d2u_dx2 = np.zeros(u. shape, np . float)
    for i in range(1, len(u) - 1):
        d2u_dx2[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / h ** 2
    # special cases at boundary : assume Neuman boundary
    # conditions , i . e . no change of u over boundary
    # so that u [0] - u [ -1]=0 and thus u [ -1]= u [0]
    # i = 0
    # d2u_dx2[i] = (u[i + 1] - 2 * u[i] + u[i]) / h ** 2
    # same at other end so that u [N -1] - u [ N ]=0
    # and thus u [ N ]= u [N -1]
    # i = len(u) - 1
    # d2u_dx2[i] = (u[i] - 2 * u[i] + u[i - 1]) / h ** 2
    return D * d2u_dx2


def compute_outgass(u, D, h):
    du_dx_0 = (u[1] - u[0]) / (2 * h)
    return D * du_dx_0


def advance_time(u, g, dt):
    """ Given the array u , the rate of change array g ,
    and a timestep dt , compute the solution for u
    after t , using simple Euler method . """
    u = u + dt * g
    return u


def time_dependent_outgass(D, time, C0, S0, S1):
    ''' Given the time to be calculated, compute the outgassing rate q(t),
     and the hydrogen concentration u(x ,t) '''
    u = C0 * np.ones(np.size(x))
    u[0] = S0
    u[-1] = S1
    steps = 1 / dt
    u_line = np.arange(0, time).reshape(time, 1)
    u_data = u * u_line
    q = []
    for i in range(time):
        for j in range(int(steps)):
            g = compute_g(u, D, h)
            u = advance_time(u, g, dt)
        q.append(compute_outgass(u, D, h))
        u_data[i, :] = u
    return q, u_data


def main():
    time = 1 * 10**4 + 1
    t = np.arange(0, time)
    T = np.array([373, 523, 673, 873, 1073, 1223])
    D = D_0 * np.exp(-E_D / (R * T))
    S1 = 1.336 * np.sqrt(760) * np.exp(-0.918 * 10**3 / T) \
        * np.sqrt(3.8 * 10**(-4))
    print(S1)
    S0 = 0
    C0 = 5 * S1
    q_data = T * np.linspace(0, time, time).reshape(time, 1)
    for i in range(np.size(T)):
        q, u_data = time_dependent_outgass(D[i], time, C0[i], S0, S1[i])
        q_data[:, i] = np.array(q)
    q, u_data = time_dependent_outgass(D[5], time, C0[5], S0, S1[5])

    plt.figure()
    plt.loglog(t, q_data[:, 0], t, q_data[:, 1], t, q_data[:, 2],
               t, q_data[:, 3], t, q_data[:, 4], t, q_data[:, 5])
    plt.xlabel('Time (s)')
    plt.ylabel('Outgassing rate ($C_0$)')
    plt.legend(['100$^\circ$', '250$^\circ$', '400$^\circ$', '600$^\circ$',
                '800$^\circ$', '950$^\circ$', '100$^\circ$'],
               loc='best', fontsize=12, frameon=False)
    plt.title('Thickness = 0.3 cm')

    plt.figure()
    plt.plot(x, u_data[0, :], x, u_data[10, :], x, u_data[100, :], x, u_data[
             500, :], x, u_data[1000, :], x, u_data[5000, :])
    plt.xlabel('depth (mm)')
    plt.ylabel('Hydrogen concentration ($torr$)')
    plt.legend(['0 s', '10 s', '100 s', '500 s', '1000 s', '5000 s'],
               loc='best', fontsize=12, frameon=False)
    plt.title('Temperature = 950$^\circ$')
    plt.show()


if __name__ == '__main__':
    main()
