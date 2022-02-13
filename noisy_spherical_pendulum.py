"""
Run via
python /Users/amoogle/Documents/Pages/pages_video/spherical_pendulum_sim.py
I have explained the animation tool in
https://youtu.be/rQwnOZFuYsU
"""

import math
import random

import numpy as np

mu = 0
sigma = 0.03
B = 10000 # Batch
T = 128 # Step

SEC_TO_MILLI_SEC = 1000
GRAVITY_ACC = 9.80665  # [m/s^2]

Y_INIT = [
    math.pi / 2.5, # theta
    0, # phi
    0, # theta'
    1 * math.pi / 4.3, # phi'
]
PENDULUM_LENGTH = 1 # [m] # used in integration scheme as well as for plots

DT = 1 / 50


def angles_to_unit_sphere(theta, phi):
    theta += random.gauss(mu, sigma)
    phi += random.gauss(mu, sigma)

    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)
    return [x, y, z]


def weighted_mean(weights, values):
    assert len(weights) == len(values)

    terms = (w * v for w, v in zip(weights, values))

    return sum(terms) / sum(weights)


def integrate(f, dt, y_init, t_init=0, t_fin=float('inf')):
    # Solves y'(t) = f(t, y) for a seqeunce of ys.

    dy = _dy_rk4_method(f, dt)
    y = y_init
    t = t_init

    while t < t_fin:
        yield y

        y += dy(t, y)
        t += dt


def _dy_rk4_method(f, dt):
    """
    See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """

    def f_(t, y):
        return np.array(f(t, y)) # Guarantee possibility of vector addition

    def dy(t, y):
        k1 = f_(t, y)
        k2 = f_(t + dt / 2, y + k1 * dt / 2)
        k3 = f_(t + dt / 2, y + k2 * dt / 2)
        k4 = f_(t + dt, y + k3 * dt)
        WEIGHTS = [1, 2, 2, 1]
        f_rk4 = weighted_mean(WEIGHTS, [k1, k2, k3, k4])

        return f_rk4 * dt
    return dy


def _pendulum_configuration_space():
    """
    Solves F = r''(t) for the pendulum with phase space. Force norm in physical space:
        |F| = m g.
    Uses generalized coordinates q = (theta, phi).
    For the second order equations q'' = g(q, q') for the pendulum, see
        https://en.wikipedia.org/wiki/Spherical_pendulum
    Potential in configuration space:
        V(theta) = |F| * l (1 - cos(theta))
        V(0) = 0
        V(pi/2) = F l
    Approach: Reduce second order ODE to first order ODE.
    I.e. solve y'(t) = f(t, y) where y = (q, q') and f(t, y) = (q', g(q, q)).
    """

    def f(_t, y):
        return dq(y) + d2q(y)

    def dq(y):
        dof = len(y) // 2
        return list(y)[dof:] # Project out LAST two components

    def d2q(y):
        theta, _phi, d_theta, d_phi = y

        if theta>=math.pi/2:
            tan = 10
        elif theta <= -math.pi/2: # theta가 0보다 작아질 일이 없지 않나..?
            tan = -10
        elif theta ==0:
            tan = 0.01
        else:
            tan = math.tan(theta)

        d2_theta = (d_phi**2 * math.cos(theta) - GRAVITY_ACC / PENDULUM_LENGTH) * math.sin(theta)
        d2_phi = -2 * d_theta * d_phi / math.tan(theta)

        return [d2_theta, d2_phi]

    return integrate(f, DT, Y_INIT)


def _pendulum_physical_space():
    """
    Note: angles_to_unit_sphere returns point in standard sphere coordinates.
    The model assums theta=0 corresponds to lowest potential energy, so we need to flip the z-axis.
    """

    def q(y):
        dof = len(y) // 2
        return y[:dof] # Project out FIRST two components

    for y in _pendulum_configuration_space():
        sphere_point = angles_to_unit_sphere(*q(y)) # Observation할 때 noise
        sphere_point[2] *= -1 # flip z-axis
        point = [PENDULUM_LENGTH * p for p in sphere_point]

        yield point

class PlotStream:
    __FPS = 60  # [1 / s]
    __INTERVAL = SEC_TO_MILLI_SEC / __FPS  # [ms]

    def __init__(self):
        self.__stream = _pendulum_physical_space()
        self.__past_points = []

    def run(self):
        """
        Warning: Render loops might work different on different OS's and so
        this might need different arguments and a return value for __next_frame
        """
        for i in range(T):
            next_point = next(self.__stream)
            self.__past_points.append(next_point)
        return self.__past_points


def generate_noisy_spherical_pendulum_dataset(mu=0, sigma=0.03,B=10000, T = 128):
    mu = mu
    sigma = sigma
    B = B # Batch
    T = T # Step


    npy_ver = 0
    batch_np = np.empty((B, T, 3))

    x = np.linspace(-math.pi / 5, math.pi / 5, int(math.sqrt(B)))
    y = np.linspace(-math.pi / 5, math.pi / 5, int(math.sqrt(B)))

    for _m in x:
        for _j in y:
            v_theta = random.uniform(0,1) # random initial theta velocity
            v_phi = random.uniform(0,1) # random initial phi velocity
            Y_INIT = [-1+_m, 0+_j, 0+v_theta, 0.3+v_phi] # change initial value

            points = PlotStream().run()
            batch_np[npy_ver, :] = np.array(points)
            npy_ver+= 1

    #np.save('./data/NoisySphericalPendulum.npy', batch_np)

    return batch_np