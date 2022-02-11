"""
Run via
python /Users/amoogle/Documents/Pages/pages_video/spherical_pendulum_sim.py
I have explained the animation tool in
https://youtu.be/rQwnOZFuYsU
"""

import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

mu = 0
sigma = 0.03

SEC_TO_MILLI_SEC = 1000
GRAVITY_ACC = 9.80665  # [m/s^2]

Y_INIT = [
    math.pi / 2.5, # theta
    0, # phi
    0, # theta'
    1 * math.pi / 4.3, # phi'
]

batch = []
# Nice divisors are divisors (2.5,4.2), (4, 6), (5, 2) and (1.1, inf)

PENDULUM_LENGTH = 3 # [m] # used in integration scheme as well as for plots

DT = 1 / 30

csv_ver=0


def transpose(points):
    if not points:
        return []
    dim = len(points[0])
    return [[point[i] for point in points] for i in range(dim)]


def angles_to_unit_sphere(theta, phi):
    x = math.sin(theta) * math.cos(phi)+random.gauss(mu,sigma)
    y = math.sin(theta) * math.sin(phi)+random.gauss(mu,sigma)
    z = math.cos(theta)+random.gauss(mu,sigma)
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
    # print('integrate_y : ',y)

    while t < t_fin:
        # print('yield y')
        yield y

        y += dy(t, y)
        t += dt
        # print('y+= ; ',y)
        # print('t+= ; ',t)


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
        # print('k1, k2, k3, k4 : ',k1,', ',k2,', ',k3,', ',k4)
        WEIGHTS = [1, 2, 2, 1]
        f_rk4 = weighted_mean(WEIGHTS, [k1, k2, k3, k4])
        # print('f_rk4 : ',f_rk4)

        return f_rk4 * dt
    # print('return dy')
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
        sphere_point = angles_to_unit_sphere(*q(y))
        sphere_point[2] *= -1 # flip z-axis
        point = [PENDULUM_LENGTH * p for p in sphere_point]
        yield point

global i
i=0
def _plot_pendulum(fig,csv_ver, ax, points):
    global i
    BOX_SIZE = 1.2 * PENDULUM_LENGTH
    BOWL_RESOLUTION = 24

    # Reset axes
    ax.cla()
    ax.set_xlim(-BOX_SIZE, BOX_SIZE)
    ax.set_ylim(-BOX_SIZE, BOX_SIZE)
    ax.set_zlim(-BOX_SIZE / 1.2, BOX_SIZE)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    # Plot bowl
    us = np.linspace(0, 2 * np.pi, BOWL_RESOLUTION)
    vs = np.linspace(0, np.pi / 3, BOWL_RESOLUTION)
    xs = np.outer(np.cos(us), np.sin(vs))
    ys = np.outer(np.sin(us), np.sin(vs))
    zs = np.outer(np.ones(np.size(us)), np.cos(vs))
    coords = [PENDULUM_LENGTH * vs for vs in [xs, ys, -zs]] # Note: scaled and inverted z-axis
    ax.plot_surface(*coords, linewidth=0, antialiased=False, cmap="cool", alpha=.12)

    # Plot lines and points
    ax.plot3D(*transpose([[0, 0, 1.2 * BOX_SIZE], [0, 0, PENDULUM_LENGTH]]), 'brown', linewidth=1)
    ax.plot3D(*transpose([[0, 0, PENDULUM_LENGTH], points[-1]]), 'brown', linewidth=1)
    ax.scatter3D(0, 0, PENDULUM_LENGTH, s=10, c="navy")
    ax.scatter3D(*transpose([points[-1]]), s=80, c="navy")
    ax.scatter3D(*transpose(points), s=1, c="red")

    csv_ver_name = format(csv_ver,'04') # 파일이름은 숫자 네자리 맞춰서 이름짓기
    # print('현재 csv ver : ',csv_ver_name)
    #  처음이면 우선 csv 파일 생성성

    batch.append(points[-1])
    if i==128:
        print('128개의 data 추출 완료')
        plt.close(fig)
        return
    i+=1

class PlotStream:
    __FPS = 60  # [1 / s]
    __INTERVAL = SEC_TO_MILLI_SEC / __FPS  # [ms]

    def __init__(self):
        self.__fig = plt.figure(figsize=(8, 8))
        self.__ax = self.__fig.gca(projection='3d')
        self.__stream = _pendulum_physical_space()
        self.__past_points = []

    def run(self):
        """
        Warning: Render loops might work different on different OS's and so
        this might need different arguments and a return value for __next_frame
        """
        print()
        _animation = matplotlib.animation.FuncAnimation(self.__fig, self.__next_frame, interval=self.__INTERVAL) # blit=True
        plt.show()

    def __next_frame(self, i):
        next_point = next(self.__stream)
        self.__past_points.append(next_point)
        _plot_pendulum(self.__fig,csv_ver,self.__ax, self.__past_points)
        # print(i, next_point) # Uncomment to print stream of pendulum points
        # return self.__ax,
        # plt.pause(0.03)


if __name__ == '__main__':
    # tmp = Y_INIT
    npy_ver=0
    for m in range(-8,8,1):
        for j in range(-312,313,1):
            _m=m/10
            _j=j/100
            # _m+=1.6 # 최댓값 확인용
            # _j+=6 # 최댓값 확인용
            print('--------------------------------------------------------')
            print('현재 m,j : ',_m,', ',_j)
            Y_INIT = [math.pi / 2.5+_m, 0+_j, 0, 1 * math.pi / 4.3] # change initial value
            print('현재 Y_INIT : ',Y_INIT)

            PlotStream().run() # csv_ver 이름
            np_batch = np.array(batch)
            np.save('./noisy_batch/data'+str(format(npy_ver,'05'))+'.npy',np_batch)
            batch.clear()
            npy_ver+=1
            i=0
            # print('한바퀴 돌았음')
