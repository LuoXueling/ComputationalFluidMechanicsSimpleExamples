import numpy as np
import matplotlib.pyplot as plt
import time
import math
import copy
from sympy import *

from scipy.optimize import fsolve

plt.rcParams["figure.dpi"] = 300

# 节点个数
N = 31
# 长度
L = 3.0
# 科朗数
C = 0.5
# 松弛因子，越小越准确，用时越长
w = 0.9
# 气体参数
gamma = 1.4
p0 = 1
T0 = 1
rho0 = 1
# 物理边界
rho_boundary = 1
T_boundary = 1
# 全亚声速压强比
pe = 0.93
# 最大迭代次数
max_iter = 15000


def sub2super():
    def A(i):
        x = X[i]
        return 1 + 2.2 * (x - 1.5) ** 2

    def left_boundary(V, T, rho):
        return 2 * V[1] - V[2], \
               T_boundary, \
               rho_boundary

    def right_boundary(V, T, rho):
        return 2 * V[-2] - V[-3], \
               2 * T[-2] - T[-3], \
               2 * rho[-2] - rho[-3]

    rho = 1 - 0.3146 * X
    T = 1 - 0.2314 * X
    V = (0.1 + 1.09 * X) * np.sqrt(T)
    rho[0] = rho_boundary
    T[0] = T_boundary

    V, T, rho, lms = process(rho, T, V, left_boundary, right_boundary, A)

    M = V / np.sqrt(T)
    p = rho * T

    # 求解析解
    # Ma = Symbol('Ma')
    # resMa = []
    # for i in range(N):
    #     print("solving " + str(i))
    #     res = solve(
    #         1 / Ma ** 2 * (2 / (1 + gamma) * (1 + (gamma - 1) / 2 * Ma ** 2)) ** ((gamma + 1) / (gamma - 1)) - A(
    #             i) ** 2, Ma)
    #     for j in res:
    #         # 提取大于0的实数解
    #         if (not isinstance(j, add.Add)) and j > 0:
    #             # 在喉道前Ma<1
    #             if i < round((N - 1) / 2) and j < 1.01:
    #                 print(j)
    #                 resMa.append(j)
    #             elif i >= round((N - 1) / 2) and j >= 0.99:
    #                 print(j)
    #                 resMa.append(j)
    # print(resMa)

    def func(Ma):
        Ma = Ma[0]
        return 1 / Ma ** 2 * (2 / (1 + gamma) * (1 + (gamma - 1) / 2 * Ma ** 2)) ** ((gamma + 1) / (gamma - 1)) - A(
            i) ** 2

    resMa = []
    for i in range(N):
        res = fsolve(func, np.array(M[-1][i]))
        for j in res:
            # 在喉道前Ma<1
            if i < round((N - 1) / 2) and j < 1.01:
                resMa.append(j)
            elif i >= round((N - 1) / 2) and j >= 0.99:
                resMa.append(j)
    print(resMa)

    plots(T, M, rho, p, resMa, lms)


def sub2sub():
    def A(i):
        x = X[i]
        if x <= 1.5:
            return 1 + 2.2 * (x - 1.5) ** 2
        else:
            return 1 + 0.2223 * (x - 1.5) ** 2

    def left_boundary(V, T, rho):
        return 2 * V[1] - V[2], \
               T_boundary, \
               rho_boundary

    def right_boundary(V, T, rho):
        return 2 * V[-2] - V[-3], \
               2 * T[-2] - T[-3], \
               pe / (2 * T[-2] - T[-3])

    rho = 1 - 0.023 * X
    T = 1 - 0.009333 * X
    V = 0.05 + 0.11 * X
    rho[0] = rho_boundary
    T[0] = T_boundary

    V, T, rho, lms = process(rho, T, V, left_boundary, right_boundary, A)

    M = V / np.sqrt(T)
    p = rho * T

    # 求解析解
    Mae = math.sqrt((pe ** (-(gamma - 1) / gamma) - 1) * 2 / (gamma - 1))
    A_star = math.sqrt(A(N - 1) ** 2 * Mae ** 2 / math.pow((2 / (gamma + 1) * (1 + (gamma - 1) / 2 * Mae ** 2)),
                                                           (gamma + 1) / (gamma - 1)))

    def func(Ma):
        Ma = Ma[0]
        return 1 / Ma ** 2 * (2 / (1 + gamma) * (1 + (gamma - 1) / 2 * Ma ** 2)) ** ((gamma + 1) / (gamma - 1)) - (A(
            i) / A_star) ** 2

    resMa = []
    for i in range(N):
        res = fsolve(func, np.array(M[-1][i]))
        for j in res:
            if 0 <= j <= 1:
                resMa.append(j)
    print(resMa)

    # Ma = Symbol('Ma')
    # resMa = []
    # for i in range(N):
    #     print("solving " + str(i))
    #     res = solve(
    #         1 / Ma ** 2 * (2 / (1 + gamma) * (1 + (gamma - 1) / 2 * Ma ** 2)) ** ((gamma + 1) / (gamma - 1)) - (
    #                     A(i) / A_star) ** 2, Ma)
    #     for j in res:
    #         # 提取大于0的实数解
    #         if (not isinstance(j, add.Add)) and j > 0 and j < 1.01:
    #             print(j)
    #             resMa.append(j)
    # print(resMa)
    #
    # resMa=[0.0769554395925505,0.0862768289191514,0.0972564726051758,0.110268344453099,0.125780021784377,
    #        0.144372361979675,0.166756541634194,0.193778538400759,0.226389141109455,0.265534246699390,
    #        0.311877963162380,0.365204508630606,0.423278688558804,0.480057986663648,0.524148297244693,
    #        0.541249752282421,0.539456102571784,0.534166438069894,0.525640461165929,0.514268423501379,
    #        0.500520814289821,0.484898067252033,0.467889248902202,0.449943549963813,0.431454237337239,
    #        0.412752509973754,0.394108142387847,0.375734173175937,0.357793597367118,0.340406708873908,0.323658280425811]

    plots(T, M, rho, p, resMa, lms)


def process(rho, T, V, left_boundary, right_boundary, A):
    start = time.process_time()

    def iter_lms(array):
        # 计算每列（同一位置）均值
        mean = np.mean(array[-50:], axis=0)
        lms = np.zeros((1, N))
        for n in range(-50, 0):
            lms = lms + np.power(array[n] - mean, 2)
        lms = np.sqrt(lms)
        return np.linalg.norm(lms)

    def all_lms():
        lms.append(iter_lms(rho) + iter_lms(V) + iter_lms(T))
        return lms[-1]

    def av_partial_rho(n, i):
        return 0.5 * (forw_partial_rho(n, i) + bkw_partial_rho(n + 1, i))

    def av_partial_V(n, i):
        return 0.5 * (forw_partial_V(n, i) + bkw_partial_V(n + 1, i))

    def av_partial_T(n, i):
        return 0.5 * (forw_partial_T(n, i) + bkw_partial_T(n + 1, i))

    def forw_partial_rho(n, i):
        return (- V[n][i] * (rho[n][i + 1] - rho[n][i])
                - rho[n][i] * (V[n][i + 1] - V[n][i])
                - rho[n][i] * V[n][i] * np.log(A(i + 1) / A(i))) / dx

    def forw_partial_V(n, i):
        return (- V[n][i] * (V[n][i + 1] - V[n][i])
                - (1 / gamma) * (T[n][i + 1] - T[n][i])
                - (1 / gamma) * T[n][i] / rho[n][i] * (rho[n][i + 1] - rho[n][i])) / dx

    def forw_partial_T(n, i):
        return (- V[n][i] * (T[n][i + 1] - T[n][i])
                - (gamma - 1) * T[n][i] * (V[n][i + 1] - V[n][i])
                - (gamma - 1) * T[n][i] * V[n][i] * np.log(A(i + 1) / A(i))) / dx

    def bkw_partial_rho(n, i):
        return (- V[n][i] * (rho[n][i] - rho[n][i - 1])
                - rho[n][i] * (V[n][i] - V[n][i - 1])
                - rho[n][i] * V[n][i] * np.log(A(i) / A(i - 1))) / dx

    def bkw_partial_V(n, i):
        return (- V[n][i] * (V[n][i] - V[n][i - 1])
                - (1 / gamma) * (T[n][i] - T[n][i - 1])
                - (1 / gamma) * T[n][i] / rho[n][i] * (rho[n][i] - rho[n][i - 1])) / dx

    def bkw_partial_T(n, i):
        return (- V[n][i] * (T[n][i] - T[n][i - 1])
                - (gamma - 1) * T[n][i] * (V[n][i] - V[n][i - 1])
                - (gamma - 1) * T[n][i] * V[n][i] * np.log(A(i) / A(i - 1))) / dx

    lms = []
    n = 0
    while (n < 50 or all_lms() > 1e-2) and n < max_iter:
        rho = np.row_stack((rho, np.zeros(N)))
        V = np.row_stack((V, np.zeros(N)))
        T = np.row_stack((T, np.zeros(N)))
        dt = C * dx / (V[n] + np.sqrt(T[n]))
        dt = np.min(dt)
        # 预估
        for i in range(1, N - 1):
            rho[n + 1][i] = rho[n][i] + w * forw_partial_rho(n, i) * dt
            V[n + 1][i] = V[n][i] + w * forw_partial_V(n, i) * dt
            T[n + 1][i] = T[n][i] + w * forw_partial_T(n, i) * dt
        # print(rho[-1])
        # 左边界
        V[n + 1][0], T[n + 1][0], rho[n + 1][0] = left_boundary(V[n + 1], T[n + 1], rho[n + 1])

        tmprho = copy.deepcopy(rho[n + 1])
        tmpV = copy.deepcopy(V[n + 1])
        tmpT = copy.deepcopy(T[n + 1])
        # 修正，av中用到预估的值
        for i in range(1, N - 1):
            tmprho[i] = rho[n][i] + w * av_partial_rho(n, i) * dt
            tmpV[i] = V[n][i] + w * av_partial_V(n, i) * dt
            tmpT[i] = T[n][i] + w * av_partial_T(n, i) * dt

        rho[n + 1] = tmprho
        V[n + 1] = tmpV
        T[n + 1] = tmpT

        # 右边界，在最后求能够利用修正值，精度稍高一些
        V[n + 1][-1], T[n + 1][-1], rho[n + 1][-1] = right_boundary(V[n + 1], T[n + 1], rho[n + 1])

        n = n + 1
    end = time.process_time()
    print("Finished in " + str(n - 1) + " iterations, time: " + str(end - start) + "s.")
    return V, T, rho, lms


def plots(T, M, rho, p, resMa, lms):
    plt.figure()
    plt.title("Iteration")
    plt.plot(T[:, round((N - 1) / 2)], label="T*")
    plt.plot(M[:, round((N - 1) / 2)], label="Ma")
    plt.plot(p[:, round((N - 1) / 2)], label="p")
    plt.plot(rho[:, round((N - 1) / 2)], label=r"$\rho*$")
    plt.xlabel('Iteration step')
    plt.ylabel('Dimensionless value')
    plt.legend()
    plt.show()
    plt.close()

    plt.figure()
    plt.title("Result along field")
    plt.plot(X, T[-1], label="T*")
    plt.plot(X, M[-1], label="Ma")
    plt.plot(X, p[-1], label="p")
    plt.plot(X, rho[-1], label=r"$\rho*$")
    plt.scatter(X, resMa, s=10, label="Ma(analytical result)")
    plt.xlabel('x*')
    plt.ylabel('Dimensionless value')
    plt.legend()
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(list(range(50, 50 + len(lms))), lms)
    plt.title("lms during iteration")
    plt.xlabel('Iteration step')
    plt.ylabel('Root-mean-square')
    plt.show()
    plt.close()


if __name__ == "__main__":
    dx = L / (N - 1)
    X = np.linspace(0, L, N)
    try:
        sub2super()
    except Exception as e:
        print("Exception occured in process")
        print(e)
