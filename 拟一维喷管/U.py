import numpy as np
import matplotlib.pyplot as plt
import time
import math
from sympy import *
import copy

from scipy.optimize import fsolve

plt.rcParams["figure.dpi"] = 300

# 节点个数
N = 61
# 长度
L = 3.0
# 科朗数
C = 0.5
# 松弛因子，越小越准确，用时越长
w = 1
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
max_iter = 5000
# 人工粘性系数
Cx = 0.4


def sub2super_u():
    def A(i):
        x = X[i]
        return 1 + 2.2 * (x - 1.5) ** 2

    def left_boundary(U1, U2, U3):
        V_ = (2 * U2[1] - U2[2]) / (rho_boundary * A(0))
        return rho_boundary * A(0), \
               2 * U2[1] - U2[2], \
               rho_boundary * A(0) * (T_boundary / (gamma - 1) + gamma / 2 * V_ ** 2)

    def right_boundary(U1, U2, U3):
        return 2 * U1[-2] - U1[-3], \
               2 * U2[-2] - U2[-3], \
               2 * U3[-2] - U3[-3]

    rho = np.zeros((1, N))
    T = np.zeros((1, N))
    for i in range(N):
        x = X[i]
        if x <= 0.5:
            rho[0][i] = 1
            T[0][i] = 1
        elif x <= 1.5:
            rho[0][i] = 1 - 0.366 * (x - 0.5)
            T[0][i] = 1 - 0.167 * (x - 0.5)
        else:
            rho[0][i] = 0.634 - 0.3879 * (x - 1.5)
            T[0][i] = 0.833 - 0.3507 * (x - 1.5)
    A_arr = np.array([A(i) for i in range(N)])
    V = 0.59 / (rho * A_arr)

    V, T, rho, lms = process(rho, T, V, left_boundary, right_boundary, A)
    M = V / np.sqrt(T)
    p = rho * T

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


def shock():
    def A(i):
        x = X[i]
        return 1 + 2.2 * (x - 1.5) ** 2

    def left_boundary(U1, U2, U3):
        V_ = (2 * U2[1] - U2[2]) / (rho_boundary * A(0))
        return rho_boundary * A(0), \
               2 * U2[1] - U2[2], \
               rho_boundary * A(0) * (T_boundary / (gamma - 1) + gamma / 2 * V_ ** 2)

    def right_boundary(U1, U2, U3):
        V_ = (2 * U2[-2] - U2[-3]) / (2 * U1[-2] - U1[-3])
        return 2 * U1[-2] - U1[-3], \
               2 * U2[-2] - U2[-3], \
               pe * A(N - 1) / (gamma - 1) + gamma / 2 * (2 * U2[-2] - U2[-3]) * V_

    rho = np.zeros((1, N))
    T = np.zeros((1, N))
    for i in range(N):
        x = X[i]
        if x <= 0.5:
            rho[0][i] = 1
            T[0][i] = 1
        elif x <= 1.5:
            rho[0][i] = 1 - 0.366 * (x - 0.5)
            T[0][i] = 1 - 0.167 * (x - 0.5)
        elif x <= 2.1:
            rho[0][i] = 0.634 - 0.702 * (x - 1.5)
            T[0][i] = 0.833 - 0.4908 * (x - 1.5)
        elif x <= 3:
            rho[0][i] = 0.5892 - 0.10228 * (x - 2.1)
            T[0][i] = 0.93968 - 0.0622 * (x - 2.1)
    A_arr = np.array([A(i) for i in range(N)])
    V = 0.59 / (rho * A_arr)

    V, T, rho, lms = process(rho, T, V, left_boundary, right_boundary, A)
    M = V / np.sqrt(T)
    p = rho * T

    resMa = []
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

    def av_partial_U1(n, i):
        return 0.5 * (forw_partial_U1(n, i) + bkw_partial_U1(n + 1, i))

    def av_partial_U2(n, i):
        return 0.5 * (forw_partial_U2(n, i) + bkw_partial_U2(n + 1, i))

    def av_partial_U3(n, i):
        return 0.5 * (forw_partial_U3(n, i) + bkw_partial_U3(n + 1, i))

    def forw_partial_U1(n, i):
        F1_f = U2[n][i + 1]
        F1_c = U2[n][i]
        return -(F1_f - F1_c) / dx

    def forw_partial_U2(n, i):
        F2_f = U2[n][i + 1] ** 2 / U1[n][i + 1] + (gamma - 1) / gamma * (
                U3[n][i + 1] - gamma / 2 * U2[n][i + 1] ** 2 / U1[n][i + 1])
        F2_c = U2[n][i] ** 2 / U1[n][i] + (gamma - 1) / gamma * (U3[n][i] - gamma / 2 * U2[n][i] ** 2 / U1[n][i])
        J = 1 / gamma * rho[n][i] * T[n][i] * (A(i + 1) - A(i)) / dx
        return -(F2_f - F2_c) / dx + J

    def forw_partial_U3(n, i):
        F3_f = gamma * U2[n][i + 1] * U3[n][i + 1] / U1[n][i + 1] - gamma * (gamma - 1) / 2 * U2[n][i + 1] ** 3 / U1[n][
            i + 1] ** 2
        F3_c = gamma * U2[n][i] * U3[n][i] / U1[n][i] - gamma * (gamma - 1) / 2 * U2[n][i] ** 3 / U1[n][i] ** 2
        return -(F3_f - F3_c) / dx

    def bkw_partial_U1(n, i):
        F1_b = U2[n][i - 1]
        F1_c = U2[n][i]
        return -(F1_c - F1_b) / dx

    def bkw_partial_U2(n, i):
        F2_b = U2[n][i - 1] ** 2 / U1[n][i - 1] + (gamma - 1) / gamma * (
                U3[n][i - 1] - gamma / 2 * U2[n][i - 1] ** 2 / U1[n][i - 1])
        F2_c = U2[n][i] ** 2 / U1[n][i] + (gamma - 1) / gamma * (U3[n][i] - gamma / 2 * U2[n][i] ** 2 / U1[n][i])
        J = 1 / gamma * rho[n][i] * T[n][i] * (A(i) - A(i - 1)) / dx
        return -(F2_c - F2_b) / dx + J

    def bkw_partial_U3(n, i):
        F3_b = gamma * U2[n][i - 1] * U3[n][i - 1] / U1[n][i - 1] - gamma * (gamma - 1) / 2 * U2[n][i - 1] ** 3 / U1[n][
            i - 1] ** 2
        F3_c = gamma * U2[n][i] * U3[n][i] / U1[n][i] - gamma * (gamma - 1) / 2 * U2[n][i] ** 3 / U1[n][i] ** 2
        return -(F3_c - F3_b) / dx

    def S(U, n, i):
        return Cx * np.abs(p[n][i + 1] - 2 * p[n][i] + p[n][i - 1]) / (p[n][i + 1] + 2 * p[n][i] + p[n][i - 1]) * (
                U[n][i + 1] - 2 * U[n][i] + U[n][i - 1])

    lms = []

    n = 0
    A_arr = np.array([A(i) for i in range(N)])
    p = rho * T
    U1 = rho * A_arr
    U2 = rho * A_arr * V
    U3 = rho * (T / (gamma - 1) + gamma / 2 * V ** 2) * A_arr
    while (n < 50 or all_lms() > 1e-2) and n < max_iter:
        U1 = np.row_stack((U1, np.zeros(N)))
        U2 = np.row_stack((U2, np.zeros(N)))
        U3 = np.row_stack((U3, np.zeros(N)))

        dt = C * dx / (V[n][:N - 1] + np.sqrt(T[n][:N - 1]))
        dt = np.min(dt)
        # 预估
        for i in range(1, N - 1):
            U1[n + 1][i] = U1[n][i] + w * forw_partial_U1(n, i) * dt + S(U1, n, i)
            U2[n + 1][i] = U2[n][i] + w * forw_partial_U2(n, i) * dt + S(U2, n, i)
            U3[n + 1][i] = U3[n][i] + w * forw_partial_U3(n, i) * dt + S(U3, n, i)

        # 左边界右边界都要确定
        U1[n + 1][0], U2[n + 1][0], U3[n + 1][0] = left_boundary(U1[n + 1], U2[n + 1], U3[n + 1])
        U1[n + 1][-1], U2[n + 1][-1], U3[n + 1][-1] = right_boundary(U1[n + 1], U2[n + 1], U3[n + 1])

        rho = np.row_stack((rho, U1[n + 1] / A_arr))
        V = np.row_stack((V, U2[n + 1] / U1[n + 1]))
        T = np.row_stack((T, (gamma - 1) * (U3[n + 1] / U1[n + 1] - gamma / 2 * V[n + 1] ** 2)))
        p = rho * T

        # 修正，av中用到预估的值
        tmpU1 = copy.deepcopy(U1[n + 1])
        tmpU2 = copy.deepcopy(U2[n + 1])
        tmpU3 = copy.deepcopy(U3[n + 1])
        for i in range(1, N - 1):
            tmpU1[i] = U1[n][i] + w * av_partial_U1(n, i) * dt + S(U1, n + 1, i)
            tmpU2[i] = U2[n][i] + w * av_partial_U2(n, i) * dt + S(U2, n + 1, i)
            tmpU3[i] = U3[n][i] + w * av_partial_U3(n, i) * dt + S(U2, n + 1, i)
        U1[n + 1] = tmpU1
        U2[n + 1] = tmpU2
        U3[n + 1] = tmpU3
        # 右边界，在最后求能够利用修正值，精度稍高一些
        U1[n + 1][-1], U2[n + 1][-1], U3[n + 1][-1] = right_boundary(U1[n + 1], U2[n + 1], U3[n + 1])

        rho[n + 1] = U1[n + 1] / A_arr
        V[n + 1] = U2[n + 1] / U1[n + 1]
        T[n + 1] = (gamma - 1) * (U3[n + 1] / U1[n + 1] - gamma / 2 * V[n + 1] ** 2)
        p = rho * T

        n = n + 1
    end = time.process_time()
    np.savetxt('U1.csv', U1, fmt='%f', delimiter=',')
    np.savetxt('U2.csv', U2, fmt='%f', delimiter=',')
    np.savetxt('U3.csv', U3, fmt='%f', delimiter=',')
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
    try:
        plt.scatter(X, resMa, s=10, label="Ma-analytical")
    except Exception as e:
        pass
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
        pe = 0.6784
        shock()
        # sub2super_u()
    except Exception as e:
        print("Exception occured in process")
        print(e)
