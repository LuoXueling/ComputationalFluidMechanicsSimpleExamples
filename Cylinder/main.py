import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import matplotlib.ticker
from scipy.interpolate import interp1d

plt.rcParams['figure.dpi'] = 150

# 步长
h = 0.25
# 长宽
L = 7
H = 4
# 半径
r = 1
# 流线数
n_sline = 10


def run():
    # 建立网格与坐标系
    mesh = np.ones((height, width))
    x = np.linspace(0, L / 2, width)
    y = np.linspace(H / 2, 0, height)
    X, Y = np.meshgrid(x, y)
    xvelo = np.linspace(0, L / 2 - h, width - 1)
    yvelo = np.linspace(H / 2 - h, 0, height - 1)
    Xvelo, Yvelo = np.meshgrid(xvelo, yvelo)

    # 设定边界
    def boundary():
        # 上边界
        mesh[0] = 2 * np.ones(width)
        # 下边界
        mesh[-1] = np.zeros(width)
        for i in range(height):
            # 右边界
            if i > 0 and i < height - 1:
                mesh[i][-1] = 0.25 * (2 * mesh[i][-2] + mesh[i - 1][-1] + mesh[i + 1][-1])
            for j in range(width):
                x = X[i][j]
                y = Y[i][j]
                # 圆柱
                if (x - L / 2) ** 2 + y ** 2 <= r ** 2:
                    mesh[i][j] = 0
                # 左边界
                mesh[i][0] = Y[i][j]

    plt.ion()
    fig = plt.figure(figsize=(11, 5))
    ax0 = fig.add_axes([0.05, 0.05, 0.57, 0.15])  # 迭代过程
    ax1 = fig.add_axes([0.05, 0.30, 0.63, 0.63])  # 流函数
    ax2 = fig.add_axes([0.05, 0.30, 0.015, 0.63])  # 流函数colorbar

    ax3 = fig.add_axes([0.73, 0.65, 0.28, 0.28])  # 速度
    ax4 = fig.add_axes([0.70, 0.65, 0.01, 0.28])  # 速度 colorbar
    ax3.set_xticks([])

    ax5 = fig.add_axes([0.73, 0.35, 0.28, 0.28])  # u
    ax6 = fig.add_axes([0.70, 0.35, 0.01, 0.28])  # u colorbar
    ax5.set_xticks([])

    ax7 = fig.add_axes([0.73, 0.05, 0.28, 0.28])  # v
    ax8 = fig.add_axes([0.70, 0.05, 0.01, 0.28])  # v colorbar

    # ax1.xaxis.set_major_locator(plt.MultipleLocator(h*2))
    # ax1.yaxis.set_major_locator(plt.MultipleLocator(h*2))
    # ax1.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(x))
    # ax1.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(y))

    # 画一个圆
    xr = np.linspace(L / 2 - r, L / 2, 50)
    yr = np.sqrt((r ** 2 - (xr - L / 2) ** 2))
    ax1.plot(xr, yr, color='white')
    ax3.plot(xr, yr, color='white')
    ax5.plot(xr, yr, color='white')
    ax7.plot(xr, yr, color='white')

    ax1.set_aspect('equal', adjustable='box')
    ax3.set_aspect('equal', adjustable='box')
    ax5.set_aspect('equal', adjustable='box')
    ax7.set_aspect('equal', adjustable='box')
    # plt.grid()
    # plt.grid(which='minor')

    def plots():
        ax0.plot(dif[1:], c='blue')

        ax1.contourf(X, Y, mesh)

        ybar = np.linspace(np.min(mesh), np.max(mesh), 100)
        ax2.scatter(xbar, ybar, c=cbar, s=300, linewidths=0.0, cmap='rainbow')
        ax2.set_xticks([])
        ax2.set_ylim([np.min(mesh), np.max(mesh)])

        # 计算速度
        u = np.zeros((height, width))
        v = np.zeros((height, width))
        velo = np.zeros((height, width))
        for i in range(height - 1):
            for j in range(width - 1):
                u[i][j] = (mesh[i][j] - mesh[i + 1][j]) / h
                v[i][j] = (mesh[i][j] - mesh[i][j + 1]) / h
        u[:, -1] = 2 * u[:, -2] - u[:, -3]
        v[:, -1] = 2 * v[:, -2] - v[:, -3]
        u[-1] = 2 * u[-2] - u[-3]
        v[-1] = 2 * v[-2] - v[-3]
        velo = np.sqrt(np.power(u, 2) + np.power(v, 2))

        ax3.contourf(X, Y, velo, cmap='rainbow')
        ybar = np.linspace(np.min(velo), np.max(velo), 100)
        ax4.scatter(xbar, ybar, c=cbar, s=300, linewidths=0.0, cmap='rainbow')
        ax4.set_xticks([])
        ax4.set_ylim([np.min(velo), np.max(velo)])

        ax5.contourf(X, Y, u, cmap='rainbow')
        ybar = np.linspace(np.min(u), np.max(u), 100)
        ax6.scatter(xbar, ybar, c=cbar, s=300, linewidths=0.0, cmap='rainbow')
        ax6.set_xticks([])
        ax6.set_ylim([np.min(u), np.max(u)])

        ax7.contourf(X, Y, v, cmap='rainbow')
        ybar = np.linspace(np.min(v), np.max(v), 100)
        ax8.scatter(xbar, ybar, c=cbar, s=300, linewidths=0.0, cmap='rainbow')
        ax8.set_xticks([])
        ax8.set_ylim([np.min(v), np.max(v)])

        plt.pause(0.001)

    xbar = [0] * 100
    cbar = np.linspace(0.0, 1.0, 100)

    boundary()
    tmp = 10 * np.ones((height, width))
    dif = [abs(np.sum(tmp - mesh))]
    while len(dif) < 10 or np.sum(dif[-10:]) > 1e-4:

        tmp = copy.deepcopy(mesh)
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                try:
                    a = (width - 1 - j) * h - math.sqrt(1 - ((height - i - 1) * h) ** 2)
                    b = (height - i - 1) * h - math.sqrt(1 - ((width - 1 - j) * h) ** 2)
                    if a <= 0 or b <= 0:
                        mesh[i][j] = 0
                    elif 0 < a < h and 0 < b < h:
                        # 下面右边都没点
                        p1 = p2 = 0
                        mesh[i][j] = (mesh[i][j - 1] / (h * (a + h)) + mesh[i - 1][j] / (h * (b + h)) + p1 / (
                                a * (a + h)) + p2 / (b * (b + h))) / (1 / (a * h) + 1 / (b * h))
                    elif a >= h and b <= h:
                        # 右边还有一个点
                        p1 = mesh[i][j + 1]
                        p2 = 0
                        mesh[i][j] = (mesh[i][j - 1] / (h * (a + h)) + mesh[i - 1][j] / (h * (b + h)) + p1 / (
                                a * (a + h)) + p2 / (b * (b + h))) / (1 / (a * h) + 1 / (b * h))
                    elif b >= h and a <= h:
                        # 下面有点
                        p1 = 0
                        p2 = mesh[i + 1][j]
                        mesh[i][j] = (mesh[i][j - 1] / (h * (a + h)) + mesh[i - 1][j] / (h * (b + h)) + p1 / (
                                a * (a + h)) + p2 / (b * (b + h))) / (1 / (a * h) + 1 / (b * h))
                    else:
                        mesh[i][j] = 0.25 * (mesh[i - 1][j] + mesh[i + 1][j] + mesh[i][j + 1] + mesh[i][j - 1])
                except:
                    mesh[i][j] = 0.25 * (mesh[i - 1][j] + mesh[i + 1][j] + mesh[i][j + 1] + mesh[i][j - 1])
        boundary()
        if len(dif) % 100 == 0:
            plots()
        dif.append(abs(np.sum(np.sum(tmp - mesh))))
        print(dif[-1])

    plots()
    # 画流线
    plines = np.linspace(np.min(mesh) + (np.max(mesh) - np.min(mesh)) / n_sline,
                         np.max(mesh) - (np.max(mesh) - np.min(mesh)) / n_sline, n_sline)
    linesx = [[] for x in range(n_sline)]
    linesy = [[] for x in range(n_sline)]
    # 循环顺序换一下后面不需要排序
    for j in range(width - 1):
        for i in range(height - 1):
            # 遍历每个三角形
            p = [mesh[i][j], mesh[i + 1][j], mesh[i + 1][j + 1]]
            # 对每个流线
            for order, k in enumerate(plines):
                # 在该三角形中
                if np.min(p) <= k <= np.max(p):
                    xp = [X[i][j], X[i + 1][j], X[i + 1][j + 1]]
                    yp = [Y[i][j], Y[i + 1][j], Y[i + 1][j + 1]]
                    # 对每条边
                    for u in range(3):
                        if p[u - 3] <= k <= p[u - 2] or p[u - 2] <= k <= p[u - 3]:
                            # u=0: 0 和 1之间
                            # u=1: 1 和 2之间
                            if abs(p[u - 2] - p[u - 3]) < 1e-10:
                                # 该边两点都是流线上的点
                                linesx[order].append(xp[u - 3])
                                linesx[order].append(xp[u - 2])
                                linesy[order].append(yp[u - 3])
                                linesy[order].append(yp[u - 2])
                            else:
                                # 一条线上一个点
                                x = xp[u - 3] + (k - p[u - 3]) / (p[u - 2] - p[u - 3]) * (xp[u - 2] - xp[u - 3])
                                y = yp[u - 3] + (k - p[u - 3]) / (p[u - 2] - p[u - 3]) * (yp[u - 2] - yp[u - 3])
                                linesx[order].append(x)
                                linesy[order].append(y)

    for order, k in enumerate(plines):
        ls = np.row_stack((linesx[order], linesy[order]))
        ls = ls.T[np.argsort(ls.T[:, 0])].T
        ax1.plot(ls[0], ls[1], c='white')

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    height = math.floor(H / h / 2) + 1
    width = math.floor(L / h / 2) + 1
    run()
