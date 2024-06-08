import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import matplotlib.ticker
from scipy.interpolate import interp1d

plt.rcParams['figure.dpi'] = 150

# 步长
h = 0.01
# 长宽
L = 1
H = 1
# 平板速度
U = 1
# 流线数
n_sline = 10
# 是否画中间图
proc_plot = False


def run():
    # 建立网格与坐标系
    vorticity = np.zeros((height, width))
    stream = np.zeros((height, width))
    x = np.linspace(0, L, width)
    y = np.linspace(H, 0, height)
    X, Y = np.meshgrid(x, y)

    # 设定边界
    def stream_boundary():
        stream[0] = np.zeros(width)
        stream[-1] = np.zeros(width)
        stream[:, 0] = np.zeros(height).T
        stream[:, -1] = np.zeros(height).T

    def vor_boundary():
        # 上边界
        vorticity[0, 1:-1] = np.zeros(width - 2)
        # 下边界
        for j in range(1, width - 1):
            vorticity[-1, j] = (1 / h ** 2) * (
                    -stream[-2, j - 1] + (8 / 3) * stream[-2, j] - stream[-2, j + 1] - (2 / 3) * stream[
                -3, j]) + 2 / 3 / h * U
        for i in range(1, height - 1):
            # 左边界
            vorticity[i, 0] = (1 / h ** 2) * (
                    -stream[i + 1, 1] + (8 / 3) * stream[i, 1] - stream[i - 1, 1] - (2 / 3) * stream[i, 2])
            # 右边界
            vorticity[i, -1] = (1 / h ** 2) * (
                    -stream[i + 1, -2] + (8 / 3) * stream[i, -2] - stream[i - 1, -2] - (2 / 3) * stream[i, -3])

        vorticity[0, 0] = 0.5 * (vorticity[0, 1] + vorticity[1, 0])
        vorticity[0, -1] = 0.5 * (vorticity[1, -1] + vorticity[0, -2])
        vorticity[-1, 0] = 0.5 * (vorticity[-2, 0] + vorticity[-1, 1])
        vorticity[-1, -1] = 0.5 * (vorticity[-2, -1] + vorticity[-1, -2])

    # 绘图函数
    def plot_vor():
        ax01.cla()
        ax01.plot(dif_vor_all, c='blue')

        ax1.contourf(X, Y, vorticity, cmap='rainbow')

        ybar = np.linspace(np.min(vorticity), np.max(vorticity), 100)
        ax2.scatter(xbar, ybar, c=cbar, s=300, linewidths=0.0, cmap='rainbow')
        ax2.set_xticks([])
        ax2.set_ylim([np.min(vorticity), np.max(vorticity)])
        plt.pause(0.01)

    def plot_str():
        ax02.cla()
        ax02.plot(dif_str_all, c='blue')
        ax3.contourf(X, Y, stream, cmap='rainbow')

        ybar = np.linspace(np.min(stream), np.max(stream), 100)
        ax4.scatter(xbar, ybar, c=cbar, s=300, linewidths=0.0, cmap='rainbow')
        ax4.set_xticks([])
        ax4.set_ylim([np.min(stream), np.max(stream)])

        # 计算速度
        u = np.zeros((height, width))
        v = np.zeros((height, width))
        velo = np.zeros((height, width))
        for i in range(height - 1):
            for j in range(width - 1):
                u[i][j] = (stream[i][j] - stream[i + 1][j]) / h
                v[i][j] = (stream[i][j] - stream[i][j + 1]) / h
        u[:, -1] = 2 * u[:, -2] - u[:, -3]
        v[:, -1] = 2 * v[:, -2] - v[:, -3]
        u[-1] = 2 * u[-2] - u[-3]
        v[-1] = 2 * v[-2] - v[-3]

        ax3.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2], color='white')

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

        plt.pause(0.01)

    # 初始化绘图
    plt.ion()
    fig = plt.figure(figsize=(13, 5))
    ax01 = fig.add_axes([0.05, 0.05, 0.3, 0.15])  # 迭代过程
    ax02 = fig.add_axes([0.40, 0.05, 0.3, 0.15])  # 迭代过程
    ax1 = fig.add_axes([-0.02, 0.30, 0.5, 0.63])  # 涡量
    ax2 = fig.add_axes([0.05, 0.30, 0.015, 0.63])  # 涡量colorbar

    ax3 = fig.add_axes([0.33, 0.30, 0.5, 0.63])  # 流函数
    ax4 = fig.add_axes([0.40, 0.30, 0.015, 0.63])  # 流函数colorbar
    ax3.contourf(X, Y, np.zeros((height, width)), cmap='rainbow')

    ax5 = fig.add_axes([0.68, 0.53, 0.40, 0.40])  # u
    ax6 = fig.add_axes([0.75, 0.53, 0.01, 0.40])  # u colorbar
    ax5.set_xticks([])

    ax7 = fig.add_axes([0.68, 0.05, 0.40, 0.40])  # v
    ax8 = fig.add_axes([0.75, 0.05, 0.01, 0.40])  # v colorbar

    ax1.set_aspect('equal', adjustable='box')
    ax3.set_aspect('equal', adjustable='box')
    ax5.set_aspect('equal', adjustable='box')
    ax7.set_aspect('equal', adjustable='box')

    xbar = [0] * 100
    cbar = np.linspace(0.0, 1.0, 100)

    stream_boundary()
    vor_boundary()

    # 计算流程
    dif_str_all=[]
    dif_vor_all=[]
    tmp_vor = 10 * np.ones((height, width))
    tmp_str = 10 * np.ones((height, width))
    tmp_str_init = 10 * np.ones((height, width))
    dif_str_outer = [abs(np.sum(tmp_str_init - stream))]
    while len(dif_str_outer) < 10 or np.sum(dif_str_outer[-10:]) > 1e-2:
        dif_vor = [abs(np.sum(tmp_vor - stream))]
        dif_str_inner = [abs(np.sum(tmp_str - stream))]
        while len(dif_vor) < 30:

            tmp_vor = copy.deepcopy(vorticity)
            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    vorticity[i][j] = 0.25 * (
                            vorticity[i - 1][j] + vorticity[i + 1][j] + vorticity[i][j + 1] + vorticity[i][j - 1])

            if len(dif_vor) % 29 == 0 and proc_plot:
                plot_vor()
            dif_vor.append(abs(np.sum(tmp_vor - vorticity)))
            dif_vor_all.append(dif_vor[-1])
        print('dif_vor: ', dif_vor[-1])

        stream_boundary()
        vor_boundary()

        tmp_str_init = copy.deepcopy(stream)
        while len(dif_str_inner) < 30:

            tmp_str = copy.deepcopy(stream)
            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    stream[i][j] = 0.25 * (
                            vorticity[i][j] * h ** 2 + stream[i - 1][j] + stream[i + 1][j] + stream[i][j + 1] +
                            stream[i][
                                j - 1])

            if len(dif_str_inner) % 29 == 0 and proc_plot:
                plot_str()
            dif_str_inner.append(abs(np.sum(tmp_str - stream)))
            dif_str_all.append(dif_str_inner[-1])
        print('dif_str_inner: ', dif_str_inner[-1])

        dif_str_outer.append(abs(np.sum(tmp_str_init - stream)))
        print('dif_str_outer: ', dif_str_outer[-1])
        stream_boundary()
        vor_boundary()

    plot_vor()
    plot_str()
    print(np.max(stream))
    print(np.max(vorticity))
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    height = math.floor(H / h) + 1
    width = math.floor(L / h) + 1
    run()
