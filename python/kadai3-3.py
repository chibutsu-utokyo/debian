#!/usr/bin/env python

import sys
import time
import numpy as np
from numba import njit


def save_plot(u, lbx, ubx, lby, uby, lbz, ubz, filename):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    vmin = np.abs(u.min()) * np.sign(u.min())
    vmax = np.abs(u.max()) * np.sign(u.max())
    xx = np.linspace(0, 1, ubx - lbx + 2)
    yy = np.linspace(0, 1, uby - lby + 2)
    zz = np.linspace(0, 1, ubz - lbz + 2)

    midx = (ubx + lbx) // 2
    midy = (uby + lby) // 2
    midz = (ubz + lbz) // 2

    slice_x = u[midx, lby : uby + 1, lbz : ubz + 1]
    slice_y = u[lbx : ubx + 1, midy, lbz : ubz + 1]
    slice_z = u[lbx : ubx + 1, lby : uby + 1, midz]

    img = [0] * 3
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    plt.subplots_adjust(left=0.075, right=0.90, bottom=0.05, top=0.95, wspace=0.3)

    X, Y = np.meshgrid(xx, yy)
    img[0] = axs[0].pcolormesh(xx, yy, slice_z, shading="auto", vmin=vmin, vmax=vmax)
    axs[0].set_title("Slice in x-y plane")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_aspect("equal")

    X, Z = np.meshgrid(xx, zz)
    img[1] = axs[1].pcolormesh(xx, zz, slice_y, shading="auto", vmin=vmin, vmax=vmax)
    axs[1].set_title("Slice in x-z plane")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("z")
    axs[1].set_aspect("equal")

    Y, Z = np.meshgrid(yy, zz)
    img[2] = axs[2].pcolormesh(yy, zz, slice_x, shading="auto", vmin=vmin, vmax=vmax)
    axs[2].set_title("Slice in y-z plane")
    axs[2].set_xlabel("y")
    axs[2].set_ylabel("z")
    axs[2].set_aspect("equal")

    # カラーバー
    pos = axs[2].get_position()
    cax = fig.add_axes([pos.x0 + pos.width * 1.1, pos.y0, pos.width * 0.1, pos.height])
    fig.colorbar(img[2], cax=cax)
    plt.savefig(filename)


def set_initial_condition(u, v, lbx, ubx, lby, uby, lbz, ubz):
    x0, y0, z0, sigma = 0.5, 0.5, 0.5, 0.1

    for ix in range(lbx, ubx + 1):
        for iy in range(lby, uby + 1):
            for iz in range(lbz, ubz + 1):
                x = (ix - lbx + 0.5) / (ubx - lbx + 1)
                y = (iy - lby + 0.5) / (uby - lby + 1)
                z = (iz - lbz + 0.5) / (ubz - lbz + 1)
                w = 0.5 * ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / sigma**2
                u[ix, iy, iz] = np.exp(-w)
                v[ix, iy, iz] = 0.0

    set_boundary_condition_naive(u, lbx, ubx, lby, uby, lbz, ubz)
    set_boundary_condition_naive(v, lbx, ubx, lby, uby, lbz, ubz)


def set_boundary_condition_naive(uv, lbx, ubx, lby, uby, lbz, ubz):
    for iy in range(lby, uby + 1):
        for iz in range(lbz, ubz + 1):
            uv[lbx - 1, iy, iz] = uv[ubx, iy, iz]
            uv[ubx + 1, iy, iz] = uv[lbx, iy, iz]
    for ix in range(lbx, ubx + 1):
        for iz in range(lbz, ubz + 1):
            uv[ix, lby - 1, iz] = uv[ix, uby, iz]
            uv[ix, uby + 1, iz] = uv[ix, lby, iz]
    for ix in range(lbx, ubx + 1):
        for iy in range(lby, uby + 1):
            uv[ix, iy, lbz - 1] = uv[ix, iy, ubz]
            uv[ix, iy, ubz + 1] = uv[ix, iy, lbz]


def push_naive(u, v, lbx, ubx, lby, uby, lbz, ubz, dt, dx, dy, dz):
    # vを更新
    for ix in range(lbx, ubx + 1):
        for iy in range(lby, uby + 1):
            for iz in range(lbz, ubz + 1):
                # fmt: off
                v[ix, iy, iz] += dt * (
                    + (u[ix + 1, iy, iz] - 2 * u[ix, iy, iz] + u[ix - 1, iy, iz]) / dx**2
                    + (u[ix, iy + 1, iz] - 2 * u[ix, iy, iz] + u[ix, iy - 1, iz]) / dy**2
                    + (u[ix, iy, iz + 1] - 2 * u[ix, iy, iz] + u[ix, iy, iz - 1]) / dz**2
                )
                # fmt: on

    # uを更新
    for ix in range(lbx, ubx + 1):
        for iy in range(lby, uby + 1):
            for iz in range(lbz, ubz + 1):
                u[ix, iy, iz] += v[ix, iy, iz] * dt

    # 境界条件の設定
    set_boundary_condition_naive(u, lbx, ubx, lby, uby, lbz, ubz)


def push_optimized(u, v, lbx, ubx, lby, uby, lbz, ubz, dt, dx, dy, dz):
    # naiveな実装を呼ぶだけ
    push_naive(u, v, lbx, ubx, lby, uby, lbz, ubz, dt, dx, dy, dz)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="3D wave equation solver")
    parser.add_argument("--delt", type=float, default=0.01, help="Time step")
    parser.add_argument("--grid", type=int, default=32, help="Number of grids")
    args = parser.parse_args()

    # nt : 時間ステップ数
    # nb : 境界条件用ののりしろ
    # nx : x方向のグリッドの数
    # ny : y方向のグリッドの数
    # nz : z方向のグリッドの数
    # lbx: x方向の下限境界の位置
    # ubx: x方向の上限境界の位置
    # lby: y方向の下限境界の位置
    # uby: y方向の上限墖界の位置
    # lbz: z方向の下限境界の位置
    # ubz: z方向の上限境界の位置
    # dt : 時間ステップ
    # dx : x方向のグリッド幅
    # dy : y方向のグリッド幅
    # dz : z方向のグリッド幅
    nt = 50
    nb = 1
    nx = args.grid
    ny = nx
    nz = nx
    lbx = nb
    ubx = nx + nb - 1
    lby = nb
    uby = ny + nb - 1
    lbz = nb
    ubz = nz + nb - 1
    dt = args.delt
    dx = 1.0 / nx
    dy = dx
    dz = dx

    # Courant数のチェック
    if dt / dx > 1.0 / np.sqrt(3):
        print("Error: Courant number exceeds the stability limit", dt / dx)
        sys.exit(1)

    # メモリ確保
    u = np.zeros((nx + 2 * nb, ny + 2 * nb, nz + 2 * nb))
    v = np.zeros((nx + 2 * nb, ny + 2 * nb, nz + 2 * nb))

    #
    # 素朴なPythonのforループによる実装
    #
    set_initial_condition(u, v, lbx, ubx, lby, uby, lbz, ubz)
    start_time = time.time()
    for i in range(nt):
        push_naive(u, v, lbx, ubx, lby, uby, lbz, ubz, dt, dx, dy, dz)
    end_time = time.time()

    print("{:30s} : {:<10.3e}".format("Elapsed time (naive)", end_time - start_time))
    save_plot(u, lbx, ubx, lby, uby, lbz, ubz, "wave3d_naive.png")

    #
    # 最適化バージョン
    #
    set_initial_condition(u, v, lbx, ubx, lby, uby, lbz, ubz)
    start_time = time.time()
    for i in range(nt):
        push_optimized(u, v, lbx, ubx, lby, uby, lbz, ubz, dt, dx, dy, dz)
    end_time = time.time()

    print(
        "{:30s} : {:<10.3e}".format("Elapsed time (optimized)", end_time - start_time)
    )
    save_plot(u, lbx, ubx, lby, uby, lbz, ubz, "wave3d_optimized.png")
