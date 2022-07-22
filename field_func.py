# field.py: ti.kernels for calculating E/M fields
# Author: Bowen Zhao (zzzhaobowen@gmail.com)

from parameters import *
from constants import *
from helper_func import *

import taichi as ti


@ti.kernel
def multi_by_number(pos : ti.template(), show_field : ti.template()):
    scalar = 1 / (xmax - xmin)
    for i in pos:
        for j in ti.static(range(3)):
            show_field[i][j] = pos[i][j] * scalar

@ti.kernel
def push_bhalf(
        B_yee: ti.template(),
        E_yee: ti.template()
):
    """
    computes the B field vector at t=t+dt/2
    """
    for Idx in ti.grouped(B_yee):
        if ti.static(DIM == 3):
            i, j, k = Idx[0], Idx[1], Idx[2]
            i_p = i + 1 if i != n_cellx - 1 else 0
            j_p = j + 1 if j != n_celly - 1 else 0
            k_p = k + 1 if k != n_cellz - 1 else 0

            c_x = inv_dy * (E_yee[i, j_p, k][2] - E_yee[i, j, k][2]) - inv_dz * (
                        E_yee[i, j, k_p][1] - E_yee[i, j, k][1])
            c_y = inv_dz * (E_yee[i, j, k_p][0] - E_yee[i, j, k][0]) - inv_dx * (
                        E_yee[i_p, j, k][2] - E_yee[i, j, k][2])
            c_z = inv_dx * (E_yee[i_p, j, k][1] - E_yee[i, j, k][1]) - inv_dy * (
                        E_yee[i, j_p, k][0] - E_yee[i, j, k][0])

            B_yee[Idx] = B_yee[Idx] - 0.5 * dt * c * ti.Vector([c_x, c_y, c_z])
            # B_yee[Idx] = B_yee[Idx] - 0.5 * dt * c * curl(E_yee, i, i_p, j, j_p, k, k_p)
        else:
            i, j = Idx[0], Idx[1]
            i_p = i + 1 if i != n_cellx - 1 else 0
            j_p = j + 1 if j != n_celly - 1 else 0

            # FIXME: for debug
            c_x = inv_dy * (E_yee[i, j_p][2] - E_yee[i, j][2])
            c_y = - inv_dx * (E_yee[i_p, j][2] - E_yee[i, j][2])
            c_z = inv_dx * (E_yee[i_p, j][1] - E_yee[i, j][1]) - inv_dy * (E_yee[i, j_p][0] - E_yee[i, j][0])

            B_yee[Idx] = B_yee[Idx] - 0.5 * dt * c * ti.Vector([c_x, c_y, c_z])
            # B_yee[Idx] = B_yee[Idx] - 0.5 * dt * c * curl_2d(E_yee, i, i_p, j, j_p)


@ti.kernel
def push_efield(
        B_yee: ti.template(),
        E_yee: ti.template(),
        J_yee: ti.template()
):
    """
    computes the E field vector at time t+dt
    """
    for Idx in ti.grouped(E_yee):
        if ti.static(DIM == 3):
            i, j, k = Idx[0], Idx[1], Idx[2]
            i_m = i - 1 if i != 0 else n_cellx - 1
            j_m = j - 1 if j != 0 else n_celly - 1
            k_m = k - 1 if k != 0 else n_cellz - 1

            c_x = inv_dy * (B_yee[i, j, k][2] - B_yee[i, j_m, k][2]) - inv_dz * (
                        B_yee[i, j, k][1] - B_yee[i, j, k_m][1])
            c_y = inv_dz * (B_yee[i, j, k][0] - B_yee[i, j, k_m][0]) - inv_dx * (
                        B_yee[i, j, k][2] - B_yee[i_m, j, k][2])
            c_z = inv_dx * (B_yee[i, j, k][1] - B_yee[i_m, j, k][1]) - inv_dy * (
                        B_yee[i, j, k][0] - B_yee[i, j_m, k][0])

            E_yee[Idx] = E_yee[Idx] + dt * (c * ti.Vector([c_x, c_y, c_z]) - 4. * pi * J_yee[Idx])
            # E_yee[Idx] = E_yee[Idx] + dt * (c * curl(B_yee, i_m, i, j_m, j, k_m, k) - 4. * pi * J_yee[Idx])
        else:
            i, j = Idx[0], Idx[1]
            i_m = i - 1 if i != 0 else n_cellx - 1
            j_m = j - 1 if j != 0 else n_celly - 1

            # FIXME: only for debug
            c_x = inv_dy * (B_yee[i, j][2] - B_yee[i, j_m][2])
            c_y = - inv_dx * (B_yee[i, j][2] - B_yee[i_m, j][2])
            c_z = inv_dx * (B_yee[i, j][1] - B_yee[i_m, j][1]) - inv_dy * (B_yee[i, j][0] - B_yee[i, j_m][0])

            E_yee[Idx] = E_yee[Idx] + dt * (c * ti.Vector([c_x, c_y, c_z]) - 4. * pi * J_yee[Idx])
            # E_yee[Idx] = E_yee[Idx] + dt * (c * curl_2d(B_yee, i_m, i, j_m, j) - 4. * pi * J_yee[Idx])


@ti.kernel
def iter_phi(
        E_yee: ti.template(),
        rho_field: ti.template(),
        phi_old: ti.template(),
        phi_new: ti.template()
):
    """
    iteratly calculate phi
    """
    for Idx in ti.grouped(phi_new):
        if ti.static(DIM == 3):
            i, j, k = Idx[0], Idx[1], Idx[2]
            i_p = i + 1 if i != n_cellx - 1 else 0
            j_p = j + 1 if j != n_celly - 1 else 0
            k_p = k + 1 if k != n_cellz - 1 else 0
            i_m = i - 1 if i != 0 else n_cellx - 1
            j_m = j - 1 if j != 0 else n_celly - 1
            k_m = k - 1 if k != 0 else n_cellz - 1

            denom = ti.static(dx * dx * dy * dy + dx * dx * dz * dz + dy * dy * dz * dz)

            phi_new[Idx] = 0.5 / denom \
                            * (
                                    (phi_old[i_p, j, k] + phi_old[i_m, j, k]) * dy * dy * dz * dz
                                    + (phi_old[i, j_p, k] + phi_old[i, j_m, k]) * dx * dx * dz * dz
                                    + (phi_old[i, j, k_p] + phi_old[i, j, k_m]) * dx * dx * dy * dy
                                    + (
                                            4. * pi * rho_field[Idx]
                                            - div(E_yee, i_m, i, j_m, j, k_m, k)
                                    ) * dx * dx * dy * dy * dz * dz
                            )
        else:
            i, j = Idx[0], Idx[1]
            i_p = i + 1 if i != n_cellx - 1 else 0
            j_p = j + 1 if j != n_celly - 1 else 0
            i_m = i - 1 if i != 0 else n_cellx - 1
            j_m = j - 1 if j != 0 else n_celly - 1

            denom = ti.static(dx * dx + dy * dy)

            phi_new[i, j] = 0.5 / denom * (
                    (phi_old[i_p, j] + phi_old[i_m, j]) * dy * dy
                    + (phi_old[i, j_p] + phi_old[i, j_m]) * dx * dx
                    + (
                            4. * pi * rho_field[i, j] - (
                                (E_yee[i, j][0] - E_yee[i_m, j][0]) / dx + (E_yee[i, j][1] - E_yee[i, j_m][1]) / dy)
                    ) * dx * dx * dy * dy
            )

    for Idx in ti.grouped(phi_new):
        phi_old[Idx] = phi_new[Idx]


@ti.kernel
def correct_efield(
        E_yee: ti.template(),
        phi_new: ti.template()
):
    """
    correct the electric field to ensure the conservation of charge,
    or to ensure that div(E)=4*pi*rho. Poission equation is solved
    using an iterative method (Gauss-Seidel method), with 5 points.
    """
    for Idx in ti.grouped(E_yee):
        if ti.static(DIM == 3):
            i, j, k = Idx[0], Idx[1], Idx[2]
            i_p = i + 1 if i != n_cellx - 1 else 0
            j_p = j + 1 if j != n_celly - 1 else 0
            k_p = k + 1 if k != n_cellz - 1 else 0

            E_yee[Idx] = E_yee[Idx] - grad(phi_new, i, i_p, j, j_p, k, k_p)
        else:
            i, j = Idx[0], Idx[1]
            i_p = i + 1 if i != n_cellx - 1 else 0
            j_p = j + 1 if j != n_celly - 1 else 0

            E_yee[Idx] = E_yee[Idx] - grad_2d(phi_new, i, i_p, j, j_p)


@ti.kernel
def eb_yee2grid(E_grid: ti.template(),
                E_yee: ti.template(),
                B_grid: ti.template(),
                B_yee: ti.template()
                ):
    """
    Transfer field from Yee lattice to grids
    """
    for Idx in ti.grouped(E_grid):
        if ti.static(DIM == 3):
            i, j, k = Idx[0], Idx[1], Idx[2]
            i_m = i - 1 if i != 0 else n_cellx - 1
            j_m = j - 1 if j != 0 else n_celly - 1
            k_m = k - 1 if k != 0 else n_cellz - 1

            Egx_ptc = (E_yee[i, j, k][0] + E_yee[i_m, j, k][0]) / 2.
            Egy_ptc = (E_yee[i, j, k][1] + E_yee[i, j_m, k][1]) / 2.
            Egz_ptc = (E_yee[i, j, k][2] + E_yee[i, j, k_m][2]) / 2.
            E_grid[i, j, k] = [Egx_ptc, Egy_ptc, Egz_ptc]

            Bgx_ptc = (B_yee[i, j, k][0] + B_yee[i, j_m, k][0] + B_yee[i, j, k_m][0] + B_yee[i, j_m, k_m][0]) / 4.
            Bgy_ptc = (B_yee[i, j, k][1] + B_yee[i_m, j, k][1] + B_yee[i, j, k_m][1] + B_yee[i_m, j, k_m][1]) / 4.
            Bgz_ptc = (B_yee[i, j, k][2] + B_yee[i, j_m, k][2] + B_yee[i_m, j, k][2] + B_yee[i_m, j_m, k][2]) / 4.
            B_grid[i, j, k] = [Bgx_ptc, Bgy_ptc, Bgz_ptc]
        else:
            i, j = Idx[0], Idx[1]
            j_m = j - 1 if j != 0 else n_celly - 1
            i_m = i - 1 if i != 0 else n_cellx - 1

            Egx_ptc = (E_yee[i, j][0] + E_yee[i_m, j][0]) / 2.
            Egy_ptc = (E_yee[i, j][1] + E_yee[i, j_m][1]) / 2.
            Egz_ptc = E_yee[i, j][2]
            E_grid[Idx] = [Egx_ptc, Egy_ptc, Egz_ptc]

            Bgx_ptc = (B_yee[i, j][0] + B_yee[i, j_m][0]) / 2.
            Bgy_ptc = (B_yee[i, j][1] + B_yee[i_m, j][1]) / 2.
            Bgz_ptc = (B_yee[i, j][2] + B_yee[i, j_m][2] + B_yee[i_m, j][2] + B_yee[i_m, j_m][2]) / 4.
            B_grid[Idx] = [Bgx_ptc, Bgy_ptc, Bgz_ptc]
