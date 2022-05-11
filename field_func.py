from parameters import *
from constants import *
from helper_func import *

import taichi as ti


@ti.kernel
def push_bhalf(
        B_yee: ti.template(),
        E_yee: ti.template()
):
    """
    computes the B field vector at t=t+dt/2
    """
    for i, j, k in B_yee:
        i_p = i + 1 if i != n_cellx - 1 else 0
        j_p = j + 1 if j != n_celly - 1 else 0
        k_p = k + 1 if k != n_cellz - 1 else 0

        B_yee[i, j, k] = B_yee[i, j, k] - 0.5 * dt * c * curl(E_yee, i, i_p, j, j_p, k, k_p)

        # Only for debug, write the component seperately
        # c_x = B_yee[i, j, k][0] - 0.5 * dt * c * inv_dy * (E_yee[i, j_p, k][2] - E_yee[i, j, k][2]) - inv_dz * (
        #             E_yee[i, j, k_p][1] - E_yee[i, j, k][1])
        # c_y = B_yee[i, j, k][1] - 0.5 * dt * c * inv_dz * (E_yee[i, j, k_p][0] - E_yee[i, j, k][0]) - inv_dx * (
        #             E_yee[i_p, j, k][2] - E_yee[i, j, k][2])
        # c_z = B_yee[i, j, k][2] - 0.5 * dt * c * inv_dx * (E_yee[i_p, j, k][1] - E_yee[i, j, k][1]) - inv_dy * (
        #             E_yee[i, j_p, k][0] - E_yee[i, j, k][0])
        #
        # B_yee[i, j, k] = ti.Vector([c_x, c_y, c_z])




@ti.kernel
def push_efield(
        B_yee: ti.template(),
        E_yee: ti.template(),
        J_yee: ti.template()
):
    """
    computes the E field vector at time t+dt
    """
    for i, j, k in E_yee:
        i_m = i - 1 if i != 0 else n_cellx - 1
        j_m = j - 1 if j != 0 else n_celly - 1
        k_m = k - 1 if k != 0 else n_cellz - 1

        E_yee[i, j, k] = E_yee[i, j, k] + dt * (c * curl(B_yee, i_m, i, j_m, j, k_m, k) - 4. * pi * J_yee[i, j, k])

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
    denom = dx * dx * dy * dy + dx * dx * dz * dz + dy * dy * dz * dz

    for i, j, k in phi_new:
        i_p = i + 1 if i != n_cellx - 1 else 0
        j_p = j + 1 if j != n_celly - 1 else 0
        k_p = k + 1 if k != n_cellz - 1 else 0
        i_m = i - 1 if i != 0 else n_cellx - 1
        j_m = j - 1 if j != 0 else n_celly - 1
        k_m = k - 1 if k != 0 else n_cellz - 1

        phi_new[i, j, k] = 0.5 / denom \
                        * (
                                (phi_old[i_p, j, k] + phi_old[i_m, j, k]) * dy * dy * dz * dz
                                + (phi_old[i, j_p, k] + phi_old[i, j_m, k]) * dx * dx * dz * dz
                                + (phi_old[i, j, k_p] + phi_old[i, j, k_m]) * dx * dx * dy * dy
                                + (
                                        4. * pi * rho_field[i, j, k]
                                        - div(E_yee, i_m, i, j_m, j, k_m, k)
                                ) * dx * dx * dy * dy * dz * dz
                        )

    for i, j, k in phi_new:
        phi_old[i, j, k] = phi_new[i, j, k]


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
    for i, j, k in E_yee:
        i_p = i + 1 if i != n_cellx - 1 else 0
        j_p = j + 1 if j != n_celly - 1 else 0
        k_p = k + 1 if k != n_cellz - 1 else 0

        E_yee[i, j, k] = E_yee[i, j, k] - grad(phi_new, i, i_p, j, j_p, k, k_p)
