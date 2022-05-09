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
    for i, j in B_yee:
        i_p = i + 1 if i != n_cellx - 1 else 0
        j_p = j + 1 if j != n_celly - 1 else 0

        Bx = B_yee[i, j][0] - c * dt / (2. * dy) * (E_yee[i, j_p][2] - E_yee[i, j][2])
        By = B_yee[i, j][1] + c * dt / (2. * dx) * (E_yee[i_p, j][2] - E_yee[i, j][2])
        Bz = B_yee[i, j][2] - c * dt / (2. * dx) * (E_yee[i_p, j][1] - E_yee[i, j][1]) \
                            + c * dt / (2. * dy) * (E_yee[i, j_p][0] - E_yee[i, j][0])

        B_yee[i, j] = ti.Vector([Bx, By, Bz])


@ti.kernel
def push_efield(
        B_yee: ti.template(),
        E_yee: ti.template(),
        J_yee: ti.template()
):
    """
    computes the E field vector at time t+dt
    """
    for i, j in E_yee:
        j_m = j - 1 if j != 0 else n_celly - 1
        i_m = i - 1 if i != 0 else n_cellx - 1

        Ex = E_yee[i, j][0] + (c * dt / dy) * (B_yee[i, j][2] - B_yee[i, j_m][2]) - 4. * pi * dt * J_yee[i, j][0]
        Ey = E_yee[i, j][1] - (c * dt / dx) * (B_yee[i, j][2] - B_yee[i_m, j][2]) - 4. * pi * dt * J_yee[i, j][1]
        Ez = E_yee[i, j][2] + (c * dt / dx) * (B_yee[i, j][1] - B_yee[i_m, j][1]) \
                            - (c * dt / dy) * (B_yee[i, j][0] - B_yee[i, j_m][0]) - 4. * pi * dt * J_yee[i, j][2]

        E_yee[i, j] = ti.Vector([Ex, Ey, Ez])


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
    denom = dx * dx + dy * dy

    for i, j in ti.ndrange(n_cellx, n_celly):
        i_p = i + 1 if i != n_cellx - 1 else 0
        j_p = j + 1 if j != n_celly - 1 else 0
        j_m = j - 1 if j != 0 else n_celly - 1
        i_m = i - 1 if i != 0 else n_cellx - 1

        phi_new[i, j] = 0.5 / denom * (
                (phi_old[i_p, j] + phi_old[i_m, j]) * dy * dy
                + (phi_old[i, j_p] + phi_old[i, j_m]) * dx * dx
                + (
                        4. * pi * rho_field[i, j] - ((E_yee[i, j][0] - E_yee[i_m, j][0]) / dx + (E_yee[i, j][1] - E_yee[i, j_m][1]) / dy)
                ) * dx * dx * dy * dy
        )

    for i, j in ti.ndrange(n_cellx, n_celly):
        phi_old[i, j] = phi_new[i, j]


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
    for i, j in E_yee:
        i_p = i + 1 if i != n_cellx - 1 else 0
        j_p = j + 1 if j != n_celly - 1 else 0

        Ex = E_yee[i, j][0] - (phi_new[i_p, j] - phi_new[i, j]) / dx
        Ey = E_yee[i, j][1] - (phi_new[i, j_p] - phi_new[i, j]) / dy

        E_yee[i, j] = ti.Vector([Ex, Ey, 0.])