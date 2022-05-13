from parameters import *
from constants import *
from helper_func import *

import taichi as ti


# TODO: Figure out why rho and J should be calculated at different time
@ti.kernel
def rhoj(q: float,
         wght_field: ti.template(),
         p_field: ti.template(),
         u_field: ti.template(),
         J_field: ti.template(),
         rho_field: ti.template(),
         ):
    """
    computes *half* charge density RHO and *half* of the current density J on grid nodes
    """
    for idx_ptc in p_field:
        pos_ptc = p_field[idx_ptc] - pos_ori
        u_ptc = u_field[idx_ptc]
        wght_ptc = wght_field[idx_ptc]

        rhop = q * e * wght_ptc / dV
        gam = ti.sqrt(1. + u_ptc.norm_sqr())
        v_ptc = u_ptc * c / gam

        if ti.static(DIM == 3):
            inv_trilerp(rho_field, pos_ptc, 0.5 * rhop)
            inv_trilerp(J_field, pos_ptc, 0.5 * rhop * v_ptc)
        else:
            inv_bilerp(rho_field, pos_ptc[0:2], 0.5 * rhop)
            inv_bilerp(J_field, pos_ptc[0:2], 0.5 * rhop * v_ptc)

@ti.kernel
def j_grid2yee(
        J_grid: ti.template(),
        J_yee: ti.template()
):
    """
    Transfer J from grid to yee lattice
    """
    for Idx in ti.grouped(J_grid):
        if ti.static(DIM == 3):
            i, j, k = Idx[0], Idx[1], Idx[2]
            i_p = i + 1 if i != n_cellx - 1 else 0
            j_p = j + 1 if j != n_celly - 1 else 0
            k_p = k + 1 if k != n_cellz - 1 else 0

            Jx = (J_grid[i, j, k][0] + J_grid[i_p, j, k][0]) / 2.
            Jy = (J_grid[i, j, k][1] + J_grid[i, j_p, k][1]) / 2.
            Jz = (J_grid[i, j, k][2] + J_grid[i, j, k_p][2]) / 2.

            J_yee[i, j, k] = ti.Vector([Jx, Jy, Jz])
        else:
            i, j = Idx[0], Idx[1]
            i_p = i + 1 if i != n_cellx - 1 else 0
            j_p = j + 1 if j != n_celly - 1 else 0

            Jx = (J_grid[i, j][0] + J_grid[i_p, j][0]) / 2.
            Jy = (J_grid[i, j][1] + J_grid[i, j_p][1]) / 2.
            Jz = J_grid[i, j][2]

            J_yee[i, j] = ti.Vector([Jx, Jy, Jz])