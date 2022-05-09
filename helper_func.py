import taichi as ti
from parameters import *

@ti.kernel
def clear_field_scalar(field: ti.template()):
    for Idx in ti.grouped(field):
        field[Idx] = 0


@ti.kernel
def clear_field_vector(field: ti.template()):
    for Idx in ti.grouped(field):
        field[Idx].fill(0)


@ti.func
def bilerp(f00, f10, f01, f11, pq):
    """

    :param f00: field at (i, j)
    :param f10: field at (i + 1, j)
    :param f01: field at (i, j + 1)
    :param f11: field at (i + 1, j + 1)
    :param pq: Weight parameters
    :return: The interpolated values
    """
    return f00 * (1. - pq[0]) * (1. - pq[1]) \
           + f11 * pq[0] * pq[1] \
           + f10 * pq[0] * (1. - pq[1]) \
           + f01 * pq[1] * (1. - pq[0])


@ti.func
def inv_bilear(f00: ti.template(),
               f10: ti.template(),
               f01: ti.template(),
               f11: ti.template(),
               pq,
               val
               ):
    f00 += val * (1. - pq[0]) * (1. - pq[1])
    f10 += val * pq[0] * (1. - pq[1])
    f01 += val * pq[1] * (1. - pq[0])
    f11 += val * pq[0] * pq[1]


@ti.kernel
def eb_yee2grid(E_grid: ti.template(),
                E_yee: ti.template(),
                B_grid: ti.template(),
                B_yee: ti.template(), ):
    """
    Transfer field from Yee lattice to grids
    """
    for i, j in E_grid:
        j_m = j - 1 if j != 0 else n_celly - 1
        i_m = i - 1 if i != 0 else n_cellx - 1

        Egx_ptc = (E_yee[i, j][0] + E_yee[i_m, j][0]) / 2.
        Egy_ptc = (E_yee[i, j][1] + E_yee[i, j_m][1]) / 2.
        Egz_ptc = E_yee[i, j][2]
        E_grid[i, j] = [Egx_ptc, Egy_ptc, Egz_ptc]

        Bgx_ptc = (B_yee[i, j][0] + B_yee[i, j_m][0]) / 2.
        Bgy_ptc = (B_yee[i, j][1] + B_yee[i_m, j][1]) / 2.
        Bgz_ptc = (B_yee[i, j][2] + B_yee[i, j_m][2] + B_yee[i_m, j][2] + B_yee[i_m, j_m][2]) / 4.
        B_grid[i, j] = [Bgx_ptc, Bgy_ptc, Bgz_ptc]


@ti.kernel
def j_grid2yee(
        J_grid: ti.template(),
        J_yee: ti.template()
):
    """
    Transfer J from grid to yee lattice
    """
    for i, j in J_grid:
        i_p = i + 1 if i != n_cellx - 1 else 0
        j_p = j + 1 if j != n_celly - 1 else 0

        Jx = (J_grid[i, j][0] + J_grid[i_p, j][0]) / 2.
        Jy = (J_grid[i, j][1] + J_grid[i, j_p][1]) / 2.
        Jz = J_grid[i, j][2]

        J_yee[i, j] = ti.Vector([Jx, Jy, Jz])



@ti.kernel
def boundary_particles(field_p: ti.template()):
    """
    Deal with boundary conditions for all the particles
    """
    for idx_ptc in field_p:
        field_p[idx_ptc][0] = (field_p[idx_ptc][0] - xmax) if field_p[idx_ptc][0] > xmax else field_p[idx_ptc][0]
        field_p[idx_ptc][0] = (field_p[idx_ptc][0] + xmax) if field_p[idx_ptc][0] < 0 else field_p[idx_ptc][0]
        field_p[idx_ptc][1] = (field_p[idx_ptc][1] - ymax) if field_p[idx_ptc][1] > ymax else field_p[idx_ptc][1]
        field_p[idx_ptc][1] = (field_p[idx_ptc][1] + ymax) if field_p[idx_ptc][1] < 0 else field_p[idx_ptc][1]