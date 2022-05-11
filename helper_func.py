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
def grad(f_field: ti.template(),
         i, i_p,
         j, j_p,
         k, k_p):
    g_x = (f_field[i_p, j, k] - f_field[i, j, k]) * inv_dx
    g_y = (f_field[i, j_p, k] - f_field[i, j, k]) * inv_dy
    g_z = (f_field[i, j, k_p] - f_field[i, j, k]) * inv_dz

    return ti.Vector([g_x, g_y, g_z])


@ti.func
def div(f_field: ti.template(),
        i, i_p,
        j, j_p,
        k, k_p):
    val = \
        (f_field[i_p, j, k][0] - f_field[i, j, k][0]) * inv_dx \
        + (f_field[i, j_p, k][1] - f_field[i, j, k][1]) * inv_dy \
        + (f_field[i, j, k_p][2] - f_field[i, j, k][2]) * inv_dz

    return val


@ti.func
def curl(f_field: ti.template(),
         i, i_p,
         j, j_p,
         k, k_p):
    """
    Calculate the curl at a specific position on yee lattice
    """
    c_x = inv_dy * (f_field[i, j_p, k][2] - f_field[i, j, k][2]) - inv_dz * (f_field[i, j, k_p][1] - f_field[i, j, k][1])
    c_y = inv_dz * (f_field[i, j, k_p][0] - f_field[i, j, k][0]) - inv_dx * (f_field[i_p, j, k][2] - f_field[i, j, k][2])
    c_z = inv_dx * (f_field[i_p, j, k][1] - f_field[i, j, k][1]) - inv_dy * (f_field[i, j_p, k][0] - f_field[i, j, k][0])

    return ti.Vector([c_x, c_y, c_z])


@ti.func
def trilerp(f_field: ti.template(),
            pos):
    """
    :param f_field: field to be interpolated
    :param pos: pos of the particle
    :return: The interpolated values
    """
    i, j, k = ti.cast(ti.floor(pos * inv_dr), int)
    i = i if i != n_cellx else n_cellx - 1
    j = j if j != n_celly else n_celly - 1
    k = k if k != n_cellz else n_cellz - 1
    assert i < n_cellx and j < n_celly and k < n_cellz
    i_p = i + 1 if i != n_cellx - 1 else 0
    j_p = j + 1 if j != n_celly - 1 else 0
    k_p = k + 1 if k != n_cellz - 1 else 0

    xq, yq, zq = pos * inv_dr - ti.cast(ti.Vector([i, j, k]), float)

    val = (1. - xq) * ((1. - yq) * (1. - zq) * f_field[i, j, k]
                       + yq * (1. - zq) * f_field[i, j_p, k]
                       + yq * zq * f_field[i, j_p, k_p]
                       + (1. - yq) * zq * f_field[i, j, k_p]) \
          + xq * ((1. - yq) * (1. - zq) * f_field[i_p, j, k]
                  + yq * (1. - zq) * f_field[i_p, j_p, k]
                  + yq * zq * f_field[i_p, j_p, k_p]
                  + (1. - yq) * zq * f_field[i_p, j, k_p])

    return val


@ti.func
def inv_trilerp(f_field: ti.template(),
                pos,
                val
                ):
    i, j, k = ti.cast(ti.floor(pos * inv_dr), int)
    i = i if i != n_cellx else n_cellx - 1
    j = j if j != n_celly else n_celly - 1
    k = k if k != n_cellz else n_cellz - 1
    assert i < n_cellx and j < n_celly and k < n_cellz
    i_p = i + 1 if i != n_cellx - 1 else 0
    j_p = j + 1 if j != n_celly - 1 else 0
    k_p = k + 1 if k != n_cellz - 1 else 0
    xq, yq, zq = pos * inv_dr - ti.cast(ti.Vector([i, j, k]), float)

    f_field[i, j, k] += val * (1. - xq) * (1. - yq) * (1. - zq)
    f_field[i, j, k_p] += val * (1. - xq) * (1. - yq) * zq
    f_field[i, j_p, k] += val * (1. - xq) * yq * (1. - zq)
    f_field[i, j_p, k_p] += val * (1. - xq) * yq * zq
    f_field[i_p, j, k] += val * xq * (1. - yq) * (1. - zq)
    f_field[i_p, j, k_p] += val * xq * (1. - yq) * zq
    f_field[i_p, j_p, k] += val * xq * yq * (1. - zq)
    f_field[i_p, j_p, k_p] += val * xq * yq * zq


@ti.kernel
def eb_yee2grid(E_grid: ti.template(),
                E_yee: ti.template(),
                B_grid: ti.template(),
                B_yee: ti.template()
                ):
    """
    Transfer field from Yee lattice to grids
    """
    for i, j, k in E_grid:
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


@ti.kernel
def j_grid2yee(
        J_grid: ti.template(),
        J_yee: ti.template()
):
    """
    Transfer J from grid to yee lattice
    """
    for i, j, k in J_grid:
        i_p = i + 1 if i != n_cellx - 1 else 0
        j_p = j + 1 if j != n_celly - 1 else 0
        k_p = k + 1 if k != n_cellz - 1 else 0

        Jx = (J_grid[i, j, k][0] + J_grid[i_p, j, k][0]) / 2.
        Jy = (J_grid[i, j, k][1] + J_grid[i, j_p, k][1]) / 2.
        Jz = (J_grid[i, j, k][2] + J_grid[i, j, k_p][2]) / 2.

        J_yee[i, j, k] = ti.Vector([Jx, Jy, Jz])


@ti.kernel
def boundary_particles(field_p: ti.template()):
    """
    Deal with boundary conditions for all the particles
    """
    for idx_ptc in field_p:
        field_p[idx_ptc][0] = (field_p[idx_ptc][0] - xmax) if field_p[idx_ptc][0] >= xmax else field_p[idx_ptc][0]
        field_p[idx_ptc][0] = (field_p[idx_ptc][0] + xmax) if field_p[idx_ptc][0] < 0 else field_p[idx_ptc][0]
        field_p[idx_ptc][1] = (field_p[idx_ptc][1] - ymax) if field_p[idx_ptc][1] >= ymax else field_p[idx_ptc][1]
        field_p[idx_ptc][1] = (field_p[idx_ptc][1] + ymax) if field_p[idx_ptc][1] < 0 else field_p[idx_ptc][1]
        field_p[idx_ptc][2] = (field_p[idx_ptc][2] - zmax) if field_p[idx_ptc][2] >= zmax else field_p[idx_ptc][2]
        field_p[idx_ptc][2] = (field_p[idx_ptc][2] + zmax) if field_p[idx_ptc][2] < 0 else field_p[idx_ptc][2]

        assert field_p[idx_ptc][0] < xmax and field_p[idx_ptc][1] < ymax and field_p[idx_ptc][2] < zmax