# helper_func.py: helper ti.kernels and ti.funcs
# Author: Bowen Zhao (zzzhaobowen@gmail.com)

import taichi as ti
from parameters import *
from constants import *


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
def grad_2d(f_field: ti.template(),
         i, i_p,
         j, j_p):
    g_x = (f_field[i_p, j] - f_field[i, j]) * inv_dx
    g_y = (f_field[i, j_p] - f_field[i, j]) * inv_dy

    return ti.Vector([g_x, g_y, 0.])


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
def div_2d(f_field: ti.template(),
        i, i_p,
        j, j_p):
    val = \
        (f_field[i_p, j][0] - f_field[i, j][0]) * inv_dx \
        + (f_field[i, j_p][1] - f_field[i, j][1]) * inv_dy

    return val


# @ti.func
# def curl(f_field: ti.template(),
#          i, i_p,
#          j, j_p,
#          k, k_p):
#     """
#     Calculate the curl at a specific position on yee lattice
#     """
#     c_x = inv_dy * (f_field[i, j_p, k][2] - f_field[i, j, k][2]) - inv_dz * (f_field[i, j, k_p][1] - f_field[i, j, k][1])
#     c_y = inv_dz * (f_field[i, j, k_p][0] - f_field[i, j, k][0]) - inv_dx * (f_field[i_p, j, k][2] - f_field[i, j, k][2])
#     c_z = inv_dx * (f_field[i_p, j, k][1] - f_field[i, j, k][1]) - inv_dy * (f_field[i, j_p, k][0] - f_field[i, j, k][0])
#
#     return ti.Vector([c_x, c_y, c_z])


# @ti.func
# def curl_2d(f_field: ti.template(),
#          i, i_p,
#          j, j_p):
#     """
#     Calculate the curl at a specific position on yee lattice
#     """
#     c_x = inv_dy * (f_field[i, j_p][2] - f_field[i, j][2])
#     c_y = - inv_dx * (f_field[i_p, j][2] - f_field[i, j][2])
#     c_z = inv_dx * (f_field[i_p, j][1] - f_field[i, j][1]) - inv_dy * (f_field[i, j_p][0] - f_field[i, j][0])
#
#     return ti.Vector([c_x, c_y, c_z])


@ti.func
def trilerp(f_field: ti.template(),
            pos):
    """
    Function of transferring the quantities on the particle to grid using tri-linear interpolation
    :param f_field: field to be interpolated on the grid
    :param pos: pos of the particle
    :return: The interpolated values
    """
    i, j, k = ti.cast(ti.floor(pos * inv_dr), int)
    i = i if i < n_cellx else n_cellx - 1
    j = j if j <  n_celly else n_celly - 1
    k = k if k <  n_cellz else n_cellz - 1
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
def bilerp(f_field: ti.template(),
            pos):
    """
    Function of transferring the quantities on the particle to grid using bi-linear interpolation
    :param f_field: field to be interpolated on the grid
    :param pos: pos of the particle
    :return: The interpolated values
    """
    i, j = ti.cast(ti.floor(pos * inv_dr), int)
    i = i if i < n_cellx else n_cellx - 1
    j = j if j < n_celly else n_celly - 1
    i = i if i >= 0 else 0
    j = j if j >= 0 else 0
    assert i < n_cellx and j < n_celly
    i_p = i + 1 if i != n_cellx - 1 else 0
    j_p = j + 1 if j != n_celly - 1 else 0

    xq, yq = pos * inv_dr - ti.cast(ti.Vector([i, j]), float)

    val = (1. - xq) * ((1. - yq) * f_field[i, j]
                       + yq * f_field[i, j_p]) \
          + xq * ((1. - yq) * f_field[i_p, j]
                  + yq * f_field[i_p, j_p])

    return val


@ti.func
def inv_trilerp(f_field: ti.template(),
                pos,
                val
                ):
    """
    Function of transferring the quantities on grid to particles using trii-linear interpolation
    :param f_field: field on the grid
    :param pos: pos of the particle
    :param val: values on the particle
    :return: the interpolated value
    """
    i, j, k = ti.cast(ti.floor(pos * inv_dr), int)
    i = i if i < n_cellx else n_cellx - 1
    j = j if j < n_celly else n_celly - 1
    k = k if k < n_cellz else n_cellz - 1
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


@ti.func
def inv_bilerp(f_field: ti.template(),
                pos,
                val
                ):
    """
    Function of transferring the quantities on grid to particles using bi-linear interpolation
    :param f_field: field on the grid
    :param pos: pos of the particle
    :param val: values on the particle
    :return: the interpolated value
    """
    i, j = ti.cast(ti.floor(pos * inv_dr), int)
    i = i if i < n_cellx else n_cellx - 1
    j = j if j < n_celly else n_celly - 1
    i = i if i >= 0 else 0
    j = j if j >= 0 else 0
    assert i < n_cellx and j < n_celly
    i_p = i + 1 if i != n_cellx - 1 else 0
    j_p = j + 1 if j != n_celly - 1 else 0
    xq, yq = pos * inv_dr - ti.cast(ti.Vector([i, j]), float)

    f_field[i, j] += val * (1. - xq) * (1. - yq)
    f_field[i, j_p] += val * (1. - xq) * yq
    f_field[i_p, j] += val * xq * (1. - yq)
    f_field[i_p, j_p] += val * xq * yq


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
        if ti.static(DIM == 3):
            field_p[idx_ptc][2] = (field_p[idx_ptc][2] - zmax) if field_p[idx_ptc][2] >= zmax else field_p[idx_ptc][2]
            field_p[idx_ptc][2] = (field_p[idx_ptc][2] + zmax) if field_p[idx_ptc][2] < 0 else field_p[idx_ptc][2]