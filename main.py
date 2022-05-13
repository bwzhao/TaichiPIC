# PIC simulation using Taichi
# Author: Bowen Zhao (zzzhaobowen@gmail.com)
# Based on Zeltron (https://ipag.osug.fr/~ceruttbe/Zeltron/features.html)
# Using Yee mesh for the fields
# Using Boris push for the particles

import taichi as ti
import numpy as np

from particle_func import *
from field_func import *
from rhoj_func import *
from helper_func import *
from gui import *


ti.init(arch=ti.cpu, debug=True, default_fp=ti.f32, default_ip=ti.i32)

########################################################################################################################
"""
Index Convention:
x, y used as real numbers in reason space
i, j used a integers when labeling the grids
"""
# Position
pos_e = ti.Vector.field(3, dtype=float, shape=n_ptc)  # electrons
pos_p = ti.Vector.field(3, dtype=float, shape=n_ptc)  # ions

# Velocity
u_e = ti.Vector.field(3, dtype=float, shape=n_ptc)  # electrons
u_p = ti.Vector.field(3, dtype=float, shape=n_ptc)  # ions

# Weight of the macro-particles
wght_e = ti.field(dtype=float, shape=n_ptc)  # electrons
wght_p = ti.field(dtype=float, shape=n_ptc)  # ions

# For illustration
colors_e = ti.Vector.field(4, dtype=float, shape=n_ptc)
colors_p = ti.Vector.field(4, dtype=float, shape=n_ptc)

# TODO: Here we only use PBC (So that we only need (n_cellx, n_celly, n_cellz) grids),
#  but in general we can also have other BCs.
# Field
B_yee = ti.Vector.field(3, dtype=float, shape=n_cells)   # Magnetic on Yee lattice
E_yee = ti.Vector.field(3, dtype=float, shape=n_cells)   # Electric on Yee lattice
B_grid = ti.Vector.field(3, dtype=float, shape=n_cells)   # Magnetic at nodes
E_grid = ti.Vector.field(3, dtype=float, shape=n_cells)   # Electric at nodes

# Charge and current
J_yee = ti.Vector.field(3, dtype=float, shape=n_cells)  # Current on Yee lattice
J_grid = ti.Vector.field(3, dtype=float, shape=n_cells)  # Current at nodes
rho_grid = ti.field(dtype=float, shape=n_cells)  # Charge

phi_new = ti.field(dtype=float, shape=n_cells)  # potential
phi_old = ti.field(dtype=float, shape=n_cells)  # potential
########################################################################################################################


########################################################################################################################
# Initialization
@ti.kernel
def initiate():
    """
    Initialize the particles and fields
    """
    # Particle spatial dim
    for idx_ptc in range(n_ptc):
        if ti.static(DIM == 3):
            pos_ptc = ti.Vector([xmin + ti.random() * (xmax - xmin) * 0.5,
                              ymin + ti.random() * (ymax - ymin) * 0.5,
                              zmin + ti.random() * (zmax - zmin) * 0.5])
            u_ptc = [0., 0., 0.]
            wght_ptc = n0 * (xmax - xmin) * (ymax - ymin) * (zmax - zmin) / n_ptc

            pos_e[idx_ptc] = pos_ptc
            pos_p[idx_ptc] = pos_ptc

            u_e[idx_ptc] = u_ptc
            u_p[idx_ptc] = u_ptc

            wght_e[idx_ptc] = wght_ptc
            wght_p[idx_ptc] = wght_ptc
        else:
            pos_ptc = ti.Vector([xmin + ti.random() * (xmax - xmin) * 0.5,
                       ymin + ti.random() * (ymax - ymin) * 0.5,
                       0.])
            u_ptc = [0.1, 0., 0.]
            wght_ptc = n0 * (xmax - xmin) * (ymax - ymin) / n_ptc

            pos_e[idx_ptc] = pos_ptc
            pos_p[idx_ptc] = pos_ptc

            u_e[idx_ptc] = u_ptc
            u_p[idx_ptc] = u_ptc

            wght_e[idx_ptc] = wght_ptc
            wght_p[idx_ptc] = wght_ptc

        colors_e[idx_ptc] = ti.Vector([1., 0., 0., 1.])
        colors_p[idx_ptc] = ti.Vector([0., 0., 1., 1.])

    # Fields:
    for Idx in ti.grouped(B_yee):
        B_yee[Idx] = [0., 0., 1.]
        E_yee[Idx] = [0., 0., 0.]


# Update
def update():
    # Transfer Fields to particle and solve Newton's equation
    eb_yee2grid(E_grid, E_yee, B_grid, B_yee)

    # Clear the J and rho field
    clear_field_vector(J_grid)
    clear_field_scalar(rho_grid)

    # Particles:
    boris_push(-1., me, pos_e, u_e, E_grid, B_grid)
    boris_push(1., mp, pos_p, u_p, E_grid, B_grid)
    rhoj(-1., wght_e, pos_e, u_e, J_grid, rho_grid)
    rhoj(1., wght_p, pos_p, u_p, J_grid, rho_grid)
    push_particles(pos_e, u_e)
    push_particles(pos_p, u_p)
    boundary_particles(pos_e)
    boundary_particles(pos_p)
    rhoj(-1., wght_e, pos_e, u_e, J_grid, rho_grid)
    rhoj(1., wght_p, pos_p, u_p, J_grid, rho_grid)

    # Fields
    j_grid2yee(J_grid, J_yee)
    push_bhalf(B_yee, E_yee)
    push_efield(B_yee, E_yee, J_yee)
    if frame % freq_poisson == 0:
        clear_field_scalar(phi_old)
        clear_field_scalar(phi_new)
        for _ in range(niter_poisson):
            iter_phi(E_yee, rho_grid, phi_old, phi_new)
        correct_efield(E_yee, phi_new)
    push_bhalf(B_yee, E_yee)
########################################################################################################################


if __name__ == '__main__':
    # window, canvas, scene, camera = gui_init()
    initiate()
    eb_yee2grid(E_grid, E_yee, B_grid, B_yee)
    initial_push(-1., me, pos_e, u_e, E_grid, B_grid)
    initial_push(1., mp, pos_p, u_p, E_grid, B_grid)

    for frame in range(10000):
        update()
        if frame % 5 == 0:
            # Only for debug
            print(E_yee[0, 0], B_yee[0, 0])
            print(rho_grid[0, 0], J_grid[0, 0])
            print(pos_e[0])
            # gui_update(window, canvas, scene, camera, pos_e, pos_p, colors_e, colors_p)



