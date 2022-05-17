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
from initialize import *


ti.init(arch=ti.gpu, debug=True)

########################################################################################################################
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
B_yee = ti.Vector.field(3, dtype=float, shape=dim_cells)   # Magnetic on Yee lattice
E_yee = ti.Vector.field(3, dtype=float, shape=dim_cells)   # Electric on Yee lattice
B_grid = ti.Vector.field(3, dtype=float, shape=dim_cells)   # Magnetic at nodes
E_grid = ti.Vector.field(3, dtype=float, shape=dim_cells)   # Electric at nodes

# Charge and current
J_yee = ti.Vector.field(3, dtype=float, shape=dim_cells)  # Current on Yee lattice
J_grid = ti.Vector.field(3, dtype=float, shape=dim_cells)  # Current at nodes
rho_grid = ti.field(dtype=float, shape=dim_cells)  # Charge

phi_new = ti.field(dtype=float, shape=dim_cells)  # potential
phi_old = ti.field(dtype=float, shape=dim_cells)  # potential

# GUI
indices_xy = ti.field(int, 6)
indices_xz = ti.field(int, 6)
indices_yz = ti.field(int, 6)
normals_xy = ti.Vector.field(3, float, 6)
normals_xz = ti.Vector.field(3, float, 6)
normals_yz = ti.Vector.field(3, float, 6)
vertices = ti.Vector.field(3, float, 7)

########################################################################################################################


########################################################################################################################
# Initialization



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
    # if frame % freq_poisson == 0:
    #     clear_field_scalar(phi_old)
    #     clear_field_scalar(phi_new)
    #     for _ in range(niter_poisson):
    #         iter_phi(E_yee, rho_grid, phi_old, phi_new)
    #     correct_efield(E_yee, phi_new)
    push_bhalf(B_yee, E_yee)
########################################################################################################################


if __name__ == '__main__':
    if ti.static(GGUI):
        window, canvas, scene, camera = ggui_init()
        set_vertics_indices(vertices, indices_xy, indices_xz, indices_yz,
                            normals_xy, normals_xz, normals_yz)
    else:
        gui = gui_init()

    initialize_random(pos_e, pos_p, u_e, u_p, wght_e, wght_p, B_yee, E_yee)
    eb_yee2grid(E_grid, E_yee, B_grid, B_yee)
    initial_push(-1., me, pos_e, u_e, E_grid, B_grid)
    initial_push(1., mp, pos_p, u_p, E_grid, B_grid)

    for frame in range(10000):
        if frame % 100 == 0:
            print("#Updates: %d" % frame)
        update()
        if ti.static(GGUI):
            ggui_update(window, canvas, scene, camera,
                        pos_e, pos_p, colors_e, colors_p,
                        vertices, indices_xy, indices_xz, indices_yz,
                        normals_xy, normals_xz, normals_yz)
        else:
            gui_update(gui, pos_e, pos_p)


