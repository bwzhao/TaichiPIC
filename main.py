# PIC simulation using Taichi
# Author: Bowen Zhao (zzzhaobowen@gmail.com)
# Based on Zeltron (https://ipag.osug.fr/~ceruttbe/Zeltron/features.html)
# Using Yee mesh for the fields
# Using Boris push for the particles

import taichi as ti
import numpy as np

from particle_func import *
from field_func import *
from helper_func import *


ti.init(arch=ti.cpu)

########################################################################################################################
"""
Index Convention:
x, y used as real numbers in reason space
i, j used a integers when labeling the grids
"""
# Position
pos_e = ti.Vector.field(3, dtype=float, shape=n_ptc) # electrons
pos_p = ti.Vector.field(3, dtype=float, shape=n_ptc) # ions

# Velocity
u_e = ti.Vector.field(3, dtype=float, shape=n_ptc) # electrons
u_p = ti.Vector.field(3, dtype=float, shape=n_ptc) # ions

# Weight of the macro-particles
wght_e = ti.field(dtype=float, shape=n_ptc)  # electrons
wght_p = ti.field(dtype=float, shape=n_ptc)  # ions

# Field
B_yee = ti.Vector.field(3, dtype=float, shape=(n_cellx, n_celly))   # Magnetic on Yee lattice
E_yee = ti.Vector.field(3, dtype=float, shape=(n_cellx, n_celly))   # Electric on Yee lattice
B_grid = ti.Vector.field(3, dtype=float, shape=(n_cellx, n_celly))   # Magnetic at nodes
E_grid = ti.Vector.field(3, dtype=float, shape=(n_cellx, n_celly))   # Electric at nodes

# Charge and current
J_yee = ti.Vector.field(3, dtype=float, shape=(n_cellx, n_celly))  # Current on Yee lattice
J_grid = ti.Vector.field(3, dtype=float, shape=(n_cellx, n_celly))  # Current at nodes
rho_grid = ti.field(dtype=float, shape=(n_cellx, n_celly))  #  Charge

phi_new = ti.field(dtype=float, shape=(n_cellx, n_celly))  # potential
phi_old = ti.field(dtype=float, shape=(n_cellx, n_celly))  # potential
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
        pos_e[idx_ptc] = [xmin + ti.random() * (xmax - xmin), ymin + ti.random() * (ymax - ymin), 0.]
        pos_p[idx_ptc] = [xmin + ti.random() * (xmax - xmin), ymin + ti.random() * (ymax - ymin), 0.]

        # u_e[idx_ptc] = [ti.random(), ti.random(), 0.]
        # u_p[idx_ptc] = [ti.random(), ti.random(), 0.]
        u_e[idx_ptc] = [0, 0, 0]
        u_p[idx_ptc] = [0, 0, 0]

        wght_e = n0 * (xmax - xmin) * (ymax - ymin) / n_ptc
        wght_p = n0 * (xmax - xmin) * (ymax - ymin) / n_ptc

    # Fields:
    for i, j in B_yee:
        B_yee[i, j] = [0., 0., 0.]
        E_yee[i, j] = [0., 0., 0.]


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
    boundary_particles(pos_e, xmax, ymax)
    boundary_particles(pos_p, xmax, ymax)
    rhoj(-1., wght_e, pos_e, u_e, J_grid, rho_grid)
    rhoj(1., wght_p, pos_p, u_p, J_grid, rho_grid)

    # Fields
    j_grid2yee(J_grid, J_yee)
    push_bhalf(B_yee, E_yee)
    push_efield(B_yee, E_yee, J_yee)
    # correct_efield(E_yee, rho_grid, phi_old, phi_new)
    clear_field_scalar(phi_old)
    push_bhalf(B_yee, E_yee)


def illustratio_init():
    gui_1 = ti.GUI("Taichi PIC Particles", res=512, background_color=0x112F41)
    gui_2 = ti.GUI("Taichi PIC E-Fields", res=(n_cellx, n_celly))
    gui_3 = ti.GUI("Taichi PIC B-Fields", res=(n_cellx, n_celly))

    return gui_1, gui_2, gui_3


def illustration_update():
    # Electrons
    gui_1.circles(pos_e.to_numpy()[:, 0:2] / (xmax - xmin),
                radius=3, color=0x068587)
    # Ions
    gui_1.circles(pos_p.to_numpy()[:, 0:2] / (xmax - xmin),
                radius=3, color=0xED553B)

    gui_2.set_image(E_yee)
    gui_3.set_image(E_yee)

    gui_1.show()
    gui_2.show()
    gui_3.show()
########################################################################################################################


if __name__ == '__main__':
    gui_1, gui_2, gui_3 = illustratio_init()
    initiate()
    initial_push(-1., me, pos_e, u_e, E_grid, B_grid)
    initial_push(1., me, pos_p, u_p, E_grid, B_grid)

    for frame in range(10000):
        update()
        illustration_update()



