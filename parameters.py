# parameters.py: model parameters
# Author: Bowen Zhao (zzzhaobowen@gmail.com)

import constants
import math

########################################################################################################################
# Parameters

# Dimensions
DIM = 3

# If use GGUI
GGUI = False

# Setup for the grid
if DIM == 3:
    n_cellx, n_celly, n_cellz = 128, 128, 128
    dim_cells = (n_cellx, n_celly, n_cellz)
    num_cells = n_cellx * n_celly * n_cellz
    xmin, xmax, ymin, ymax, zmin, zmax = 0., 128., 0., 128., 0., 128.   # Spatial region
    pos_ori = [xmin, ymin, zmin]
    dx, dy, dz = (xmax - xmin) / n_cellx, (ymax - ymin) / n_celly, (zmax - zmin) / n_cellz   # Spatial step
    inv_dx, inv_dy, inv_dz = 1. / dx, 1. / dy, 1. / dz   # Inverse dx and dy
    dr = [dx, dy, dz]
    dV = dx * dy * dz
    inv_dr = [inv_dx, inv_dy, inv_dz]
    # Time step
    dt = 0.99 / math.sqrt(inv_dx * inv_dx + inv_dy * inv_dy + inv_dz * inv_dz) / constants.c
else:
    n_cellx, n_celly = 128, 128
    dim_cells = (n_cellx, n_celly)
    num_cells = n_cellx * n_celly
    xmin, xmax, ymin, ymax = 0., 128., 0., 128.  # Spatial region
    pos_ori = [xmin, ymin, 0.]
    dx, dy = (xmax - xmin) / n_cellx, (ymax - ymin) / n_celly  # Spatial step
    inv_dx, inv_dy = 1. / dx, 1. / dy  # Inverse dx and dy
    dr = [dx, dy]
    dV = dx * dy
    inv_dr = [inv_dx, inv_dy]
    # Time step
    dt = 0.99 / math.sqrt(inv_dx * inv_dx + inv_dy * inv_dy) / constants.c

# Particles
# n_ptc = num_cells
n_ptc = 128

# Physical density of the particles
n0 = 1.

# External radiation field energy density
# TODO: Figure out what those parameters are!
# udens_ratio = 0.
# B0 = 1.  # Guide Field strength
# Uph = udens_ratio * (B0 * B0 / (8. * constants.pi))

# Number of iteration for correction of E fields
niter_poisson = 500
freq_poisson = 25
