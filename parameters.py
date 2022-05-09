import constants
import math

########################################################################################################################
# Parameters
# Particles
n_ptc = 1

# Setup for the grid
n_cellx, n_celly = 512, 512
xmin, xmax, ymin, ymax = 0., 1., 0., 1.   # Spatial region
pos_ori = [xmin, ymin]
dx, dy = (xmax - xmin) / n_cellx, (ymax - ymin) / n_celly   # Spatial step
inv_dx, inv_dy = 1. / dx, 1. / dy   # Inverse dx and dy

# Time step
dt = 0.99*dx*dy / math.sqrt(dx*dx+dy*dy) / constants.c

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
