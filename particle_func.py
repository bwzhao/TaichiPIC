from parameters import *
from constants import *
from helper_func import *

import taichi as ti

@ti.kernel
def initial_push(q: float,
                 mass: float,
                 p_field: ti.template(),
                 u_field: ti.template(),
                 Eg: ti.template(),
                 Bg: ti.template()):
    """
    Calculate u(0 - dt/2)
    """
    for idx_ptc in p_field:
        pos_t = p_field[idx_ptc][0:2] - pos_ori
        u_t = u_field[idx_ptc]

        # Transfer fields from grid to particle
        i, j = ti.cast(ti.floor(pos_t * inv_dx), int)
        i_p = i + 1 if i != n_cellx - 1 else 0
        j_p = j + 1 if j != n_celly - 1 else 0
        pq = pos_t * inv_dx - ti.cast(ti.Vector([i, j]), float)
        E_ptc = bilerp(Eg[i, j], Eg[i_p, j], Eg[i, j_p], Eg[i_p, j_p], pq)
        B_ptc = bilerp(Bg[i, j], Bg[i_p, j], Bg[i, j_p], Bg[i_p, j_p], pq)

        # Doing "half" of a boris push
        gam = ti.sqrt(1. + u_t.norm_sqr())
        t = q * e * B_ptc * dt / (4. * gam * mass * c)
        s = 2. * t / (1. + t.norm_sqr())
        u0 = -u_t + u_t.cross(t)
        up = u_t + u0.cross(s)

        # uL at t = -dt/2
        uL = up - q * e * E_ptc * dt / (2. * mass * c)

        # # Synchrotron radiatative power losses, and Radiative inverse Compton energy losses (Thomson regime) at t=0
        # Psyn = (2. / 3.) * e ** 4. / (mass ** 2. * c ** 4.) * c *\
        #       ((gam * E_ptc + u_t.cross(B_ptc)).norm_sqr() - u_t.dot(E_ptc).norm_sqr())
        # Pics = (32. / 9.) * pi * e ** 4. / (mass ** 2. * c ** 4.) * c * Uph * u_t.norm_sqr()

        # Write final results back to u-field
        u_field[idx_ptc] = uL  # + dt * (Psyn + Pics) / (2. * mass * c * c) * u_t / gam


@ti.kernel
def boris_push(q: float,
               mass: float,
               p_field: ti.template(),
               u_field: ti.template(),
               Eg: ti.template(),
               Bg: ti.template()
               ):
    """
    Calculate u(t + dt/2) based on u(t - dt/2)
    """
    for idx_ptc in p_field:
        pos_t = p_field[idx_ptc][0:2] - pos_ori
        u_tmhalf = u_field[idx_ptc]

        # Transfer fields from grid to particle
        i, j = ti.cast(ti.floor(pos_t * inv_dx), int)
        i_p = i + 1 if i != n_cellx - 1 else 0
        j_p = j + 1 if j != n_celly - 1 else 0
        pq = pos_t * inv_dx - ti.cast(ti.Vector([i, j]), float)
        E_ptc = bilerp(Eg[i, j], Eg[i_p, j], Eg[i, j_p], Eg[i_p, j_p], pq)
        B_ptc = bilerp(Bg[i, j], Bg[i_p, j], Bg[i, j_p], Bg[i_p, j_p], pq)

        # Doing "half" of a boris push
        gam = ti.sqrt(1. + u_tmhalf.norm_sqr())
        um = u_tmhalf + q * e * E_ptc * dt / (2. * mass * c)
        t = q * e * B_ptc * dt / (2. * gam * mass * c)
        s = 2. * t / (1. + t.norm_sqr())
        u0 = um + um.cross(t)
        up = u_tmhalf + u0.cross(s)

        # u
        u_tphalf = up + q * e * E_ptc * dt / (2. * mass * c)


        # Synchrotron radiatative power losses, and Radiative inverse Compton energy losses (Thomson regime) at t=0
        # u_t = (u_tphalf + u_tmhalf) / 2.
        # Psyn = (2. / 3.) * e ** 4. / (mass ** 2. * c ** 4.) * c *\
        #        ((gam * E_ptc + u_t.cross(B_ptc)).norm_sqr() - u_t.dot(E_ptc).norm_sqr())
        # Pics = (32. / 9.) * pi * e ** 4. / (mass ** 2. * c ** 4.) * c * Uph * u_t.norm_sqr()

        # Write final results back to u-field
        u_field[idx_ptc] = u_tphalf # - dt * (Psyn + Pics) / (mass * c * c) * u_t / gam


@ti.kernel
def push_particles(p_field: ti.template(),
                   u_field: ti.template()):
    """
    Update the positions of the particles using the 4-velocity known at t+dt/2
    """
    for idx_ptc in p_field:
        gam = ti.sqrt(1. + u_field[idx_ptc].norm_sqr())
        p_field[idx_ptc] = p_field[idx_ptc] + (c * dt) / gam * u_field[idx_ptc]


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
        pos_ptc = p_field[idx_ptc][:2] - pos_ori
        u_ptc = u_field[idx_ptc]
        wght_ptc = wght_field[idx_ptc]

        rhop = q * e * wght_ptc / (dx * dy)
        gam = ti.sqrt(1. + u_ptc.norm_sqr())
        v_ptc = u_ptc * c / gam

        i, j = ti.cast(ti.floor(pos_ptc * inv_dx), int)
        i_p = i + 1 if i != n_cellx - 1 else 0
        j_p = j + 1 if j != n_celly - 1 else 0
        pq = pos_ptc * inv_dx - ti.cast(ti.Vector([i, j]), float)

        # Calculate rho and J on the grid
        inv_bilear(rho_field[i, j], rho_field[i_p, j], rho_field[i, j_p], rho_field[i_p, j_p], pq, 0.5 * rhop)
        inv_bilear(J_field[i, j], J_field[i_p, j], J_field[i, j_p], J_field[i_p, j_p], pq, 0.5 * rhop * v_ptc)