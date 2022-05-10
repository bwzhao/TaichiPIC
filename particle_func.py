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
        pos_t = p_field[idx_ptc] - pos_ori
        u_t = u_field[idx_ptc]

        E_ptc = trilerp(Eg, pos_t)
        B_ptc = trilerp(Bg, pos_t)

        # Doing "half" of a boris push
        gam = ti.sqrt(1. + u_t.norm_sqr())
        t = q * e * B_ptc * dt / (4. * gam * mass * c)
        s = 2. * t / (1. + t.norm_sqr())
        u0 = -u_t + u_t.cross(t)
        up = u_t + u0.cross(s)

        # uL at t = -dt/2
        uL = up - q * e * E_ptc * dt / (2. * mass * c)

        # Write final results back to u-field
        u_field[idx_ptc] = uL


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
        pos_t = p_field[idx_ptc] - pos_ori
        u_tmhalf = u_field[idx_ptc]

        # Transfer fields from grid to particle
        E_ptc = trilerp(Eg, pos_t)
        B_ptc = trilerp(Bg, pos_t)

        # Doing "half" of a boris push
        gam = ti.sqrt(1. + u_tmhalf.norm_sqr())
        um = u_tmhalf + q * e * E_ptc * dt / (2. * mass * c)
        t = q * e * B_ptc * dt / (2. * gam * mass * c)
        s = 2. * t / (1. + t.norm_sqr())
        u0 = um + um.cross(t)
        up = u_tmhalf + u0.cross(s)

        # u
        u_tphalf = up + q * e * E_ptc * dt / (2. * mass * c)

        # Write final results back to u-field
        u_field[idx_ptc] = u_tphalf


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
        pos_ptc = p_field[idx_ptc] - pos_ori
        u_ptc = u_field[idx_ptc]
        wght_ptc = wght_field[idx_ptc]

        rhop = q * e * wght_ptc / (dx * dy * dz)
        gam = ti.sqrt(1. + u_ptc.norm_sqr())
        v_ptc = u_ptc * c / gam

        inv_trilerp(rho_field, pos_ptc, 0.5 * rhop)
        inv_trilerp(J_field, pos_ptc, 0.5 * rhop * v_ptc)