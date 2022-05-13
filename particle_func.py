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

        if ti.static(DIM == 3):
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
        else:
            E_ptc = bilerp(Eg, pos_t[0:2])
            B_ptc = bilerp(Bg, pos_t[0:2])

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
        if ti.static(DIM == 3):
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
        else:
            E_ptc = bilerp(Eg, pos_t[0:2])
            B_ptc = bilerp(Bg, pos_t[0:2])

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
