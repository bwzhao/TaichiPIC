# initialize.py: different ways of initialization
# Author: Bowen Zhao (zzzhaobowen@gmail.com)


import taichi as ti

from parameters import *
from constants import *


@ti.kernel
def initialize_random(pos_e: ti.template(),
                      pos_p: ti.template(),
                      u_e: ti.template(),
                      u_p: ti.template(),
                      wght_e: ti.template(),
                      wght_p: ti.template(),
                      B_yee: ti.template(),
                      E_yee: ti.template()):
    """
    Initialize the particles and fields
    """
    # Particle spatial dim
    for idx_ptc in range(n_ptc):
        if ti.static(DIM == 3):
            pos_ptc = ti.Vector([xmin + ti.random() * (xmax - xmin),
                                 ymin + ti.random() * (ymax - ymin),
                                 zmin + ti.random() * (zmax - zmin)])
            u_ptc = ti.Vector([ti.random(), ti.random(), ti.random()])
            wght_ptc = n0 * (xmax - xmin) * (ymax - ymin) * (zmax - zmin) / n_ptc

            pos_e[idx_ptc] = pos_ptc
            pos_p[idx_ptc] = pos_ptc

            u_e[idx_ptc] = u_ptc
            u_p[idx_ptc] = -u_ptc

            wght_e[idx_ptc] = wght_ptc
            wght_p[idx_ptc] = wght_ptc
        else:
            pos_ptc = ti.Vector([xmin + ti.random() * (xmax - xmin) * 0.5,
                       ymin + ti.random() * (ymax - ymin) * 0.5,
                       0.])
            u_ptc = [0.9, 0., 0.]
            wght_ptc = n0 * (xmax - xmin) * (ymax - ymin) / n_ptc

            pos_e[idx_ptc] = pos_ptc
            pos_p[idx_ptc] = pos_ptc

            u_e[idx_ptc] = u_ptc
            u_p[idx_ptc] = u_ptc

            wght_e[idx_ptc] = wght_ptc
            wght_p[idx_ptc] = wght_ptc

    # Fields:
    for Idx in ti.grouped(B_yee):
        B_yee[Idx] = [0., 0., 0.]
        E_yee[Idx] = [0., 0., 0.]


# @ti.kernel
# def initialize_maxwellian(pos_e: ti.template(),
#                       pos_p: ti.template(),
#                       u_e: ti.template(),
#                       u_p: ti.template(),
#                       wght_e: ti.template(),
#                       wght_p: ti.template(),
#                       B_yee: ti.template(),
#                       E_yee: ti.template(),
#                       u_d: ti.template(),
#                           theta: float):
#     """
#     Initialize the particles and fields
#     """
#     cmin = -1.0
#     cmax = 1.0
#     pmin = 0.
#     pmax = 2.0 * pi
#     ND = 800
#     EPS = 1e-6
#
#     uMode = ti.sqrt(2.) * theta * ti.sqrt(1. + ti.sqrt(1. + 1. / theta ** 2))
#     gMode = ti.sqrt(uMode ** 2 + 1.)
#
#     umint = uMode * ti.sqrt(EPS)
#     gmint = ti.sqrt(umint ** 2 + 1.)
#     umint = umint * ti.exp(0.5 * (gmint - gMode) / theta)
#     gmint = ti.sqrt(umint ** 2 + 1.)
#
#     gmaxt = gMode + theta * ti.log(1. / EPS)
#     umaxt = ti.sqrt((gmaxt - 1.) * (gmaxt + 1.))
#     gmaxt = gMode + theta * ti.log((umaxt / uMode) ** 2 / EPS)
#     umaxt = ti.sqrt((gmaxt - 1.) * (gmaxt + 1.))
#
#     for idx in range(ND):
#         u_d[idx] = 1e1**((idx - 1)*1. / ((ND - 1)*1.) * (ti.log10(umaxt) - ti.log10(umint)) + ti.log10(umint))