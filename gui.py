# gui.py: functions for illustrating using GGUI in Taichi
# Author: Bowen Zhao (zzzhaobowen@gmail.com)

import taichi as ti
import numpy as np

from parameters import *


def gui_init():
    gui = ti.GUI('Taichi PIC', background_color=0x000000)

    return gui


def gui_update(gui, pos_e, pos_p):
    gui.triangles(a=T_field(np.array([[0., 0., 0.], [0., 0., 0.]])),
                  b=T_field(np.array([[1., 0., 0.], [0., 1., 0.]])),
                  c=T_field(np.array([[1., 1., 0.], [1., 1., 0.]])),
                  color=0x0000ff)
    gui.triangles(a=T_field(np.array([[0., 0., 0.], [0., 0., 0.]])),
                  b=T_field(np.array([[1., 0., 0.], [0., 0., 1.]])),
                  c=T_field(np.array([[1., 0., 1.], [1., 0., 1.]])),
                  color=0xff0000)
    gui.triangles(a=T_field(np.array([[0., 0., 0.], [0., 0., 0.]])),
                  b=T_field(np.array([[0., 1., 0.], [0., 0., 1.]])),
                  c=T_field(np.array([[0., 1., 1.], [0., 1., 1.]])),
                  color=0x00ff00)

    gui.circles(T_field(pos_e.to_numpy() / (xmax - xmin)), radius=2., color=0x000000)
    gui.circles(T_field(pos_p.to_numpy() / (xmax - xmin)), radius=3., color=0xffffff)

    gui.lines(begin=T_field(np.array([[1., 1., 0.], [1., 0., 1.], [0., 1., 1.]])),
              end=T_field(np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])),
              radius=2, color=0x068587)

    gui.show()


def T_field(a):
    phi, theta = np.radians(-28), np.radians(-32)

    a = (a - 0.5) * 0.5
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    c, s = np.cos(phi), np.sin(phi)
    C, S = np.cos(theta), np.sin(theta)
    x, z = x * c + z * s, z * c - x * s
    u, v = x, y * C + z * S
    return np.array([u, v]).swapaxes(0, 1) + 0.5


@ti.kernel
def set_vertics_indices(vertices: ti.template(),
                        indices_xy: ti.template(),
                        indices_xz: ti.template(),
                        indices_yz: ti.template(),
                        normals_xy: ti.template(),
                        normals_xz: ti.template(),
                        normals_yz: ti.template()
                        ):
    vertices[0] = [0., 0., 0.]

    vertices[1] = [1., 0., 0.]
    vertices[2] = [0., 1., 0.]
    vertices[3] = [0., 0., 1.]

    vertices[4] = [1., 1., 0.]
    vertices[5] = [1., 0., 1.]
    vertices[6] = [0., 1., 1.]

    # 0: x-y plane; 1: x-z plane; 2: y-z plane
    indices_xy[0] = 0
    indices_xy[1] = 1
    indices_xy[2] = 4
    indices_xy[3] = 0
    indices_xy[4] = 2
    indices_xy[5] = 4

    indices_xz[0] = 0
    indices_xz[1] = 1
    indices_xz[2] = 5
    indices_xz[3] = 0
    indices_xz[4] = 3
    indices_xz[5] = 5

    indices_yz[0] = 0
    indices_yz[1] = 2
    indices_yz[2] = 6
    indices_yz[3] = 0
    indices_yz[4] = 3
    indices_yz[5] = 6

    for idx in ti.static(range(6)):
        normals_xy[idx] = [0., 0., 1.]
        normals_xz[idx] = [0., 1., 0.]
        normals_yz[idx] = [1., 0., 0.]


def ggui_init():
    """
    Initialize GGUI in Taichi
    :return:
    """
    
    window = ti.ui.Window("Taichi PIC Particles", res=(1024, 1024), vsync=True)

    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.make_camera()
    camera.position(3, 3, 3)
    camera.lookat(0.5, 0.5, 0.5)
    camera.fov(45)

    return window, canvas, scene, camera


def ggui_update(window, canvas, scene, camera,
                show_field_e, show_field_p, colors_e, colors_p,
                vertices, indices_xy, indices_xz, indices_yz,
                normals_xy, normals_xz, normals_yz):
    """
    Updates the fields used in illustration
    """
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0.1, 0.1, 0.1))
    particles_radius_e = 0.03
    particles_radius_p = 0.07

    # Here 4 lines below are too slow, because it would fetch data from gpu to cpu,
    # and then send data from cpu to gpu. And the gc of  field in python scope is 
    # also a problem that lead your project run slowly.
    
    # show_field_e = ti.Vector.field(3, float, shape=n_ptc)
    # show_field_p = ti.Vector.field(3, float, shape=n_ptc)
    # show_field_e.from_numpy(pos_e.to_numpy() / (xmax - xmin))
    # show_field_p.from_numpy(pos_p.to_numpy() / (xmax - xmin))
    scene.particles(show_field_e, radius=particles_radius_e)
    scene.particles(show_field_p, radius=particles_radius_p)

    scene.mesh(vertices,
               indices=indices_xy,
               normals=normals_xy,
               color=(1, 0., 0.),
               two_sided=True)

    scene.mesh(vertices,
               indices=indices_xz,
               normals=normals_xz,
               color=(0., 1, 0.),
               two_sided=True)

    scene.mesh(vertices,
               indices=indices_yz,
               normals=normals_yz,
               color=(0., 0., 1),
               two_sided=True)

    scene.point_light(pos=(2, 2, 2), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)

    window.show()
