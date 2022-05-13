import taichi as ti

from parameters import *


def gui_init():
    window = ti.ui.Window("Taichi PIC Particles", res=(512, 512), vsync=True)

    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.make_camera()
    camera.position(1.5, 1.5, 2)
    camera.lookat(0.5, 0.5, 0.5)
    camera.fov(55)

    return window, canvas, scene, camera


def gui_update(window, canvas, scene, camera,
               pos_e, pos_p, colors_e, colors_p):
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))
    particles_radius_e = 0.05
    particles_radius_p = 0.1

    # Electrons
    show_field_e = ti.Vector.field(2, float, shape=n_ptc)
    show_field_p = ti.Vector.field(2, float, shape=n_ptc)
    show_field_e.from_numpy(pos_e.to_numpy()[:, 0:2] / (xmax - xmin))
    show_field_p.from_numpy(pos_p.to_numpy()[:, 0:2] / (xmax - xmin))
    scene.particles(show_field_e, per_vertex_color=colors_e, radius=particles_radius_e)
    scene.particles(show_field_p, per_vertex_color=colors_p, radius=particles_radius_p)

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)

    window.show()