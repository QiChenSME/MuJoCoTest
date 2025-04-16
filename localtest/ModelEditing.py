import mujoco
import mujoco.viewer as viewer
import numpy as np

wafer_stage = """
<mujoco model="wafer stage">
  <option integrator="RK4"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
  </asset>

  <worldbody>
    <geom size="1 1 .1" type="plane" material="grid"/>
    <light pos="0 0 2"/>
    <camera name="closeup" pos="0 -.1 .07" xyaxes="1 0 0 0 1 2"/>
  </worldbody>

  <actuator>
    <motor forcerange="0 1" gear="0 0 1 0 0 0" site="act_o" name="v_1"/>
    <motor forcerange="0 1" gear="0 0 1 0 0 0" site="act_p" name="v_2"/>
    <motor forcerange="0 1" gear="0 0 1 0 0 0" site="act_q" name="v_3"/>
    <motor forcerange="-1 1" gear="1 0 0 0 0 0" site="act_o" name="h_1"/>
    <motor forcerange="-1 1" gear="-0.5 -0.866 0 0 0 0" site="act_p" name="h_2"/>
    <motor forcerange="-1 1" gear="-0.5 0.866 0 0 0 0" site="act_q" name="h_3"/>
    <motor forcerange="0 1" gear="0 0 68.6 0 0 0" site="center" name="total"/>
  </actuator>

  <sensor>
    <rangefinder name="sensor_z1" site="sensor_z1"/>
    <rangefinder name="sensor_z2" site="sensor_z2"/>
    <rangefinder name="sensor_z3" site="sensor_z3"/>
    <rangefinder name="sensor_x1" site="sensor_x1"/>
    <rangefinder name="sensor_x2" site="sensor_x2"/>
    <rangefinder name="sensor_y" site="sensor_y"/>
    <gyro name="body_gyro" site="center"/>
    <accelerometer name="body_linacc" site="center"/>
  </sensor>
</mujoco>
"""
spec = mujoco.MjSpec.from_string(wafer_stage)

def add_stage(spec=None, len_a=.075, len_b=.3, height = 0.02, name="stage", pos=None, mass=7, edge=None):
      if spec is None:
        spec = mujoco.MjSpec()
      if pos is None:
        pos = [0, 0, 0.3]
      if edge is None:
        edge = np.sqrt(3)*len_a/4

      a = len_a
      b = len_b
      h = height
      sr2 = np.sqrt(2)
      sr3 = np.sqrt(3)
      m = np.clip(edge, 0, sr3/3 * a + sr3/6 * b)

      # Defaults
      main = spec.default
      main.geom.type = mujoco.mjtGeom.mjGEOM_MESH

      rgba = [1, 1, 1, 0.5]

      # Create mesh vertices
      mesh = np.asarray([
            [-a/2,      sr3/6 * a + sr3/3 * b, 0],
            [a/2,       sr3/6 * a + sr3/3 * b, 0],
            [-(a+b)/2,  sr3/6 * a - sr3/6 * b, 0],
            [(a+b)/2,   sr3/6 * a - sr3/6 * b, 0],
            [-b/2,     -sr3/3 * a - sr3/6 * b, 0],
            [b/2,      -sr3/3 * a - sr3/6 * b, 0],
            [-a/2,      sr3/6 * a + sr3/3 * b, h],
            [a/2,       sr3/6 * a + sr3/3 * b, h],
            [-(a+b)/2,  sr3/6 * a - sr3/6 * b, h],
            [(a+b)/2,   sr3/6 * a - sr3/6 * b, h],
            [-b/2,     -sr3/3 * a - sr3/6 * b, h],
            [b/2,      -sr3/3 * a - sr3/6 * b, h]
      ])
      # Create Body and add mesh to the Geom of the Body
      spec.add_mesh(name=name, uservert=mesh.flatten())
      body = spec.worldbody.add_body(pos=pos, name=name, mass=1)
      body.add_geom(meshname=name, rgba=rgba, mass=mass)
      body.add_freejoint()

      # Create mirror Geoms
      # body.add_geom(name="mirror_o",
      #               type=mujoco.mjtGeom.mjGEOM_BOX,
      #               rgba=[0, 1, 1, 0.3],
      #               mass=0,
      #               size=[b * 0.4, h * 0.25, h],
      #               pos=[0, -sr3 / 3 * a - sr3 / 6 * b, h * 0.5],
      #               # quat=[sr3/2, 0, 0, 0.5]
      #               )
      # body.add_geom(name="mirror_p",
      #               type=mujoco.mjtGeom.mjGEOM_BOX,
      #               rgba=[0, 1, 1, 0.3],
      #               mass=0,
      #               size=[b * 0.4, h * 0.25, h],
      #               pos=[-a/2 - b/4, sr3/6 * a + sr3/12 * b, h * 0.5],
      #               quat=[sr3/2, 0, 0, 0.5]
      #               )
      # body.add_geom(name="mirror_q",
      #               type=mujoco.mjtGeom.mjGEOM_BOX,
      #               rgba=[0, 1, 1, 0.3],
      #               mass=0,
      #               size=[b * 0.4, h * 0.25, h],
      #               pos=[a/2 + b/4, sr3/6 * a + sr3/12 * b, h * 0.5],
      #               quat=[sr3/2, 0, 0, -0.5]
      #               )
      body.add_geom(name="mirror_body",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    rgba=[0, 0.6, 1, 0.3],
                    mass=0,
                    size=[b * 0.3, b * 0.3, h*1.25],
                    pos=[0, 0, -h*1.25],
                    )
      # Create restriction Geoms
      body.add_geom(name="act_restrict_z1",
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    rgba=[1, 1, 0, 1],
                    mass=0,
                    size=[0.001, 0, 0],
                    pos=[0, 0, -0.25],
                    friction=[0,0,0],
                    )
      body.add_geom(name="act_restrict_z2",
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    rgba=[1, 1, 0, 1],
                    mass=0,
                    size=[0.001, 0, 0],
                    pos=[0, 0, 0.25],
                    friction=[0,0,0],
                    )
      body.add_geom(name="act_restrict_y1",
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    rgba=[1, 1, 0, 1],
                    mass=0,
                    size=[0.001, 0, 0],
                    pos=[0, 1.5, 0],
                    friction=[0,0,0],
                    )
      body.add_geom(name="act_restrict_y2",
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    rgba=[1, 1, 0, 1],
                    mass=0,
                    size=[0.001, 0, 0],
                    pos=[0, -1.5, 0],
                    friction=[0,0,0],
                    )

      # Create sites
      body.add_site(name="act_o",
                    pos=[0,
                         -sr3/3 * a - sr3/6 * b + m,
                         0],
                    rgba = [1, 0, 0, 1])
      body.add_site(name="act_p",
                    pos=[-a/2 - b/4 + sr3/2 * m,
                         sr3/6 * a + sr3/12 * b - 1/2 * m,
                         0],
                    rgba = [0, 1, 0, 1])
      body.add_site(name="act_q",
                    pos=[a/2 + b/4 - sr3/2 * m,
                         sr3/6 * a + sr3/12 * b - 1/2 * m,
                         0],
                    rgba = [0, 0, 1, 1])
      body.add_site(name="center",
                    pos=[0, 0, 0],
                    rgba = [0, 0, 0, 1])

      return body


def add_res(spec=None,
            name="restriction",
            pos=None,
            visible=False,
            up_ball_pos=0.25,
            down_ball_pos=-0.25,
            theta_ball_pos=1.5,
            thickness=0.005,
            bnd_x=0.03,
            bnd_y=0.03,
            bnd_z=0.005,
            res_ball=0.001,
            theta=5):
    if spec is None:
        spec = mujoco.MjSpec()
    if pos is None:
        pos = [0, 0, 0.3]

    sr2 = np.sqrt(2)
    sr3 = np.sqrt(3)

    # Defaults
    main = spec.default
    main.geom.type = mujoco.mjtGeom.mjGEOM_MESH

    bnd_t=theta_ball_pos * np.tan(theta/180*np.pi) * np.cos(theta/180*np.pi)**2 + bnd_y
    pos_t=theta_ball_pos - theta_ball_pos * np.tan(theta/180*np.pi) * np.cos(theta/180*np.pi) * np.sin(theta/180*np.pi)

    if visible:
        opacity = 0.1
    else:
        opacity = 0
    # Create Body and add mesh to the Geom of the Body
    body = spec.worldbody.add_body(pos=pos, name=name, mass=1)

    # Create restriction Geoms
    body.add_geom(name="restriction_x1_max",
                  type=mujoco.mjtGeom.mjGEOM_BOX,
                  rgba=[1, 0, 0, opacity],
                  mass=1,
                  size=[bnd_z+res_ball+thickness*2, bnd_y+res_ball+thickness*2, thickness],
                  pos=[(bnd_x+thickness+res_ball), 0, down_ball_pos],
                  quat=[sr2/2, 0, sr2/2, 0],
                  friction=[0,0,0],
                  )
    body.add_geom(name="restriction_x1_min",
                  type=mujoco.mjtGeom.mjGEOM_BOX,
                  rgba=[0, 1, 1, opacity],
                  mass=1,
                  size=[bnd_z+res_ball+thickness*2, bnd_y+res_ball+thickness*2, thickness],
                  pos=[-(bnd_x+thickness+res_ball), 0, down_ball_pos],
                  quat=[sr2/2, 0, sr2/2, 0],
                  friction=[0,0,0],
                  )
    body.add_geom(name="restriction_y1_max",
                  type=mujoco.mjtGeom.mjGEOM_BOX,
                  rgba=[0, 1, 0, opacity],
                  mass=1,
                  size=[bnd_x+res_ball+thickness*2, bnd_z+res_ball+thickness*2, thickness],
                  pos=[0, (bnd_y+thickness+res_ball), down_ball_pos],
                  quat=[sr2/2, sr2/2, 0, 0],
                  friction=[0,0,0],
                  )
    body.add_geom(name="restriction_y1_min",
                  type=mujoco.mjtGeom.mjGEOM_BOX,
                  rgba=[1, 0, 1, opacity],
                  mass=1,
                  size=[bnd_x+res_ball+thickness*2, bnd_z+res_ball+thickness*2, thickness],
                  pos=[0, -(bnd_y+thickness+res_ball), down_ball_pos],
                  quat=[sr2/2, sr2/2, 0, 0],
                  friction=[0,0,0],
                  )
    body.add_geom(name="restriction_z1_max",
                  type=mujoco.mjtGeom.mjGEOM_BOX,
                  rgba=[0, 0, 1, opacity],
                  mass=1,
                  size=[bnd_x+res_ball+thickness*2, bnd_y+res_ball+thickness*2, thickness],
                  pos=[0, 0, down_ball_pos + (bnd_z+thickness+res_ball)],
                  friction=[0,0,0],
                  )
    body.add_geom(name="restriction_z1_min",
                  type=mujoco.mjtGeom.mjGEOM_BOX,
                  rgba=[1, 1, 0, opacity],
                  mass=1,
                  size=[bnd_x+res_ball+thickness*2, bnd_y+res_ball+thickness*2, thickness],
                  pos=[0, 0, down_ball_pos - (bnd_z+thickness+res_ball)],
                  friction=[0,0,0],
                  )
    body.add_geom(name="restriction_x2_max",
                  type=mujoco.mjtGeom.mjGEOM_BOX,
                  rgba=[1, 0, 0, opacity],
                  mass=1,
                  size=[bnd_z+res_ball+thickness*2, bnd_y+res_ball+thickness*2, thickness],
                  pos=[(bnd_x+thickness+res_ball), 0, up_ball_pos],
                  quat=[sr2/2, 0, sr2/2, 0],
                  friction=[0,0,0],
                  )
    body.add_geom(name="restriction_x2_min",
                  type=mujoco.mjtGeom.mjGEOM_BOX,
                  rgba=[0, 1, 1, opacity],
                  mass=1,
                  size=[bnd_z+res_ball+thickness*2, bnd_y+res_ball+thickness*2, thickness],
                  pos=[-(bnd_x+thickness+res_ball), 0, up_ball_pos],
                  quat=[sr2/2, 0, sr2/2, 0],
                  friction=[0,0,0],
                  )
    body.add_geom(name="restriction_y2_max",
                  type=mujoco.mjtGeom.mjGEOM_BOX,
                  rgba=[0, 1, 0, opacity],
                  mass=1,
                  size=[bnd_x+res_ball+thickness*2, bnd_z+res_ball+thickness*2, thickness],
                  pos=[0, (bnd_y+thickness+res_ball), up_ball_pos],
                  quat=[sr2/2, sr2/2, 0, 0],
                  friction=[0,0,0],
                  )
    body.add_geom(name="restriction_y2_min",
                  type=mujoco.mjtGeom.mjGEOM_BOX,
                  rgba=[1, 0, 1, opacity],
                  mass=1,
                  size=[bnd_x+res_ball+thickness*2, bnd_z+res_ball+thickness*2, thickness],
                  pos=[0, -(bnd_y+thickness+res_ball), up_ball_pos],
                  quat=[sr2/2, sr2/2, 0, 0],
                  friction=[0,0,0],
                  )
    body.add_geom(name="restriction_z2_max",
                  type=mujoco.mjtGeom.mjGEOM_BOX,
                  rgba=[0, 0, 1, opacity],
                  mass=1,
                  size=[bnd_x+res_ball+thickness*2, bnd_y+res_ball+thickness*2, thickness],
                  pos=[0, 0, up_ball_pos + (bnd_z+thickness+res_ball)],
                  friction=[0,0,0],
                  )
    body.add_geom(name="restriction_z2_min",
                  type=mujoco.mjtGeom.mjGEOM_BOX,
                  rgba=[1, 1, 0, opacity],
                  mass=1,
                  size=[bnd_x+res_ball+thickness*2, bnd_y+res_ball+thickness*2, thickness],
                  pos=[0, 0, up_ball_pos - (bnd_z+thickness+res_ball)],
                  friction=[0,0,0],
                  )
    body.add_geom(name="restriction_x3_max",
                  type=mujoco.mjtGeom.mjGEOM_BOX,
                  rgba=[1, 0, 0, opacity],
                  mass=1,
                  size=[15*(bnd_z+res_ball+thickness*2), 5*(bnd_y+res_ball+thickness*2), thickness*10],
                  pos=[(bnd_t+res_ball), pos_t-2.5*(bnd_y+res_ball+thickness*2), 0],
                  quat=[sr2/2, 0, sr2/2, 0],
                  friction=[0,0,0],
                  )
    body.add_geom(name="restriction_x3_min",
                  type=mujoco.mjtGeom.mjGEOM_BOX,
                  rgba=[0, 1, 1, opacity],
                  mass=1,
                  size=[15*(bnd_z+res_ball+thickness*2), 5*(bnd_y+res_ball+thickness*2), thickness*10],
                  pos=[-(bnd_t+res_ball), pos_t-2.5*(bnd_y+res_ball+thickness*2), 0],
                  quat=[sr2/2, 0, sr2/2, 0],
                  friction=[0,0,0],
                  )
    body.add_geom(name="restriction_x4_max",
                  type=mujoco.mjtGeom.mjGEOM_BOX,
                  rgba=[1, 0, 0, opacity],
                  mass=1,
                  size=[15*(bnd_z+res_ball+thickness*2), 5*(bnd_y+res_ball+thickness*2), thickness*10],
                  pos=[(bnd_t+res_ball), -(pos_t-2.5*(bnd_y+res_ball+thickness*2)), 0],
                  quat=[sr2/2, 0, sr2/2, 0],
                  friction=[0,0,0],
                  )
    body.add_geom(name="restriction_x4_min",
                  type=mujoco.mjtGeom.mjGEOM_BOX,
                  rgba=[0, 1, 1, opacity],
                  mass=1,
                  size=[15*(bnd_z+res_ball+thickness*2), 5*(bnd_y+res_ball+thickness*2), thickness*10],
                  pos=[-(bnd_t+res_ball), -(pos_t-2.5*(bnd_y+res_ball+thickness*2)), 0],
                  quat=[sr2/2, 0, sr2/2, 0],
                  friction=[0,0,0],
                  )

    return body

def add_sensor(spec=None,
            name="sensor",
            pos=None,
            visible=False,):
    if spec is None:
        spec = mujoco.MjSpec()
    if pos is None:
        pos = [0, 0, 0.3]

    sr2 = np.sqrt(2)
    sr3 = np.sqrt(3)

    main = spec.default
    main.geom.type = mujoco.mjtGeom.mjGEOM_MESH

    if visible:
        opacity = 1
    else:
        opacity = 0
    # Create Body and add mesh to the Geom of the Body
    body = spec.worldbody.add_body(pos=pos, name=name, mass=1)

    body.add_site(name="sensor_z1",
                  type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  pos=[0.02,
                       0.02,
                       -0.1],
                  rgba=[1, 0, 0, opacity])
    body.add_site(name="sensor_z2",
                  type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  pos=[0.02,
                       -0.02,
                       -0.1],
                  rgba=[0, 1, 0, opacity])
    body.add_site(name="sensor_z3",
                  type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  pos=[-0.02,
                       0.02,
                       -0.1],
                  rgba=[0, 0, 1, opacity])
    body.add_site(name="sensor_x1",
                  type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  pos=[-0.15,
                       -0.02,
                       -0.025],
                  rgba=[1, 0, 0, opacity],
                  quat=[sr2/2, 0, sr2/2, 0],)
    body.add_site(name="sensor_x2",
                  type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  pos=[-0.15,
                       0.02,
                       -0.025],
                  rgba=[0, 1, 0, opacity],
                  quat=[sr2/2, 0, sr2/2, 0],)
    body.add_site(name="sensor_y",
                  type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  pos=[0,
                       -0.15,
                       -0.025],
                  rgba=[0, 0, 1, opacity],
                  quat=[sr2/2, -sr2/2, 0, 0],)

    return body


stage = add_stage(spec=spec)
base = add_res(spec=spec)
sensor = add_sensor(spec=spec, visible=True)

model = spec.compile()
print(spec.to_xml())
spec.to_file(r"wafer_stage.xml")

viewer.launch(model)

