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
    <geom size=".5 .5 .01" type="plane" material="grid"/>
    <light pos="0 0 2"/>
    <camera name="closeup" pos="0 -.1 .07" xyaxes="1 0 0 0 1 2"/>
  </worldbody>
</mujoco>
"""
spec = mujoco.MjSpec.from_string(wafer_stage)

def add_stage(spec=None, len_a=.025, len_b=.1, name="stage", pos=[0, 0, 0.1]):
  if spec is None:
    spec = mujoco.MjSpec()

  # Defaults
  main = spec.default
  main.geom.type = mujoco.mjtGeom.mjGEOM_MESH

  rgba = [1, 0, 0, 1]

  # Create mesh vertices
  mesh = np.asarray([
    [-len_a/2, np.sqrt(3)*len_a/6+np.sqrt(3)*len_b/3, 0],
    [len_a/2, np.sqrt(3)*len_a/6+np.sqrt(3)*len_b/3, 0],
    [-(len_a+len_b)/2, np.sqrt(3)*len_a/6-np.sqrt(3)*len_b/6, 0],
    [(len_a+len_b)/2, np.sqrt(3)*len_a/6-np.sqrt(3)*len_b/6, 0],
    [-len_b/2, -np.sqrt(3)*len_a/3-np.sqrt(3)*len_b/6, 0],
    [len_b/2, -np.sqrt(3)*len_a/3-np.sqrt(3)*len_b/6, 0],
    [-len_a/2, np.sqrt(3)*len_a/6+np.sqrt(3)*len_b/3, 0.01],
    [len_a/2, np.sqrt(3)*len_a/6+np.sqrt(3)*len_b/3, 0.01],
    [-(len_a+len_b)/2, np.sqrt(3)*len_a/6-np.sqrt(3)*len_b/6, 0.01],
    [(len_a+len_b)/2, np.sqrt(3)*len_a/6-np.sqrt(3)*len_b/6, 0.01],
    [-len_b/2, -np.sqrt(3)*len_a/3-np.sqrt(3)*len_b/6, 0.01],
    [len_b/2, -np.sqrt(3)*len_a/3-np.sqrt(3)*len_b/6, 0.01]
  ])
  # Create Body and add mesh to the Geom of the Body
  spec.add_mesh(name=name, uservert=mesh.flatten())
  body = spec.worldbody.add_body(pos=pos, name=name, mass=1)
  body.add_geom(meshname=name, rgba=rgba)
  body.add_freejoint()

  return body


def add_rock(spec=None, scale=1.0, name="rock", pos=[0, 0, 0]):
  if spec is None:
    spec = mujoco.MjSpec()

  # Defaults
  main = spec.default
  main.mesh.scale = np.array([scale]*3 , dtype = np.float64)
  main.geom.type = mujoco.mjtGeom.mjGEOM_MESH

  # Random gray-brown color
  gray = np.array([.5, .5, .5, 1])
  light_brown = np.array([200, 150, 100, 255]) / 255.0
  mix = np.random.uniform()
  rgba = light_brown*mix + gray*(1-mix)

  # Create mesh vertices
  mesh = np.random.normal(size = (20, 3))
  mesh /= np.linalg.norm(mesh, axis=1, keepdims=True)

  # Create Body and add mesh to the Geom of the Body
  spec.add_mesh(name=name, uservert=mesh.flatten())
  body = spec.worldbody.add_body(pos=pos, name=name, mass=1)
  body.add_geom(meshname=name, rgba=rgba)
  body.add_freejoint()

  return body

rock = add_stage(spec=spec)

model = spec.compile()
print(spec.to_xml())

viewer.launch(model)

