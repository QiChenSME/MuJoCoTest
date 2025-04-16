import mujoco
import mujoco.viewer as viewer
import numpy as np

spec = mujoco.MjSpec.from_file(r"..\bitcraze_crazyflie_2\scene.xml")

model = spec.compile()

viewer.launch(model)