from typing import Optional
import time
import numpy as np
import gymnasium as gym
import casadi as cs

import mujoco
import mujoco.viewer

DEFAULT_MODEL = """
<mujoco model="tippe top">
  <option integrator="RK4"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
  </asset>

  <worldbody>
    <geom size=".2 .2 .01" type="plane" material="grid"/>
    <light pos="0 0 .6"/>
    <camera name="closeup" pos="0 -.1 .07" xyaxes="1 0 0 0 1 2"/>
    <body name="top" pos="0 0 .02">
      <freejoint/>
      <geom name="ball" type="sphere" size=".02" />
      <geom name="stem" type="cylinder" pos="0 0 .02" size="0.004 .008"/>
      <geom name="ballast" type="box" size=".023 .023 0.005"  pos="0 0 -.015"
       contype="0" conaffinity="0" group="3"/>
    </body>
  </worldbody>

  <keyframe>
    <key name="spinning" qpos="0 0 0.02 1 0 0 0" qvel="0 0 0 0 1 200" />
  </keyframe>
</mujoco>
"""


class MoJoCoEnv(gym.Env):
    def __init__(
            self,
            model_path=None,
            model_string=None,
            model_object=None,
            para_dict=None,
            action_ub=None,
            action_lb=None,
            observation_ub=None,
            observation_lb=None,
            w=None
    ):
        self.PARAMETERS = para_dict
        self.XML_PATH = None
        if model_path is not None:
            self.XML_PATH = model_path
            self.m = mujoco.MjModel.from_xml_path(self.XML_PATH)
            self.d = mujoco.MjData(self.m)
            self.model_source = "local file"
        elif model_string is not None:
            self.m = mujoco.MjModel.from_xml_string(model_string)
            self.d = mujoco.MjData(self.m)
            self.model_source = "xml string"
        elif model_object is not None:
            self.m = model_object
            self.d = mujoco.MjData(self.m)
            self.model_source = "internal object"
        else:
            self.m = mujoco.MjModel.from_xml_string(DEFAULT_MODEL)
            self.d = mujoco.MjData(self.m)
            self.model_source = "default model"

        if action_ub is not None and action_lb is not None:
            self.action_lb = action_lb
            self.action_ub = action_ub
        else:
            self.action_lb = -np.ones(self.m.nu)
            self.action_ub = np.ones(self.m.nu)
        self.action_space = gym.spaces.Box(self.action_lb, self.action_ub)

        if observation_ub is not None and observation_lb is not None:
            self.observation_lb = observation_lb
            self.observation_ub = observation_ub
        else:
            self.observation_lb = -np.ones(self.m.nv)*np.inf
            self.observation_ub = np.ones(self.m.nv)*np.inf
        self.observation_space = gym.spaces.Box(self.observation_lb, self.observation_ub)

        if w is not None:
            self.w = w
        else:
            self.w = np.ones(self.m.nv).reshape(self.m.nv, 1)

        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)

        self.state: np.ndarray | None = None
        self.step_start = None
        self.isopen = True


    def step(self, action:cs.DM, ref=None):
        assert self.state is not None, "Call reset before using step method."

        if ref is not None:
            ref = ref.flatten()
        else:
            ref = np.zeros(self.m.nv)

        time_until_next_step = self.m.opt.timestep - (time.time() - self.step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        self.step_start = time.time()
        if type(action) == cs.DM:
            action=action.full().flatten()
        else:
            action = action.flatten()
        self.d.ctrl = action

        mujoco.mj_step(self.m, self.d)

        self.state = self.d.sensordata[:self.m.nv].copy().reshape(self.m.nv,1)

        lb, ub = self.observation_lb.reshape(self.m.nv, 1), self.observation_ub.reshape(self.m.nv, 1)

        print(np.sum((self.state.flatten() - ref.flatten()) ** 2))
        print(0.1 * sum(action ** 2))
        print(self.w.T @ np.maximum(0, lb - self.state))
        print(self.w.T @ np.maximum(0, self.state - ub))
        reward = float(
            0.5
            * (
                    np.sum((self.state.flatten() - ref.flatten()) ** 2)
                    + 0.1 * sum(action ** 2)
                    + self.w.T @ np.maximum(0, lb - self.state)
                    + self.w.T @ np.maximum(0, self.state - ub)
            )
        )

        self.viewer.sync()

        return np.array(self.state, dtype=np.float32), reward, False, False, {}


    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.m, self.d)
        self.step_start = 0
        self.state = self.d.sensordata[:self.m.nv].copy()

        return np.array(self.state, dtype=np.float32), {}


    def close(self):
        if self.viewer.is_running():
            self.viewer.close()
            self.isopen = False

