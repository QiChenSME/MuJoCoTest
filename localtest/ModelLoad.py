import time
import numpy as np

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path(r"wafer_stage.xml")
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
    p_flag = True
    CTRL_VALUE = 1

    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(m, d)

        # System input
        if time.time() - start > 9:
            d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "v_1")] = 0
            d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "v_2")] = 0
            d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "v_3")] = CTRL_VALUE
        elif time.time() - start > 6:
            d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "v_1")] = 0
            d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "v_2")] = CTRL_VALUE
            d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "v_3")] = 0
        elif time.time() - start > 3:
            d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "v_1")] = CTRL_VALUE
            d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "v_2")] = 0
            d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "v_3")] = 0
        d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "center")] = 1

        # Print sensor data for every 2 second.
        if int(d.time % 2):
            if p_flag:
                print(f"{"\033[32m"}[{time.strftime("%H:%M:%S", time.localtime())}] returned sensor data{"\033[0m"}")
                print("sensor_z1:", f"{d.sensordata[0]:.6f}")
                print("sensor_z2:", f"{d.sensordata[1]:.6f}")
                print("sensor_z3:", f"{d.sensordata[2]:.6f}")
                print("sensor_x1:", f"{d.sensordata[3]:.6f}")
                print("sensor_x2:", f"{d.sensordata[4]:.6f}")
                print("sensor_y: ", f"{d.sensordata[5]:.6f}")
                print("gyro:     ", f"{d.sensordata[6]:.6f}", f"{d.sensordata[7]:.6f}", f"{d.sensordata[8]:.6f}")
                print("acc:      ", f"{d.sensordata[9]:.6f}", f"{d.sensordata[10]:.6f}", f"{d.sensordata[11]:.6f}")
                print("nv:", m.nv)
                print("nu:", m.nu)
                print("qpos:", d.qpos)
                p_flag = False
        else:
            p_flag = True

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
          time.sleep(time_until_next_step)

