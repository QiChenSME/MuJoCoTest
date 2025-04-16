from MuJoCoEnv import *


if __name__ == "__main__":
    env = MoJoCoEnv(model_path="wafer_stage.xml")
    env.reset()

    while env.viewer.is_running():
        print("reward:", env.step(env.action_space.sample())[1])
        print(env.state)

