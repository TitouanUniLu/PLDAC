from gym.envs.registration import register
from mon_env.envs.cartpole import MonCartPoleEnv

register(
    id='MonCartPole-v1',
    entry_point='mon_env.envs.cartpole:MonCartPoleEnv',
)