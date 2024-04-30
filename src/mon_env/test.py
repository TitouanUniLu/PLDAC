import gym
import mon_env

env = gym.make('MonCartPole-v1',render_mode="rbg_array")
print("Environnement chargé avec succès : ", env)
img = env.render()