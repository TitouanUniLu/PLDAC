import gymnasium
import mon_env
import matplotlib.pyplot as plt 

images = []

env = gymnasium.make('MonCartPole-v1', render_mode='rgb_array')
env.reset()
images.append(env.render())

while True:
    action = env.action_space.sample() 
    observation, reward, done, info, _ = env.step(action)
    img = env.render()
    images.append(img)
    if done:
        break


# # fig, axes = plt.subplots(1,len(images), figsize = (15,20))
# # for i in range(len(images)):
# #     axes[i].imshow(images[i])
# #     axes[i].axis("off")
# plt.show()

plt.imshow(images[-1])