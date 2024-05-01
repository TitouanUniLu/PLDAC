import gymnasium
import mon_env
import matplotlib.pyplot as plt 

images = []

env = gymnasium.make('MonCartPole-v1', render_mode='rgb_array')
env.reset()
images.append(env.render())
max_theta_dot,min_theta_dot=(0,0)
min_x_dot,max_x_dot=(0,0)

while True:
    action = env.action_space.sample() 
    observation, reward, done, info, _ = env.step(action)
    print(observation,reward,done,info,_)
    if observation[1]>max_x_dot:
        max_x_dot = observation[1]
    if observation[1]<min_x_dot:
        min_x_dot = observation[1]

    if observation[3]>max_theta_dot:
        max_theta_dot = observation[3]
    if observation[3]<min_theta_dot:
        min_theta_dot = observation[3]
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
print(max_theta_dot,min_theta_dot,min_x_dot,max_x_dot)