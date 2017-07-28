import gym
import scipy.misc

env = gym.make('ReacherPixel-v1')
for i_episode in range(20):
    observation = env.reset()
    import ipdb; ipdb.set_trace()
    for t in range(100):
        env.render()
        scipy.misc.imsave('obs.jpg', observation)
        print(observation)
        import ipdb; ipdb.set_trace()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
