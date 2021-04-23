import time
import gym

# env = gym.make('Acrobot-v1')
# env = gym.make('FetchReach-v1')
env = gym.make('CartPole-v0')

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        time.sleep(0.1)
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

# for i in range(env.action_space.n):
#     print(env.action_space.sample())
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)
