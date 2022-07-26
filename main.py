import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Loading environment
environment_name = 'CartPole-v0'
env = gym.make(environment_name)



env.action_space                   # 0: Left, 1: Right
env.action_space.sample()
env.observation_space              # List: Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity
env.observation_space.sample()

log_path = os.path.join('Training', 'Logs')       # Make your directories first

env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)  #MlpPolicy means using standard nueral network
model.learn(total_timesteps=20000)

PPO_Path = os.path.join('Training', 'Saved Models', 'PPP_Model_Cartpole')
model.save(PPO_Path)
del model

model = PPO.load(PPO_Path, env=env)

evaluate_policy(model, env, n_eval_episodes=10, render=True) #n_eval_episodes = 10 means we are using 10 episodes, render = true means that it will be visualized
#will return tuple, first number is average score, and second is standard deviation

episodes = 5
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()            # Shows the environment visuall
        action, _ = model.predict(obs)             #Now using model here, second value is not needed therefore it is assigned to _
        obs, reward, done, info = env.step(action)      # env.step returns a set of observations, the reward, whether or not its done, and then information(?)
        score += reward
    print('Episode: {} Score: {}'.format(episode, score))
env.close()
