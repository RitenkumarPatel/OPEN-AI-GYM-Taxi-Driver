import gym
import numpy as np
import random

#create taxi environment
env = gym.make('Taxi-v3')

#create a new instance of taxi, and get the initial state
state = env.reset()

state_size = env.observation_space.n #total number of states (S)
action_size = env.action_space.n    	#total number of actions (A)
# initialize a qtalbe iwth 0's for all Q values
qtable = np.zeros((state_size, action_size))

#hyperparameters to tune
learning_rate = 0.9
discount_rate = 0.8
epsilon = 1.0   	  # probability that our agent will explore
decay_rate = 0.01     # of epsilon

# training variables
num_episodes = 1000
max_steps = 99 # per episode

for episode in range(num_episodes):

    # reset env
    state = env.reset()
    done = False

    for s in range(max_steps):

        # exploration-exploitation tradeoff
        if random.uniform(0,1) < epsilon:
            # explore
            action = env.action_space.sample()
        else:
            # exploit
            action = np.argmax(qtable[state,:])

        # take action and observe reward
        new_state, reward, done, info = env.step(action)

        # Qlearning algorithm: Q(s,a) := Q(s,a) + learning_rate * (reward + discount_rate * max Q(s',a') - Q(s,a))
        qtable[state,action] = qtable[state, action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:]) - qtable[state,action])
        # ^^^ The above line adds values to the qtable based on possible actions

        # update to our new state
        state = new_state

        # if done, finish episode
        if done == True:
            break

        #decrease epsilon
        epsilon = np.exp(-decay_rate*episode)
print(f"Training completed over {num_episodes} episodes")
input("Press Enter to watch trained agent...")
print("SLJS:DLFJD")
#watch trained agent
state = env.reset()
done = False
rewards = 0
for s in range(max_steps):
    print(f"TRAINED AGENT")
    print("Step {}".format(s+1))

    action = np.argmax(qtable[state,:])
    new_state, reward, done, info = env.step(action)
    rewards += reward
    env.render()
    print(f"score: {rewards}")
    state = new_state

    if done == True:
        break

    env.close()
