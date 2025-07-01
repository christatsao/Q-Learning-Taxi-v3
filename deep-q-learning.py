import random
import gym
import numpy as np

env = gym.make('Taxi-v3')

alpha = 0.9 # alpha is how quickly the agent replaces old knowledge with new knowledge: 1 being most flexible
gamma = 0.95 # gamma is the discount factor that controls how much the agent values future rewards compared to immediate ones
epsilon = 1.0 # epsilon is the probability that the next action taken will be completely random
epsilon_decay = 0.9995 # we decay epsilon so eventually we mainly exploit and do not explore
min_epsilon = 0.01 # we set the minimum epsilon to ensure that we always have a small chance to explore randomly even towards the end after most exploration is done
num_episodes = 10000 # how many epochs we train for (how many times agent plays the game)
max_steps = 100 # this is per episode the max number of steps the taxi can take, even if it never reaches the goal

# 5 rows × 5 columns × 5 passenger locations × 4 destinations = 500 different states of the board
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# this function determines the agents action given a state
def choose_action(state):
  if random.uniform(0,1) < epsilon: # if a randomly generated number is less than epsilon
    return env.action_space.sample() # then take a random sample of the action space
  else:
    return np.argmax(q_table[state, :]) # otherwise take the best action according to qtable at current state out of all the actions

 # example of what q_table looks like
# State →       Action0   Action1   Action2   Action3 ...
# ---------------------------------------------------
# State 0   |    0.0       0.0       0.0       0.0
# State 1   |    0.0       0.0       0.0       0.0
# State 2   |    0.0       0.0       0.0       0.0
# ...

# step through each episode of training
for episode in range(num_episodes): 
  state = env.reset() #reset the board at the beginning of each episode

  done = False # mark not done

  # take steps until max is hit
  for step in range(max_steps): 
    action = choose_action(state) # choose an action
    next_state, reward, done, info = env.step(action) # actually take the action

    old_value = q_table[state,action]
    next_max = np.max(q_table[next_state,:]) # take the next most rewarding action

    # update the q_table: keeps part of the old value and adds part of the new value's expected reward
    q_table[state,action] = (1-alpha) * old_value + alpha * (reward + gamma*next_max )

    state = next_state
    
    if done:
      break
      
  epsilon = max(min_epsilon, epsilon * epsilon_decay)
    

env = gym.make('Taxi-v3', render_mode = 'human')

# validation phase
for episode in range(5):
  state = env.reset()
  done = False

  print('Episode ', episode)

  for step in range(max_steps):
    env.render()

    action = np.argmax(q_table[state,:])

    next_state, reward, done, info = env.step(action)
    state = next_state

    if done:
      env.render()
      print('Finished episode', episode, 'with reward', reward)
      break

env.close()