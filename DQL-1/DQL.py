# Import Libraries;
import numpy as np
import tqdm as tq     # Visulalize Loops.
import gymnasium as gym  # Env for Reinforcement Learning (updated from gym).
import random
import time as tm     # Time module for sleep.
from collections import deque   # Data Structure for Memory.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# Deep Q-Learning Agent Class;
class DQLAgent:
    def __init__(self,env):
        
        # Environment;
        self.env = env
        self.state_size = int(env.observation_space.shape[0])
        self.action_size = int(env.action_space.n) # Action space is in environment.
        
        # Hyperparameters;
        self.gamma = 0.95               # Discount Factor.
        self.epsilon = 1.0              # Exploration Rate.
        self.epsilon_decay = 0.995      # Decay Rate for Exploration.
        self.epsilon_min = 0.01         # Minimum Exploration Rate.
        self.learning_rate = 0.001      # Learning Rate for NN.
        self.batch_size = 64            # Batch Size for Training.
        self.memory = deque(maxlen=2000)  # Memory for Experience Replay.
        self.model = self.build_model()   # Build the Deep Q-Network.
    
    def build_model(self): # Artificial Neural Network for Deep Q-Learning.
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self,state):
        if random.uniform(0,1) < self.epsilon:
            return self.env.action_space.sample()  # Explore: Random Action.

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # Exploit: Best Action from Q-Table.

    def replay(self,):
        # If memory has not enough samples, return.
        if len(self.memory) < self.batch_size:
            return

        mini_batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in mini_batch:
            target=reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state,verbose=0)[0])

            target_f = self.model.predict(state,verbose=0) # Reward of the Model Prediction.
            target_f[0][action] = target # Update the target for the action taken.
            self.model.fit(state, target_f, epochs=1, verbose=0)
    
    def AdaptiveEGreedy(self,):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# GYM Enviroment;
env = gym.make("CartPole-v1",render_mode="human")
agent = DQLAgent(env)

batch_size = 32
episodes = 10

for e in tq.tqdm(range(episodes)):
    state = env.reset()
    state = np.reshape(state[0], [1, agent.state_size])
    time = 0
    while True:
        # Agent Acts;
        action = agent.act(state)
        
        # Agent make results and it take the next state, reward, done from environment;  
        (next_state, reward, done, _, _)= env.step(action)
        next_state = np.reshape(next_state, [1, agent.state_size])
        # Agent save the experience in memory;
        agent.remember(state, action, reward, next_state, done)
        # Update state;
        state = next_state
        # Replay from memory; -> Training
        agent.replay()
        # Update exploration rate;
        agent.AdaptiveEGreedy()

        time += 1
        if done:
            print(f"episode: {e+1}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
            break

# Test the Agent;
trained_model = agent
env = gym.make("CartPole-v1",render_mode="human")
state = env.reset()[0]
state = np.reshape(state, [1, agent.state_size])
time_t = 0

while True:
    env.render()
    action = trained_model.act(state) # Acts from trained model.
    (next_state, reward, done, _, _)= env.step(action)
    next_state = np.reshape(next_state, [1, agent.state_size])
    state = next_state
    time_t += 1
    print(f"Time Step: {time_t}")
    tm.sleep(0.05)

    if done:
        print(f"Test score: {time_t}")
        break



# Finished.