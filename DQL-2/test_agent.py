# Test trained DQL agent without retraining;
import numpy as np
import gymnasium as gym
import tensorflow as tf
import imageio
from tensorflow.keras import layers

# Environment setup;
env_name = 'LunarLander-v3'
state_size = 8
number_actions = 4

# Network definition (same as training);s
class Network(tf.keras.Model):
    def __init__(self, state_size, action_size, seed=42):
        super(Network, self).__init__()
        tf.random.set_seed(seed)
        self.fc1 = layers.Dense(64, activation='relu', kernel_initializer='he_uniform')
        self.fc2 = layers.Dense(64, activation='relu', kernel_initializer='he_uniform')
        self.fc3 = layers.Dense(action_size, kernel_initializer='glorot_uniform')
    
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.fc3(x)

# Create and build model;
model = Network(state_size, number_actions)
dummy_state = tf.zeros((1, state_size))
model(dummy_state)  # Build model;

# Load trained weights;
model.load_weights('dql_lunarlander_weights.weights.h5')
print("Model weights loaded successfully!")

# Record video;
def record_video(model, env_name, filename='video_test.mp4', num_episodes=1):
    env = gym.make(env_name, render_mode='rgb_array')
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        frames = []
        total_reward = 0
        
        while not done:
            frame = env.render()
            frames.append(frame)
            
            # Get action from model (greedy - no exploration)
            state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
            action_values = model(state_tensor, training=False)
            action = int(np.argmax(action_values.numpy()))
            
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}")
        
        # Save video;
        imageio.mimsave(filename, frames, fps=30)
        print(f"Video saved as {filename}")
    
    env.close()

# Record the video;
record_video(model, env_name)
print("Done!")





# Finished.