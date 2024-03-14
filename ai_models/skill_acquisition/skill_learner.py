import numpy as np
import pickle
from collections import deque
from sklearn.neural_network import MLPRegressor

class SkillLearner:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1, max_memory=10000, batch_size=32):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory = deque(maxlen=max_memory)
        self.batch_size = batch_size
        self.q_network = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=10000)

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            q_values = self.q_network.predict([state])
            action = np.argmax(q_values)
        return action

    def train_q_network(self):
        if len(self.memory) < self.batch_size:
            return

        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])

        q_values = self.q_network.predict(states)
        next_q_values = self.q_network.predict(next_states)

        for i in range(self.batch_size):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]
            else:
                q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        self.q_network.fit(states, q_values)

    def save_model(self, file_path):
        with open(file_path, "wb") as file:
            pickle.dump(self.q_network, file)

    def load_model(self, file_path):
        with open(file_path, "rb") as file:
            self.q_network = pickle.load(file)

    def transfer_learning(self, source_model_path):
        with open(source_model_path, "rb") as file:
            source_model = pickle.load(file)
        self.q_network.coefs_ = source_model.coefs_
        self.q_network.intercepts_ = source_model.intercepts_