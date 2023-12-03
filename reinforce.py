import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from models import MLPPolicy, ICM, FeatureEncoder
from environment_interface import MCEnvironment

device = torch.device("cpu")

# Implementation of the REINFORCE algorithm
class Agent():
    def __init__(self):
        self.env = MCEnvironment(84)

        self.state_size = self.env.state_size
        self.action_size = self.env.action_size
        self.mse = torch.nn.MSELoss()

        self.encoder = FeatureEncoder((1, self.state_size, self.state_size)).to(device)
        self.policy_network = MLPPolicy(self.encoder.feature_size(), self.action_size, self.encoder).to(device)
        self.icm = ICM(self.encoder.feature_size(), self.action_size, self.encoder).to(device)

        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=0.001)
        self.policy_optimiser = optim.Adam(self.policy_network.parameters(), lr=0.01)

        self.score = []

    def random_action(self):
        return self.env.random_action_index()

    def sample_trajectory(self, curious, max_steps=50):
        state = self.env.reset()
        terminal = False
        trajectory = []
        step = 0
        total_curiosity_score = 0
        while not terminal and step < max_steps:
            prob_a = self.policy_network.pi(torch.FloatTensor(state)).to(device)
            
            if not curious:
                action = self.random_action()
            else:
                action = torch.distributions.Categorical(prob_a).sample().item()

            next_state, env_reward, terminal = self.env.step(action)
            curiosity_reward = self.icm.reward(state, action, next_state)
            total_curiosity_score += curiosity_reward
            
            reward = 1 * env_reward + int(curious) * 10 * curiosity_reward
            trajectory.append((state, action, reward))

            state = next_state
            step += 1

        self.score.append(total_curiosity_score)
        return trajectory

    def calculated_discounted_returns(self, rewards, gamma=0.9):
        returns = []
        R = 0
        for reward in reversed(rewards):
            R = reward + gamma * R
            returns.insert(0, R)
        
        return returns
    
    def update_policy(self, trajectories):
        policy_loss = 0
        for trajectory in trajectories:
            states, actions, rewards = zip(*trajectory)
            discounted_returns = self.calculated_discounted_returns(rewards)
            
            for (state, action, _), G_t in zip(trajectory, discounted_returns):
                state = torch.FloatTensor(state)
                prob_a = self.policy_network.pi(state)
                policy_loss += -torch.log(prob_a[0, action]) * G_t

        self.policy_optimiser.zero_grad()
        policy_loss.backward()
        self.policy_optimiser.step()

    def update_icm(self, trajectories):
        icm_loss = 0
        for trajectory in trajectories:
            states, actions, rewards = zip(*trajectory)
            next_states = states[1:]
            actions = actions[:-1]
            states = states[:-1]
            for state, action, next_state in zip(states, actions, next_states):
                icm_loss += self.icm.error(state, action, next_state)
        
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        self.icm_optimizer.step()

    def train(self, num_episodes, batch_size=5, curious=True):
        print('Training...')
        self.env.start()
        for episode in range(num_episodes):
            print('Running episode:', episode)
            trajectories = []
            for batch_id in range(batch_size):
                trajectory = self.sample_trajectory(curious)
                trajectories.append(trajectory)

            self.update_icm(trajectories)
            self.update_policy(trajectories)
        
        print('Training complete.')
        self.env.close()
        self.plot_results(self.score)
    
    def plot_results(self, score):
        plt.plot(range(len(score)), score)
        plt.title('Curiousity Score')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.savefig('curiosity_score.png')
        plt.close()
    
