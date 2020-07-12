# Autodidactic Iteration (ADI)
# Modified from https://arxiv.org/pdf/1805.08966v1.pdf
# But reframe as Q

# TODO Add Q training when testing
# TODO Bagging

from collections import deque
from env.tsp_env import TspEnv
from utils import tsp_plots
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim

class Params(object):
    def __init__(self):
        # Set number of cities to visit
        self.number_of_cities = 10

        # Set training details
        self.learning_depth = 50

        # Discount rate of future rewards
        self.gamma = 0.99
        # Learing rate for neural network
        self.learning_rate = 0.001
        # Maximum memory of game steps (state, action, reward, next state) 
        self.memory_size = 100000
        # number of episodes to play before starting training
        self.initial_training_episodes = 5

        # Set stopping conditions
        self.maximum_runs = 20000
        self.maximum_time_mins = 180
        self.no_improvement_runs = 1000
        self.no_improvement_time = 180


class ADI(nn.Module):

    def __init__(self, model_params, observation_space, n_actions):
        """Constructor method. Set up memory and neural nets."""

        self.n_actions = n_actions

        # Set starting exploration rate
        self.exploration_rate = model_params.exploration_max

        # Set up memory for state/action/reward/next_state/done
        # Deque will drop old data when full
        self.memory = deque(maxlen=model_params.memory_size)

        # Set up neural net
        super(ADI, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_space, model_params.number_of_cities * 5),
            nn.ReLU(),
            nn.Linear(model_params.number_of_cities * 5, 
                      model_params.number_of_cities * 5),
            nn.ReLU(),
            nn.Linear(model_params.number_of_cities * 5, 
                      model_params.number_of_cities * 5),
            nn.ReLU(),
            nn.Linear(model_params.number_of_cities * 5, n_actions))

        # Set loss function and optimizer
        self.objective = nn.MSELoss()
        self.optimizer = optim.Adam(
            params=self.net.parameters(), lr=self.params.learning_rate)

    def act(self, state):
        q_values = self.net(torch.FloatTensor(state))
        # Get index of action with best Q
        action = np.argmax(q_values.detach().numpy()[0])

        return action


    def experience_replay(self, params, runs):
        """Update model by sampling (s/a/r/s') from memory. As the memory
        accumulates knowledge of rewards, this will lead to a better ability to
        estimate Q (which depends on future possible rewards."""

        # Do not try to train model if memory is less than reqired batch size
        if len(self.memory) < params.batch_size:
            return

        # Reduce exploration rate down to minimum
        self.exploration_rate *= params.exploration_decay
        self.exploration_rate = max(params.exploration_min, self.exploration_rate)

        # Sample a random batch from memory
        batch = random.sample(self.memory, params.batch_size)

        for state, action, reward, state_next, terminal in batch:

            if terminal:
                q_update = reward

            else:
                # Get best possible Q for next action
                action_q = self.net(torch.Tensor(state_next))
                action_q = action_q.detach().numpy().flatten()
                best_next_q = np.amax(action_q)
                # Calculate current Q using Bellman equation
                q_update = (reward + params.gamma * best_next_q)

            # Get predicted Q values for current state
            q_values = self.net(torch.Tensor(state))

            # Update predicted Q for current state/action
            q_values[0][action] = q_update

            # Update neural net to better predict the updated Q value

            # Reset net gradients
            self.optimizer.zero_grad()
            # calculate loss
            loss_v = nn.MSELoss()(self.net(torch.FloatTensor(state)), q_values)
            # Backpropogate loss
            loss_v.backward()
            # Update network gradients
            self.optimizer.step()
            

    def forward(self, x):
        """Feed forward function for neural net"""

        return self.net(x)

    def remember(self, state, action, reward, next_state, done):
        """state/action/reward/next_state/done"""

        self.memory.append((state, action, reward, next_state, done))


class Model(object):

    def __init__(self):
        # Set up environment
        self.env = TspEnv(number_of_cities=self.model_params.number_of_cities)
        self.time_start = time.time()

        # Get number of observations returned for state
        self.observation_space = self.env.observation_space.shape[0] * 2

        # Get number of actions possible
        self.n_actions = len(self.env.action_space)

        # Set up neural net
        self.adi = ADI(model_params =self.model_params,
                       observation_space = self.observation_space, 
                       n_actions = self.n_actions)


    def initial_train(self):
        # Build up memory
        for episode in range(self.model_params.initial_training_episodes):
            self.reverse_search()





    def reverse_search(self):
        self.env.reset(reverse=True)
        # Loop up to maximum search depth
        for step in range(self.model_params.learning_depth):
            # Loop through all possible actions
            for action in range(self.n_actions):
                # Record current state
                current_location = self.adi.state.agent_location.copy()
                current_cities_visited = self.adi.state.visited_status.copy()
                # Set current location to ot visited
                self.adi.state.visited_status[current_location] = 0
                print()




def main():
    model = Model()
    model.model_params = Params()
    model.initial_train()

main()


