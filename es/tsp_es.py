# See https://github.com/staturecrane/PyTorch-ES/blob/master/pytorch_es/examples/cartpole/train_pytorch.py


from env.tsp_env import TspEnv
from env import tsp_plots
import copy
import numpy as np
import torch
import torch.nn as nn

NUMBER_OF_CITIES = 10

# Set stopping conditions
MAXIMUM_ITERATIONS = 500

# ES parameters
POPULATION_SIZE = 100
MAX_SIGMA = 0.2
SIGMA_DECAY = 0.999
MIN_SIGMA = 0.2
LEARNING_RATE = 0.05


class Net(nn.Module):
    """
    Pytorch neural net using the flexible pytorch nn.Modlule class.
    Note: the neural net output is linear. To convert these to probabilities for
    each action (sum to 1.0) a SoftMax activation on the final output is
    required, but this is applied outside of the net itself, which improves
    speed and stability of training.

    Layers in model:
    * Input layer (implied, takes the number of observations)
    * Densely connected layer (size = 4 x number of observations)
    * ReLU activation
    * Densely connected layer (size = 4 x number of observations)
    * ReLU activation
    * Output layer (size = number of possible actions)
    * Softmax normalisation

    """

    def __init__(self, observation_space, action_space):
        """Define layers of sequential net"""
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_space, NUMBER_OF_CITIES * 4),
            nn.ReLU(),
            nn.Linear(NUMBER_OF_CITIES * 4, NUMBER_OF_CITIES * 4),
            nn.ReLU(),
            nn.Linear(NUMBER_OF_CITIES * 4, action_space),
            nn.Softmax(dim=1)
    )
        
        
    def forward(self, x):
        """Define forward pass (simple, as using a pre-defined sequential
        model)"""
        
        # Move input to required device (GPU or CPU)
        # Pass through net
        return self.net(x)


def play_episode(env, model, observation_space):
    """Play an episode"""

    # Reset trackers and environment
    episode_reward = 0
    obs_tracker = []
    action_tracker = []

    # Reset environment (returns first observation)
    obs, reward, is_terminal, info = env.reset()

    # Loop up to 1000 steps
    for step in range(NUMBER_OF_CITIES * 50):

        # Track observations
        obs = np.float32(obs)
        obs_tracker.append(obs)

        # Get action probability (put obs in Tensor first)
        obs = torch.FloatTensor([obs])
        act_probs = model(obs)
        act_probs = act_probs.data.numpy()[0]
        action = np.argmax(act_probs)
        action_tracker.append(action)

        # Take action
        obs, reward, done, info = env.step(action)
        episode_reward += reward

        # Pole has fallen over if done is True
        if done:
            break

    # Put results in dictionary
    results = {'episode_reward': episode_reward,
               'episode_obs': obs_tracker,
               'episode_actions': action_tracker}

    return results


def jitter_weights(model, sigma):
    cloned_model = copy.deepcopy(model)
    for param in cloned_model.parameters():
        shape = param.size()
        jitter = torch.normal(mean=0, std=sigma, size=shape)
        param.data += jitter
        
    return cloned_model


def adjust_model(model, rewards, alt_models):
    
    new_model = copy.deepcopy(model)
    
    # Normalise rewards
    norm_rewards = (np.array(rewards) - np.mean(rewards)) / np.std(rewards)    
        
    # Extract weights of alternative models
    weights = [[] for params in new_model.parameters()]    
    for alt_model in alt_models:
        for layer, params in enumerate(alt_model.parameters()):
            weights[layer].append(params.data.detach().numpy())


    # Update model
    for layer, params in enumerate(new_model.parameters()):
        layer_weights = np.array(weights[layer])
        new_weights = np.dot(layer_weights.T, norm_rewards).T
        params.data = params.data + (torch.Tensor(new_weights) * LEARNING_RATE)
 
    return new_model




def main():
     # Set up environment
    env = TspEnv(number_of_cities = NUMBER_OF_CITIES)
    overall_best_reward = 0
    overall_best_route = None

    # Get number of observations from environemt(allows the env to change)
    # Obs = array of visited cities and on-ehot array of current city
    obs_size = env.observation_space.shape[0] * 2

    # Get number of actins from environemnt
    n_actions = len(env.action_space)

    # Set up Neural Net (only needs eval mode)
    model = Net(obs_size, n_actions)
    model.eval()
    
    # Set up sigma
    sigma = MAX_SIGMA
    
    # Loop through iterations
    for iter in range(MAXIMUM_ITERATIONS):
        
        # Loop through population
        rewards = []; routes = []; alt_models = []
        for pop in range(POPULATION_SIZE):
            # Get jittered model
            cloned_model = jitter_weights(model, sigma)
            # Play episode
            results = play_episode(env, cloned_model, obs_size)
            rewards.append(results['episode_reward'])
            routes.append(results['episode_actions'])
            alt_models.append(cloned_model)
         
        best_reward = np.max(rewards)
        mean_reward = np.mean(rewards)
        print(f'{iter}: Best: {best_reward:.0f}, mean: {mean_reward: .0f}')
        
        # Check for new best
        if best_reward > overall_best_reward:
            overall_best_reward = best_reward
            overall_best_route = routes[np.argmax(rewards)]
            # Plot new best route
            co_ords = env.state.city_locations
            overall_best_route = [0] + overall_best_route
            route_co_ords = [co_ords[i] for i in overall_best_route]
            tsp_plots.plot_route(overall_best_route, route_co_ords)
        
        # Adjust model
        model = adjust_model(model, rewards, alt_models)
            
        # Decay sigma
        sigma *= SIGMA_DECAY
        sigma = max(sigma, MIN_SIGMA)

            
            

main()
            
            
    
    
    