import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

class DuellingDQN(nn.Module):

    """
    Duelling DQN
    """
    
    def __init__(self, observation_space, action_space, learning_rate,
                 exploration_max):
        """Constructor method. Set up neural nets."""

        # Set starting exploration rate
        self.exploration_rate = exploration_max
        
        # Set up action space (choice of possible actions)
        self.action_space = action_space
              
        super(DuellingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(observation_space, observation_space * 4),
            nn.ReLU(),
            nn.Linear(observation_space * 4, observation_space * 4),
            nn.ReLU()
            )
        
        self.advantage = nn.Sequential(
            nn.Linear(observation_space * 4, observation_space * 4),
            nn.ReLU(),
            nn.Linear(observation_space * 4, action_space)
        )
        
        self.value = nn.Sequential(
            nn.Linear(observation_space * 4, observation_space * 4),
            nn.ReLU(),
            nn.Linear(observation_space * 4, 1)
        )
        
        # Set optimizer
        self.optimizer = optim.Adam(
                params=self.parameters(), lr=learning_rate)
        

    def act(self, state):
        """Act either randomly or by redicting action that gives max Q"""
        
        # Act randomly if random number < exploration rate
        if np.random.rand() < self.exploration_rate:
            action = random.randrange(self.action_space)
            
        else:
            # Otherwise get predicted Q values of actions
            q_values = self.forward(torch.FloatTensor(state))
            # Get index of action with best Q
            action = np.argmax(q_values.detach().numpy()[0])
        
        return  action
        
  
    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value     = self.value(x)
        action_q = value + advantage  - advantage.mean()
        return action_q
    
    

def optimize_duelling_dqn(policy_net, target_net, memory, batch_size, 
                          exploration_decay, exploration_min, gamma):
    """
    Update  model by sampling from memory.
    Uses policy network to predict best action (best Q).
    Uses target network to provide target of Q for the selected next action.
    """
      
    # Do not try to train model if memory is less than reqired batch size
    if len(memory) < batch_size:
        return    
 
    # Reduce exploration rate
    policy_net.exploration_rate *= exploration_decay
    policy_net.exploration_rate = max(exploration_min, 
                                      policy_net.exploration_rate)
    # Sample a random batch from memory
    batch = random.sample(memory, batch_size)
    for state, action, reward, state_next, terminal in batch:
        
        state_action_values = policy_net(torch.FloatTensor(state))
       
        if not terminal:
            # For non-terminal actions get Q from policy net
            expected_state_action_values = policy_net(torch.FloatTensor(state))
            # Detach next state values from gradients to prevent updates
            expected_state_action_values = expected_state_action_values.detach()
            # Get next state action with best Q from the policy net (double DQN)
            policy_next_state_values = policy_net(torch.FloatTensor(state_next))
            policy_next_state_values = policy_next_state_values.detach()
            best_action = np.argmax(policy_next_state_values[0].numpy())
            # Get targen net next state
            next_state_action_values = target_net(torch.FloatTensor(state_next))
            # Use detach again to prevent target net gradients being updated
            next_state_action_values = next_state_action_values.detach()
            best_next_q = next_state_action_values[0][best_action].numpy()
            updated_q = reward + (gamma * best_next_q)      
            expected_state_action_values[0][action] = updated_q
        else:
            # For termal actions Q = reward (-1)
            expected_state_action_values = policy_net(torch.FloatTensor(state))
            # Detach values from gradients to prevent gradient update
            expected_state_action_values = expected_state_action_values.detach()
            # Set Q for all actions to reward (-1)
            expected_state_action_values[0] = reward

        # Update neural net
        
        # Reset net gradients
        policy_net.optimizer.zero_grad()  
        # calculate loss
        loss_v = nn.MSELoss()(state_action_values, expected_state_action_values)
        # Backpropogate loss
        loss_v.backward()
        # Update network gradients
        policy_net.optimizer.step()  

    return