'''
Code name: models.py
Author: Marco Antonio Esquivel Basaldua

Description:
Defines every Artificial Neural Network architecture (for pursuer and evader).

Input arguments:
    - none

Output:
    - none
'''

import torch
from torch import nn
from torch.distributions.categorical import Categorical
import copy


############ Evader Model ############################
class evader_NN(nn.Module):
    def __init__(self, input_size, n_actions, num_neurons_hidden, device):
        super(evader_NN, self).__init__()

        # Define network
        self.predict = nn.Sequential(
            nn.Linear(input_size, num_neurons_hidden),
            nn.ReLU(),
            nn.Linear(num_neurons_hidden, num_neurons_hidden),
            nn.ReLU(),
            nn.Linear(num_neurons_hidden, num_neurons_hidden),
            nn.ReLU(),
            nn.Linear(num_neurons_hidden, num_neurons_hidden),
            nn.ReLU(),
            nn.Linear(num_neurons_hidden, n_actions),
            nn.Softmax(dim=-1)         
        )

        self.device = device

    def forward(self, state):
        dist = self.predict(torch.FloatTensor(state).to(self.device))

        return dist

    def best_action(self, state):
        return torch.argmax(self.predict(state)).item()


############# Pursuer Model ######################
class pursuer_NN(nn.Module):
    def __init__(self, input_size, n_actions, num_neurons_hidden, device):
        super(pursuer_NN, self).__init__()

        # Define network
        self.predict = nn.Sequential(
            nn.Linear(input_size, num_neurons_hidden),
            nn.ReLU(),
            nn.Linear(num_neurons_hidden, num_neurons_hidden),
            nn.ReLU(),
            nn.Linear(num_neurons_hidden, num_neurons_hidden),
            nn.ReLU(),
            nn.Linear(num_neurons_hidden, num_neurons_hidden),
            nn.ReLU(),
            nn.Linear(num_neurons_hidden, n_actions),
            nn.Softmax(dim=-1)    
        )

        self.device = device

    def forward(self, state):
        dist = self.predict(torch.FloatTensor(state).to(self.device))

        return dist
    
    def best_action(self, state):
        return torch.argmax(self.predict(state)).item()