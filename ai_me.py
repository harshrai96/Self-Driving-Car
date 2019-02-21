# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:21:27 2018

@author: Vaibhav
"""
# Importing the libraries

import numpy as np # For normal calculations
import random # For creating a random sample from memory
import os # For load/save
import torch
import torch.nn as nn # For creating the Neural Network Model
import torch.nn.functional as F # For activation functions
import torch.optim as optim # For optimizers
import torch.autograd as autograd
from torch.autograd import Variable # For creating a variable, which has a torch tensor and a gradient associated with it.

# Creating the Neural Network architecture

class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)

    def forward(self, state):
        h = F.relu(self.fc1(state))
        q_values = self.fc2(h)
        return q_values


# Experience Replay

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        '''
        From each event, collecting all the state, next_state, action and reward together

        e.g. x = [[1,2,3],[4,5,6]]  ----> samples = [[1,4],[2,5],[3,6]]
        '''
        samples = zip(*random.sample(self.memory, batch_size))
        # making all synchronised according to time. (i.e. 3rd index next state will correspond to the next state of 3rd index state)
        return map(lambda x: Variable(torch.cat(x,0)), samples)

# Creating the Brain (Implementing Deep Q Learning)

class Dqn:

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(3000000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile=True))*100) # T=100
        action = probs.multinomial()
        return action.data[0,0]

    def learn(self, batch_state, batch_next_state, batch_action, batch_reward):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables=True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_action, batch_reward)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)

    def save(self):
        torch.save({'state_dict':self.model.state_dict(),
                    'optimizer':self.optimizer.state_dict()
                    }, 'last_brain.pth')

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
