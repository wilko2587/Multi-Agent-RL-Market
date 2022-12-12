import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import pandas as pd
from copy import deepcopy
from torch.optim.lr_scheduler import MultiStepLR

np.random.seed(1)
torch.manual_seed(1)
random.seed(1)

class FFnet(nn.Module):

    def __init__(self, lr, fc2_dims, fc3_dims, out_dims, activation = nn.ReLU):
        super(FFnet, self).__init__()
        # input size of 500
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(4*16, fc2_dims),
            nn.LeakyReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(fc2_dims, fc3_dims),
            nn.LeakyReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(fc3_dims, out_dims),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = F.mse_loss
        self.activation3 = activation()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[300, 500, 700, 900], gamma=0.1)

        self.loss_curve = [] # initialise container to keep track of training loss
        self.double()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).to(self.device).double()
            x = x.view(-1, 1, 512)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def fit(self, x, target, epochs=1):
        x = torch.tensor(x).type(torch.float)
        x = x.view(x.shape[0], -1, x.shape[1])
        target = torch.tensor(target).type(torch.float)
        for epoch in range(epochs):
            self.optimizer.zero_grad()  # zero the gradient buffers
            output = self(x)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()  # Does the update
            self.scheduler.step()
            self.loss_curve.append(float(loss.item()))
        return float(loss.item())


class SmartTrader:
    counter = 0
    def __init__(self, lr, state_size, eps_decay = 0.99, batch_size=32,
                 fc2_dims=8, fc3_dims=8, confidence_epsilon_thresh=0.01,
                 gamma = 0.99):

        self.state_size = int(state_size)
        self.memory = deque(maxlen=1000)
        self.state_size = state_size

        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = eps_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.counter = 0 # counter to track soft update frequency
        self.id = int(SmartTrader.counter)
        SmartTrader.counter += 1

        self.action_dict = {
            0: ["sell", 3, 100],  # buy/sell, size, holding time
            1: ["sell", 3, 250],
            2: ["sell", 3, 500],
            3: ["do nothing", 0, 100],
            4: ["buy", 3, 100],
            5: ["buy", 3, 250],
            6: ["buy", 3, 500],
        }

        self.action_size = len(self.action_dict)

        self.experience = 0
        self.totalreward = 0
        self.totalrewards = [self.totalreward]
        self.confident = False # bool to turn True when epsilon is low enough
        self.confidence_epsilon_thresh = confidence_epsilon_thresh
        self.epsilons = []
        self.extra_counter = 0

        self.states = []
        self.actions = []

        # create main model
        self.model = FFnet(lr, fc2_dims, fc3_dims, self.action_size)
        self.model_target = deepcopy(self.model)

    def remember(self, state, action, next_state, reward):

        if self.confident:
            self.states.append(pd.Series(state))
            self.actions.append(action)

        self.totalreward += reward  # tracker to keep hold of rewards resulting from intentional actions
        self.totalrewards.append(self.totalrewards[-1] + reward)
        self.epsilons.append(self.epsilon)
        if len(state) == self.state_size:
            self.memory.append((state, action, next_state, reward))

            if self.epsilon <= self.confidence_epsilon_thresh:
                if self.confident == False and self.extra_counter == 100:
                    self.confident = True
                self.extra_counter += 1
            if len(self.memory) > self.batch_size:
                if self.epsilon >= self.confidence_epsilon_thresh:
                    self.epsilon *= self.epsilon_decay
                self.learn()

    def act(self, state):
        state = np.array(state) #[::self.state_reduction]
        state = state.reshape((1, 1, len(state)))

        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            q_out = self.model(state).detach().numpy()
            action = np.argmax(q_out)
            return action

    def get_action_details(self, action_index):
        return self.action_dict[action_index]

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # assign data into state, next_state, action, reward and done from minibatch
        for i in range(self.batch_size):
            ministate, miniaction, mininext, minireward = minibatch[i]
            state[i] = ministate
            next_state[i] = mininext
            action.append(miniaction)
            reward.append(minireward)

        state = torch.tensor(state).double()
        next_state = torch.tensor(next_state).double()
        target = self.model_target(state.unsqueeze(dim=1)).detach().clone()
        target_next = self.model_target(next_state.unsqueeze(dim=1))
        for i in range(self.batch_size):
            target[i, action[i]] = reward[i] + self.gamma * torch.max(target_next[i])

        output = self.model(state.unsqueeze(dim=1))
        mask = torch.zeros_like(output)
        icoords = torch.arange(start=0, end=len(action)).type(torch.int64)
        jcoords = torch.tensor(action).type(torch.int64)
        mask[(icoords, jcoords)] = 1
        output_filt = output*mask
        target_filt = target*mask
        self.model.optimizer.zero_grad()  # zero the gradient buffers
        loss = self.model.criterion(output_filt.double(), target_filt.double())
        loss.backward()
        self.model.optimizer.step()  # Does the update
        self.model.scheduler.step()
        self.model.loss_curve.append(float(loss.item()))

        # soft update of target network
        tau = 1e-3
        if self.counter % 1 == 0:
            for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * param.data.detach() + (1.0 - tau) * target_param.data.detach())
        #       #target_param.data.copy_(param.data) # hard update

        self.counter += 1

    def save_model(self, name): # unused in current implementation. Could be useful
        if name[-3:] != '.pt':
            raise NameError("Model Name needs to end in .pt. Currently: {}".format(name))
        torch.save(self.model.state_dict(), os.path.join('TorchModels/', name))

    def load_model(self, modelname):
        x = torch.load(os.path.join('TorchModels/', modelname))
        self.model.load_state_dict(x)
        self.model.eval()
        self.confident = True
        self.epsilon = self.epsilon_min


if __name__ == '__main__':
    # dummy run through
    model = SmartTrader(1e-3, 500, 2)
    for i in range(2100):
        state = np.random.uniform(-1, 1, size=500)
        model.remember(state, np.random.randint(0, 2), np.random.uniform(-1, 1))
        model.learn()

    model.act(state)
