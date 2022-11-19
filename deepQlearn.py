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

counter = 0

def NNN():
    global counter
    counter += 1
    return counter

np.random.seed(1)
torch.manual_seed(1)
random.seed(1)

class FFnet(nn.Module):

    def __init__(self, lr, fc2_dims, fc3_dims, out_dims, activation = F.relu):
        super(FFnet, self).__init__()
        # input size of 500
        self.conv1 = nn.Conv1d(1, 32, kernel_size=8, stride=8)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.max1 = nn.MaxPool1d(1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=4, stride=1)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.max2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(32, 16, kernel_size=6)
        self.batch_norm3 = nn.BatchNorm1d(16)
        self.max3 = nn.MaxPool1d(10)
        self.fc1 = nn.Linear(16, fc2_dims, bias=True)
        self.fc2 = nn.Linear(fc2_dims, fc3_dims, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.fc3 = nn.Linear(fc3_dims, out_dims, bias=True)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss
        self.activation = activation
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[400, 800], gamma=0.1)

        self.loss_curve = [] # initialise container to keep track of training loss

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).type(torch.float).to(self.device)

        x = self.conv1(x)
        #x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.max1(x)
        x = self.conv2(x)
        #x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.max2(x)
        x = self.conv3(x)
        #x = self.batch_norm3(x)
        x = self.activation(x)
        x = self.max3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        x = self.fc3(x)
        return x

    def fit(self, x, target, epochs=1):
        x = torch.tensor(x).type(torch.float)
        x = x.view(x.shape[0], -1, x.shape[1])
        for epoch in range(epochs):
            self.optimizer.zero_grad()  # zero the gradient buffers
            output = self(x)
            loss = self.criterion()(output, torch.tensor(target).type(torch.float))
            loss.backward()
            self.optimizer.step()  # Does the update
            self.scheduler.step()
            self.loss_curve.append(float(loss.item()))
        return float(loss.item())


class SmartTrader:
    def __init__(self, lr, state_size, eps_decay = 0.99, batch_size=32,
                 fc2_dims=32, fc3_dims=32, confidence_epsilon_thresh=0.05,
                 gamma = 0.99, trade_size=3):

        self.state_size = int(state_size)
        self.memory = deque(maxlen=2000)
        self.state_size = state_size

        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = eps_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.NNN = NNN()

        self.action_dict = {
            0: ["sell", trade_size, 250], # buy/sell, size, holding time
            1: ["buy", trade_size, 250],
            2: ["do nothing", 0, 1], # give no action a trade duration of 500, to give the bot some room to make other actions
            }

        self.action_size = len(self.action_dict)

        self.experience = 0
        self.totalreward = 0
        self.confident = False # bool to turn True when epsilon is low enough
        self.confidence_epsilon_thresh = confidence_epsilon_thresh
        self.counter = 0

        # create main model
        self.model = FFnet(lr, fc2_dims, fc3_dims, self.action_size)
        self.model_losses = deque(maxlen=20)
        self.model_target = deepcopy(self.model)

    def remember(self, state, action, next_state, reward):

        if len(state) == self.state_size:
            self.memory.append((state, action, next_state, reward))

            if self.epsilon < self.confidence_epsilon_thresh:
                if self.confident==False:
                    pd.Series(self.model.loss_curve).to_csv('{}.csv'.format(NNN()))

                self.confident = True
                self.totalreward += reward # tracker to keep hold of rewards resulting from intentional actions

            if len(self.memory) > self.batch_size:
                if self.epsilon >= self.confidence_epsilon_thresh:
                    self.epsilon *= self.epsilon_decay

    def act(self, state):
        state = np.array(state)#[::self.state_reduction]
        state = state.reshape((1, 1, len(state)))
        if state.shape[2] != self.state_size: # catch for dimensionality issues
            return np.random.randint(0, self.action_size)

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

        target = np.zeros((self.batch_size, self.action_size))
        # compute value function of current(call it target) and value function of next state(call it target_next)
        for i in range(self.batch_size):
            a = action[i]
            target[i] = self.model(state[i].reshape((1, 1, self.state_size))).detach().numpy()
            target[i, a] = reward[i]

        training_loss = self.model.fit(state, target, epochs=1)

        #tau = 1e-2
        #if counter % 10 == 0:
        #    for idx, param  in enumerate(self.model.parameters()):
        #        self.model_target.parameters()[idx] = ((1-tau) * param) + \
        #                                       (tau * self.model_target.parameters()[idx])

        #self.model_losses.append(training_loss)
        #if len(self.model_losses) > 8:
        #    print(sum(self.model_losses)/len(self.model_losses))

    def save_model(self, name): # unused in current implementation. Could be useful
        if name[-3:] != '.pt':
            raise NameError("Model Name needs to end in .pt. Currently: {}".format(name))
        torch.save(self.model.state_dict(), os.path.join('./TorchModels/', name))

    def load_model(self, modelname):
        x = torch.load(os.path.join('./TorchModels/', modelname))
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
