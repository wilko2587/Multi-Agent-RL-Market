import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os

class FFnet(nn.Module):

    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, out_dims, activation = F.relu):
        super(FFnet, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims, bias=True)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims, bias=True)
        self.fc3 = nn.Linear(fc2_dims, out_dims, bias=True)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss
        self.activation = activation
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).type(torch.float)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

    def fit(self, x, target, epochs=1):
        x = torch.tensor(x).type(torch.float)
        for epoch in range(epochs):
            self.optimizer.zero_grad()  # zero the gradient buffers
            output = self(x)
            loss = self.criterion()(output, torch.tensor(target).type(torch.float))
            loss.backward()
            self.optimizer.step()  # Does the update
        return


class SmartTrader:
    def __init__(self, lr, state_size, eps_decay = 0.999, batch_size=64,
                 fc1_dims=50, fc2_dims=50, confidence_epsilon_thresh = 0.1, state_reduction=10,
                 gamma = 0.99):

        self.state_size = int(state_size/state_reduction)
        self.state_reduction = state_reduction
        self.memory = deque(maxlen=2000)

        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = eps_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.action_dict = {
            0: ["sell", 2, 100], # buy/sell, size, holding time
            1: ["sell", 2, 250],
            2: ["sell", 2, 500],
            3: ["sell", 1, 100],
            4: ["sell", 1, 250],
            5: ["sell", 1, 500],
            6: ["do nothing", 0, 500], # give no action a trade duration of 500, to give the bot some room to make other actions
            7: ["buy", 2, 100],
            8: ["buy", 2, 250],
            9: ["buy", 2, 500],
            10: ["buy", 1, 100],
            11: ["buy", 1, 250],
            12: ["buy", 1, 500]
            }

        self.action_size = len(self.action_dict)
                        

        self.experience = 0
        self.totalreward = 0
        self.confident = False # bool to turn True when epsilon is low enough
        self.confidence_epsilon_thresh = confidence_epsilon_thresh

        # create main model
        self.model = FFnet(lr, self.state_size, fc1_dims, fc2_dims, self.action_size)

    def remember(self, state, action, next_state, reward):
        state = np.array(state)[::self.state_reduction]
        next_state = np.array(next_state)[::self.state_reduction]

        if len(state) == self.state_size:
            self.memory.append((state, action, next_state, reward))

            if self.epsilon < self.confidence_epsilon_thresh:
                self.confident = True
                self.totalreward += reward

            if len(self.memory) > self.batch_size:
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

    def act(self, state):
        state = np.array(state)[::self.state_reduction]
        if len(state) != self.state_size: # catch for dimensionality issues
            return np.random.randint(0, self.action_size)

        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            q_out = self.model.forward(state).detach().numpy()
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
            action.append(self.act(state[i]))
            reward.append(minireward)

        target = np.zeros((self.batch_size, self.action_size))
        # compute value function of current(call it target) and value function of next state(call it target_next)
        for i in range(self.batch_size):
            a = action[i]
            Q_target_next = np.max(self.model.forward(next_state[i].reshape([1, len(next_state[i])])).detach().numpy())
            target[i] = self.model.forward(state[i].reshape([1, len(state[i])])).detach().numpy()
            target[i, a] = reward[i] + self.gamma * Q_target_next

        self.model.fit(state, target, epochs=1)

    def save_model(self, name):
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
