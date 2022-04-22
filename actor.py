import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as functional


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = functional.relu(self.linear1(state))
        output = functional.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(functional.softmax(output, dim=-1))
        return distribution
