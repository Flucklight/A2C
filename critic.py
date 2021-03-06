import torch.nn as nn
import torch.nn.functional as functional


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = functional.relu(self.linear1(state))
        output = functional.relu(self.linear2(output))
        value = self.linear3(output)
        return value
