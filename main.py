import gym
import torch
import torch.optim as optim

from itertools import count
from actor import Actor
from critic import Critic


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def training(actor, critic, n_iterations):
    optimizer_actor = optim.Adam(actor.parameters())
    optimizer_critic = optim.Adam(critic.parameters())
    for iteration in range(n_iterations):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0

        for i in count():
            env.render()
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)

            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy())

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1 - done], dtype=torch.float, device=device))

            state = next_state

            if done:
                print('Iteration: {}, Score: {}'.format(iteration, i))
                break

        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizer_actor.step()
        optimizer_critic.step()
    env.close()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1").unwrapped

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    learning_rate = 0.0001

    training(actor=Actor(state_size, action_size).to(device), critic=Critic(state_size, action_size).to(device),
             n_iterations=100)
