import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class NNPolicy(nn.Module):

    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Args:
            x: input tensor (first dimension is a batch dimension)

        Return:
            Probabilities of performing all actions in given input states x. Shape: batch_size x action_space_size
        """
        # YOUR CODE HERE
        # raise NotImplementedError
        action_probs = F.softmax(self.l2(nn.ReLU()(self.l1(x))))
        return action_probs

    def get_probs(self, obs, actions):
        """
        This function takes a tensor of states and a tensor of actions and returns a tensor that contains
        a probability of performing corresponding action in all states (one for every state action pair).

        Args:
            obs: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: batch_size x 1

        Returns:
            A torch tensor filled with probabilities. Shape: batch_size x 1.
        """
        # YOUR CODE HERE
        # raise NotImplementedError
        action_probs = torch.gather(self.forward(obs),1,actions)
        return action_probs

    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.

        Args:
            obs: state as a tensor. Shape: 1 x obs_dim or obs_dim

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        # raise NotImplementedError
        action = torch.multinomial(self.forward(obs.reshape((1,-1))),1)
        return int(action)

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function as tensors.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of tensors (states, actions, rewards, dones). All tensors should have same first dimension and 
        should have dim=2. This means that vectors of length N (states, rewards, actions) should be Nx1.
        Hint: Do not include the state after termination in states.
    """
    states = []
    actions = []
    rewards = []
    dones = []
    done = False
    state = env.reset()
    while not done:
        state = torch.FloatTensor(state)
        states.append(state)
        action = policy.sample_action(state)
        state, reward, done, _ = env.step(action)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
    # YOUR CODE HERE
    # raise NotImplementedError
    states, actions, rewards, dones = torch.stack(states), torch.tensor(actions, dtype=torch.int64).reshape((-1,1)), torch.FloatTensor(rewards).reshape((-1,1)), torch.tensor(dones).reshape((-1,1))
    return states, actions, rewards, dones
import sys
def compute_reinforce_loss(policy, episode, discount_factor):
    """
    Computes reinforce loss for given episode.

    Args:
        policy: A policy which allows us to get probabilities of actions in states with its get_probs method.

    Returns:
        loss: reinforce loss
    """
    # Compute the reinforce loss
    # Make sure that your function runs in LINEAR TIME
    # Note that the rewards/returns should be maximized 
    # while the loss should be minimized so you need a - somewhere
    # YOUR CODE HERE
    # raise NotImplementedError
    states, actions, rewards, _ = episode
    probs = policy.get_probs(states, actions)
    G = []
    G.append(rewards[0])
    for r in rewards[1:]:
        G.append(r+discount_factor*G[-1])
    loss = -torch.sum(torch.stack(tuple(reversed(G))) * torch.log(probs))
    return loss

# # YOUR CODE HERE
# raise NotImplementedError

def run_episodes_policy_gradient(policy, env, num_episodes, discount_factor, learn_rate, 
                                 sampling_function=sample_episode):
    optimizer = optim.Adam(policy.parameters(), learn_rate)
    
    episode_durations = []
    for i in range(num_episodes):
        # YOUR CODE HERE
        optimizer.zero_grad()
        episode = sampling_function(env, policy)
        loss = compute_reinforce_loss(policy, episode, discount_factor)
        loss.backward()
        optimizer.step()
        # raise NotImplementedError
                           
        if i % 10 == 0:
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))
        episode_durations.append(len(episode[0]))
        
    return episode_durations
