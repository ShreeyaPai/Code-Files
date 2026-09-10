from dino import BasicEnv
import torch
import torch.nn as nn
import sympy

class PolicyNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(7,8),nn.ReLU(),nn.Linear(8,1))

  def forward(self,state):
    return self.net(state)

def choose_action(model, obs):
  state = torch.as_tensor(obs)
  logit = model(state)
  dist = torch.distributions.Bernoulli(logits=logit)
  action = dist.sample()
  log_prob = dist.log_prob(action)
  return int(action.item()), log_prob

def compute_returns(rewards, discount_factor):
  returns = rewards[:]
  for step in range(len(returns)-1, 0, -1):
    returns[step-1] += returns[step]*discount_factor
  return torch.tensor(returns)

def run_episode(model, env, seed=None):
  log_probs, rewards = [], []
  obs, info = env.reset(seed=seed)
  while True:
    action, log_prob = choose_action(model, obs)
    obs, reward, done, truncated, _info = env.step(action)
    log_probs.append(log_prob)
    rewards.append(reward)
    if done or truncated:
      return log_probs, rewards

def train_reinforce(model, optimizer, env, n_episodes, discount_factor):
  for episode in range(n_episodes):
    seed = torch.randint(0,2**32,size=()).item()
    log_probs, rewards = run_episode(model, env, seed=seed)
    returns = compute_returns(rewards, discount_factor)
    returns = returns.float()
    std_returns = ( returns - returns.mean() ) / (returns.std(unbiased=False) + 1e-7)
    losses = [-logp * rt for logp, rt in zip(log_probs, std_returns)]
    loss = torch.cat(losses).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Episode {episode} Reward: {sum(rewards):.2f}", end=" ")

  
env = BasicEnv()

observation, info = env.reset()

torch.manual_seed(42)
model = PolicyNetwork()
optimizer = torch.optim.NAdam(model.parameters(), lr=0.06)
train_reinforce(model, optimizer, env, n_episodes=200, discount_factor=0.95)