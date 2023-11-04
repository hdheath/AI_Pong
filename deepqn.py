from collections import deque
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()
        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):
      '''
      chooses an action based on the epsilon value and the Q-values of the current state. If a random number 
      generated is greater than epsilon, it chooses the action with the highest Q-value. Otherwise, it chooses a random action
      '''
      if random.random() > epsilon:
          # converts the state from a numpy array to a PyTorch tensor, and then adds an extra dimension to the tensor to make 
          # it compatible with the input shape expected by the neural network
          state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0))
          # computes the Q-values for the given state using the neural network
          q_value = self.forward(state)
          # selects the action with the highest Q-value for the given state
          action = torch.argmax(self(state)[0]).item()
      else:
          # If the random number generated is less than or equal to epsilon, then a random action is chosen
          action = random.randrange(self.env.action_space.n)
      return action


    def copy_from(self, target):
        self.load_state_dict(target.state_dict())


def compute_td_loss(model, target_model, batch_size, gamma, replay_buffer):
  '''
  takes in the current model, target model, batch size, gamma, and replay buffer as inputs. It samples a batch of 
  experiences from the replay buffer, and then converts them into PyTorch tensors. The Q-values for the current state-action 
  pairs are computed using the model. The target Q-values are computed using the target model and the Bellman equation. 
  Finally, the mean squared error loss is computed between the predicted and target Q-values. The resulting loss value is returned as output
  '''
  state, action, reward, next_state, done = replay_buffer.sample_batch(batch_size)
  state = Variable(torch.FloatTensor(np.float32(state)))
  next_state = Variable(torch.FloatTensor(np.float32(next_state)).squeeze(1), requires_grad=True)
  action = Variable(torch.LongTensor(action))
  reward = Variable(torch.FloatTensor(reward))
  done = Variable(torch.FloatTensor(done))

  # computes the Q-values for the current state-action pairs using the neural network model
  q_value = model(torch.squeeze(state,1))
  # computes the Q-values for the next state using the target neural network model. 
  next_q_value = target_model(next_state)
  # computes the maximum Q-value for the next state.
  next_q_value = torch.max(target_model(next_state),1)[0]
  # computes the target Q-values for the current state-action pairs using the Bellman equation
  target_q_value = reward + gamma * next_q_value * (1 - done)
  # computes the Q-value for the action taken in the current state.
  q_value = model(state.squeeze(1)).gather(1,action.unsqueeze(1)).squeeze(1)
  # Compute mean squared error loss between predicted and target Q-values
  loss = torch.nn.MSELoss(reduction="sum")(q_value,target_q_value)
  return loss

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size):
        # TODO: Randomly sampling data with specific batch size from the buffer
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
