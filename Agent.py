import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import copy
import pickle
from model import *

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.00001
APPLE_REWARD = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float)

# 32x24 grid

# The agent is the component that does the learning and performs the actions
class Agent:
  def __init__(self):
    self.n_games = 0
    self.epsilon = 0.5
     # exploration
    self.gamma = 0.99 # discount rate
    self.memory = deque(maxlen=MAX_MEMORY)
    self.memory_weights = [i for i in range(MAX_MEMORY)]
    self.counter = 0
    self.model = DuelingCNN(1, 3) # input depth = 1 ie black white, outsize = 3 ie 3 actions left, stright, right
    # self.model = LinearNet(11, 256, 3)
    # self.model = LinearNet(32*24*3, 1024, 3)
    self.model_target = copy.deepcopy(self.model)
    self.model_target.to(device)
    self.model.to(device)
    self.trainer = DQTrainer(self.model, self.model_target, lr=LR, gamma=self.gamma)

  def get_state(self, game):
    return game.return_grid() # returns 1x32x24 grid

  def update_target_model(self):
    self.model_target.load_state_dict(self.model.state_dict())
    
  def load_user_training_data(self):
    file2 = open(r'user_play.pkl', 'rb')

    new_d = pickle.load(file2)
    # print(len(new_d))
    for tup in new_d:
        self.add_memory(*tup)
        
    self.update_memory_episode_rewards()
    
  def add_memory(self, state, action, reward, next_state, done):
    self.memory.append([state, action, reward, next_state, done])
    # self.counter += 1
    # self.memory_weights.append(self.counter)

  def update_memory_episode_rewards(self):
    idx = len(self.memory)-2 # most recent one is done
    bonus = 0
    counter = 1
    apple = False
    while idx >= 0:
      state, action, reward, next_state, done = self.memory[idx]
      
      if done or not counter:
        break
      
      if apple: 
        self.memory[idx] = [state, action, reward+bonus, next_state, done]
        counter -= 1

      if reward == APPLE_REWARD:
        bonus += APPLE_REWARD * 0.5
        apple = True
        counter += 5      
      idx -= 1
      

  def train_long_memory(self):
    if len(self.memory) < BATCH_SIZE:
      samples = self.memory
    else:
      samples = random.choices(self.memory, weights=self.memory_weights[:len(self.memory)], k=BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*samples)
    # print(actions)
    states  = torch.stack(list(states), dim=0)
    next_states = torch.stack(list(next_states), dim=0)
    actions = torch.stack(list(torch.tensor(actions)), dim=0)
    rewards = torch.stack(list(torch.tensor(rewards)), dim=0)
    dones = torch.stack(list(torch.tensor(dones)), dim=0)

    # states = torch.unsqueeze(states, 1)
    # next_states = torch.unsqueeze(next_states, 1)
    # actions = torch.unsqueeze(actions, 1)
    # rewards = torch.unsqueeze(rewards, 1)
    # dones = torch.unsqueeze(dones, 1)

    # print(states.shape)
    self.trainer.train(states, actions, rewards, next_states, dones)

  def train_single_step(self, state, action, reward, next_state, done):
    self.trainer.train(state, action, reward, next_state, done)

  def get_action(self, state):
    # in the beginning will do some random moves, tradeoff between exploration and exploataition
    # self.epsilon = 80 - self.n_games
    final_move = [0, 0, 0] # put a 1 for the action we want to take
    
    # only explore in beginning
    # if self.epsilon > 0 and random.randint(0, 200) < self.epsilon:
    if random.random() < self.epsilon:
      move = random.randint(0, 2)
      final_move[move] = 1
    else:
    #   state_tensor = torch.tensor(state, dtype=torch.float)
      state_tensor = state
      # if len(state_tensor.shape) == 3:
      #   state_tensor = torch.unsqueeze(state_tensor, dim=0)
        # print(len(state_tensor.shape))
        # state_tensor = state_tensor[None, :]

      state_tensor = state_tensor.to(device)
      # print(type(state_tensor))
    #   state_tensor = torch.flatten(state_tensor)
    #   print(state_tensor.shape)
      # print(torch.flatten(state_tensor).shape)
      with torch.no_grad():
        prediction = self.model(state_tensor)

      move = torch.argmax(prediction).item()
      final_move[move] = 1

    return final_move



# trainer for dqn
class DQTrainer:
    def __init__(self, model, model_target, lr, gamma):
      self.model = model
      self.model_target = model_target
      self.lr = lr
      self.gamma = gamma
      self.lamda = 0.05

      self.optimizer = optim.AdamW(model.parameters(), lr=self.lr)
      self.scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True)
      self.loss_function = nn.MSELoss()
      # self.loss_function = nn.HuberLoss()
      
    def train(self, state_tensor, action, reward, new_state_tensor, done):
      # state_tensor = torch.tensor(state, dtype=torch.float)
      action_tensor = torch.tensor(action, dtype=torch.float)
      reward_tensor = torch.tensor(reward, dtype=torch.float)
      # new_state_tensor = torch.tensor(new_state, dtype=torch.float)
      if len(state_tensor.shape) == 3: # convert to batch form
        state_tensor = torch.unsqueeze(state_tensor, dim=0)
        action_tensor = torch.unsqueeze(action_tensor, dim=0)
        reward_tensor = torch.unsqueeze(reward_tensor, dim=0)
        new_state_tensor = torch.unsqueeze(new_state_tensor, dim=0)
        done = (done,) # convert done to a 1d tensor ie if batch we'd have 1 done for each state vector

      for i in range(len(done)): # for idx in batch
        q_pred = self.model.forward(state_tensor[i])[torch.argmax(action_tensor[i])] # current q val est for action taken
        
        # if(len(q_pred.shape) == 2):
        #   q_pred = q_pred[0]
        # train_next_argmax = torch.argmax(self.model.forward(new_state_tensor[i]))
        with torch.no_grad():
          # q_target = q_pred.clone()
          
          if done[i]: 
            q_next = torch.zeros(1) # no next state
          else: 
            # next_state_q_curr = self.model.forward(new_state_tensor[i])
            # next_state_action = torch.argmax(next_state_q_curr).item()
            next_state_q = self.model_target.forward(new_state_tensor[i])
            q_next = torch.max(next_state_q)#.max(dim=1)[0]#[0][train_next_argmax] # .max(dim=1)[0] -> get max value
          q_target = reward_tensor[i] + self.gamma*q_next
          # q_target[torch.argmax(action_tensor[i])] = reward_tensor[i] + self.gamma*q_next
          
        # loss = (q_pred - q_target.detach()).pow(2).mean()
        
        self.optimizer.zero_grad()
        loss = self.loss_function(q_pred, q_target.detach())
        loss.backward()
        self.optimizer.step()
        
        for model_param, target_param in zip(self.model.parameters(), self.model_target.parameters()):
            target_param.data.copy_((1 - self.lamda) * target_param.data + self.lamda * model_param.data)

