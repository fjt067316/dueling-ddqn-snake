
import torch
from Agent import *
from Snake import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float)

score_history = []
mean_scores = []
agent = Agent()
agent.model.load_state_dict(torch.load("conv_net_77000.pth"))
# agent.trainer.optimizer.load_state_dict(torch.load("opt50k.pth"))
# agent.update_target_model()

def train():
  total_score = 0
  best_score = 0
    
  # past_frame_buf = np.array(None)#deque(maxlen=3)

  game = SnakeGame()

  # train for 200 games
  while agent.n_games < 200_000:

    old_state = agent.get_state(game)
    move = agent.get_action(old_state)

    reward, done, score = game.playStep(move) # reward = 10 per food snake eats, -10 if taking too long or collision

    new_state = agent.get_state(game)
    
    if done:

      game.reset()
 
        
train()