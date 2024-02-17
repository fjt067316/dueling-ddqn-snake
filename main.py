
import torch
from Agent import *
from Snake import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float)

score_history = []
mean_scores = []
agent = Agent()
agent.model.load_state_dict(torch.load("50k.pth"))
agent.trainer.optimizer.load_state_dict(torch.load("opt50k.pth"))
agent.update_target_model()

# agent.load_user_training_data()

# for _ in range(5):
#   agent.train_long_memory()
def save_reward_stat(reward):
  with open('rewards.txt', 'a') as file:
    line_to_write = str(reward) + '\n'
    file.write(line_to_write)

def train():
  total_score = 0
  best_score = 0
    
  # past_frame_buf = np.array(None)#deque(maxlen=3)

  game = SnakeGame()

  # train for 200 games
  while agent.n_games < 200_000:

    old_state = agent.get_state(game)

    # if past_frame_buf.size == 3: 
    #     past_frame_buf[2] = past_frame_buf[1]
    #     past_frame_buf[1] = past_frame_buf[0]
    #     past_frame_buf[0] = old_state
    # else:
    #     # past_frame_buf = np.array([old_state] * 3)
    #     past_frame_buf = torch.tensor((old_state))
    #     past_frame_buf = past_frame_buf.repeat((3,1,1))

    # old_state = past_frame_buf.clone() # past 3 frames in last state ie depth of 3

    move = agent.get_action(old_state)

    reward, done, score = game.playStep(move) # reward = 10 per food snake eats, -10 if taking too long or collision

    new_state = agent.get_state(game)
    
    # past_frame_buf[2] = past_frame_buf[1]
    # past_frame_buf[1] = past_frame_buf[0]
    # past_frame_buf[0] = new_state

    # new_state = past_frame_buf.clone()

    # agent.train_single_step(old_state, move, reward, new_state, done) # new_state = past_frame_buf
    # print(old_state.shape)
    # print(new_state.shape)
    agent.add_memory(old_state, move, reward, new_state, done)
    # agent.train_single_step(old_state, move, reward, new_state, done)

    if done:
      # train longterm memory
      # agent.update_memory_episode_rewards()
      game.reset()
    #   past_frame_buf = np.array(None)
      agent.n_games += 1
      agent.train_long_memory()
      
      # if  agent.n_games % 100 == 0: agent.update_target_model()

      if score > best_score:
        best_score = score

      total_score += score
      # mean_score = total_score / agent.n_games

      # score_history.append(score)
      # mean_scores.append(mean_score)

      if score == best_score or agent.n_games % 100 == 0:
        if agent.n_games % 100 ==0: agent.epsilon = max(agent.epsilon*0.99, 0.1)
        if agent.n_games % 1000 == 0: 
            torch.save(agent.model.state_dict(), f"conv_net_{agent.n_games}.pth")
            torch.save(agent.trainer.optimizer.state_dict(), f"opt_{agent.n_games}.pth")
            # score_history.clear()
            # mean_scores.clear()
        print("Game number: ", agent.n_games, "Score: ", score, "Best Score: ", best_score, "Last 100 Avg: ", total_score/100)

        if agent.n_games % 100 == 0: 
          save_reward_stat(total_score/100)
          total_score = 0
        
train()