import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingCNN(nn.Module):
  def __init__(self, input_depth, output_size): # 16x12
    super().__init__()
    
    self.conv = nn.Sequential( # (n-k)/s + 1
      nn.Conv2d(input_depth, 16, (5,5)), # 16x12x8
      nn.ReLU(),
      nn.BatchNorm2d(16),
      nn.Conv2d(16, out_channels=32, kernel_size=(3,3)), # 32x10x6
      nn.ReLU(),
      nn.BatchNorm2d(32),
      nn.Conv2d(32, 32, (3,3)), # 32x8x4
      nn.ReLU(),
      nn.BatchNorm2d(32),
      nn.Conv2d(32, 16, (3,3)), # 16x6x2
      nn.ReLU(),
    )
    
    self.value_stream = nn.Sequential(
      nn.Linear(16*6*2, 64),
      nn.ReLU(),
      nn.Linear(64, 1)
    )
    
    self.advantage_stream = nn.Sequential(
      nn.Linear(16*6*2, 64),
      nn.ReLU(),
      nn.Linear(64, 3)
    )

  def forward(self, x):
    x = x.to(torch.float)
    if len(x.shape) == 3:
        x = torch.unsqueeze(x, 0)

    x = self.conv(x)
    x = torch.flatten(x)
    values = self.value_stream(x)
    advantages = self.advantage_stream(x)

    q_vals = values + (advantages - advantages.mean())
    return q_vals
  