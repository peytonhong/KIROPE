import torch
from torch import nn
import matplotlib.pyplot as plt

# row_embed = torch.nn.Embedding(50, 256)
# col_embed = torch.nn.Embedding(50, 256)
# h, w = 20, 30
# i = torch.arange(w)
# j = torch.arange(h)
# x_embed = col_embed(i)
# y_embed = row_embed(j)
# print(x_embed.unsqueeze(0).repeat(h, 1, 1).shape)
# print(y_embed.unsqueeze(1).repeat(1, w, 1).shape)
# pos = torch.cat([x_embed.unsqueeze(0).repeat(h, 1, 1),
#                 y_embed.unsqueeze(1).repeat(1, w, 1),
#                 ], dim=-1)
# print(pos.permute(2,0,1).unsqueeze(0).repeat(4, 1, 1, 1).shape)

# x1 = x_embed[1].detach().numpy().reshape(16,16)

hidden_dim = 256
kp_prediction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), 2),
            nn.Sigmoid(),
        )
x = torch.rand(4, 7, 256)
y = kp_prediction(x)
print(y.shape)