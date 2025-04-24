import torch.nn as nn


class DiffusionModel(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, hidden_dim, 3, 1, 1), nn.GroupNorm(8, hidden_dim), nn.SiLU())
        self.down2 = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim*2, 3, 2, 1), nn.GroupNorm(8, hidden_dim*2), nn.SiLU())
        self.down3 = nn.Sequential(nn.Conv2d(hidden_dim*2, hidden_dim*4, 3, 2, 1), nn.GroupNorm(8, hidden_dim*4), nn.SiLU())
        
        self.mid = nn.Sequential(nn.Conv2d(hidden_dim*4, hidden_dim*4, 3, 1, 1), nn.GroupNorm(8, hidden_dim*4), nn.SiLU())
        
        self.up1 = nn.Sequential(nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, 3, 2, 1, 1), nn.GroupNorm(8, hidden_dim*2), nn.SiLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(hidden_dim*2, hidden_dim, 3, 2, 1, 1), nn.GroupNorm(8, hidden_dim), nn.SiLU())
        self.up3 = nn.Sequential(nn.Conv2d(hidden_dim, in_channels, 3, 1, 1))

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h1 = self.down1(x)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        h = self.mid(h3)
        h = self.up1(h + h3)
        h = self.up2(h + h2)
        h = self.up3(h + h1)
        return h