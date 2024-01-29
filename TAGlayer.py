import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np



class Convs(nn.Module):
    def __init__(self, emb_dim):
        super(Convs, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=emb_dim, out_channels=1280, kernel_size=(3, 3), stride=(3, 3))
        self.BR1 = nn.Sequential(nn.BatchNorm2d(1280), nn.PReLU())
        self.conv2 = nn.Conv2d(in_channels=1280, out_channels=128, kernel_size=(3, 3), stride=(3, 3))
        self.BR2 = nn.Sequential(nn.BatchNorm2d(128), nn.PReLU())
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features = self.conv1(x)
        features = self.BR1(features)
        features = self.conv2(features)
        features = self.BR2(features)
        features = self.pool(features)
#        features = features.view(-1, 64)
        features = torch.flatten(features, start_dim=1)
        return features


class KSGPPI(torch.nn.Module):
    def __init__(self, args):
        super(KSGPPI, self).__init__()
        self.embedding_size = 5120
        self.drop = 0.2
        self.Convs2 = Convs(self.embedding_size)
        self.fc_seqg = torch.nn.Linear(228, 128)

        self.fc1 = nn.Linear(128 * 2, 64)
        self.fc2 = nn.Linear(64, 16)
        self.out = nn.Linear(16, 1)
        self.dropout = nn.Dropout(self.drop)
        self.Prelu = nn.PReLU()

    def forward(self, g1, g2, pad_dmap1, pad_dmap2):
        seq1 = self.Convs2(pad_dmap1)
        gs1 = torch.cat([seq1, g1], dim=1)
        seq1 = self.Prelu(self.fc_seqg(gs1))

        seq2 = self.Convs2(pad_dmap2)
        gs2 = torch.cat([seq2, g2], dim=1)
        seq2 = self.Prelu(self.fc_seqg(gs2))

        seq = torch.cat([seq1, seq2], dim=1)
        seq = self.fc1(seq)
        seq = self.Prelu(seq)
        seq = self.dropout(seq)
        seq = self.fc2(seq)
        seq = self.Prelu(seq)
        seq = self.dropout(seq)
        out = self.out(seq)

        output = F.sigmoid(out)

        return output