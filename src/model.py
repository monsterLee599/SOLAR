import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")

class LinearSVM(nn.Module):
    """Support Vector Machine"""

    def __init__(self,input_size, output_size):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        h = self.fc(x.float())
        return h

class CNN(nn.Module):
        def __init__(self, embed_num, embed_dim=128, class_num=2, kernel_num=100, kernel_sizes=[3, 4, 5], dropout=0.5):
            super(CNN, self).__init__()

            V = embed_num
            D = embed_dim
            C = class_num
            Ci = 1
            Co = kernel_num
            Ks = kernel_sizes

            self.embed = nn.Embedding(V, D)
            self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(len(Ks) * Co, C)

        def conv_and_pool(self, x, conv):
            x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
            x = F.max_pool1d(x, x.size(2)).squeeze(2)
            return x

        def forward(self, x):
            x = self.embed(x)  # (N, W, D)
            if self.args.static:
                x = Variable(x)
            x = x.unsqueeze(1)  # (N, Ci, W, D)
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
            x = torch.cat(x, 1)
            x = self.dropout(x)  # (N, len(Ks)*Co)
            logit = self.fc1(x)  # (N, C)
            return logit

class lstm(nn.Module):
    def __init__(self):
        pass

    def forward(self,x):
        pass

