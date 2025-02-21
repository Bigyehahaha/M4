import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class MMoE(nn.Module):
    def __init__(self, feat_input, experts_out, towers_out, towers_hidden, tasks, num_expert):
        super(MMoE, self).__init__()
        self.feat_input = feat_input
        self.tasks = tasks
        self.experts_out = experts_out
        self.towers_hidden = towers_hidden
        self.towers_out = towers_out
        self.num_expert = num_expert
        #layer1 
        self.specific_experts1 = nn.ModuleList([AMIL(self.feat_input, is_expert = True) for i in range(self.num_expert)])

        self.gate1 = nn.ModuleList([nn.Sequential(nn.Linear(self.feat_input, self.num_expert), nn.Softmax(dim=-1)) for i in range(self.tasks)])

        #tower
        self.towers = nn.ModuleList([Tower(self.experts_out, self.towers_out, self.towers_hidden) for i in range(self.tasks)])
    def forward(self, x):
        #layer1
        expert_output = [self.specific_experts1[i](x_path=x) for i in range(self.num_expert)]
        expert_output = torch.stack(expert_output).squeeze(1)#10*256
        gate_out = [self.gate1[i](x.mean(dim=0)) for i in range(self.tasks)]
        tower_in = []
        for i in range(self.tasks):
            tower_in_i = torch.mm(gate_out[i].unsqueeze(0),expert_output)#[1,256]
            tower_in.append(tower_in_i)

        #task specific tower
        tower_in_stack = torch.stack(tower_in)#(self.tasks,1,256)
        tower_in_stack = tower_in_stack.squeeze(1)#(self.tasks,256)
        final_output = [t(ti) for t, ti in zip(self.towers, tower_in_stack)]
        return final_output



class AMIL(nn.Module):
    def __init__(self, feat_input=2048, size_arg = "small", n_classes=4, is_expert=False):
        super(AMIL, self).__init__()
        self.is_expert = is_expert
        self.size_dict = {"small": [feat_input, 512, 128], "big": [feat_input, 512, 384]}
        size = self.size_dict[size_arg]
        
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=0.25, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Linear(size[1], n_classes)
        initialize_weights(self)
                


    def forward(self, **kwargs):
        h = kwargs['x_path']
        A, h = self.attention_net(h)  
        A = torch.transpose(A, 1, 0)

        # if 'attention_only' in kwargs.keys():
        #     if kwargs['attention_only']:
        #         return A

        A_raw = A 
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h)
        
        if self.is_expert:
            return M
        else:
            h = self.classifier(M)
            h = self.softmax(h)
            return h

class Attn_Net_Gated(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
