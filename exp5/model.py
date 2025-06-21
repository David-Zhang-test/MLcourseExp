import torch.nn as nn
import torch.nn.functional as F
import torch

class SurnameModel(nn.Module):
    def __init__(self, num_input_channels, num_output_channels, num_classes, dropout_p):
        super(SurnameModel, self).__init__()
        
        # Conv weights
        self.conv = nn.ModuleList([
            nn.Conv1d(num_input_channels, num_output_channels, kernel_size=f) 
            for f in [2, 3, 4]
        ])
        
        # BatchNorm layers (1 per Conv1d)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_output_channels)
            for _ in [2, 3, 4]
        ])
        
        self.dropout = nn.Dropout(dropout_p)
       
        # FC weights
        self.fc1 = nn.Linear(num_output_channels * 3, num_classes)

    def forward(self, x, channel_first=False, apply_softmax=False):
        if not channel_first:
            x = x.transpose(1, 2)  # (N, C, L)

        # Conv → BN → ReLU → MaxPool
        z = []
        for conv, bn in zip(self.conv, self.batch_norms):
            zz = conv(x)
            zz = bn(zz)
            zz = F.relu(zz)
            zz = F.max_pool1d(zz, zz.size(2)).squeeze(2)
            z.append(zz)

        z = torch.cat(z, 1)
        z = self.dropout(z)

        y_pred = self.fc1(z)
        
        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return y_pred
