import numpy as np
import torch.nn as nn
import torch

class fund_GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(fund_GRU, self).__init__()

        self.hidden_size = hidden_size

        self.daily_rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.weekly_rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.monthly_rnn = nn.GRU(input_size, hidden_size, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, daily_input, weekly_input, monthly_input):
        # packed_input = nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first=True)
        out, daily_hidden = self.daily_rnn(daily_input, self.daily_hidden)
        out, weekly_hidden = self.weekly_rnn(weekly_input, self.weekly_hidden)
        out, monthly_hidden = self.monthly_rnn(monthly_input, self.monthly_hidden)
        merged_hidden = torch.cat((daily_hidden[0], weekly_hidden[0], monthly_hidden[0]), 1)
        # logits = self.classifier(hidden[0].view(hidden[0].size(0), -1))
        logits = self.classifier(merged_hidden)

        return logits

    def init_hidden(self, curr_batch):
        self.daily_hidden = torch.zeros(1, curr_batch, self.hidden_size).to('cuda:0')
        self.weekly_hidden = torch.zeros(1, curr_batch, self.hidden_size).to('cuda:0')
        self.monthly_hidden = torch.zeros(1, curr_batch, self.hidden_size).to('cuda:0')


class DefaultPredict:
    def __init__(self, code: str):
        pass


