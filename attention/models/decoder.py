import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

class DecoderRNN(nn.Module):
    """
    Func:
        decode the code produced by encdoer 
        first input token SOS, the end token EOS
    """
    def __init__(self, hidden_size, output_size, dropout_p = 0.1, max_length = MAX_LENGTH):
        """
        Func:
            init decoder
        Args:
            hidden_size: input and hidden status size
            output_size: output size
            dropout: dropout
            max_length: the max length sentence
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p   = dropout_p
        self.max_length  = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(self.hidden_size << 1, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size << 1, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(input_size = hidden_size, hidden_size = hidden_size, num_layers = 1)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        """
        Func:
            1. embedded input word
            2. get the attention weights, formula from link https://arxiv.org/pdf/1409.0473.pdf
        """
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # concatnate the embedded and hidden
        # attn_weights for encoder output vectors to create a wieghted combination
        # linear [embedded; hidden] => max_length
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim = 1)
        # torch.bmm: batch matrix multiply
        # 多维矩阵的乘法, [10, 3, 4] * [10, 4, 5] => [10, 3, 5]
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim = 1)

        return output, hidden, attn_weights

    def initHidden(self):
        """
        Func:
            init the hidden status (zeros)
        """
        return torch.zeros(1, 1, self.hidden_size)
