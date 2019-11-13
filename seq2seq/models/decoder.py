import torch
import torch.nn as nn

class DecoderRNN(nn.Module):
    """
    Func:
        decode the code produced by encdoer 
        first input token SOS, the end token EOS
    """
    def __init__(self, hidden_size, output_size, batch_size, num_layers=1):
        """
        Func:
            init decoder
        Args:
            hidden_size: input and hidden status size
            output_size: output size
            num_layers: layers
            batch_size: the mini-batch optimization
        """
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output, (h_n, c_n) = self.lstm(output, hidden)
        output = self.out(output[0])
        return output, (h_n, c_n)

    def initHidden(self):
        """
        Func:
            init the hidden status (zeros)
        """
        return [torch.zeros(self.num_layers, 1, self.hidden_size),
                torch.zeros(self.num_layers, 1, self.hidden_size)]
