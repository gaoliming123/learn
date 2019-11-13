import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    """
    Func:
        encode the sentence
    """
    def __init__(self, hidden_size, input_size, batch_size, num_layers = 1):
        """
        Func:
            init the encoder
        Args:
            hidden_size: hidden status size
            input_size: input size
            batch_size: mimi-batch optimization
            num_layers: layers
        """
        super(EncoderRNN, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        #embedding the input words
        self.embedding = nn.Embedding(input_size, embedding_dim=hidden_size)

        # LSTM
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)


    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, (h_n, c_n) = self.lstm(embedded, hidden)
        return output, (h_n, c_n)


    def initHidden(self):
        """
        Func:
            init the hidden status(zeros)
        """
        encoder_state = [torch.zeros(self.num_layers, 1, self.hidden_size),
                          torch.zeros(self.num_layers, 1, self.hidden_size)]
        return encoder_state

