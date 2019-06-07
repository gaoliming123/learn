import torch

# the first word token
SOS_token = 1
# the end word token
EOS_token = 2
EOS_word  = 'EOS'

device = torch.device("cpu")

# some training args
EPOCH = 10 
BATCH_SIZE = 32
AUTO_ENCODER = False
MAX_LENGTH = 10
HIDDEN_SIZE_ENCODER = 256
NUM_LAYER_ENCODER = 1 
HIDDEN_SIZE_DECODER = 256
NUM_LAYER_DECODER = 1
