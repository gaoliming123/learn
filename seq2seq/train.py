import torch
import torch.utils.data
import torch.nn as nn
from torch import optim
from visdom import Visdom
from data.data_loader import Dataset
from models.encoder import EncoderRNN
from models.decoder import DecoderRNN
from models.linear import Linear 
from _utils.transformer import reformat_tensor_mask
from _utils.transformer import timeSince

from config import *

# vistual loss function
viz    = Visdom()

# down load trainset
trainset = Dataset(phase='train', max_input_length = MAX_LENGTH, auto_encoder = AUTO_ENCODER)
input_lang, output_lang = trainset.langs()
trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, 
                                          pin_memory = False, drop_last = True)
dataiter = iter(trainloader)

def train(input_tensor, target_tensor, mask_input, mask_target, encoder, decoder, bridge, 
          encoder_optimizer, decoder_optimizer, bridge_optimizer, criterion, max_length = MAX_LENGTH):

    # zero optimizer
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    bridge_optimizer.zero_grad()

    # save the the last of encoder hidden status
    encoder_hiddens_last = []
    loss = 0

    ### train encoder
    for step_idx in range(BATCH_SIZE):
        # must reset the hidden status
        encoder_hidden = encoder.initHidden()
        input_tensor_step = input_tensor[:, step_idx][input_tensor[:, step_idx] != 0]
        input_length = input_tensor_step.size(0)

        encoder_outputs = torch.zeros(BATCH_SIZE, max_length, encoder.hidden_size, device = device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder( input_tensor_step[ei], encoder_hidden)
            encoder_outputs[step_idx, ei, :] = encoder_output[0, 0]

        # only get the last hidden status
        hn, cn = encoder_hidden
        encoder_hn_last_layer = hn[-1].view(1,1,-1)
        encoder_cn_last_layer = cn[-1].view(1,1,-1)
        encoder_hidden = [encoder_hn_last_layer, encoder_cn_last_layer]

        # normalize the last hidden status to the decoder input hidden status
        encoder_hidden = [bridge(item) for item in encoder_hidden]
        encoder_hiddens_last.append(encoder_hidden)

    ### decoder
    decoder_input = torch.tensor([SOS_token], device=device)
    decoder_hiddens = encoder_hiddens_last

    for step_idx in range(BATCH_SIZE):
        target_tensor_step = target_tensor[:, step_idx][target_tensor[:, step_idx] != 0]
        target_length = target_tensor_step.size(0)
        decoder_hidden = decoder_hiddens[step_idx]

        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            # detach from calculation 
            decoder_input = topi.squeeze().detach()  

            loss += criterion(decoder_output, target_tensor_step[di].view(1))
            if decoder_input.item() == EOS_token:
                break

    loss = loss / BATCH_SIZE
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, bridge, print_every = 1000, learning_rate = 0.1):

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    bridge_optimizer = optim.SGD(bridge.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    n_iters_per_epoch = int(len(trainset) / BATCH_SIZE)
    for i in range(EPOCH):

        for iteration, data in enumerate(trainloader, 1):
            training_pair = data

            input_tensor = training_pair['sentence'][:,:,0,:]
            input_tensor, mask_input = reformat_tensor_mask(input_tensor)

            target_tensor = training_pair['sentence'][:,:,1,:]
            target_tensor, mask_target = reformat_tensor_mask(target_tensor)

            loss = train(input_tensor, target_tensor, mask_input, mask_target, encoder,
                         decoder, bridge, encoder_optimizer, decoder_optimizer, bridge_optimizer, criterion)
            
            viz.line([loss], [iteration + i * n_iters_per_epoch], win = 'train_loss', update = 'append')
            if ((iteration + i * n_iters_per_epoch) % print_every == 0):
                print ('the step %d loss is %.4f' % (iteration + i * n_iters_per_epoch, loss))


        print ('Finished epoch %d of %d' % (i + 1, EPOCH))


viz.line([0.], [0.], win = 'train_loss', opts = dict(title = 'train loss'))
encoder = EncoderRNN(HIDDEN_SIZE_ENCODER, input_lang.n_words, BATCH_SIZE, num_layers=NUM_LAYER_ENCODER).to(device)
bridge = Linear(HIDDEN_SIZE_ENCODER, HIDDEN_SIZE_DECODER).to(device)
decoder = DecoderRNN(HIDDEN_SIZE_DECODER, output_lang.n_words, BATCH_SIZE, num_layers=NUM_LAYER_DECODER).to(device)
trainIters(encoder, decoder, bridge, print_every = 10)

print ('save model...')
torch.save(encoder, './weights/encoder.model.pkl')
torch.save(decoder, './weights/decoder.model.pkl')
torch.save(bridge, './weights/bridge.model.pkl')

