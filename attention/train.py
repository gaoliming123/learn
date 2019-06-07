import pdb
import torch
import torch.utils.data
import torch.nn as nn
from torch import optim
from visdom import Visdom
from data.data_loader import Dataset
from models.encoder import EncoderRNN
from models.decoder import DecoderRNN

from config import *

# vistual loss function
viz    = Visdom()

# down load trainset
trainset = Dataset(phase='train', max_input_length = MAX_LENGTH, auto_encoder = AUTO_ENCODER)
input_lang, output_lang = trainset.langs()
trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, 
                                          pin_memory = False, drop_last = True)
dataiter = iter(trainloader)

def train(input_tensor, target_tensor, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion, max_length = MAX_LENGTH):
    """
    Func:
        train encoder, decoder
    """
    # init args
    encoder_hidden  = encoder.initHidden()
    target_length   = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
    # 输入数据压缩一维
    # 矩阵转置(实际的存储方式是[batch_size * sentence]而是[sentence * batch_size])
    input  = torch.squeeze(input_tensor)
    input  = input.transpose(0 ,1)
    target = torch.squeeze(target_tensor)
    target = target.transpose(0, 1)
    # recode total loss
    total_loss = 0
    
    # encoder
    for idx in range(BATCH_SIZE):
        # 每一个句子
        input_tensor_step = input[:, idx][input[:, idx] != 0]
        input_length = input_tensor_step.size(0)

        target_tensor_step = target[:, idx][target[:, idx] != 0]
        target_length = target_tensor_step.size(0)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor_step[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
    
        # 输入的第一个单词: SOS_token 
        decoder_input  = torch.tensor([SOS_token])
        # encoder hidden 输入的第一个状态是encoder的最后一个状态
        decoder_hidden = encoder_hidden
        loss           = 0

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, 
                                                                        decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            if decoder_input.item() == EOS_token:
                break
            
            loss += criterion(decoder_output, target_tensor_step[di].view(1))

        # update weights
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward(retain_graph = True)
        encoder_optimizer.step()
        decoder_optimizer.step()
        total_loss += loss

    total_loss /= BATCH_SIZE
    return total_loss.item()

def trainIters(encoder, decoder, print_every = 1000, learning_rate = 0.01):

    encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss()
    n_iters_per_epoch = int (len(trainset) / BATCH_SIZE)

    for i in range(EPOCH):
        for iteration, data in enumerate(trainloader, 1):
            training_pair = data

            input_tensor = training_pair['sentence'][:,:,0,:]
            target_tensor = training_pair['sentence'][:,:,1,:]

            loss = train(input_tensor, target_tensor, encoder, decoder, 
                         encoder_optimizer, decoder_optimizer, criterion)
            
            viz.line([loss], [iteration + i * n_iters_per_epoch], win = 'train_loss', update = 'append')
            if ((iteration + i * n_iters_per_epoch) % print_every == 0):
                print ('the step %d loss is %.4f' % (iteration + i * n_iters_per_epoch, loss))

        print ('Finished epoch %d of %d' % (i + 1, EPOCH))


viz.line([0.], [0.], win = 'train_loss', opts = dict(title = 'train loss'))
encoder = EncoderRNN(HIDDEN_SIZE_ENCODER, input_lang.n_words, num_layers = NUM_LAYER_ENCODER).to(device)
decoder = DecoderRNN(HIDDEN_SIZE_DECODER, output_lang.n_words, dropout_p = 0.1).to(device)
trainIters(encoder, decoder, print_every = 10)

print ('save model...')
torch.save(encoder, './weights/encoder.model.pkl')
torch.save(decoder, './weights/decoder.model.pkl')

