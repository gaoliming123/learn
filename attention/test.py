import pdb
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from data.data_loader import Dataset
from models.encoder import EncoderRNN
from models.decoder import DecoderRNN
from _utils.transformer import reformat_tensor_mask
from _utils.transformer import SentenceFromTensor_
from tensorboardX import SummaryWriter

from config import *

# load test dataset
testset = Dataset(phase='test', max_input_length = MAX_LENGTH, auto_encoder = AUTO_ENCODER)
testloader = torch.utils.data.DataLoader(testset, batch_size = 1,
                                          shuffle = True, num_workers = 1, pin_memory = False, drop_last = True)
input_lang, output_lang = testset.langs()

def showAttention(input_sentence, output_words, attentions):
    """
    Func:
        visulize attention
    """
    fig = plt.figure()
    # 111, 前两个1表示1 * 1的网格, 最后一个1表示最后一个子图
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap = 'bone')
    fig.colorbar(cax)

    ax.set_xticklabels([''] + input_sentence.split(' '), rotation = 90)
    ax.set_yticklabels([''] + output_words)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def evaluate(encoder, decoder, input_tensor, max_length = MAX_LENGTH):
    """
    Func:
        预测句子
    """

    with torch.no_grad():

        # Initialize the encoder hidden.
        input_length    = input_tensor.size(0)
        encoder_hidden  = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input      = torch.tensor([SOS_token])
        decoder_hidden     = encoder_hidden 
        decoded_words      = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            with SummaryWriter(comment = 'decoder') as writer:
                writer.add_graph(decoder, (decoder_input, decoder_hidden, encoder_outputs, ))

            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append(EOS_word)
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[: di + 1]

def evaluateRandomly(encoder, decoder, n = 10):
    for i in range(n):
        pair          = testset[i]['sentence']
        input_tensor  = pair[:, 0, :].view(1, 1, -1)
        input_tensor  = input_tensor[input_tensor != 0]
        output_tensor = pair[:, 1, :].view(1, 1, -1)
        output_tensor = output_tensor[output_tensor != 0]

        input_sentence  = ' '.join(SentenceFromTensor_(input_lang, input_tensor))
        output_sentence = ' '.join(SentenceFromTensor_(output_lang, output_tensor))
        print ('%s input sentence: %s' % (input_lang.name, input_sentence))
        print ('%s output sentence: %s' % (output_lang.name, output_sentence))
        output_words, attentions = evaluate(encoder, decoder, input_tensor)
        output_sentence = ' '.join(output_words)
        print ('%s predicted output sentence: %s' % (output_lang.name, output_sentence))
        print ('attentions')
        showAttention(input_sentence, output_words, attentions)

def showModelParam(encoder, decoder):
    """
    Func:
        show trained weights, encoder and decoder
    """
    print ('----------------------------------encoder params-------------------------------------')
    encoder_params = encoder.named_parameters()
    for name, param in encoder_params:
        print (name, param)

    print ('----------------------------------decoder params-------------------------------------')
    decoder_params = decoder.named_parameters()
    for name, param in decoder_params:
        print (name, param)


encoder = torch.load('./weights/encoder.model.pkl')
decoder = torch.load('./weights/decoder.model.pkl')

# showModelParam(encoder, decoder)
evaluateRandomly(encoder, decoder)
