import torch
from models.linear import Linear 
from models.encoder import EncoderRNN
from models.decoder import DecoderRNN
from data.data_loader import Dataset
from _utils.transformer import reformat_tensor_mask
from _utils.transformer import SentenceFromTensor_

from config import *

# load test dataset
testset = Dataset(phase='test', max_input_length = MAX_LENGTH, auto_encoder = AUTO_ENCODER)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=True, num_workers=1, pin_memory=False, drop_last=True)
input_lang, output_lang = testset.langs()

def evaluate(encoder, decoder, bridge, input_tensor, max_length=MAX_LENGTH):

    with torch.no_grad():

        # Initialize the encoder hidden.
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden()

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)

        # only return the hidden and cell states for the last layer and pass it to the decoder
        hn, cn = encoder_hidden
        encoder_hn_last_layer = hn[-1].view(1, 1, -1)
        encoder_cn_last_layer = cn[-1].view(1, 1, -1)
        encoder_hidden_last = [encoder_hn_last_layer, encoder_cn_last_layer]

        decoder_input = torch.tensor([SOS_token], device = device)
        encoder_hidden_last = [bridge(item) for item in encoder_hidden_last]
        decoder_hidden = encoder_hidden_last

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append(EOS_word)
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

def evaluateRandomly(encoder, decoder, bridge, n=10):
    for i in range(n):
        pair = testset[i]['sentence']
        input_tensor, mask_input = reformat_tensor_mask(pair[:,0,:].view(1,1,-1))
        input_tensor = input_tensor[input_tensor != 0]
        output_tensor, mask_output = reformat_tensor_mask(pair[:,1,:].view(1,1,-1))
        output_tensor = output_tensor[output_tensor != 0]
        if device == torch.device("cuda"):
            input_tensor = input_tensor.cuda()
            output_tensor = output_tensor.cuda()

        input_sentence = ' '.join(SentenceFromTensor_(input_lang, input_tensor))
        output_sentence = ' '.join(SentenceFromTensor_(output_lang, output_tensor))
        print('Input: ', input_sentence)
        print('Output: ', output_sentence)
        output_words = evaluate(encoder, decoder, bridge, input_tensor)
        output_sentence = ' '.join(output_words)
        print('Predicted Output: ', output_sentence)
        print('')

encoder = torch.load('./weights/encoder.model.pkl')
decoder = torch.load('./weights/decoder.model.pkl')
bridge = torch.load('./weights/bridge.model.pkl')
evaluateRandomly(encoder, decoder, bridge)
