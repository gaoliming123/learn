import unicodedata
import re
import random

import torch
import torch.utils.data
from _utils.transformer import tensorsFromPair

# define some args
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 10
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


# the decoder first input token
SOS_token = 1
# the token of the end sentence
EOS_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<pad>", SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    """
    Func:
        unicode to ascii
    Args:
        s: string
    Ref:
        http://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    """
    Func:
        Lowercase, trim, and remove non-letter characters
    Args:
        s: string
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub("[.!?]", '', s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)

    return s


def readLangs(lang1, lang2, auto_encoder = False, reverse = False):
    """
    Func:
        read the words from file
    Args:
        lang1: needed translate language
        lang2: translate to this language
        auto_encoder: if auto_encoder, the lang1 == lang2, so not translate
        reverse: reverse lang1 and lang2
    """
    print ('read lines form the file of ./data/eng-fra.txt, and encode with utf-8')
    lines = open('./data/eng-fra.txt', encoding='utf-8').read().strip().split('\n')

    print ('split %s  and %s for every lang' % (lang1, lang2))
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # autoencoder have the same data as the output
    if auto_encoder:
        pairs = [[pair[0], pair[0]] for pair in pairs]

    # reverse the language
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p, max_input_length):
    return len(p[0].split(' ')) < max_input_length and \
           len(p[1].split(' ')) < max_input_length and \
           p[1].startswith(eng_prefixes)

def filterPairs(pairs, max_input_length):
    pairs = [pair for pair in pairs if filterPair(pair, max_input_length)]
    return pairs


def prepareData(lang1, lang2, max_input_length, auto_encoder = False, reverse = False):
    """
    Func:
        read language from file and normalize the language 
    Args:
        lang1 (string):
        lang2 (string):
        max_input_length (int):
        auto_encoder (bool):
        reverse (bool):
    """
    input_lang, output_lang, pairs = readLangs(lang1, lang2, auto_encoder, reverse)
    pairs = filterPairs(pairs, max_input_length)
    print("过滤后训练集的样本数: %s " % len(pairs))
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print ("语言和单词总数的信息:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    print ('随机抽取的一个句子:')
    print(random.choice(pairs))

    return input_lang, output_lang, pairs


class Dataset():

    def __init__(self, phase, num_embeddings = None, max_input_length = None, 
                transform = None, auto_encoder = False):
        """
        Func:
            init the dataset
        Args:
            phase (string): 'train' or 'test'
            num_embeddings (int): the embedding dimentionality.
            max_input_length (int): max sentence length
            transform: 
            auto_encoder (bool): if auto_encoder
        """
        if auto_encoder:
            lang_in = 'eng'
            lang_out = 'eng'
        else:
            lang_in = 'eng'
            lang_out = 'fra'

        # prepare data
        input_lang, output_lang, pairs = prepareData(lang_in, lang_out, max_input_length,
                                                    auto_encoder = auto_encoder, reverse = True)

        # randomize list
        random.shuffle(pairs)

        # get different data
        if phase == 'train':
            selected_pairs = pairs[0:int(0.8 * len(pairs))]
        else:
            selected_pairs = pairs[int(0.8 * len(pairs)):]

        # getting the tensors
        selected_pairs_tensors = [tensorsFromPair(selected_pairs[i], input_lang, output_lang, max_input_length)
                     for i in range(len(selected_pairs))]

        self.transform = transform
        self.num_embeddings = num_embeddings
        self.data = selected_pairs_tensors
        self.input_lang = input_lang
        self.output_lang = output_lang

    def langs(self):
        """
        Func:
            get the language info
        """
        return self.input_lang, self.output_lang

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        pair = self.data[idx]
        sample = {'sentence': pair}
        if self.transform:
            sample = self.transform(sample)
        return sample


# # test
# trainset = Dataset(phase='train', max_input_length=10)
# 
# print ('loop the dataset')
# for i in range(len(trainset)):
#     sample = trainset[i]
# 
#     print(i, sample['sentence'].shape)
#     print(sample['sentence'][0])
#     break
