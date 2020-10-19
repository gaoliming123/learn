import torch
import argparse
import torch.utils.data as Data

from dkste import DKSTE
from dataloader import TrainDataset
from dataloader import TestDataset
from train import train
from res import Result


def parse_args(args = None):
    parse = argparse.ArgumentParser()
    parse.add_argument('-lr', '--learning_rate',  help = 'learning rate', type = float)
    parse.add_argument('-d','--dim', help = 'relation or entitiy dim', type = int)
    parse.add_argument('-b', '--batch_size', help = 'batch size', type = int)
    parse.add_argument('-e', '--epoch', default = 1000, help = 'epoch', type = int)
    parse.add_argument('-neg', '--negative_sample_size',  default = 10, help = 'negative sample size', type = int)
    parse.add_argument('-data', '--dataset', default = 'wn18', help = 'dataset')
    parse.add_argument('-r', '--regularization', help = 'regularization', type = float)
    parse.add_argument('-c', '--cuda_index', help = 'cuda index')
    parse.add_argument('-se', '--save_epoch', default = 50, help = 'every save epoch', type = int)
    return parse.parse_args()

def main(args):
    device           = torch.device('cuda:' + args.cuda_index)
    train_dataset    = TrainDataset('./data/' + args.dataset + '/', args.negative_sample_size, device)
    valid_dataset    = TestDataset('./data/' + args.dataset + '/', device, 'valid')
    test_dataset     = TestDataset('./data/' + args.dataset + '/', device, 'test')
    dataloader       = Data.DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True)
    model            = DKSTE(train_dataset.nentity, train_dataset.nrelation, args.dim, device).to(device)
    result           = Result()
    '''train model'''
    train(dataloader, model, args, device, result, valid_dataset)
    '''test model'''
    best_model_name  = './models' + args.dataset + '/' + str(int(Result.BEST_EPOCH)) + '_checkpoint'
    test_model       = torch.load(best_model_name)
    test(test_model, result, args, Result.BEST_EPOCH, test_dataset, device)

if __name__ == '__main__':
    main(parse_args())

