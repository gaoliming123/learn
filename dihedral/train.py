from dataloader import TrainDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import pdb

def train(dataloader, model, args, device, result ,valid_dataset):
    '''train the model'''
    current_learning_rate = args.learning_rate
    regularization        = args.regularization
    optimizer = torch.optim.Adagrad(model.parameters(), lr = current_learning_rate, weight_decay = 0)
    for epoch in range(args.epoch):

        if epoch % args.save_epoch == 0:
            print ('epoch: ' + str(epoch) + ' -> saving model')
            save_model(epoch, model, args)
            print ('epoch: ' + str(epoch) + ' -> evaluating model')
            test(model, result, args, epoch, valid_dataset, device)

        model.train()
        for step, (positive_sample, negative_sample) in enumerate(dataloader):
            optimizer.zero_grad()

            '''positive_sample size([1024, 3]) negative_sample size([1024, 10, 3])'''
            batch_size      = negative_sample.shape[0]
            negative_sample = negative_sample.view(args.negative_sample_size * batch_size, 3)
            positive_score  = model(positive_sample[:, 0], positive_sample[:, 1], positive_sample[:, 2])
            negative_score  = model(negative_sample[:, 0], negative_sample[:, 1], negative_sample[:, 2])
            positive_loss   = torch.sum(F.softplus(torch.tensor(-1.).to(device) * positive_score))
            negative_loss   = torch.sum(F.softplus(torch.tensor( 1.).to(device) * negative_score))
            loss            = positive_loss + negative_loss + args.regularization * model.l2_loss()

            loss.backward()
            optimizer.step()

            print ('epoch: ' + str(epoch) + ' and  step: ' + str(step) + ' -> loss: ' + str(loss.item()))


def test(model, result, args, epoch, valid_dataset, device):
    model.eval()
    for step, (positive_sample, negative_sample) in enumerate(valid_dataset):
        negative_sample.insert(0, positive_sample)
        negative_sample = torch.LongTensor(negative_sample).to(device)
        score  = model(negative_sample[:, 0], negative_sample[:, 1], negative_sample[:, 2])
        result.statistic(score)
    result.show(valid_dataset.len * 2)
    result.record(epoch)

def save_model(epoch, model, args):
    '''save model'''
    directory = './models/' + args.dataset + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model, directory + str(epoch) + '_checkpoint')
