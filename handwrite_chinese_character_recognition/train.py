#!/usr/bin/env python
# coding=utf-8

import os
import cv2
import pdb
import torch
import struct
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from cnn import CnnHwdb
from dataset import DataSetHwdb

# some arguments
EPOCH      = 10
BATCH_SIZE = 50
LR         = 0.001

train_data_dir = './dataset/train'
test_data_dir  = './dataset/test'
# define string encode with unicode
recognize_str  = u'心中那自由的世界如此的清澈高原盛开着永不凋零蓝莲花'

# read handwrite chinese and label
def read_data(data_dir = train_data_dir):

    print ('read data...')
    # data format link: http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html
    def one_file(f):
        header_size = 10
        while True:
            header = np.fromfile(f, dtype = 'uint8', count = header_size)
            if not header.size: break
            # get data and tag
            sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
            tagcode     = header[5] + (header[4] << 8)
            width       = header[6] + (header[7] << 8)
            heigh       = header[8] + (header[9] << 8)
            if header_size + width * heigh != sample_size: break
            
            img = np.fromfile(f, dtype = 'uint8', count = width * heigh).reshape((heigh, width))
            yield img, tagcode
    
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(data_dir, file_name)
            with open(file_path, 'rb') as f:
                for img, tagcode in one_file(f):
                    # return the result but the function continue to execute
                    yield img, tagcode 


def convert_to_one_hot(char):
    """
    function: one hot encode to handwrite
    """
    vector = np.zeros(len(recognize_str))
    vector[recognize_str.index(char)] = 1
    return vector

def resize_and_normalize_image(img):
    """
    function: resize image to 64 * 64
    """
    img   = cv2.resize(img, (50, 50))
    img   = cv2.copyMakeBorder(img, 7, 7, 7, 7, cv2.BORDER_CONSTANT, value = [255, 255, 255])
    input = torch.FloatTensor(img) / 255

    input = input.unsqueeze(0).unsqueeze(0)

    return input

def test():
    # load test data and model
    cnn_gwdb_model = torch.load('cnn_hwdb_model.pkl')
    
    for img, tagcode in read_data(data_dir = test_data_dir):
        # use big-endian(>) unsigned short(H), encode with unicode
        tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
        if tagcode_unicode in recognize_str:
    
            predict_y = torch.max(cnn_gwdb_model(resize_and_normalize_image(img)), 1)[1].data.numpy().squeeze()
            real_y    = convert_to_one_hot(tagcode_unicode)
            print (predict_y, real_y)
            pdb.set_trace()


def train():
    """
    Function:
        trian the network
    """
    # the cnn network information
    cnn           = CnnHwdb()
    train_dataset = DataSetHwdb(train_data_dir)
    train_loader  = Data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    optimizer     = torch.optim.Adam(cnn.parameters(), lr = LR)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, (train_x, train_y) in enumerate(train_loader):  # itertor every batch
            output = cnn(train_x)
            loss   = loss_function(output, train_y)
            optimizer.zero_grad()   # clear this step gradient
            loss.backward()         # backpropagation, compute gradient
            optimizer.step()        # apply gradient
            
    # save model
    torch.save(cnn, 'cnn_hwdb_model.pkl')
    print ('save model success')

# train the model
train()
test()

