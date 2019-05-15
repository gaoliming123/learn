#!/usr/bin/env python
# coding=utf-8

"""
Module:   cnn_gwdb_dataset.py
function: custom dataset class of hwdb extends DataSet
source:   https://github.com/utkuozbulak/pytorch-custom-dataset-examples
"""

import os
import cv2
import pdb
import torch 
import struct
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


class DataSetHwdb(Dataset):

    def __init__(self, data_path, transformation = None):
        """
        arguments:
            data_path (string): filepath to data 
            heigh (int): image heigh
            width (int):  image width
            transforms: pytorch transforms for transforms and tensor conversion
        """
        self.heigh         = 64 
        self.width         = 64 
        self.data          = []
        self.labels        = []
        # encode string with unicode
        self.recognize_str = u'心中那自由的世界如此的清澈高原盛开着永不凋零蓝莲花'
        self.len           = 0
        self.transforms    = transformation if transformation is not None else transforms.Compose([transforms.ToTensor()])
        self.load_data(data_path)
        

    def __getitem__(self, index):
        """
        function: return item of data as tensor
        """
        # read the image label and get it
        vector_image_label = self.labels[index]
        label = vector_image_label.tolist().index(1.)

        # read the image
        img_as_array       = self.data[index]

        # transform image and label to tensor
        img_as_tensor   = self.transforms(img_as_array)

        return (img_as_tensor, label)


    def __len__(self):
        """
        function: return the len of data
        """
        return self.len
        

    # read handwrite chinese and label
    def read_data_from_file(self, data_dir):
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


    def convert_to_one_hot(self, char):
        """
        function: one hot encode to handwrite
        """
        vector = np.zeros(len(self.recognize_str))
        vector[self.recognize_str.index(char)] = 1
        return vector


    def resize_and_normalize_image(self, img):
        """
        function: resize image to 64 * 64
        """
        img = cv2.resize(img, (50, 50))
        img = cv2.copyMakeBorder(img, 7, 7, 7, 7, cv2.BORDER_CONSTANT, value = [255, 255, 255])

        return img

    def load_data(self, filepath):
        """
        function: read the data form filepath
        """
        for img, tagcode in self.read_data_from_file(filepath):
            # use big-endian(>) unsigned short(H) , encode with unicode
            tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
            
            if tagcode_unicode in self.recognize_str:
                self.data.append(self.resize_and_normalize_image(img))
                self.labels.append(self.convert_to_one_hot(tagcode_unicode))
                self.len = self.len + 1

