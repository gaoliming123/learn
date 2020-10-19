import pdb
import random
import torch
from torch.utils.data import Dataset
import numpy as np

class TrainDataset(Dataset):
    def __init__(self, data_path, negative_sample_size, device):
        super(TrainDataset).__init__()
        self.data_path    = data_path
        self.device       = device
        self.entities2id  = self.read_entities(data_path + 'entities.dict')
        self.relations2id = self.read_relations(data_path + 'relations.dict')
        self.triples      = self.read_triple(data_path + 'train.txt')
        self.len          = len(self.triples)
        self.nentity      = len(self.entities2id)
        self.nrelation    = len(self.relations2id)
        self.negative_sample_size              = negative_sample_size
        self.all_true_head, self.all_true_tail = self.get_all_true_head_and_tail(data_path)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        positive_sample      = self.triples[idx]
        head, relation, tail = positive_sample
        negative_sample_list = []
        tmp_negative_sample  = []
        negative_sample_size = 0
        if random.random() < 0.5:
            '''replace head'''
            while negative_sample_size < self.negative_sample_size:
                negative_sample = np.random.randint(self.nentity, size = self.negative_sample_size)
                mask = np.in1d(
                        negative_sample,
                        self.all_true_head[(relation, tail)],
                        assume_unique = True,
                        invert = True
                    )
                negative_sample       = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                negative_sample_size += negative_sample.size
            negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
            [tmp_negative_sample.append((h, relation, tail)) for h in negative_sample]
        else:
            '''replace tail'''
            while negative_sample_size < self.negative_sample_size:
                negative_sample = np.random.randint(self.nentity, size = self.negative_sample_size)
                mask = np.in1d(
                        negative_sample,
                        self.all_true_tail[(head, relation)],
                        assume_unique = True,
                        invert = True
                    )
                negative_sample = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                negative_sample_size += negative_sample.size
            negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
            [tmp_negative_sample.append((head, relation, t)) for t in negative_sample]

        positive_sample     = torch.LongTensor(positive_sample)
        tmp_negative_sample = torch.from_numpy(np.array(tmp_negative_sample))
        return (positive_sample.to(self.device), tmp_negative_sample.to(self.device))

    def get_all_true_triples(self, dir_path):
        triples = []
        data = {part: self.read_triple(dir_path + part + '.txt') for part in ['train', 'test', 'valid']}
        triples = data['train'] + data['valid'] + data['test']
        return triples

    def get_all_true_head_and_tail(self, dir_path):
        triples   = self.get_all_true_triples(dir_path)
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail

    def read_triple(self, file_path):
        triples = []
        with open(file_path, 'r') as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                triples.append((self.entities2id[h], self.relations2id[r], self.entities2id[t]))
        return triples

    @staticmethod
    def read_relations(file_path):
        relations2id = dict()
        with open(file_path, 'r') as fin:
            for line in fin:
                rid, relation = line.strip().split('\t')
                relations2id[relation] = int(rid)
        return relations2id

    @staticmethod
    def read_entities(file_path):
        entities2id = dict()
        with open(file_path, 'r') as fin:
            for line in fin:
                eid, entity = line.strip().split('\t')
                entities2id[entity] = int(eid)
        return entities2id

class TestDataset(Dataset):
    def __init__(self, data_path, device, valid_or_test):
        super(TestDataset, self).__init__()
        self.data_path    = data_path
        self.device       = device
        self.entities2id  = TrainDataset.read_entities(data_path + 'entities.dict')
        self.relations2id = TrainDataset.read_relations(data_path + 'relations.dict')
        self.triples      = self.read_triple(data_path + valid_or_test + '.txt')
        self.len          = len(self.triples)
        self.nentity      = len(self.entities2id)
        self.nrelation    = len(self.relations2id)
        self.all_true_triples = set(self.get_all_true_triples(data_path))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample      = self.triples[idx]
        h, r, t              = positive_sample
        negative_sample      = []

        [negative_sample.append((h, r, tail)) if (h, r, tail) not in self.all_true_triples else tail for tail in range(self.nentity)]
        [negative_sample.append((head, r, t)) if (head, r, t) not in self.all_true_triples else head for head in range(self.nentity)]
        #positive_smaple = torch.tensor(np.array(positive_sample))
        #negative_smaple = torch.tensor(np.array(negative_sample))

        return (positive_sample, negative_sample)

    def read_triple(self, file_path):
        triples = []
        with open(file_path, 'r') as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                triples.append((self.entities2id[h], self.relations2id[r], self.entities2id[t]))
        return triples

    def get_all_true_triples(self, dir_path):
        triples = []
        data = {part: self.read_triple(dir_path + part + '.txt') for part in ['train', 'test', 'valid']}
        triples = data['train'] + data['valid'] + data['test']
        return triples
