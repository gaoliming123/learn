import logging

class Result():
    '''static variable'''
    BEST_EPOCH = 0
    BEST_MRR   = 0.0

    def __init__(self):
        self.h1    = 0.0
        self.h3    = 0.0
        self.h10   = 0.0
        self.mr    = 0.0
        self.mrr   = 0.0

    def statistic(self, score):
        rank = (score >= score[0]).sum()
        if rank == 1:
            self.h1 += 1
        elif rank <= 3:
            self.h3 += 1
        elif rank <= 10:
            self.h10 += 1
        self.mr  += rank
        self.mrr += (1. / rank)

    def show(self, num):
        print ('hits@1: ' + str(self.h1 / num))
        print ('hits@3: ' + str(self.h3 / num))
        print ('hits@10: ' + str(self.h10 / num))
        print ('MR: ' + str((self.mr / num).item()))
        print ('MRR: ' + str((self.mrr / num).item()))

    def record(self, epoch):
        if Result.BEST_MRR > self.mrr:
            Result.BEST_MRR   = self.mrr
            Result.BEST_EPOCH = epoch
        self.clear()

    def clear(self):
        self.h1  = 0.0
        self.h3  = 0.0
        self.h10 = 0.0
        self.mr  = 0.0
        self.mrr = 0.0

