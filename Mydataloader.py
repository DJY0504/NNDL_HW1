#Mydataloader
import numpy as np
class Mydataloader(object):
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.len = len(y)
        self.batch_size = batch_size
        self.arr = np.array(range(len(y)))
    
    def __iter__(self):
        self.num = 0
        self.seq = np.random.permutation(self.arr)
        return self
    
    def __next__(self):
        self.num += self.batch_size
        if self.num <= self.len:
            sample = self.seq[self.num-self.batch_size:(self.num)]
            self.data = self.x[sample]
            self.label = self.y[sample]
            return self.data, self.label
        else:
            raise StopIteration
    def __len__(self):
        return self.len
