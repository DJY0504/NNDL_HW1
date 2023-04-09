#loss
class SoftmaxLossLayer(object):
    def __init__(self):
        pass
    def forward(self,input):
        input_max=np.max(input,axis=1,keepdims=True)
        input_exp=np.exp(input-input_max)
        self.prob=input_exp/np.sum(input_exp,axis=1,keepdims=True)
        return self.prob
    def get_loss(self,label):
        self.batch_size=self.prob.shape[0]
        self.label_onehot=np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size),label]=1.0
        loss = -np.sum(np.log(self.prob)*self.label_onehot/self.batch_size)
        return loss
    def backward(self):
        bottom_diff=(self.prob-self.label_onehot)/self.batch_size
        return bottom_diff
