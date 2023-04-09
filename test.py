import numpy as np
from mlxtend.data import loadlocal_mnist
from Mydataloader import MydataLoader
from model import twolayers_mlp 

x_test, y_test = loadlocal_mnist(
    images_path="/Users/djy/Desktop/研究生/课程/神经网络和深度学习/hw1/我的/mnist/t10k-images-idx3-ubyte",
    labels_path="/Users/djy/Desktop/研究生/课程/神经网络和深度学习/hw1/我的/mnist/t10k-labels-idx1-ubyte")



            
def normalize(x):
    m = np.mean(x,axis=0,keepdims=True)
    var = np.var(x,axis=0,keepdims=True)
    return (x-m)/np.sqrt(var+1e-05)
x_test = normalize(x_test)

input_size = 28*28
hidden_size = 128
output_size = 10
batch_size = 64

test_iter = MydataLoader(x_test,y_test,batch_size) 
network = twolayers_mlp(input_size,hidden_size,output_size,lr = 0,l2 = 0)
param = np.load('/Users/djy/Desktop/研究生/课程/神经网络和深度学习/hw1/我的/mnist/mlp2-256hidden-0.1lr-0.0001l2.npy', allow_pickle=True)
network.load_model(param)

accuracy = 0
for x,y in test_iter:
    y_hat = network(x)
    accuracy += (np.argmax(y_hat,axis=1)==y).sum()
print('accuracy:',accuracy/len(test_iter))










