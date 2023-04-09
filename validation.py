#validation
import numpy as np
from loss import CrossEntropyLoss

def validation(iter_,model):
    accuracy = 0
    l = []
    for x,y in iter_:
        x = np.asarray(x)
        y = np.asarray(y)
        y_hat = model(x)
        loss = CrossEntropyLoss(y_hat,y)
        accuracy += (np.argmax(y_hat,axis=1)==y).sum()
        l.append(loss.item())
    return accuracy/len(iter_), np.mean(l)
