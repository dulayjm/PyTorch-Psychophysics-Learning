import numpy as np
import torch

# reaction time psychophysical loss
def RtPsychCrossEntropyLoss(outputs, targets, psych):
    num_examples = targets.shape[0]
    batch_size = outputs.shape[0]

    # converting reaction time to penalty
    # 10002 is close to the max penalty time seen in the data
    for idx in range(len(psych)):   
        psych[idx] = abs(10002 - psych[idx])

    # adding penalty to each of the output logits 
    for i in range(len(outputs)):
        val = psych[i] / 300
        if np.isnan(val.cpu()):
            val = 0 
            
        outputs[i] += val 

    outputs = _log_softmax(outputs)
    outputs = outputs[range(batch_size), targets]

    return - torch.sum(outputs) / num_examples

# mean accuracy psychophysical loss
def AccPsychCrossEntropyLoss(outputs, targets, psych):
    num_examples = targets.shape[0]
    batch_size = outputs.shape[0]

    # converting accuracy to penalty
    for idx in range(len(psych)):   
        psych[idx] = abs(1 - psych[idx])

    for i in range(len(outputs)):
        outputs[i] += (psych[i])

    outputs = _log_softmax(outputs)
    outputs = outputs[range(batch_size), targets]

    return - torch.sum(outputs)/num_examples

def _softmax(x):
    exp_x = torch.exp(x)
    sum_x = torch.sum(exp_x, dim=1, keepdim=True)

    return exp_x/sum_x

def _log_softmax(x):
    return torch.log(_softmax(x))
