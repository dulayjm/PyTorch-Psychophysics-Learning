import torch

def PsychCrossEntropyLoss(outputs, targets, psych):
    num_examples = targets.shape[0]
    batch_size = outputs.shape[0]

    # converting reaction time to penalty
    # 30000 is close to the max penalty time seen in the data
    for idx in range(len(psych)):   
        psych[idx] = abs(30000 - psych[idx])

    # adding penalty to each of the output logits 
    # but it's too severe and outweighs the rest of the loss
    # scaling seems to somewhat work
    for i in range(len(outputs)):
        outputs[i] += (psych[i] / 1000)

    outputs = _log_softmax(outputs)
    outputs = outputs[range(batch_size), targets]

    return - torch.sum(outputs)/num_examples


def _softmax(x):
    exp_x = torch.exp(x)
    sum_x = torch.sum(exp_x, dim=1, keepdim=True)

    return exp_x/sum_x

def _log_softmax(x):
    return torch.log(_softmax(x))