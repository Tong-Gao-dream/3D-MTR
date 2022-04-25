from torch.autograd import Variable

def var_or_cuda(x):

    x = x.cuda()
    return Variable(x)
