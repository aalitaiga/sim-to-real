from torch import autograd


def makeIntoVariables(a, b, CUDA):
    x, y = autograd.Variable(a, requires_grad=False), \
           autograd.Variable(b, requires_grad=False )
    if CUDA:
        return x.cuda()[0], y.cuda()[0]
    return x[0], y[0]  # because we have minibatch_size=1

def makeDataSliceIntoVariables(dataslice, CUDA):
    return makeIntoVariables(
        dataslice["state_next_sim_joints"],
        dataslice["state_next_real_joints"],
        CUDA
    )


