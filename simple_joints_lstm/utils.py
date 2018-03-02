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

PUSHER3DOF_POS_MAX = 2.6
PUSHER3DOF_POS_MIN = -0.2
PUSHER3DOF_VEL_MAX = 12.5
PUSHER3DOF_VEL_MIN = -2.5
PUSHER3DOF_POS_DIFF = PUSHER3DOF_POS_MAX - PUSHER3DOF_POS_MIN
PUSHER3DOF_VEL_DIFF = PUSHER3DOF_VEL_MAX - PUSHER3DOF_VEL_MIN

def normalizePusher3Dof(state):
    ## measured:
    ## max_pos: 2.5162039
    ## min_pos: -0.1608184
    ## max_vel: 12.24464
    ## min_vel: -2.2767675

    state[:,:3] -= PUSHER3DOF_POS_MIN # add the minimum
    state[:,:3] /= PUSHER3DOF_POS_DIFF # divide by range to bring into [0,1]
    state[:,:3] *= 2 # double and
    state[:,:3] -= 1 # shift left by one to bring into range [-1,1]

    state[:,3:] -= PUSHER3DOF_VEL_MIN
    state[:,3:] /= PUSHER3DOF_VEL_DIFF
    state[:,3:] *= 2
    state[:,3:] -= 1

    return state


