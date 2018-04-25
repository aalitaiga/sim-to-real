from torch import nn, torch
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction
from torch.autograd import Variable

class Zoneout(InplaceFunction):

    @staticmethod
    def _make_mask(input):
        return input.new().resize_as_(input).expand_as(input)

    @classmethod
    def forward(cls, ctx, current_input, previous_input, p=None, mask=None,
                train=False, inplace=False):
        if p is None and mask is None:
            raise ValueError('Either p or mask must be provided')
        if p is not None and mask is not None:
            raise ValueError('Only one of p and mask can be provided')
        if p is not None and (p < 0 or p > 1):
            raise ValueError('zoneout probability has to be between 0 and 1, '
                             'but got {}'.format(p))
        if mask is not None and \
                not isinstance(mask, torch.ByteTensor) and \
                not isinstance(mask, torch.cuda.ByteTensor):
            raise ValueError("mask must be a ByteTensor")
        if current_input.size() != previous_input.size():
            raise ValueError(
                'Current and previous inputs must be of the same '
                'size, but current has size {current} and '
                'previous has size {previous}.'.format(
                    current='x'.join(str(size) for size in current_input.size()),
                    previous='x'.join(str(size) for size in previous_input.size()))
            )
        if type(current_input) != type(previous_input):
            raise ValueError('Current and previous inputs must be of the same '
                             'type, but current is {current} and previous is '
                             '{previous}'.format(current=type(current_input),
                                                 previous=type(previous_input))
                             )

        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(current_input)
            output = current_input
        else:
            output = current_input.clone()

        if not ctx.train:
            return output

        ctx.current_mask = cls._make_mask(current_input)
        ctx.previous_mask = cls._make_mask(current_input)
        if mask is None:
            ctx.current_mask.bernoulli_(1 - ctx.p)
        else:
            ctx.current_mask.fill_(0).masked_fill_(mask, 1)
        ctx.previous_mask.fill_(1).sub_(ctx.current_mask)
        output.mul_(ctx.current_mask).add_(previous_input.mul(ctx.previous_mask))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.train:
            return grad_output * Variable(ctx.current_mask), grad_output * Variable(ctx.previous_mask), \
                None, None, None, None
        else:
            return grad_output, None, None, None, None, None

class Zoneout(nn.Module):
    r"""During training of an RNN, randomly swaps some of the elements of the
    input tensor with its values from a previous time-step with probability *p*
    using samples from a Bernoulli distribution. The elements to be swapped are
    randomized on every time-step by default, but a shared mask can be
    provided.
    Zoneout is a variant of dropout designed specifically for regularizing
    recurrent connections of LSTMs or GRUs. While dropout applies a zero mask
    to its inputs, zoneout applies an identity mask when incrementing a
    time-step.
    It has proven to be an effective technique for regularization of LSTMs
    and GRUs as, contrary to dropout, gradient information and state
    information are more readily propagated through time. For further
    information, consult the paper
    `Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activation`_ .
    Similarly to dropout, during evaluation the module simply computes an
    identity function.
    Args:
        p: probability of an element to be zeroed. Default: None.
        inplace: If set to ``True``, will do this operation in-place.
        Default: ``False``
        mask: `ByteTensor`. A mask used to select elements to be swapped.
        The intended use case for this argument is sharing a zoneout mask
        across several time-steps.
    Shape:
        - Input: `Any`. A pair of tensors of the same shape
        - Output: `Same`. Output is of the same shape as input
    Examples::
        >>> zoneout = nn.Zoneout(p=0.15)
        >>> current_hidden_state = Variable(torch.Tensor([1, 2, 3]))
        >>> previous_hidden_state = Variable(torch.Tensor([4, 5, 6]))
        >>> output = zoneout(current_hidden_state, previous_hidden_state)
    Using a shared mask:
        >>> mask = torch.ByteTensor(1, 3).bernoulli()
        >>> zoneout = nn.Zoneout(mask=mask)
        >>> current_hidden_state = Variable(torch.Tensor([1, 2, 3]))
        >>> previous_hidden_state = Variable(torch.Tensor([4, 5, 6]))
        >>> output = zoneout(current_hidden_state, previous_hidden_state)
    Wrapping around a `GRUCell`:
        >>> rnn = nn.GRUCell(10, 20)
        >>> input = Variable(torch.randn(6, 3, 10))
        >>> h = Variable(torch.randn(3, 20))
        >>> h_prev = Variable(torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        ...     h = zoneout(h, h_prev)
        ...     h, h_prev = rnn(input[i], h_prev), h
        ...     output.append(h)
    .. _Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activation:
    https://arxiv.org/abs/1606.01305
    """

    def __init__(self, p=None, inplace=False, mask=None):
        super(Zoneout, self).__init__()
        if p is None and mask is None:
            raise ValueError("Either p or mask must be provided")
        if p is not None and mask is not None:
            raise ValueError("Only one of p and mask can be provided")
        if p is not None and (p < 0 or p > 1):
            raise ValueError("zoneout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        if mask is not None and \
                not isinstance(mask, torch.ByteTensor) and \
                not isinstance(mask, torch.cuda.ByteTensor):
            raise ValueError("mask must be a ByteTensor")
        self.p = p
        self.inplace = inplace
        self.mask = mask

    def forward(self, previous_input, current_input):
        return F.zoneout(previous_input, current_input, self.p, self.mask,
                         self.training, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        if self.mask is not None:
            mask_str = 'mask=ByteTensor of size ' + \
                       'x'.join(str(size) for size in self.mask.size())
        else:
            mask_str = 'p=' + str(self.p)
        return self.__class__.__name__ + '(' + mask_str + inplace_str + ')'

def zoneout(previous_input, current_input, p=None, mask=None, training=False,
            inplace=False):
    return Zoneout.apply(previous_input, current_input, p,
                          mask, training, inplace)
