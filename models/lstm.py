from blocks.bricks import application, lazy, LinearLike, Feedforward
from blocks.bricks.conv import Convolutional, ConvolutionalTranspose
from blocks.bricks.recurrent import LSTM, recurrent
from blocks.bricks.wrappers import WithExtraDims
from blocks.utils import shared_floatx_nans, shared_floatx_zeros
from blocks.roles import add_role, WEIGHT, INITIAL_STATE, BIAS
from theano import tensor

class ConvLSTM(LSTM):
    """ Convolutional LSTM """

    @lazy(allocation=['filter_size', 'num_filters', 'num_channels'])
    def __init__(self, filter_size, num_filters, num_channels, batch_size=None,
        image_size=(None,None), step=(1,1), border_mode='valid', tied_biases=None,
        activation=None, gate_activation=None, convolution_type='conv', **kwargs):
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.image_size = image_size
        self.batch_size = batch_size
        self.border_mode = border_mode
        self.tied_biases = tied_biases
        self.feature_map_size = tuple(int(i/2) for i in image_size)

        if convolution_type == 'conv':
            conv = Convolutional
            add_kwargs = {}
        elif convolution_type == 'deconv':
            conv = ConvolutionalTranspose
            add_kwargs = {}#{'original_image_size': (2,2)}
        else:
            raise ValueError

        self.input_convolution = conv(
            filter_size, 4*num_filters, num_channels,
            batch_size=batch_size, image_size=image_size,
            step=step, border_mode=border_mode, tied_biases=tied_biases,
            name='convolution_input', **add_kwargs
        )
        self.state_convolution = conv(
            filter_size, 4*num_filters, num_filters,
            batch_size=batch_size, image_size=self.feature_map_size,
            border_mode=border_mode, tied_biases=tied_biases,
            name='convolution_state', **add_kwargs
        )

        super(ConvLSTM, self).__init__(self.num_filters, activation, gate_activation, **kwargs)
        self.children.extend([self.input_convolution, self.state_convolution])

    def push_allocation_config(self):
        self._push_allocation_config()
        self.allocation_config_pushed = True

        self.input_convolution.num_channels = self.num_channels
        self.state_convolution.num_channels = self.num_filters
        self.input_convolution.image_size = self.image_size
        self.state_convolution.image_size = self.feature_map_size

        for layer in [self.input_convolution, self.state_convolution]:
            layer.num_filters = 4*self.num_filters
            layer.filter_size = self.filter_size
            layer.batch_size = self.batch_size
            layer.tied_biases = self.tied_biases
            layer.push_allocation_config()

    def _allocate(self):
        self.W_cell_to_in = shared_floatx_nans(
            (self.num_filters,) + self.feature_map_size, name='W_cell_to_in'
        )
        self.W_cell_to_forget = shared_floatx_nans(
            (self.num_filters,) + self.feature_map_size, name='W_cell_to_forget'
        )
        self.W_cell_to_out = shared_floatx_nans(
            (self.num_filters,) + self.feature_map_size, name='W_cell_to_out'
        )
        # The underscore is required to prevent collision with
        # the `initial_state` application method
        self.initial_state_ = shared_floatx_zeros(
            (self.num_filters,) + self.feature_map_size, name="initial_state"
        )
        self.initial_cells = shared_floatx_zeros(
            (self.num_filters,) + self.feature_map_size, name="initial_cells"
        )
        add_role(self.W_cell_to_in, WEIGHT)
        add_role(self.W_cell_to_forget, WEIGHT)
        add_role(self.W_cell_to_out, WEIGHT)
        add_role(self.initial_state_, INITIAL_STATE)
        add_role(self.initial_cells, INITIAL_STATE)

        self.parameters.extend([
            self.W_cell_to_in, self.W_cell_to_forget,
            self.W_cell_to_out, self.initial_state_, self.initial_cells
        ])

    @recurrent(sequences=['inputs', 'mask'], states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells'])
    def apply(self, inputs, states, cells, mask=None):
        def slice_last(x, no):
            return x[:, no*self.num_filters: (no+1)*self.num_filters, :, :]

        activation = self.input_convolution.apply(inputs) + self.state_convolution.apply(states)
        in_gate = self.gate_activation.apply(
            slice_last(activation, 0) + cells * self.W_cell_to_in)
        forget_gate = self.gate_activation.apply(
            slice_last(activation, 1) + cells * self.W_cell_to_forget)
        next_cells = (
            forget_gate * cells +
            in_gate * self.activation.apply(slice_last(activation, 2)))
        out_gate = self.gate_activation.apply(
            slice_last(activation, 3) + next_cells * self.W_cell_to_out)
        next_states = out_gate * self.activation.apply(next_cells)

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
            next_cells = (mask[:, None] * next_cells +
                          (1 - mask[:, None]) * cells)

        return next_states, next_cells

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.initial_state_[None, :, :, :], batch_size, 0),
                tensor.repeat(self.initial_cells[None, :, :, :], batch_size, 0)]

# class LinearPlus(Linear)
#     decorators = [WithExtraDims()]

class Linear2(LinearLike, Feedforward):
    @lazy(allocation=['input_dim', 'output_dim'])
    def __init__(self, input_dim, output_dim, batch_size, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

    def _allocate(self):
        W = shared_floatx_nans((self.input_dim, self.output_dim), name='W')
        add_role(W, WEIGHT)
        self.parameters.append(W)
        self.add_auxiliary_variable(W.norm(2), name='W_norm')
        if getattr(self, 'use_bias', True):
            b = shared_floatx_nans((self.output_dim,), name='b')
            add_role(b, BIAS)
            self.parameters.append(b)
            self.add_auxiliary_variable(b.norm(2), name='b_norm')

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the linear transformation.
        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            The input on which to apply the transformation
        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            The transformed input plus optional bias
        """
        # From (time, batch, n_channel, 1, 1) to (batch, time, n_channel)
        input_ = input_.dimshuffle(1,0,2,3,4)
        input_ = input_.reshape([self.batch_size, self.time, -1])
        output = tensor.dot(input_, self.W)
        if getattr(self, 'use_bias', True):
            output += self.b
        return output

    def get_dim(self, name):
        if name == 'input_':
            return self.input_dim
        if name == 'output':
            return self.output_dim
        super(Linear, self).get_dim(name)
