from blocks.bricks import application, lazy
from blocks.bricks.conv import Convolutional
from blocks.bricks.recurrent import LSTM, recurrent
from blocks.utils import shared_floatx_nans, shared_floatx_zeros
from blocks.roles import add_role, WEIGHT, INITIAL_STATE
from theano import tensor

class ConvLSTM(LSTM):
    """ Convolutional LSTM """

    @lazy(allocation=['filter_size', 'num_filters', 'num_channels'])
    def __init__(self, filter_size, num_filters, num_channels, batch_size=None,
        image_size=(None,None), step=(1,1), border_mode='valid', tied_biases=None,
        activation=None, gate_activation=None, **kwargs):
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.image_size = image_size
        self.batch_size = batch_size
        self.border_mode = border_mode
        self.tied_biases = tied_biases
        self.feature_map_size = tuple(int(i/2) for i in image_size)

        self.input_convolution = Convolutional(
            filter_size, 4*num_filters, num_channels,
            batch_size=batch_size, image_size=image_size,
            step=step, border_mode=border_mode, tied_biases=tied_biases,
            name='convolution_input'
        )
        self.state_convolution = Convolutional(
            filter_size, 4*num_filters, num_filters,
            batch_size=batch_size, image_size=self.feature_map_size,
            border_mode=border_mode, tied_biases=tied_biases,
            name='convolution_state'
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


# class StackedConvLSTM(BaseRecurrent):
#     """ Stacked convolutional lstm
#         Can be implemented easily with only ConvLSTM
#         However this class might be useful to add skip_connections
#     """
#
#     def __init__(self, dim, n_lstm, filter_size, num_filters, num_channels, batch_size=None,
#         image_size=(None,None), step=(1,1), border_mode='valid', activation=None,
#         gate_activation=None, skip_connection=False, **kwargs):
#
#         self.dim = dim
#         self.skip_connection = skip_connection
#         self.layers = []
#         for _ in range(n_lstm):
#             self.layers.append(
#                 ConvLSTM(dim, n_lstm, filter_size, num_filters, num_channels,
#                 batch_size=batch_size, image_size=image_size, step=step, border_mode=border_mode,
#                 activation=activation, gate_activation=gate_activation, **kwargs)
#             )
#         self.children = self.layers
#
#
#     @recurrent(sequences=['inputs', 'mask'], states=['states_{}'.format(i) for i in len(self.layers)] + ['cells'],
#                contexts=[], outputs=['states', 'cells'])
#     def apply(self, inputs, states, cells, mask=None):
#         pass



if __name__ == '__main__':
    # Exemple of StackedConvLSTM:

    # TODO: add real parameters to make the code run
    # first_lstm = ConvLSTM(dim, n_lstm, filter_size, num_filters, num_channels,
    # batch_size=batch_size, image_size=image_size, step=step, border_mode=border_mode,
    # activation=activation, gate_activation=gate_activation)
    # second_lstm = ConvLSTM(dim, n_lstm, filter_size, num_filters, num_channels,
    # batch_size=batch_size, image_size=image_size, step=step, border_mode=border_mode,
    # activation=activation, gate_activation=gate_activation, **kwargs)
    #
    # first_output = first_lstm.apply(input)
    # second_output = second_lstm.apply(first_output)
    print 'hello'
