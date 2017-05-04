from theano import tensor
from blocks.brick.base import lazy
from blocks.bricks.conv import Convolutional
from blocks.bricks.recurrent import LSTM, recurrent, BaseRecurrent
from blocks.utils import shared_floatx_nans, shared_floatx_zeros
from blocks.roles import add_role, WEIGHT, INITIAL_STATE


class ConvLSTM(LSTM):
    """ Convolutional LSTM """

    @lazy(allocation=['filter_size', 'num_filters', 'num_channels'])
    def __init__(self, dim, filter_size, num_filters, num_channels, batch_size=None,
        image_size=(None,None), step=(1,1), border_mode='valid', activation=None,
        gate_activation=None, **kwargs):
        self.dim = dim
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_channels = num_channels

        self.input_convolution = Convolutional(
            filter_size, 4*num_filters, num_channels, batch_size=batch_size,
            image_size=image_size, step=step, border_mode=border_mode,
            name='convolution_input'
        )
        self.state_convolution = Convolutional(
            num_filters, 4*num_filters, num_channels, batch_size=batch_size,
            image_size=image_size, step=step, border_mode=border_mode,
            name='convolution_state'
        )

        super(ConvLSTM, self).__init__(activation, gate_activation, **kwargs)
        self.children.extend([self.input_convolution, self.output_convolution])

    def _allocate(self):
        self.W_cell_to_in = shared_floatx_nans((self.dim,),
                                               name='W_cell_to_in')
        self.W_cell_to_forget = shared_floatx_nans((self.dim,),
                                                   name='W_cell_to_forget')
        self.W_cell_to_out = shared_floatx_nans((self.dim,),
                                                name='W_cell_to_out')
        # The underscore is required to prevent collision with
        # the `initial_state` application method
        self.initial_state_ = shared_floatx_zeros((self.dim,),
                                                  name="initial_state")
        self.initial_cells = shared_floatx_zeros((self.dim,),
                                                 name="initial_cells")
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
            return x[:, no*self.dim: (no+1)*self.dim, :, :]

        activation = self.state_convolution.apply(states) + self.input_convolution.apply(inputs)
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
