from theano.tensor.nnet import conv2d
from theano.tensor.nnet.abstract_conv import (AbstractConv2d_gradInputs,
                                              get_conv_output_shape)
from theano import tensor as T

from blocks.bricks import LinearLike
from blocks.bricks.base import application, Brick, lazy
from blocks.initialization import Constant
from blocks.roles import add_role, FILTER, BIAS, WEIGHT
from blocks.utils import shared_floatx_nans


class Convolutional(LinearLike):
    """Performs a 2D convolution.
    Parameters
    ----------
    filter_size : tuple
        The height and width of the filter (also called *kernels*).
    num_filters : int
        Number of filters per channel.
    num_channels : int
        Number of input channels in the image. For the first layer this is
        normally 1 for grayscale images and 3 for color (RGB) images. For
        subsequent layers this is equal to the number of filters output by
        the previous convolutional layer. The filters are pooled over the
        channels.
    batch_size : int, optional
        Number of examples per batch. If given, this will be passed to
        Theano convolution operator, possibly resulting in faster
        execution.
    image_size : tuple, optional
        The height and width of the input (image or feature map). If given,
        this will be passed to the Theano convolution operator, resulting
        in possibly faster execution times.
    step : tuple, optional
        The step (or stride) with which to slide the filters over the
        image. Defaults to (1, 1).
    border_mode : {'valid', 'full'}, optional
        The border mode to use, see :func:`scipy.signal.convolve2d` for
        details. Defaults to 'valid'.
    tied_biases : bool
        Setting this to ``False`` will untie the biases, yielding a
        separate bias for every location at which the filter is applied.
        If ``True``, it indicates that the biases of every filter in this
        layer should be shared amongst all applications of that filter.
        Defaults to ``True``.
    """
    # Make it possible to override the implementation of conv2d that gets
    # used, i.e. to use theano.sandbox.cuda.dnn.dnn_conv directly in order
    # to leverage features not yet available in Theano's standard conv2d.
    # The function you override with here should accept at least the
    # input and the kernels as positionals, and the keyword arguments
    # input_shape, subsample, border_mode, and filter_shape. If some of
    # these are unsupported they should still be accepted and ignored,
    # e.g. with a wrapper function that swallows **kwargs.
    conv2d_impl = staticmethod(conv2d)

    # Used to override the output shape computation for a given value of
    # conv2d_impl. Should accept 4 positional arguments: the shape of an
    # image minibatch (with 4 elements: batch size, number of channels,
    # height, and width), the shape of the filter bank (number of filters,
    # number of output channels, filter height, filter width), the border
    # mode, and the step (vertical and horizontal strides). It is expected
    # to return a 4-tuple of (batch size, number of channels, output
    # height, output width). The first element of this tuple is not used
    # for anything by this brick.
    get_output_shape = staticmethod(get_conv_output_shape)

    @lazy(allocation=['filter_size', 'num_filters', 'num_channels'])
    def __init__(self, filter_size, num_filters, num_channels, batch_size=None,
                 image_size=(None, None), step=(1, 1), border_mode='valid',
                 tied_biases=True, weightnorm=False, **kwargs):
        super(Convolutional, self).__init__(**kwargs)

        self.filter_size = filter_size
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.step = step
        self.border_mode = border_mode
        self.tied_biases = tied_biases
        self.weightnorm = weightnorm

    def _allocate(self):
        W = shared_floatx_nans((self.num_filters, self.num_channels) +
                               self.filter_size, name='W')
        add_role(W, FILTER)
        self.parameters.append(W)
        self.add_auxiliary_variable(W.norm(2), name='W_norm')
        if getattr(self, 'use_bias', True):
            if self.tied_biases:
                b = shared_floatx_nans((self.num_filters,), name='b')
            else:
                # this error is raised here instead of during initializiation
                # because ConvolutionalSequence may specify the image size
                if self.image_size == (None, None) and not self.tied_biases:
                    raise ValueError('Cannot infer bias size without '
                                     'image_size specified. If you use '
                                     'variable image_size, you should use '
                                     'tied_biases=True.')

                b = shared_floatx_nans(self.get_dim('output'), name='b')
            add_role(b, BIAS)

            self.parameters.append(b)
            self.add_auxiliary_variable(b.norm(2), name='b_norm')
        if self.weightnorm:
            g = shared_floatx_nans((self.num_filters,), name='g')
            add_role(g, WEIGHT)
            self.parameters.append(g)


    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Perform the convolution.
        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            A 4D tensor with the axes representing batch size, number of
            channels, image height, and image width.
        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            A 4D tensor of filtered images (feature maps) with dimensions
            representing batch size, number of filters, feature map height,
            and feature map width.
            The height and width of the feature map depend on the border
            mode. For 'valid' it is ``image_size - filter_size + 1`` while
            for 'full' it is ``image_size + filter_size - 1``.
        """
        if self.image_size == (None, None):
            input_shape = None
        else:
            input_shape = (self.batch_size, self.num_channels)
            input_shape += self.image_size

        if self.weightnorm:
            w_norm = T.sqrt(T.sum(T.square(self.W),axis=(1,2,3)))
            output = self.conv2d_impl(
                input_, self.W * (self.g / w_norm).dimshuffle(0,'x','x','x'),
                input_shape=input_shape,
                subsample=self.step,
                border_mode=self.border_mode,
                filter_shape=((self.num_filters, self.num_channels) +
                              self.filter_size))
        else:
            output = self.conv2d_impl(
                input_, self.W,
                input_shape=input_shape,
                subsample=self.step,
                border_mode=self.border_mode,
                filter_shape=((self.num_filters, self.num_channels) +
                              self.filter_size))
        if getattr(self, 'use_bias', True):
            if self.tied_biases:
                output += self.b.dimshuffle('x', 0, 'x', 'x')
            else:
                output += self.b.dimshuffle('x', 0, 1, 2)
        return output

    def get_dim(self, name):
        if name == 'input_':
            return (self.num_channels,) + self.image_size
        if name == 'output':
            input_shape = (None, self.num_channels) + self.image_size
            kernel_shape = ((self.num_filters, self.num_channels) +
                            self.filter_size)
            out_shape = self.get_output_shape(input_shape, kernel_shape,
                                              self.border_mode, self.step)
            assert len(out_shape) == 4
            return out_shape[1:]
        return super(Convolutional, self).get_dim(name)

    @property
    def num_output_channels(self):
        return self.num_filters

    @property
    def g(self):
        return self.parameters[2]

    def initialize(self):
        self._initialize()
        if self.weightnorm:
            Constant(1).initialize(self.parameters[2], self.rng)


class ConvolutionalTranspose(Convolutional):
    """Performs the transpose of a 2D convolution.
    Parameters
    ----------
    num_filters : int
        Number of filters at the *output* of the transposed convolution,
        i.e. the number of channels in the corresponding convolution.
    num_channels : int
        Number of channels at the *input* of the transposed convolution,
        i.e. the number of output filters in the corresponding
        convolution.
    step : tuple, optional
        The step (or stride) of the corresponding *convolution*.
        Defaults to (1, 1).
    image_size : tuple, optional
        Image size of the input to the *transposed* convolution, i.e.
        the output of the corresponding convolution. Required for tied
        biases. Defaults to ``None``.
    unused_edge : tuple, optional
        Tuple of pixels added to the inferred height and width of the
        output image, whose values would be ignored in the corresponding
        forward convolution. Must be such that 0 <= ``unused_edge[i]`` <=
        ``step[i]``. Note that this parameter is **ignored** if
        ``original_image_size`` is specified in the constructor or manually
        set as an attribute.
    original_image_size : tuple, optional
        The height and width of the image that forms the output of
        the transpose operation, which is the input of the original
        (non-transposed) convolution. By default, this is inferred
        from `image_size` to be the size that has each pixel of the
        original image touched by at least one filter application
        in the original convolution. Degenerate cases with dropped
        border pixels (in the original convolution) are possible, and can
        be manually specified via this argument. See notes below.
    See Also
    --------
    :class:`Convolutional` : For the documentation of other parameters.
    Notes
    -----
    By default, `original_image_size` is inferred from `image_size`
    as being the *minimum* size of image that could have produced this
    output. Let ``hanging[i] = original_image_size[i] - image_size[i]
    * step[i]``. Any value of ``hanging[i]`` greater than
    ``filter_size[i] - step[i]`` will result in border pixels that are
    ignored by the original convolution. With this brick, any
    ``original_image_size`` such that ``filter_size[i] - step[i] <
    hanging[i] < filter_size[i]`` for all ``i`` can be validly specified.
    However, no value will be output by the transposed convolution
    itself for these extra hanging border pixels, and they will be
    determined entirely by the bias.
    """
    @lazy(allocation=['filter_size', 'num_filters', 'num_channels'])
    def __init__(self, filter_size, num_filters, num_channels,
                 original_image_size=None, unused_edge=(0, 0),
                 **kwargs):
        super(ConvolutionalTranspose, self).__init__(
            filter_size, num_filters, num_channels, **kwargs)
        self.original_image_size = original_image_size
        self.unused_edge = unused_edge

    @property
    def original_image_size(self):
        if self._original_image_size is None:
            if all(s is None for s in self.image_size):
                raise ValueError("can't infer original_image_size, "
                                 "no image_size set")
            if isinstance(self.border_mode, tuple):
                border = self.border_mode
            elif self.border_mode == 'full':
                border = tuple(k - 1 for k in self.filter_size)
            elif self.border_mode == 'half':
                border = tuple(k // 2 for k in self.filter_size)
            else:
                border = [0] * len(self.image_size)
            tups = zip(self.image_size, self.step, self.filter_size, border,
                       self.unused_edge)
            return tuple(s * (i - 1) + k - 2 * p + u for i, s, k, p, u in tups)
        else:
            return self._original_image_size

    @original_image_size.setter
    def original_image_size(self, value):
        self._original_image_size = value

    def conv2d_impl(self, input_, W, input_shape, subsample, border_mode,
                    filter_shape):
        # The AbstractConv2d_gradInputs op takes a kernel that was used for the
        # **convolution**. We therefore have to invert num_channels and
        # num_filters for W.
        W = W.transpose(1, 0, 2, 3)
        imshp = (None,) + self.get_dim('output')
        kshp = (filter_shape[1], filter_shape[0]) + filter_shape[2:]
        return AbstractConv2d_gradInputs(
            imshp=imshp, kshp=kshp, border_mode=border_mode,
            subsample=subsample)(W, input_, self.get_dim('output')[1:])

    def get_dim(self, name):
        if name == 'output':
            return (self.num_filters,) + self.original_image_size
        return super(ConvolutionalTranspose, self).get_dim(name)
