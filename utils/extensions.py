import logging

from blocks.extensions import SimpleExtension
import numpy as np
from visdom import Visdom

logger = logging.getLogger(__name__)

# https://gist.github.com/dmitriy-serdyuk/83f4130e53590bec908e16260ff6ee26

class VisdomExt(SimpleExtension):
    r""" Extension doing visdom plotting.
    """
    def __init__(self, channels, plot_options=[],
                visdom_kwargs={}, **kwargs):
        self.viz = Visdom(**visdom_kwargs)
        self.list_names = channels
        self.env = visdom_kwargs.get('env', 'main')

        # `line` method squeezes the input, in order to maintain the shape
        # we have to repeat it twice making its shape (2, M), where M is
        # the number of lines
        self.windows = []
        self.p = {}

        for i, channel_set in enumerate(channels):
            try:
                channel_set_opts = plot_options[i]
            except IndexError:
                channel_set_opts = {}
            # we have to initialize the plot with some data, but NaNs are ignored
            dummy_data = [np.nan] * len(channel_set)
            dummy_ind = [0.] * len(channel_set)
            channel_set_opts.update(dict(legend=channel_set))
            for channel in channel_set:
                self.p[channel] = i

            self.windows.append(self.viz.line(np.vstack([dummy_data, dummy_data]),
                                    np.vstack([dummy_ind, dummy_ind]),
                                    opts=channel_set_opts,
                                    env=self.env
                                ))
        super(VisdomExt, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        log = self.main_loop.log
        if 'batch' in which_callback:
            iteration = log.status['iterations_done']
        else:
            iteration = log.status['epochs_done']
        for channel in self.p:
            if channel in log.current_row:
                values = np.array(
                    [log.current_row[channel]], dtype='float32')
                iterations = np.array([iteration], dtype='float32')
                self.viz.updateTrace(iterations, values, append=True,
                                     name=channel, win=self.windows[self.p[channel]],
                                     env=self.env)

class GenerateSamples(SimpleExtension):
    """ Extension to generate samples during training """

    def __init__(self, theano_func, file_name='', **kwargs):
        super(GenerateSamples, self).__init__(**kwargs)
        self.file_name = file_name
        self.theano_func = theano_func

    def do(self, which_callback, *args):
        epochs_done = str(self.main_loop.log.status['epochs_done'])
        batch = next(self.main_loop.data_stream.get_epoch_iterator(as_dict=True))
        image_source, image_target = batch['image_source'], batch['image_target']
        image_generated = self.theano_func(image_source)[-2]
        # import ipdb; ipdb.set_trace()
        np.savez(
            self.file_name+'/epoch_'+epochs_done,
            source=image_source,
            generated=image_generated,
            target=image_target
        )
