import numpy as np
from visdom import Visdom



class VisdomExt():
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

    def update(self,iteration, value, channel):
        val = np.array([value], dtype='float32')
        iter = np.array([iteration], dtype='float32')
        self.viz.updateTrace(iter, val, append=True, name=channel,
            win=self.windows[self.p[channel]], env=self.env)

    @classmethod
    def load(cls, windows, p, visdom_kwargs={}):
        vis = cls([])
        vis.windows = windows
        vis.p = p
        vis.env = visdom_kwargs.get('env', 'main')
        vis.viz = Visdom(**visdom_kwargs)
        return vis
