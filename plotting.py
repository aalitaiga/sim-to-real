import logging
import signal
from subprocess import Popen, PIPE
import time

# from abc import ABCMeta
# from six import add_metaclass
import numpy as np

from visdom import Visdom

from blocks.extensions import SimpleExtension

logger = logging.getLogger(__name__)

# @add_metaclass(ABCMeta)
class Plot(SimpleExtension):
    """Base class for extensions doing visdom plotting.

    Parameters
    ----------
    document_name : str
        The name of the visdom document. Use a different name for each
        experiment if you are storing your plots.
    start_server : bool, optional
        Whether to try and start the Bokeh plotting server. Defaults to
        ``False``. The server started is not persistent i.e. after shutting
        it down you will lose your plots. If you want to store your plots,
        start the server manually using the ``bokeh-server`` command. Also
        see the warning above.
    clear_document : bool, optional
        Whether or not to clear the contents of the server-side document
        upon creation. If `False`, previously existing plots within the
        document will be kept. Defaults to `True`.

    """
    # Tableau 10 colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


    def __init__(self, env_name, channels, start_server=False, **kwargs):
        self.env_name = env_name
        self.start_server = start_server
        self.port = kwargs.pop('port', 8097)
        self.viz = Visdom()
        self.plots = {}
        # Create figures for each group of channels
        self.p = []
        self.p_indices = {}
        self.color_indices = {}
        for i, channel_set in enumerate(channels):
            channel_set_opts = {}
            if isinstance(channel_set, dict):
                channel_set_opts = channel_set
                channel_set = channel_set_opts.pop('channels')
            channel_set_opts.setdefault('title',
                                        '{} #{}'.format('Test', i + 1))
            channel_set_opts.setdefault('xlabel', 'iterations')
            channel_set_opts.setdefault('ylabel', 'value')
            self.p.append((channel_set, channel_set_opts))

        kwargs.setdefault('after_epoch', True)
        kwargs.setdefault('before_first_epoch', True)
        kwargs.setdefault('after_training', True)
        super(Plot, self).__init__(**kwargs)

    def _start_server_process(self):
        def preexec_fn():
            """Prevents the server from dying on training interrupt."""
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        # Only memory works with subprocess, need to wait for it to start
        logger.info('Starting plotting server on localhost:8097')
        self.sub = Popen('python -m visdom.server',
                         stdout=PIPE, stderr=PIPE, preexec_fn=preexec_fn)
        time.sleep(2)
        logger.info('Plotting server PID: {}'.format(self.sub.pid))

    def do(self, which_callback, *args):
        super(Plot, self).do(which_callback, *args)
        log = self.main_loop.log
        iteration = log.status['iterations_done']
        log_items = log.current_row
        log_keys = set(log_items.keys())
        for channel_set, opts in self.p:
            if set(channel_set).issubset(log_keys):
                if str(channel_set) not in self.plots:
                    # opts.setdefault('legend', channel_set)
                    # import ipdb; ipdb.set_trace()
                    Y = np.column_stack([[log_items[v]] for v in channel_set])
                    # import pdb; pdb.set_trace()
                    win = self.viz.line(
                        Y=Y,
                        X=np.repeat(iteration, len(channel_set)),
                        env=self.env_name,
                        opts=opts
                    )
                    self.plots[str(channel_set)] = win
                else:
                    win = self.plots[str(channel_set)]
                    self.viz.line(
                        Y=np.hstack([log_items[v] for v in channel_set]),
                        X=np.repeat(iteration, len(channel_set)),
                        win=win,
                        update='append'
                    )
