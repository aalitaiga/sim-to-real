"""The event-based main loop of Blocks."""
import signal
import logging
import traceback

from blocks.config import config
from blocks.log import BACKENDS
from blocks.utils import reraise_as, unpack, change_recursion_limit
from blocks.utils.profile import Profile, Timer
from blocks.algorithms import GradientDescent
from blocks.extensions import CallbackName
from blocks.model import Model
from rllab.envs.normalized_env import NormalizedEnv

from buffer_ import FIFO

logger = logging.getLogger(__name__)

error_message = """

Blocks will attempt to run `on_error` extensions, potentially saving data, \
before exiting and reraising the error. Note that the usual `after_training` \
extensions will *not* be run. The original error will be re-raised and also \
stored in the training log. Press CTRL + C to halt Blocks immediately."""

error_in_error_handling_message = """

Blocks will now exit. The remaining `on_error` extensions will not be run."""


epoch_interrupt_message = """

Blocks will complete this epoch of training and run extensions \
before exiting. If you do not want to complete this epoch, press CTRL + C \
again to stop training after the current batch."""

batch_interrupt_message = """

Blocks will complete the current batch and run extensions before exiting. If \
you do not want to complete this batch, press CTRL + C again. WARNING: Note \
that this will end training immediately, and extensions that e.g. save your \
training progress won't be run."""

no_model_message = """

A possible reason: one of your extensions requires the main loop to have \
a model. Check documentation of your extensions."""


class RLMainLoop(object):
    """ MainLoop better suited to do RL """

    def __init__(self, algorithm, d, model=None, log=None,
                log_backend=None, extensions=None, render=True, generator_algo=None):
        if log is None:
            if log_backend is None:
                log_backend = config.log_backend
            log = BACKENDS[log_backend]()
        if extensions is None:
            extensions = []

        self.env = d["env"]
        self.env2 = d["env2"]
        self.observation_dim = int(self.env.observation_space.shape[0])
        self.action_dim = int(self.env.action_space.shape[0])
        self.buffer = d.get("buffer_", None)
        self.history = d["history"]
        self.render = d.get("render", True)
        self.episode_len = d["episode_len"]
        self.trajectory_len = d["trajectory_len"]
        self.d_iter = d.get("d_iter", None)

        self.algorithm = algorithm
        self.generator_algorithm = generator_algo
        self.log = log
        self.extensions = extensions

        self.profile = Profile()

        self._model = model

        self.status['training_started'] = False
        self.status['epoch_started'] = False
        self.status['epoch_interrupt_received'] = False
        self.status['batch_interrupt_received'] = False

    @property
    def model(self):
        if not self._model:
            raise AttributeError("no model in this main loop" +
                                 no_model_message)
        return self._model

    @property
    def status(self):
        """A shortcut for `self.log.status`."""
        return self.log.status

    def _match_env(self):
        # make env1 state match env2 state (simulator matches real world)
        if isinstance(self.env, NormalizedEnv):
            self.env.wrapped_env.env.env.set_state(
                self.env2.wrapped_env.env.env.model.data.qpos.ravel(),
                self.env2.wrapped_env.env.env.model.data.qvel.ravel()
            )
        else:
            self.env.env.set_state(
                self.env2.env.model.data.qpos.ravel(),
                self.env2.env.model.data.qvel.ravel()
            )

    def run(self):
        """Starts the main loop.

        The main loop ends when a training extension makes
        a `training_finish_requested` record in the log.

        """
        # This should do nothing if the user has already configured
        # logging, and will it least enable error messages otherwise.
        logging.basicConfig()

        # If this is resumption from a checkpoint, it is crucial to
        # reset `profile.current`. Otherwise, it simply does not hurt.
        self.profile.current = []

        # Sanity check for the most common case
        # if (self._model and isinstance(self._model, Model) and
        #         isinstance(self.algorithm, GradientDescent)):
        #     if not (set(self._model.get_parameter_dict().values()) ==
        #             set(self.algorithm.parameters)):
        #         logger.warning("different parameters for model and algorithm")

        with change_recursion_limit(config.recursion_limit):
            self.original_sigint_handler = signal.signal(
                signal.SIGINT, self._handle_epoch_interrupt)
            self.original_sigterm_handler = signal.signal(
                signal.SIGTERM, self._handle_batch_interrupt)
            try:
                logger.info("Entered the main loop")
                if not self.status['training_started']:
                    for extension in self.extensions:
                        extension.main_loop = self
                    self._run_extensions('before_training')
                    with Timer('initialization', self.profile):
                        self.algorithm.initialize()
                        if self.generator_algorithm:
                            self.generator_algorithm.initialize()
                    self.status['training_started'] = True
                # We can not write "else:" here because extensions
                # called "before_training" could have changed the status
                # of the main loop.
                if self.log.status['iterations_done'] > 0:
                    self.log.resume()
                    self._run_extensions('on_resumption')
                    self.status['epoch_interrupt_received'] = False
                    self.status['batch_interrupt_received'] = False
                with Timer('training', self.profile):
                    while self._run_episode():
                        pass
            except TrainingFinish:
                self.log.current_row['training_finished'] = True
            except Exception as e:
                self._restore_signal_handlers()
                self.log.current_row['got_exception'] = traceback.format_exc()
                logger.error("Error occured during training." + error_message)
                try:
                    self._run_extensions('on_error', e)
                except Exception:
                    logger.error(traceback.format_exc())
                    logger.error("Error occured when running extensions." +
                                 error_in_error_handling_message)
                reraise_as(e)
            finally:
                self._restore_signal_handlers()
                if self.log.current_row.get('training_finished', False):
                    self._run_extensions('after_training')
                if config.profile:
                    self.profile.report()

    def find_extension(self, name):
        """Find an extension with a given name.

        Parameters
        ----------
        name : str
            The name of the extension looked for.

        Notes
        -----
        Will crash if there no or several extension found.

        """
        return unpack([extension for extension in self.extensions
                       if extension.name == name], singleton=True)

    def _run_episode(self):
        # Here one epoch is an episode
        if not self.status.get('epoch_started', False):
            self.log.status['received_first_batch'] = False
            # return False
            self.status['epoch_started'] = True
            self._run_extensions('before_epoch')
        with Timer('epoch', self.profile):
            for _ in range(self.episode_len):
                self.env.reset()
                self.env2.reset()
                self._match_env()
                self._run_trajectory()
        self.status['epoch_started'] = False
        self.status['epochs_done'] += 1
        # Log might not allow mutating objects, so use += instead of append
        self.status['_epoch_ends'] += [self.status['iterations_done']]
        self._run_extensions('after_epoch')
        self._check_finish_training('epoch')
        return True

    def _run_trajectory(self):
        # Here one iteration is a trajectory
        self.log.status['received_first_batch'] = True
        prev_observations = FIFO(self.history)
        actions = FIFO(self.history)
        d_iter_done = 0
        for _ in range(self.trajectory_len):
            if self.render:
                self.env.render()
                self.env2.render()
            # TODO modify to follow a given policy and not choose random actions
            action = self.env.action_space.sample()
            s_obs, s_reward, s_done, _ = self.env.step(action)
            r_obs, r_reward, r_done, _ = self.env2.step(action)
            actions.push(action)

            if len(prev_observations) == self.history and len(actions) == self.history:
                if self.buffer:
                    self.buffer.add_sample(
                        prev_observations.copy(), actions.copy(), s_obs, r_obs, s_reward, r_reward
                    )
                batch = {
                    'actions': actions.copy(),
                    'previous_obs': prev_observations.copy(),
                    'observation_sim': s_obs,
                    'observation_real': r_obs
                }
                self._run_extensions('before_batch', batch)
                with Timer('train', self.profile):
                    self.algorithm.process_batch(batch)
                    if self.generator_algorithm and d_iter_done % self.d_iter == 0:
                        self.generator_algorithm.process_batch(batch)
                d_iter_done += 1
                self._run_extensions('after_batch', batch)
            prev_observations.push(r_obs)
            self._match_env()

            # if r_done:
            #     break
        self.status['iterations_done'] += 1
        self._check_finish_training('batch')
        return False

    def _run_extensions(self, method_name, *args):
        with Timer(method_name, self.profile):
            for extension in self.extensions:
                with Timer(type(extension).__name__, self.profile):
                    extension.dispatch(CallbackName(method_name), *args)

    def _check_finish_training(self, level):
        """Checks whether the current training should be terminated.

        Parameters
        ----------
        level : {'epoch', 'batch'}
            The level at which this check was performed. In some cases, we
            only want to quit after completing the remained of the epoch.

        """
        # In case when keyboard interrupt is handled right at the end of
        # the iteration the corresponding log record can be found only in
        # the previous row.
        if (self.log.current_row.get('training_finish_requested', False) or
                self.status.get('batch_interrupt_received', False)):
            raise TrainingFinish
        if (level == 'epoch' and
                self.status.get('epoch_interrupt_received', False)):
            raise TrainingFinish

    def _handle_epoch_interrupt(self, signal_number, frame):
        # Try to complete the current epoch if user presses CTRL + C
        logger.warning('Received epoch interrupt signal.' +
                       epoch_interrupt_message)
        signal.signal(signal.SIGINT, self._handle_batch_interrupt)
        self.log.current_row['epoch_interrupt_received'] = True
        # Add a record to the status. Unlike the log record it will be
        # easy to access at later iterations.
        self.status['epoch_interrupt_received'] = True

    def _handle_batch_interrupt(self, signal_number, frame):
        # After 2nd CTRL + C or SIGTERM signal (from cluster) finish batch
        self._restore_signal_handlers()
        logger.warning('Received batch interrupt signal.' +
                       batch_interrupt_message)
        self.log.current_row['batch_interrupt_received'] = True
        # Add a record to the status. Unlike the log record it will be
        # easy to access at later iterations.
        self.status['batch_interrupt_received'] = True

    def _restore_signal_handlers(self):
        signal.signal(signal.SIGINT, self.original_sigint_handler)
        signal.signal(signal.SIGTERM, self.original_sigterm_handler)


class TrainingFinish(Exception):
    """An exception raised when a finish request is found in the log."""
    pass
