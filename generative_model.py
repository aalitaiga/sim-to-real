from collections import OrderedDict

from blocks.algorithms import GradientDescent, Restrict, StepRule, CompositeRule
from blocks.bricks.base import application
from blocks.bricks.interfaces import Initializable, Random
from blocks.extensions import SimpleExtension
from blocks.roles import add_role, ALGORITHM_HYPERPARAMETER, ALGORITHM_BUFFER
from blocks.select import Selector
from blocks.utils import shared_floatx

from picklable_itertools.extras import equizip
import numpy as np
import theano
from theano import tensor as T
from theano.ifelse import ifelse

class GAN(Initializable, Random):
    """ Generative adversarial generative model
    Parameters
    ----------
        generator: :class:`blocks.bricks.Brick`
            Generator network.
        discriminator : :class:`blocks.bricks.Brick`
            Discriminator network taking :math:`x` and :math:`z` as input.
    """
    def __init__(self, generator, discriminator, alpha=0, **kwargs):
        self.generator = generator
        self.discriminator = discriminator
        super(GAN, self).__init__(**kwargs)
        self.children.extend([self.generator, self.discriminator])
        self.alpha = alpha

    @property
    def discriminator_parameters(self):
        return list(Selector([self.discriminator]).get_parameters().values())

    @property
    def generator_parameters(self):
        return list(Selector([self.generator]).get_parameters().values())

    @application(inputs=['x', 'y', 'y_tilde'], outputs=['data_preds', 'sample_preds'])
    def get_predictions(self, x, y, y_tilde, application_call):
        input_D_real = T.concatenate([x, y], axis=1)
        input_D_fake = T.concatenate([x, y_tilde], axis=1)

        pred_real = self.discriminator.apply(input_D_real)
        pred_fake = self.discriminator.apply(input_D_fake)

        application_call.add_auxiliary_variable(
            T.nnet.sigmoid(pred_real).mean(), name='data_accuracy')
        application_call.add_auxiliary_variable(
            (1 - T.nnet.sigmoid(pred_fake)).mean(),
            name='sample_accuracy')

        return pred_real, pred_fake

    @application(inputs=['context', 'obs_sim', 'obs_real'], outputs=['discriminator_loss', 'generator_loss'])
    def losses(self, context, obs_sim, obs_real, application_call):
        # TODO: add rewards later

        x_fake = T.concatenate([context, obs_sim], axis=0)
        obs_generated = self.generator.apply(x_fake)

        pred_real, pred_fake = self.get_predictions(context, obs_real, obs_generated)

        discriminator_loss = (T.nnet.softplus(-pred_real) +
                              T.nnet.softplus(pred_fake)).mean()
        sqr_error = self.alpha * T.sqr(obs_real - obs_generated).mean()
        sqr_error.name = 'squared_error'

        generator_loss = T.nnet.softplus(-pred_fake).mean() + sqr_error

        application_call.add_auxiliary_variable(
            abs((obs_generated - obs_real) / obs_real).mean(),
            name="percent_error"
        )
        return discriminator_loss, generator_loss

    def algorithm(self, discriminator_loss, generator_loss, discriminator_step_rule, generator_step_rule):
        discriminator_parameters = self.discriminator_parameters
        generator_parameters = self.generator_parameters

        gradients = OrderedDict()
        gradients.update(
            zip(discriminator_parameters,
                theano.grad(discriminator_loss, discriminator_parameters)))
        gradients.update(
            zip(generator_parameters,
                theano.grad(generator_loss, generator_parameters)))
        step_rule = CompositeRule(Restrict(discriminator_step_rule,
                                            discriminator_parameters),
                                   Restrict(generator_step_rule,
                                            generator_parameters))
        return GradientDescent(
            cost=generator_loss + discriminator_loss,
            gradients=gradients,
            parameters=discriminator_parameters + generator_parameters,
            step_rule=step_rule,
            on_unused_sources='ignore',
            theano_func_kwargs={'allow_input_downcast': True}
        )

class WGAN(GAN):
    """ Wasserstein GAN """
    @application(inputs=['context', 'obs_sim', 'obs_real'], outputs=['discriminator_loss', 'generator_loss'])
    def losses(self, context, obs_sim, obs_real, application_call):
        # TODO: add rewards later

        x_fake = T.concatenate([context, obs_sim], axis=1)
        obs_generated = self.generator.apply(x_fake)

        pred_real, pred_fake = self.get_predictions(context, obs_real, obs_generated)

        discriminator_loss = pred_fake.mean() - pred_real.mean()

        sqr_error = self.alpha * T.sqr(obs_real - obs_generated).mean()
        sqr_error.name = 'squared_error'
        generator_loss = -pred_fake.mean() + sqr_error

        application_call.add_auxiliary_variable(
            abs((obs_generated - obs_real) / obs_real).mean(),
            name="percent_error"
        )
        return discriminator_loss, generator_loss

    def algorithm(self, discriminator_loss, generator_loss, discriminator_step_rule, generator_step_rule):
        discriminator_parameters = self.discriminator_parameters
        generator_parameters = self.generator_parameters

        discriminator_gradients = OrderedDict()
        discriminator_gradients = discriminator_gradients.update(
            zip(discriminator_parameters,
                theano.grad(discriminator_loss, discriminator_parameters)))


        generator_gradients = OrderedDict()
        generator_gradients = generator_gradients.update(
            zip(generator_parameters,
                theano.grad(generator_loss, generator_parameters)))

        discriminator_algo = GradientDescent(
            cost=discriminator_loss,
            gradients=discriminator_gradients,
            parameters=discriminator_parameters,
            step_rule=discriminator_step_rule,
            on_unused_sources='ignore',
            theano_func_kwargs={'allow_input_downcast': True}
        )

        generator_algo = GradientDescent(
            cost=generator_loss,
            gradients=generator_gradients,
            parameters=generator_parameters,
            step_rule=generator_step_rule,
            on_unused_sources='ignore',
            theano_func_kwargs={'allow_input_downcast': True}
        )
        return discriminator_algo, generator_algo

class ParameterPrint(SimpleExtension):
    """ Check that parameters are indeed clipped """
    def do(self, which_callback, *args):
        gan = self.main_loop.model.top_bricks[0]
        for param in gan.discriminator_parameters:
            print(param.get_value())

class WeightClipping(SimpleExtension):
    """ Extension used for weight clipping in Wasserstein GANSs"""

    def __init__(self, parameters, c=0.01, **kwargs):
        super(WeightClipping, self).__init__(**kwargs)
        self.c = c
        updates = OrderedDict((param, T.clip(param, -c, c)) for param in parameters)
        self.f = theano.function([], [], updates=updates)

    def do(self, which_callback, *args):
        self.f()
