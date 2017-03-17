from collections import OrderedDict

from blocks.bricks import Softplus
from blocks.algorithms import GradientDescent, Adam, RMSProp, CompositeRule, Restrict
from blocks.bricks.base import application
from blocks.bricks.interfaces import Initializable, Random
from blocks.select import Selector

import theano
from theano import tensor as T

class GAN(Initializable, Random):
    """ Generative adversarial generative model
    Parameters
    ----------
        generator: :class:`blocks.bricks.Brick`
            Generator network.
        discriminator : :class:`blocks.bricks.Brick`
            Discriminator network taking :math:`x` and :math:`z` as input.
    """
    def __init__(self, generator, discriminator, **kwargs):
        self.generator = generator
        self.discriminator = discriminator
        super(GAN, self).__init__(**kwargs)
        self.children.extend([self.generator, self.discriminator])

    @property
    def discriminator_parameters(self):
        return list(
            Selector([self.discriminator]).get_parameters().values())

    @property
    def generator_parameters(self):
        return list(
            Selector([self.generator]).get_parameters().values())

    @application(inputs=['x', 'y', 'y_tilde'], outputs=['data_preds', 'sample_preds'])
    def get_predictions(self, x, y, y_tilde):
        input_D_real = T.concatenate([x, y])
        input_D_fake = T.concatenate([x, y_tilde])

        pred_real = self.discriminator.apply(input_D_real)
        pred_fake = self.discriminator.apply(input_D_fake)
        return pred_real, pred_fake

    @application(inputs=['x', 'y'], outputs=['discriminator_loss', 'generator_loss'])
    def compute_losses(self, x, y):
        y_tilde = self.generator.apply(x)
        pred_real, pred_fake = self.get_predictions(x, y, y_tilde)

        discriminator_loss = (T.nnet.softplus(-pred_real) +
                              T.nnet.softplus(pred_fake)).mean()
        generator_loss = T.nnet.softplus(-pred_fake).mean()
        return discriminator_loss, generator_loss

    def algorithm(self, discriminator_loss, generator_loss, discriminator_step_rule, generator_step_rule):
        discriminator_parameters = self.discriminator_parameters
        generator_parameters = self.generator_parameters

        gradients = OrderedDict()
        gradients.update(
            zip(discriminator_parameters,
                theano.grad(discriminator_loss, self.discriminator_parameters)))
        gradients.update(
            zip(generator_parameters,
                theano.grad(generator_loss, generator_parameters)))
        step_rule = CompositeRule([Restrict(discriminator_step_rule,
                                            discriminator_parameters),
                                   Restrict(generator_step_rule,
                                            generator_parameters)])
        return GradientDescent(
            cost=generator_loss + discriminator_loss,
            gradients=gradients,
            parameters=discriminator_parameters + generator_parameters,
            step_rule=step_rule,
            on_unused_sources='ignore'
        )

# model = Model([discriminator_loss, generator_loss])
#
# extensions = [
#     FinishAfter(after_n_epochs=nb_epoch),
#     TrainingDataMonitoring(
#         [algorithm.cost],
#         prefix="train",
#         after_epoch=True),
#     DataStreamMonitoring(
#         [algorithm.cost],
#         test_stream,
#         prefix="test"),
#     Printing(),
#     ProgressBar(),
# ]
