from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn

from ...common import get_root_logger, print_log


class BaseLocation(nn.Module, metaclass=ABCMeta):
    """Base class for locations."""

    def __init__(self):
        super(BaseLocation, self).__init__()

    @abstractmethod
    def extract_feat(self, x):
        """Extract features from signal data."""
        pass

    @abstractmethod
    def forward_train(self, x, px, py, **kwargs):
        """
        Args:
            x (list[Tensor]): List of tensors of shape (1, 1, 8, W).
            px (list[Tensor]): List of tensors of shape (1, C).
            py (list[Tensor]): List of tensors of shape (1, C).
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        pass

    @abstractmethod
    def simple_test(self, x, **kwargs):
        pass

    @abstractmethod
    def init_weights(self, pre_trained=None):
        """Initialize the weights in task.

       Args:
           pre_trained (str, optional): Path to pre-trained weights.
               Defaults to None.
       """

        if pre_trained is not None:
            logger = get_root_logger()
            print_log(f'load model from: {pre_trained}', logger=logger)

    def forward_test(self, x, **kwargs):
        """
        Args:
            x (list[Tensor]): List of tensors of shape (1, 1, 8, W).
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        return self.simple_test(x, **kwargs)

    def forward(self, x, px=None, py=None, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.
        """
        if return_loss:
            return self.forward_train(x, px, py, **kwargs)
        else:
            return self.forward_test(x, **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss, log_vars=log_vars,
                       num_samples=len(data['px']))

        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss, log_vars=log_vars,
                       num_samples=len(data['px']))

        return outputs

    # TODO: a method to show the classification results
    # def show_result(self, ):
