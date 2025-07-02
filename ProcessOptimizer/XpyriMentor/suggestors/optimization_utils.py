from collections.abc import Callable, Sequence
from typing import Any
import torch
import gpytorch
from torch import Tensor


import botorch
from botorch.utils.dispatcher import Dispatcher, type_bypassing_encoder
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood



class EarlyStopper:
  def __init__(self, patience=5, min_delta=0):
      self.patience = patience
      self.min_delta = min_delta
      self.counter = 0
      self.best_loss = None
      self.early_stop = False

  def __call__(self, val_loss): # val_loss is the validation loss
      if self.best_loss is None:
          self.best_loss = val_loss
      elif self.best_loss - val_loss > self.min_delta:
          self.best_loss = val_loss
          self.counter = 0  # Reset counter when loss improves
      elif self.best_loss - val_loss < self.min_delta: # No significant improvement
          self.counter += 1
          if self.counter >= self.patience:
              print(f"Early stopping triggered after {self.patience} epochs.")
              self.early_stop = True


FitGPyTorchMLL = Dispatcher("fit_gpytorch_mll_wth_stopper", encoder=type_bypassing_encoder)
def fit_gpytorch_mll_wth_stopper(
        # TODO: implemnt the stopper withing the optimisation loop
    mll: MarginalLogLikelihood,
    closure: Callable[[], tuple[Tensor, Sequence[Tensor | None]]] | None = None,
    optimizer: Callable | None = None,
    closure_kwargs: dict[str, Any] | None = None,
    optimizer_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> MarginalLogLikelihood:
    r"""Clearing house for fitting models passed as GPyTorch MarginalLogLikelihoods.

    Args:
        mll: A GPyTorch MarginalLogLikelihood instance.
        closure: Forward-backward closure for obtaining objective values and gradients.
            Responsible for setting parameters' `grad` attributes. If no closure is
            provided, one will be obtained by calling `get_loss_closure_with_grads`.
        optimizer: User specified optimization algorithm. When `optimizer is None`,
            this keyword argument is omitted when calling the dispatcher.
        closure_kwargs: Keyword arguments passed when calling `closure`.
        optimizer_kwargs: A dictionary of keyword arguments passed when
            calling `optimizer`.
        **kwargs: Keyword arguments passed down through the dispatcher to
            fit subroutines. Unexpected keywords are ignored.

    Returns:
        The `mll` instance. If fitting succeeded, then `mll` will be in evaluation mode,
        i.e. `mll.training == False`. Otherwise, `mll` will be in training mode.
    """
    if optimizer is not None:  # defer to per-method defaults
        kwargs["optimizer"] = optimizer

    return FitGPyTorchMLL(
        mll,
        type(mll.likelihood),
        type(mll.model),
        closure=closure,
        closure_kwargs=closure_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        **kwargs,
    )
