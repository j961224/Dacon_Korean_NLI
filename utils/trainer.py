import math
import torch
import torch.nn as nn
import contextlib
import torch.nn.functional as F
import sys
import os
import inspect

from transformers import Trainer
from torch.nn.modules import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from functools import partial
from utils.collate_functions import collate_to_max_length
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoConfig, AdamW, get_linear_schedule_with_warmup
import json


class Custom_Trainer(Trainer):
  def __init__(self, **kwargs):
    super(Custom_Trainer, self).__init__(**kwargs)

  def create_optimizer_and_scheduler(self, num_training_steps: int):
     
    self.create_optimizer()
    self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

  def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
    if not self.args.is_noam:
        super().create_scheduler(num_training_steps, optimizer)
    else:
      if self.lr_scheduler is None:
          self.lr_scheduler = self.get_noam_schedule_with_warmup(
              optimizer=self.optimizer if optimizer is None else optimizer,
              num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
          )
      return self.lr_scheduler

  def get_noam_schedule_with_warmup(self, optimizer: torch.optim.Optimizer, num_warmup_steps: int, last_epoch=-1) -> LambdaLR:
    def lr_lambda(current_step: int):
        return 1 / math.sqrt(self.args.d_model) * min(1/math.sqrt(current_step+1), (current_step+1) /(num_warmup_steps**(1.5)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

###############

from packaging import version

if version.parse(torch.__version__) >= version.parse("1.6"):
  from torch.cuda.amp import autocast

def nested_detach(tensors):
  "Detach `tensors` (even if it's a nested list/tuple of tensors)."
  if isinstance(tensors, (list, tuple)):
      return type(tensors)(nested_detach(t) for t in tensors)
  return tensors.detach()

def is_sagemaker_mp_enabled():
  # Get the sagemaker specific mp parameters from smp_options variable.
  smp_options = os.getenv("SM_HP_MP_PARAMETERS", "{}")
  try:
      # Parse it and check the field "partitions" is included, it is required for model parallel.
      smp_options = json.loads(smp_options)
      if "partitions" not in smp_options:
          return False
  except json.JSONDecodeError:
      return False




class ExplainableModel_Trainer(Trainer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.loss_fn = nn.CrossEntropyLoss()
    self.args.lamb = 1.0
  

  
  def smp_forward_only(model, inputs):
      return model(**inputs)

  def smp_nested_concat(tensor):
      if isinstance(tensor, (list, tuple)):
          return type(tensor)(smp_nested_concat(t) for t in tensor)
      elif isinstance(tensor, dict):
          return type(tensor)({k: smp_nested_concat(v) for k, v in tensor.items()})
      # It doesn't seem possible to check here if `tensor` is a StepOutput because StepOutput lives in `smp.step`
      # which is also the name of the decorator so Python is confused.
      return tensor.concat().detach().cpu()
  
  def autocast_smart_context_manager(self):
    """
    A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
    arguments, depending on the situation.
    """
    if self.use_amp:
        if version.parse(torch.__version__) >= version.parse("1.10"):
            ctx_manager = autocast(dtype=self.amp_dtype)
        else:
            ctx_manager = autocast()
    else:
        ctx_manager = contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()

    return ctx_manager

  def compute_loss(self, model, inputs, return_outputs=False):
    if self.label_smoother is not None and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None
    outputs = model(**inputs)
    # Save past state if it exists
    # TODO: this needs to be fixed and made cleaner later.
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]

    if labels is not None:
        y_hat, a_ij = outputs
        labels = labels.view(-1)
        ce_loss = self.loss_fn(y_hat, labels)
        reg_loss = self.args.lamb * a_ij.pow(2).sum(dim=1).mean()
        loss = ce_loss + reg_loss
    else:
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    return (loss, outputs) if return_outputs else loss
  
  def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
      has_labels = all(inputs.get(k) is not None for k in self.label_names)
      inputs = self._prepare_inputs(inputs)
      if ignore_keys is None:
          if hasattr(self.model, "config"):
              ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
          else:
              ignore_keys = []

      # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
      if has_labels:
          labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
          if len(labels) == 1:
              labels = labels[0]
      else:
          labels = None

      with torch.no_grad():
          if is_sagemaker_mp_enabled():
            print("Yyyyyy")
            raw_outputs = smp_forward_only(model, inputs)
            if has_labels:
                if isinstance(raw_outputs, dict):
                    loss_mb = raw_outputs["loss"]
                    logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    loss_mb = raw_outputs[0]
                    logits_mb = raw_outputs[1:]

                loss = loss_mb.reduce_mean().detach().cpu()
                logits = smp_nested_concat(logits_mb)
            else:
                loss = None
                if isinstance(raw_outputs, dict):
                    logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                else:
                    logits_mb = raw_outputs
                logits = smp_nested_concat(logits_mb)
          else:

            if has_labels:
                with self.autocast_smart_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[0]
            else:
                loss = None
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

      if prediction_loss_only:
          return (loss, None, None)

      logits = nested_detach(logits)
      if len(logits) == 1:
          logits = logits[0]
      return (loss, logits, labels)


  
