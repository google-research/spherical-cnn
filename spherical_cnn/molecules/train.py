# Copyright 2023 The spherical_cnn Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for training Spin-Weighted Spherical CNNs."""

import functools
import io
import os
from typing import Any, Callable, Dict, Sequence, Tuple, Union

from absl import logging
from clu import checkpoint
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from clu import periodic_actions
import flax
import flax.jax_utils as flax_utils
import flax.linen as nn
import jax
from jax.experimental.compilation_cache import compilation_cache
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from spherical_cnn.molecules import input_pipeline
from spherical_cnn.molecules import models
import tensorflow as tf

import shutil


_PMAP_AXIS_NAME = "batch"
Optimizer = optax.GradientTransformation


@flax.struct.dataclass
class TrainState:
  step: int
  optimizer_state: optax.OptState
  params: optax.Params
  batch_stats: Any
  constants: Any


def create_train_state(
    config: ml_collections.ConfigDict,
    rng: jnp.ndarray,
    metadata: Dict[str, Any],
    inputs: Sequence[Any],
    learning_rate_schedule: Callable[[int], float]
) -> Tuple[nn.Module, Optimizer, TrainState]:
  """Create and initialize the model.

  Args:
    config: Configuration for model.
    rng: JAX PRNG Key.
    metadata: Dataset metadata.
    inputs: Shapes of the inputs fed into the model.
    learning_rate_schedule: Function that computes the learning rate given the
      step number.

  Returns:
    The model and initialized TrainState with the optimizer.
  """
  config.spins = tuple([tuple(s) for s in config.spins])
  # We pass metadata as FrozenDict because nn.Module must be hashable.
  metadata = nn.FrozenDict(metadata)
  # Use of getattr could simplify this but is discouraged by the style guide. We
  # should consider using it anyway if sequence of ifs grows too big.
  model_options = dict(
      resolutions=config.resolutions,
      spins=config.spins,
      widths=config.widths,
      spectral_pooling=config.spectral_pooling,
      num_filter_params=config.num_filter_params,
      tail_module=config.tail_module,
      use_atom_type_embedding=config.use_atom_type_embedding,
      use_distance_to_center=config.use_distance_to_center,
      atom_aggregation_mode=config.atom_aggregation_mode,
      num_transformer_layers=config.num_transformer_layers,
      num_transformer_heads=config.num_transformer_heads,
      dropout_rate=config.dropout_rate,
      attention_dropout_rate=config.attention_dropout_rate,
      metadata=metadata,
      axis_name=_PMAP_AXIS_NAME,
      input_transformer=None)
  # Gaussian representation is computed on device.
  if config.molecule_to_sphere_method in ["gaussian", "gaussian_on_device"]:
    extra_options = dict(
        sphere_resolution=config.sphere_resolution,
        cutoff_angle=np.pi/4,
        distance_powers=config.distance_powers,
        rotation_augmentation=config.rotation_augmentation)
    model_options = {**model_options, **extra_options}
    model_fun = models.SpinSphericalRegressorFromPositions
  else:
    model_fun = models.SpinSphericalRegressor
  if config.model_name == "spin_spherical_regressor":
    model = model_fun(**model_options, residual=False)
  elif config.model_name == "spin_spherical_residual_regressor":
    model = model_fun(**model_options, residual=True)
  else:
    raise ValueError(f"Model {config.model_name} not supported.")

  params_rng, dropout_rng, augmentation_rng = jax.random.split(rng, num=3)
  model_rng = {"params": params_rng,
               "dropout": dropout_rng,
               "augmentation": augmentation_rng}

  jit_init = jax.jit(functools.partial(model.init,
                                       train=False))
  variables = jit_init(model_rng, *inputs)

  params = variables["params"]
  batch_stats = variables.get("batch_stats", {})
  constants = variables["constants"]

  abs_if_complex = lambda x: jnp.abs(x) if x.dtype == jnp.complex64 else x
  parameter_overview.log_parameter_overview(
      jax.tree_util.tree_map(abs_if_complex, params))
  optimizer = optax.adam(learning_rate_schedule)  # pytype: disable=wrong-arg-types  # numpy-scalars
  optimizer_state = optimizer.init(params)
  return model, optimizer, TrainState(
      step=0,
      optimizer_state=optimizer_state,
      params=params,
      batch_stats=batch_stats,
      constants=constants)


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
  eval_loss: metrics.Average.from_output("loss")
  eval_mean_squared_error: metrics.Average.from_output("mean_squared_error")
  eval_mean_absolute_error: metrics.Average.from_output("mean_absolute_error")


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  loss: metrics.Average.from_output("loss")
  mean_squared_error: metrics.Average.from_output("mean_squared_error")
  mean_absolute_error: metrics.Average.from_output("mean_absolute_error")
  loss_std: metrics.Std.from_output("loss")


def cosine_decay(lr: float, step: float, total_steps: int):
  ratio = jnp.maximum(0., step / total_steps)
  mult = 0.5 * (1. + jnp.cos(jnp.pi * ratio))
  return mult * lr


def get_learning_rate(step: int,
                      *,
                      base_learning_rate: float,
                      steps_per_epoch: int,
                      num_epochs: int,
                      warmup_epochs: int = 5):
  """Cosine learning rate schedule."""
  logging.info(("get_learning_rate(step=%s, "
                "base_learning_rate=%s, "
                "steps_per_epoch=%s, "
                "num_epochs=%s)"),
               step, base_learning_rate, steps_per_epoch, num_epochs)

  if steps_per_epoch <= 0:
    raise ValueError(f"steps_per_epoch should be a positive integer but was "
                     f"{steps_per_epoch}.")
  if warmup_epochs >= num_epochs:
    raise ValueError(f"warmup_epochs should be smaller than num_epochs. "
                     f"Currently warmup_epochs is {warmup_epochs}, "
                     f"and num_epochs is {num_epochs}.")
  epoch = step / steps_per_epoch
  lr = cosine_decay(base_learning_rate, epoch - warmup_epochs,
                    num_epochs - warmup_epochs)
  if warmup_epochs == 0:
    warmup = 1.0
  else:
    warmup = jnp.minimum(1., epoch / warmup_epochs)
  return lr * warmup


def _normalize(x, mean, standard_deviation):
  return (x - mean) / standard_deviation


def _denormalize(x, mean, standard_deviation):
  return x * standard_deviation + mean


def _compute_loss(predictions: jnp.ndarray,
                  labels: jnp.ndarray,
                  loss_type: str,
                  average_over_batch: bool) -> jnp.ndarray:
  """Computes the loss value."""
  if loss_type == "l1":
    loss = jnp.abs(predictions - labels)
  elif loss_type == "l2":
    loss = jnp.square(predictions - labels) / 2.0
  else:
    raise ValueError("`loss_type` must be either 'l1' or 'l2'.")
  if average_over_batch:
    loss = loss.mean()

  return loss


def _compute_metrics(predictions, labels):
  difference = predictions - labels
  mean_squared_error = jnp.square(difference)
  mean_absolute_error = jnp.abs(difference)
  return mean_squared_error, mean_absolute_error


def _unpad_batch_to_max_atoms(batch, max_atoms):
  """Removing padding from batch up to `max_atoms`."""
  if "spheres" in batch:
    batch["spheres"] = batch["spheres"][
        jax.process_index(), :, :, :max_atoms]
  if "positions" in batch:
    batch["positions"] = batch["positions"][
        jax.process_index(), :, :, :max_atoms]
  batch["charges"] = batch["charges"][
      jax.process_index(), :, :, :max_atoms]
  batch["label"] = batch["label"][jax.process_index()]
  return batch


def train_step(model: nn.Module,
               optimizer: Optimizer,
               state: TrainState,
               batch: Dict[str, jnp.ndarray],
               weight_decay: float,
               loss_type: str,
               rng: jnp.ndarray) -> Tuple[TrainState, metrics.Collection]:
  """Perform a single training step.

  Args:
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    optimizer: Optax optimizer module.
    state: State of the model (optimizer and state).
    batch: Training inputs for this step.
    weight_decay: Weighs L2 regularization term.
    loss_type: Which loss to use (follows `config.loss_type`).
    rng: JAX PRNG Key.

  Returns:
    The new model state and dictionary with metrics.
  """
  logging.info("train_step(batch=%s)", batch)

  step = state.step + 1

  rng = jax.random.fold_in(rng, state.step)
  rng = jax.random.fold_in(rng, jax.process_index())
  dropout_rng, augmentation_rng = jax.random.split(rng)

  def loss_fn(params):
    variables = {"params": params,
                 "batch_stats": state.batch_stats,
                 "constants": state.constants}
    rngs = {"dropout": dropout_rng, "augmentation": augmentation_rng}
    # Batch may contain positions only or the spherical representation.
    if "spheres" in batch:
      inputs = (batch["spheres"], batch["charges"], batch["positions"])
    else:
      inputs = (batch["positions"], batch["charges"])
    outputs, new_variables = model.apply(variables,
                                         *inputs,
                                         mutable=["batch_stats"],
                                         train=True,
                                         rngs=rngs)

    normalized_labels = _normalize(
        batch["label"],
        mean=model.metadata["label_mean"],
        standard_deviation=model.metadata["label_standard_deviation"])

    loss = _compute_loss(outputs, normalized_labels,
                         loss_type=loss_type,
                         average_over_batch=True)
    weight_penalty_params = jax.tree_util.tree_leaves(variables["params"])
    # Loss must be real and weights are complex, so we use the magnitudes.
    weight_l2 = sum([jnp.sum(x * x.conj()).real
                     for x in weight_penalty_params if x.ndim > 1])
    # TODO(machc): better to do weight decay as in AdamW.
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (new_variables["batch_stats"], outputs)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (new_batch_stats, outputs)), grad = grad_fn(state.params)

  # Compute average gradient across multiple workers.
  grad = jax.lax.pmean(grad, axis_name=_PMAP_AXIS_NAME)

  # NOTE(machc): JAX uses different conventions than TensorFlow for the
  # gradients of complex functions. They differ by a conjugate, so we conjugate
  # all gradients here in order to make gradient descent work seamlessly. This
  # is crucial if there are complex weights in the model, and makes no
  # difference for real weights. See https://github.com/google/jax/issues/4891.
  grad = jax.tree_util.tree_map(jnp.conj, grad)

  updates, optimizer_state = optimizer.update(grad, state.optimizer_state)
  params = optax.apply_updates(state.params, updates)
  new_state = state.replace(  # pytype: disable=attribute-error
      step=step,
      optimizer_state=optimizer_state,
      params=params,
      batch_stats=new_batch_stats)

  predictions = _denormalize(
      outputs,
      mean=model.metadata["label_mean"],
      standard_deviation=model.metadata["label_standard_deviation"])

  # Compute MSE and MAD with respect to unnormalized data
  mean_squared_error, mean_absolute_error = _compute_metrics(
      predictions, batch["label"])

  mean_absolute_error *= model.metadata["unit_conversion_factor"]
  mean_squared_error *= model.metadata["unit_conversion_factor"] ** 2

  metrics_update = TrainMetrics.gather_from_model_output(
      loss=loss,
      mean_squared_error=mean_squared_error,
      mean_absolute_error=mean_absolute_error)
  return new_state, metrics_update


@functools.partial(jax.pmap,
                   axis_name=_PMAP_AXIS_NAME,
                   static_broadcasted_argnums=(0, 3))
def eval_step(model: nn.Module, state: TrainState,
              batch: Dict[str, jnp.ndarray],
              loss_type: str) -> Tuple[jnp.ndarray, metrics.Collection]:
  """Compute the metrics for the given model in inference mode.

  The model is applied to the inputs with train=False using all devices on the
  host. Afterwards metrics are averaged across *all* devices (of all hosts).

  Args:
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    state: Replicate model state.
    batch: Inputs that should be evaluated.
    loss_type: Which loss to use (follows `config.loss_type`).

  Returns:
    Dictionary of the replicated metrics.
  """
  logging.info("eval_step(batch=%s)", batch)
  variables = {
      "params": state.params,
      "batch_stats": state.batch_stats,
      "constants": state.constants,
  }
  # Batch may contain positions only or the spherical representation.
  if "spheres" in batch:
    inputs = (batch["spheres"], batch["charges"], batch["positions"])
  else:
    inputs = (batch["positions"], batch["charges"])
  outputs = model.apply(
      variables, *inputs, mutable=False, train=False)

  normalized_labels = _normalize(
      batch["label"],
      mean=model.metadata["label_mean"],
      standard_deviation=model.metadata["label_standard_deviation"])

  predictions = _denormalize(
      outputs,
      mean=model.metadata["label_mean"],
      standard_deviation=model.metadata["label_standard_deviation"])

  # Eval batches can be zero-padded so normalization may cause NaNs. Zero-padded
  # entries will be ignored by `metrics.Average` according to batch['mask'], so
  # we avoid averaging over the batch dimension here.
  loss = _compute_loss(outputs, normalized_labels,
                       loss_type=loss_type,
                       average_over_batch=False)
  mean_squared_error, mean_absolute_error = _compute_metrics(
      predictions, batch["label"])

  mean_absolute_error *= model.metadata["unit_conversion_factor"]
  mean_squared_error *= model.metadata["unit_conversion_factor"] ** 2

  return (predictions,
          EvalMetrics.gather_from_model_output(
              loss=loss,
              mean_squared_error=mean_squared_error,
              mean_absolute_error=mean_absolute_error,
              mask=batch.get("mask"),
          ))


class StepTraceContextHelper:
  """Helper class to use jax.profiler.StepTraceAnnotation."""

  def __init__(self, name: str, init_step_num: int):
    self.name = name
    self.step_num = init_step_num
    self.context = None

  def __enter__(self):
    self.context = jax.profiler.StepTraceAnnotation(
        self.name, step_num=self.step_num
    )
    self.step_num += 1
    self.context.__enter__()
    return self

  def __exit__(self, exc_type, exc_value, tb):
    assert self.context is not None, "Exited context without entering."
    self.context.__exit__(exc_type, exc_value, tb)
    self.context = None

  def next_step(self):
    if self.context is None:
      raise ValueError("Must call next_step() within a context.")
    self.__exit__(None, None, None)
    self.__enter__()


def evaluate(model: nn.Module,
             state: TrainState,
             eval_ds: tf.data.Dataset,
             workdir: str,
             num_eval_steps: int = -1,
             loss_type: str = "l1") -> Union[None, EvalMetrics]:
  """Evaluate the model on the given dataset."""
  logging.info("Starting evaluation.")
  eval_metrics = None
  outputs = []
  with StepTraceContextHelper("eval", 0) as trace_annotation:
    for step, batch in enumerate(eval_ds):  # pytype: disable=wrong-arg-types
      batch = jax.tree_util.tree_map(np.asarray, batch)
      predictions, metrics_update = eval_step(model, state, batch, loss_type)

      output = {k: batch[k] for k in ["positions", "charges", "label"]
                if k in batch}
      output["predictions"] = predictions
      # `mask` indicates whether the batch was padded.
      mask = batch.get("mask")
      if mask is not None:
        output = {key: value[mask] for key, value in output.items()}
      outputs.append(output)

      metrics_update = flax_utils.unreplicate(metrics_update)
      eval_metrics = (
          metrics_update
          if eval_metrics is None else eval_metrics.merge(metrics_update))
      if num_eval_steps > 0 and step + 1 == num_eval_steps:
        break
      trace_annotation.next_step()

    outputs = {k: np.concatenate([output[k] for output in outputs])
               for k in outputs[0]}
    filename = f"{workdir}/outputs_{jax.process_index()}.npz"
    logging.info("Saving %s...", filename)
    with tf.io.gfile.GFile(filename, "wb") as outfile:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, **outputs)
      outfile.write(io_buffer.getvalue())

  return eval_metrics


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.

  Raises:
    ArithmeticError: In case NaN loss is detected. Only checked every
    `config.log_loss_every_steps`.
  """
  tf.io.gfile.makedirs(workdir)

  # Set JAX compilation cache.
  if not compilation_cache.is_initialized():
    compilation_cache_dir = os.path.join(os.path.dirname(workdir),
                                         "compilation_cache")
    compilation_cache.initialize_cache(compilation_cache_dir)

  rng = jax.random.PRNGKey(config.seed)

  # Build input pipeline.
  rng, data_rng = jax.random.split(rng)
  if not config.per_batch_padding:
    # To allow per-batch padding, `data_rng` must be the same for all hosts;
    # this is not necessary otherwise.
    data_rng = jax.random.fold_in(data_rng, jax.process_index())
  splits = input_pipeline.create_datasets(config, data_rng)
  train_iter = iter(splits.train)  # pytype: disable=wrong-arg-types

  # Learning rate schedule.
  num_train_steps = config.num_train_steps
  if num_train_steps == -1:
    num_train_steps = splits.train.cardinality().numpy()
  steps_per_epoch = num_train_steps // config.num_epochs
  logging.info("num_train_steps=%d, steps_per_epoch=%d", num_train_steps,
               steps_per_epoch)
  # We treat the learning rate in the config as the learning rate for batch size
  # 32 but scale it according to our batch size.
  global_batch_size = config.per_device_batch_size * jax.device_count()
  base_learning_rate = config.learning_rate * global_batch_size / 32.0
  learning_rate_fn = functools.partial(
      get_learning_rate,
      base_learning_rate=base_learning_rate,
      steps_per_epoch=steps_per_epoch,
      num_epochs=config.num_epochs,
      warmup_epochs=config.warmup_epochs)

  keys = jax.random.split(rng, len(splits.train.element_spec))
  random_batch = {}
  for k, rng in zip(splits.train.element_spec.keys(), keys):
    shape = splits.train.element_spec[k].shape
    dtype = splits.train.element_spec[k].dtype
    if dtype == tf.int64:
      random_batch[k] = tf.random.uniform(shape,
                                          minval=1, maxval=10,
                                          dtype=tf.int64)
    else:
      random_batch[k] = tf.random.normal(shape=shape, dtype=dtype, seed=rng[0])
  random_batch = jax.tree_util.tree_map(np.asarray, random_batch)

  # Gaussian representation is computed from positions.
  if config.molecule_to_sphere_method in ["gaussian", "gaussian_on_device"]:
    inputs = (random_batch["positions"], random_batch["charges"])
  else:
    inputs = (random_batch["spheres"], random_batch["charges"],
              random_batch["positions"])
  # First two dimensions are host and device when
  # config.per_batch_padding==True; else it's just the device.
  inputs = (jax.tree_util.tree_map(lambda x: x[0, 0], inputs)
            if config.per_batch_padding
            else jax.tree_util.tree_map(lambda x: x[0], inputs))
  rng, model_rng = jax.random.split(rng)
  model, optimizer, state = create_train_state(
      config,
      model_rng,
      metadata=splits.info.metadata,
      inputs=inputs,
      learning_rate_schedule=learning_rate_fn)

  # Set up checkpointing of the model and the input pipeline.
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Checkpoint the data loader (only works when molecule_to_sphere is computed
  # on device.).
  tf_state = (
      {"train_iter": train_iter}
      if config.molecule_to_sphere_method in ["gaussian", "gaussian_on_device"]
      else None)

  # To load from an arbitrary checkpoint, we simply copy it to the current
  # workdir.
  if config.load_checkpoint_from:
    target_dir = f"{checkpoint_dir}-{jax.process_index()}"
    load_checkpoint_from = os.path.join(config.load_checkpoint_from,
                                        f"checkpoints-{jax.process_index()}")
    shutil.copytree(load_checkpoint_from, target_dir)

  ckpt = checkpoint.MultihostCheckpoint(
      checkpoint_dir,
      tf_state=tf_state,
      max_to_keep=2)
  state = ckpt.restore_or_initialize(state)
  initial_step = int(state.step) + 1

  # Count number of trainable parameters. This must be done before replicating
  # the state to avoid double-counting replicated parameters.
  param_count = sum(p.size for p in jax.tree_util.tree_leaves(state.params))

  # Distribute training over local devices.
  state = flax_utils.replicate(state)

  _, model_rng = jax.random.split(rng)
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          model=model,
          optimizer=optimizer,
          weight_decay=config.weight_decay,
          loss_type=config.loss_type,
          rng=model_rng),
      axis_name=_PMAP_AXIS_NAME)

  # This triggers XLA compilation of the model for all possible input shapes. It
  # ensures that the compiled train_steps appear in the same order after restart
  # so the compilation_cache is hit. It also fails early in case of OOMs on
  # larger inputs.
  if config.per_batch_padding and config.precompile:
    for max_atoms in range(splits.info.metadata["min_atoms"],
                           splits.info.metadata["max_atoms"] + 1):
      batch = _unpad_batch_to_max_atoms(random_batch.copy(), max_atoms)
      p_train_step(state=state, batch=batch)

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0)
  if initial_step == 1:
    writer.write_hparams(dict(config))
    # Log the number of trainable params.
    writer.write_scalars(initial_step, {"param_count": param_count})

  logging.info("Starting training loop at step %d.", initial_step)
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer)
  if jax.process_index() == 0:
    hooks += [
        report_progress,
        periodic_actions.Profile(num_profile_steps=5, logdir=workdir)
    ]
  train_metrics = None
  with metric_writers.ensure_flushes(writer):
    for step in range(initial_step, num_train_steps + 1):
      # `step` is a Python integer. `state.step` is JAX integer on the GPU/TPU
      # devices.
      is_last_step = step == num_train_steps

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        batch = jax.tree_util.tree_map(np.asarray, next(train_iter))

        # Each host sees all data to decide on padding, then takes its own
        # part. Now batch has as extra dimension of size jax.process_count(),
        # and mol2sph is repeated in all hosts. This improves the training
        # speed.
        if config.per_batch_padding:
          max_atoms = int(np.count_nonzero(batch["charges"], axis=-1).max())
          batch = _unpad_batch_to_max_atoms(batch, max_atoms)

        state, metrics_update = p_train_step(state=state, batch=batch)
        metric_update = flax_utils.unreplicate(metrics_update)
        train_metrics = (
            metric_update
            if train_metrics is None else train_metrics.merge(metric_update))

      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
      for h in hooks:
        h(step)

      if step % config.log_loss_every_steps == 0 or is_last_step:
        # Write learning rate.
        if jax.process_index() == 0:
          writer.write_scalars(step, {"learning_rate": learning_rate_fn(step)})
        train_metrics = train_metrics.compute()
        if jnp.any(jnp.isnan(train_metrics["loss"])):
          raise ArithmeticError("NaNs found!")
        writer.write_scalars(step, train_metrics)
        train_metrics = None

      if step % config.eval_every_steps == 0 or is_last_step:
        with report_progress.timed("eval"):
          eval_metrics = evaluate(model, state,
                                  splits.validation,
                                  workdir,
                                  config.num_eval_steps,
                                  config.loss_type)
        if eval_metrics is not None:
          writer.write_scalars(step, eval_metrics.compute())

      if step % config.checkpoint_every_steps == 0 or is_last_step:
        with report_progress.timed("checkpoint"):
          ckpt.save(flax_utils.unreplicate(state))

      # Evaluate a single time on the test set when requested.
    if config.eval_on_test:
      with report_progress.timed("test"):
        test_metrics = evaluate(model, state,
                                splits.test,
                                workdir,
                                config.num_eval_steps,
                                config.loss_type)
      if test_metrics is not None:
        writer.write_scalars(num_train_steps, test_metrics.compute())

  logging.info("Finishing training at step %d", num_train_steps)
