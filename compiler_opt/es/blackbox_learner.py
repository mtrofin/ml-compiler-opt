# coding=utf-8
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Class for coordinating blackbox optimization."""

import functools
import os
import typing
from absl import logging
import concurrent.futures
import dataclasses
import gin
import math
import numpy as np
import numpy.typing as npt
import subprocess
import tempfile
import tensorflow as tf
from typing import Iterable, List, Optional, Protocol, TypeVar

from compiler_opt.distributed import buffered_scheduler
from compiler_opt.distributed.worker import FixedWorkerPool
from compiler_opt.es import blackbox_optimizers
from compiler_opt.es import policy_utils
from compiler_opt.rl import corpus
from compiler_opt.rl import policy_saver

# If less than 40% of requests succeed, skip the step.
_SKIP_STEP_SUCCESS_RATIO = 0.1
_REWARD_IF_TIMEOUT = -2.0


T = TypeVar('T')


def _interleave_list(values: Iterable[T],
                     interleaver=lambda x: -x) -> Iterable[T]:
  return [p for p in values for p in (p, interleaver(p))]


@gin.configurable
@dataclasses.dataclass(frozen=True)
class BlackboxLearnerConfig:
  """Hyperparameter configuration for BlackboxLearner."""

  # Total steps to train for
  total_steps: int

  # Name of the blackbox optimization algorithm
  blackbox_optimizer: blackbox_optimizers.Algorithm

  # What kind of ES training?
  #   - antithetic: for each perturbtation, try an antiperturbation
  #   - forward_fd: try total_num_perturbations independent perturbations
  est_type: blackbox_optimizers.EstimatorType

  # Should the rewards for blackbox optimization in a single step be normalized?
  fvalues_normalization: bool

  # How to update optimizer hyperparameters
  hyperparameters_update_method: blackbox_optimizers.UpdateMethod

  # Number of top performing perturbations to select in the optimizer
  # 0 means all
  num_top_directions: int

  # How many IR files to try a single perturbation on?
  num_ir_repeats_within_worker: int

  # How many times should we reuse IR to test different policies?
  num_ir_repeats_across_worker: int

  # How many IR files to sample from the test corpus at each iteration
  num_exact_evals: int

  # How many perturbations to attempt at each perturbation
  total_num_perturbations: int

  # How much to scale the stdev of the perturbations
  precision_parameter: float

  # Learning rate
  step_size: float


def _prune_skipped_perturbations(perturbations: List[npt.NDArray[np.float32]],
                                 rewards: List[Optional[float]],
                                 is_antithetic: bool):
  """Remove perturbations that were skipped during the training step.

  Perturbations may be skipped due to an early exit condition or a server error
  (clang timeout, malformed training example, etc). The blackbox optimizer
  assumes that each perturbations has a valid reward, so we must remove any of
  these "skipped" perturbations.

  Pruning occurs in-place.

  Args:
    perturbations: the model perturbations used for the ES training step.
    rewards: the rewards for each perturbation.

  Returns:
    The number of perturbations that were pruned.
  """
  indices_to_prune = []
  step = 2 if is_antithetic else 1
  next_to_check_offset = 1 if is_antithetic else 0
  for i in range(0, len(rewards), step):
    if rewards[i] is None or rewards[i + next_to_check_offset] is None:
      indices_to_prune.extend(set([i, i + next_to_check_offset]))

  # Iterate in reverse so that the indices remain valid
  for i in reversed(indices_to_prune):
    del perturbations[i]
    del rewards[i]

  return len(indices_to_prune)


class PolicySaverCallableType(Protocol):
  """Protocol for the policy saver function.
  A Protocol is required to type annotate
  the function with keyword arguments"""

  def __call__(self, parameters: npt.NDArray[np.float32],
               policy_name: str) -> None:
    ...


class BlackboxLearner:
  """Implementation of blackbox learning."""

  def __init__(self,
               blackbox_opt: blackbox_optimizers.BlackboxOptimizer,
               sampler: corpus.Corpus,
               tf_policy_path: str,
               output_dir: str,
               policy_saver_fn: PolicySaverCallableType,
               model_weights: npt.NDArray[np.float32],
               config: BlackboxLearnerConfig,
               initial_step: int = 0,
               seed: Optional[int] = None):
    """Construct a BlackboxLeaner.

    Args:
      blackbox_opt: the blackbox optimizer to use
      train_sampler: corpus_sampler for training data.
      tf_policy_path: where to write the tf policy
      output_dir: the directory to write all outputs
      policy_saver_fn: function to save a policy to cns
      model_weights: the weights of the current model
      config: configuration for blackbox optimization.
      stubs: grpc stubs to inlining/regalloc servers
      initial_step: the initial step for learning.
      deadline: the deadline in seconds for requests to the inlining server.
    """
    self._blackbox_opt = blackbox_opt
    self._sampler = sampler
    self._tf_policy_path = tf_policy_path
    self._output_dir = output_dir
    self._policy_saver_fn = policy_saver_fn
    self._model_weights = model_weights
    self._config = config
    self._step = initial_step
    self._seed = seed

    self._summary_writer = tf.summary.create_file_writer(output_dir)
    # hack
    self._baseline_scores: dict[str, float] = {
        'score-and-snippet-op2.o': 277067.5
    }

  def _get_perturbations(self) -> List[npt.NDArray[np.float32]]:
    """Get perturbations for the model weights."""
    perturbations = []
    rng = np.random.default_rng(seed=self._seed)
    for _ in range(self._config.total_num_perturbations):
      perturbations.append(
          rng.normal(size=(len(self._model_weights))) *
          self._config.precision_parameter / np.sqrt(len(self._model_weights)))
    return perturbations

  def _get_rewards(
      self, results: List[concurrent.futures.Future]) -> List[Optional[float]]:
    """Convert ES results to reward numbers."""
    rewards = [None] * len(results)

    for i in range(len(results)):
      if not results[i].exception():
        rewards[i] = results[i].result().reward
      # elif isinstance(results[i].exception(), subprocess.TimeoutExpired):
      #   logging.info('Timeout, recording penalty')
      #   rewards[i] = _REWARD_IF_TIMEOUT
      else:
        logging.info('Error retrieving result from future: %s',
                     str(results[i].exception()))

    return rewards

  def _update_model(self, perturbations: List[npt.NDArray[np.float32]],
                    rewards: List[float]) -> None:
    """Update the model given a list of perturbations and rewards."""
    self._model_weights = self._blackbox_opt.run_step(
        perturbations=np.array(perturbations),
        function_values=np.array(rewards),
        current_input=self._model_weights,
        current_value=np.mean(rewards))

  def _log_rewards(self, rewards: List[float]) -> None:
    """Log reward to console."""
    logging.info('Train reward: [%f]', np.mean(rewards))

  def _log_tf_summary(self, rewards: List[float]) -> None:
    """Log tensorboard data."""
    with self._summary_writer.as_default():
      tf.summary.scalar(
          'reward/average_reward_train', np.mean(rewards), step=self._step)

      tf.summary.histogram('reward/reward_train', rewards, step=self._step)

      train_regressions = [reward for reward in rewards if reward < 0]
      tf.summary.scalar(
          'reward/regression_probability_train',
          len(train_regressions) / len(rewards),
          step=self._step)

      tf.summary.scalar(
          'reward/regression_avg_train',
          np.mean(train_regressions) if len(train_regressions) > 0 else 0,
          step=self._step)

      # The "max regression" is the min value, as the regressions are negative.
      tf.summary.scalar(
          'reward/regression_max_train',
          min(train_regressions, default=0),
          step=self._step)

      train_wins = [reward for reward in rewards if reward > 0]
      tf.summary.scalar(
          'reward/win_probability_train',
          len(train_wins) / len(rewards),
          step=self._step)

  def _save_model(self) -> None:
    """Save the model."""
    logging.info('Saving the model.')
    self._policy_saver_fn(
        parameters=self._model_weights, policy_name=f'iteration{self._step}')

  def get_model_weights(self) -> npt.NDArray[np.float32]:
    return self._model_weights

  def _get_baseline_score(self, mod_name:str)->Optional[float]:
    if mod_name not in self._baseline_scores:
      return None
    return self._baseline_scores[mod_name]

  def _get_results(
      self, pool: FixedWorkerPool,
      perturbations: List[bytes]) -> List[concurrent.futures.Future]:
    samples:list[corpus.LoadedModuleSpec] = []
    samples = [
        self._sampler.load_module_spec(m)
        for m in self._sampler.sample(self._config.total_num_perturbations)
    ]
    if self._config.est_type == (
        blackbox_optimizers.EstimatorType.ANTITHETIC):
      samples = _interleave_list(samples, interleaver=lambda x: x)

    compile_args = zip(perturbations, samples)

    _, futures = buffered_scheduler.schedule_on_worker_pool(
        action=lambda w, v: w.es_compile(
            params=self._model_weights + v[0],
            loaded_module_spec=v[1],
            baseline_score=self._get_baseline_score(v[1].name)),
        jobs=compile_args,
        worker_pool=pool)

    not_done = futures
    # wait for all futures to finish
    while not_done:
      # update lists as work gets done
      _, not_done = concurrent.futures.wait(
          not_done, return_when=concurrent.futures.FIRST_COMPLETED)
    for i in range(len(futures)):
      name = samples[i].name
      if futures[i].exception() is None and name not in self._baseline_scores:
        self._baseline_scores[name] = futures[i].result().baseline

    return futures

  def run_step(self, pool: FixedWorkerPool) -> None:
    """Run a single step of blackbox learning.
    This does not instantaneously return due to several I/O
    and executions running while this waits for the responses"""
    logging.info('-' * 80)
    logging.info('Step [%d]', self._step)

    initial_perturbations = self._get_perturbations()
    # positive-negative pairs
    if self._config.est_type == blackbox_optimizers.EstimatorType.ANTITHETIC:
      initial_perturbations = _interleave_list(initial_perturbations)

    results = self._get_results(pool, initial_perturbations)
    rewards = self._get_rewards(results)

    num_pruned = _prune_skipped_perturbations(
        initial_perturbations, rewards,
        self._config.est_type == blackbox_optimizers.EstimatorType.ANTITHETIC)
    logging.info('Pruned [%d]', num_pruned)
    min_num_rewards = math.ceil(_SKIP_STEP_SUCCESS_RATIO * len(results))
    if len(rewards) < 1:
      logging.warning(
          'Skipping the step, too many requests failed: %d of %d '
          'train requests succeeded (required: %d)', len(rewards), len(results),
          min_num_rewards)
      return

    if np.std(np.array(rewards)) == 0:
      logging.warning('All the rewards have the same value. Skipping')
      return

    self._update_model(initial_perturbations, rewards)
    self._log_rewards(rewards)
    self._log_tf_summary(rewards)

    self._save_model()

    self._step += 1
