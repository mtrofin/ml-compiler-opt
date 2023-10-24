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
"""Module for collect data of inlining-for-size."""

import os
import tempfile
from typing import Dict, Tuple

import gin
import tensorflow as tf

from compiler_opt.rl import compilation_runner
from compiler_opt.rl import corpus
from compiler_opt.rl import log_reader

_DEFAULT_IDENTIFIER = 'default'


@gin.configurable(module='runners')
class InliningRunner(compilation_runner.CompilationRunner):
  """Class for collecting data for inlining-for-size.

  Usage:
  inliner = InliningRunner(
                clang_path, llvm_size_path, launcher_path,
                moving_average_decay_rate)
  serialized_sequence_example, default_reward, moving_average_reward,
  policy_reward = inliner.collect_data(
      ir_path, tf_policy_path, default_reward, moving_average_reward)
  """

  def __init__(self, llvm_size_path: str, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._llvm_size_path = llvm_size_path

  def compile_fn(
      self, command_line: corpus.FullyQualifiedCmdLine, tf_policy_path: str,
      reward_only: bool,
      workdir: str) -> Dict[str, Tuple[tf.train.SequenceExample, float]]:
    """Run inlining for the given IR file under the given policy.

    Args:
      command_line: the fully qualified command line.
      tf_policy_path: path to TF policy direcoty on local disk.
      reward_only: whether only return native size.

    Returns:
      A dict mapping from example identifier to tuple containing:
        sequence_example: A tf.SequenceExample proto describing compilation
        trace, None if reward_only == True.
        native_size: Native size of the final native code.

    Raises:
      subprocess.CalledProcessError: if process fails.
      compilation_runner.ProcessKilledError: (which it must pass through) on
      cancelled work.
      RuntimeError: if llvm-size produces unexpected output.
    """

    working_dir = tempfile.mkdtemp(dir=workdir)

    log_path = '/dev/null'
    output_native_path = '/dev/null'
    score_dump = os.path.join(working_dir, 'scores')

    score = 0
    cmdline = []
    if self._launcher_path:
      cmdline.append(self._launcher_path)
    cmdline.extend([self._clang_path] + list(command_line) + [
        '-mllvm', '-enable-ml-inliner=development', '-mllvm', '-training-log=' +
        log_path, '-o', output_native_path, '-mllvm', '-dump-reward=' +
        score_dump
    ])
    if tf_policy_path:
      cmdline.extend([
          '-mllvm',
          '-ml-inliner-model-under-training=' + tf_policy_path,
          '-mllvm',
          '-inliner-scc-level=1',
      ])
    compilation_runner.start_cancellable_process(cmdline,
                                                 self._compilation_timeout,
                                                 self._cancellation_manager)
    with open(score_dump) as f:
      lines = f.readlines()
      latency = float(lines[0].strip())
      cache_pressure = float(lines[1].strip())
      print(f'latency: {latency}, pressure: {cache_pressure}')
    score = latency

    if score == 0:
      return {}

    if reward_only:
      return {_DEFAULT_IDENTIFIER: (None, score)}

    result = log_reader.read_log_as_sequence_examples(log_path)
    if len(result) != 1:
      return {}
    sequence_example = next(iter(result.values()))

    if not sequence_example.HasField('feature_lists'):
      return {}

    return {_DEFAULT_IDENTIFIER: (sequence_example, score)}
