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
"""Build the whole corpus."""

import os
import tempfile
from typing import Dict, List, Optional, Union, Tuple  # pylint:disable=unused-import

from absl import app
from absl import flags
from absl import logging
import gin
import time
import tensorflow as tf

from compiler_opt.rl import compilation_runner
from compiler_opt.rl import corpus
from compiler_opt.rl import registry
from compiler_opt.distributed.local import local_worker_manager
from compiler_opt.distributed import buffered_scheduler, worker
import concurrent.futures

# see https://bugs.python.org/issue33315 - we do need these types, but must
# currently use them as string annotations

_CORPUS_DIR = flags.DEFINE_string('corpus_dir', None,
                                 'Path to folder containing IR files.')
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path', None, 'A dir to produce output under.')
_GIN_FILES = flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.')
_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')


def main(_):

  gin.parse_config_files_and_bindings(
      _GIN_FILES.value, bindings=_GIN_BINDINGS.value, skip_unknown=False)
  logging.info(gin.config_str())

  config = registry.get_configuration()

  csfdo_path = os.path.join(_CORPUS_DIR.value, 'merged.profdata')
  replace_flags = config.flags_to_replace()
  assert '-fprofile-instrument-use-path' not in replace_flags
  replace_flags['-fprofile-instrument-use-path'] = csfdo_path
  cps = corpus.Corpus(
      data_path=_CORPUS_DIR.value,
      additional_flags=config.flags_to_add() +
      ('-mllvm',
       f'-regalloc-profile-path={os.path.join(_CORPUS_DIR.value,"muppet_regalloc_perf_real.afdo")}'
      ),
      delete_flags=config.flags_to_delete(),
      replace_flags=replace_flags)
  logging.info('Done loading module specs from corpus.')

  corpus_elements = cps.module_specs
  class TheWorker(config.get_runner_type()):

    def __init__(self, output_path_root: str, *args, **kwargs):
      self._output_path = output_path_root
      super().__init__(*args, **kwargs)

    def build(self, lms: corpus.LoadedModuleSpec):
      with tempfile.TemporaryDirectory() as tmp:
        cmdline = lms.build_command_line(tmp)
        reldir = os.path.dirname(lms.name)
        modname = os.path.basename(lms.name)
        outputdir = os.path.join(self._output_path, reldir)
        os.makedirs(outputdir, exist_ok=True)
        stats_output = os.path.join(outputdir, modname + ".stats")
        cmdline += (
            '-mllvm',
            '-regalloc-dump-profile=' + stats_output,
        )
        try:
          t1 = time.time()
          self.compile_fn(
              command_line=cmdline, tf_policy_path='', reward_only=True)
          if os.stat(stats_output).st_size == 0:
            os.remove(stats_output)
          return time.time() - t1
        except BaseException as e:
          return e

  def work_factory(ms: corpus.ModuleSpec):
    def work(w: TheWorker):
      lms = cps.load_module_spec(module_spec=ms)
      return w.build(lms)
    return work

  with local_worker_manager.LocalWorkerPoolManager(
      worker_class=TheWorker, count=None,
      output_path_root=_OUTPUT_PATH.value,
      compilation_timeout=600) as pool:
    workers = pool.get_currently_active()
    work = [work_factory(ms) for ms in corpus_elements]
    all_work = buffered_scheduler.schedule(
        work=work, workers=workers, buffer=pool.get_worker_concurrency())
    completion_times = [v.result() for v in all_work]
    print(f'{max(completion_times)}')


if __name__ == '__main__':
  app.run(main)
