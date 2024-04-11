"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

# pylint: disable=g-bad-todo, abstract-method, consider-using-with

import os
import sys
import pathlib
from typing import Sequence
import numpy as np
from absl import app
import entrypoint_helper
import tokenizer
from generate_helper import generate
from decode import init_decode, validate_config
sys.path.append(os.path.join(str(pathlib.Path(__file__).parent.parent),
                             'bigcode-evaluation-harness'))
from run_bigcode_evals import evaluate_code

"""Contains functions to initialize the model and decode the sequences for the Bigcode Evaluation Harness tasks.
   Contains the MaxTextWrapperLMEvalBatched class, which acts a wrapper
   for a MaxText model to be evaluated on lm_eval tasks.
   The function run_lm_eval in this file is used to evaluate the model on lm_eval tasks.
"""

class MaxTextWrapperBigcode:
  """
  The class is a wrapper for a MaxText model for lm_eval tasks.
  The class is used to evaluate the model on lm_eval tasks.
  """
  def __init__(self,config):
    """
    The function initializes the model, model variables, tokenizer and random
    number generator for lm_eval tasks
    """
    self.batch_sz = int(config.eval_per_device_batch_size)
    #Note: max_prefill_predict_length should be greater than
    #largest possible prompt fed to the model in the eval
    self.model, state, _ , self.rng, _ = init_decode(config)
    self.model_vars = {'params':state.params}
    self.tokenizer = tokenizer.load_tokenizer(tokenizer_path=config.tokenizer_path,
                                          add_bos=True,
                                          add_eos=False)

  def update_state(self, model_vars):
    """ used in training loop to set latest checkpoint
        model_vars is a pytree (dict of all layers : weights)
    """
    self.model_vars = model_vars

  def generate(self,inputs,num_return_sequences,**kwargs):
    """
    Generates the completions for the given input
    """
    stopping_criteria = kwargs.get('stopping_criteria',None)
    max_length = kwargs.get('max_length',self.model.config.max_target_length)
    prompts = np.array([inp['prompt'] for inp in inputs]).repeat(num_return_sequences)
    num_batches = int(np.ceil(len(prompts)/self.batch_sz))
    if stopping_criteria:
      until = [stopping_criteria]*self.batch_sz
    else:
      until = None
    outputs = []
    for i in range(num_batches):
      batch_prompts = prompts[i*self.batch_sz: (i+1)*self.batch_sz]
      outputs += generate(self.model,self.model_vars,self.tokenizer,
                          self.rng,batch_prompts,until=until,max_gen_toks=max_length)

    outputs = np.array(outputs).reshape((len(inputs),num_return_sequences))
    return outputs


def run_bigcode_eval(argv : Sequence[str]) -> None:
  """
  Entrypoint for lm_eval_harness
  """
  config = entrypoint_helper.read_cmd_line_get_config(argv)
  validate_config(config)
  bigcode_obj = MaxTextWrapperBigcode(config)
  evaluate_code(bigcode_obj,'humaneval',n_samples=1)


if __name__ == "__main__":
  app.run(run_bigcode_eval)
