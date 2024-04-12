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
from typing import Sequence, List
import numpy as np
from absl import app
from types import SimpleNamespace
import pyconfig
sys.path.append(os.path.join(str(pathlib.Path(__file__).parent.parent),
                             'bigcode-evaluation-harness'))
from run_bigcode_evals import evaluate_code
from lm_eval_wrapper import MaxTextWrapperLMEvalBatched
from lm_eval.logging_utils import WandbLogger
import pandas as pd
import json
import wandb

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

    self.lm_wrapper = MaxTextWrapperLMEvalBatched(config)


  def generate(self,inputs,num_return_sequences,**kwargs):
    """
    Generates the completions for the given input
    """
    stopping_criteria = kwargs.get('stopping_criteria',None)
    prompts = np.array([inp['prompt'] for inp in inputs]).repeat(num_return_sequences)
    if stopping_criteria:
      until = stopping_criteria
    else:
      until = []

    requests = [SimpleNamespace(args=[prompts[i],{'until':until}]) for i in range(len(prompts))]
    
    outputs = self.lm_wrapper.generate_until(requests)
    outputs = np.array(outputs).reshape((len(inputs),num_return_sequences))
    return outputs



def log_eval_samples(logger):
  "Write out the generations and references as a table into wandb"
  # generations will be written to json files as "generations_humaneval.json" etc
  # references to an equivalent references_humaneval.json
  task_names: List[str] = [
      x for x in logger.task_names if x not in logger.group_names
  ]
  for task in task_names:
    # open references_{task}.json and generations_{task}.json
    with open (f"references_{task}.json", "r", encoding='utf-8') as f:
      references = json.load(f)
    with open (f"generations_{task}.json", "r", encoding='utf-8') as g:
      generations = json.load(g)
    data = [x[0] for x  in generations]
    ids = list(range(len(data)))
    input_lens = [len(x) for x in data]
    df_data = {
      "generation" : data,
      "test" : references,
      "input_len" : input_lens,
      "id" : ids
    }
    df = pd.DataFrame(df_data)
    logger.run.log({f"{task}_eval_results": df})

def log_to_wandb(results_dict):
  wandb.define_metric("infer_eval_step")
  wandb_logger = WandbLogger()  # or empty if wandb.init(...) already called before
  wandb_logger.post_init(results_dict)
  step = 1
  wandb_logger.run.log({"infer_eval_step" : step}, commit=False)
  wandb_logger.log_eval_result()
  log_eval_samples(wandb_logger) # create dataframe with generations

def run_bigcode_eval(argv : Sequence[str]) -> None:
  """
  Entrypoint for lm_eval_harness
  """
  import time
  start_time = time.time()
  pyconfig.initialize(argv)
  config = pyconfig.config
  bigcode_obj = MaxTextWrapperBigcode(config)
  taskname = 'mbpp'
  results = evaluate_code(eai_wrapper=bigcode_obj,
                          tasks=taskname,
                          n_samples=1,
                          save_references=True,
                          save_generations=True)

  # convert into a form suitable matching lm_eval
  # and reusing its own wandb logger
  task_names = [taskname]
  results_all_tasks = {'results' : {}, 'groups' : {}, 'versions' : {},
          'configs': {}, 'config' : {}, 'n-shot' : {}}
  results_all_tasks['config'] = results['config']
  for task in task_names:
    results_all_tasks['results'].update({task : results[task]})
    results_all_tasks['n-shot'].update({task : 0})
    results_all_tasks['versions'].update({task : 1.0})
  wandb.init() 
  log_to_wandb(results_all_tasks)
  end_time = time.time()
  print(f'Total time: {end_time - start_time}')


if __name__ == "__main__":
  app.run(run_bigcode_eval)
