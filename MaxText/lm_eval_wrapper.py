"""
Entrypoint for lm_eval harness
"""
import os
import time
import pathlib
import functools
from typing import Sequence, List
from types import SimpleNamespace
import numpy as np
from generate_helper import generate
from decode import init_decode, validate_config
from flax.linen import partitioning as nn_partitioning
from jax.sharding import PartitionSpec as P
import jax
import lm_eval
from absl import app
from lm_eval.logging_utils import WandbLogger
from lm_eval.api.model import LM
from loglikelihood_helper import get_dataloader, model_ll
import tokenizer
import entrypoint_helper


""" Contains the MaxTextWrapperLMEvalBatched class, which acts a wrapper
    for a MaxText model to be evaluated on lm_eval tasks.
    The function run_lm_eval in this file is used to evaluate the model on lm_eval tasks."""

class MaxTextWrapperLMEvalBatched(LM):
  """
  The class is a wrapper for a MaxText model for lm_eval tasks.
  The class is used to evaluate the model on lm_eval tasks.
  """
  def __init__(self, config, model = None,
               model_vars = None, rng = None, state_mesh_annotations = None):
    """
    The function initializes the model, model variables, tokenizer and random
    number generator for lm_eval tasks
    """
    super().__init__()

    self.batch_sz = int(config.eval_per_device_batch_size)

    #Note: max_prefill_predict_length should be greater than largest possible
    #prompt fed to the model in the eval
    self.max_len = config.max_prefill_predict_length
    self.tokenizer = tokenizer.load_tokenizer(tokenizer_path=config.tokenizer_path,
                                          add_bos=True,
                                          add_eos=True)

    if model is not None and\
       model_vars is not None and\
       rng is not None and\
       state_mesh_annotations is not None:
      self.model = model
      self.model_vars = model_vars
      self.rng = rng
      self.state_mesh_annotations = state_mesh_annotations
    else:
      self.model, self.model_vars, self.tokenizer, self.rng, state_mesh_annotations = init_decode(config)
    self.state_mesh_shardings = jax.tree_map(
        lambda p: jax.sharding.NamedSharding(self.model.mesh, p), state_mesh_annotations)
    self.data_sharding = jax.tree_map(
      lambda p: jax.sharding.NamedSharding(self.model.mesh, p), P(*self.model.config.data_sharding))

  def update_state(self, model_vars):
    """ used in training loop to set latest checkpoint
        model_vars is a pytree (dict of all layers : weights)
    """
    self.model_vars = model_vars

  def loglikelihood(self, requests: List) -> list[tuple[float, bool]]:
    """
    Computes the loglikelihood of completions given the context. is_greedy
    is a boolean indicating whether the completion is the most likely output of the model.
    Called: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/evaluator.py#L294

    :param requests: list[Request]
        - the list of requests to be processed and is contains the entire dataset in the list

    :return: list[tuple[float, bool]]
        - the list of loglikelihoods and is_greedy for each request
    """
    outputs = []

    import ipdb; ipdb.set_trace()

    # this will contain the entire dataset
    num_requests = len(requests)

    # Pad up requests to be an integral multiple of global batch size
    if num_requests%self.batch_sz != 0:
      add_requests = self.batch_sz - num_requests % self.batch_sz
      requests += requests[:add_requests]

    # convert requests: List into a sharded dataloader that divides up the dataset
    # between the various devices
    eval_iter = get_dataloader(self.model, self.tokenizer, requests, self.batch_sz, self.max_len)
    partial_model_ll = functools.partial(model_ll, model=self.model)
    p_model_ll = jax.jit(
    partial_model_ll,
    in_shardings=(self.state_mesh_shardings, self.data_sharding, None),
    out_shardings=(self.data_sharding,self.data_sharding),
    )

    for batch in eval_iter:
      with self.model.mesh, nn_partitioning.axis_rules(self.model.config.logical_axis_rules):
        ll, is_greedy = p_model_ll(self.model_vars, batch, self.rng)
        ll = jax.device_get(jax.experimental.multihost_utils.process_allgather(ll))
        is_greedy = jax.device_get(jax.experimental.multihost_utils.process_allgather(is_greedy))
        # output is will identical across all host processes
        outputs += [(ll[j].item(),is_greedy[j].item()) for j in range(len(ll))]

    return outputs[:num_requests]

  def loglikelihood_rolling(self, requests : List) -> list[float]:
    """
    Computes the loglikelihoods of the whole sequence.
    Hence, we can use self.loglikelihood as a subroutine.
    self.loglikelihood expects requests with (prompt,completion) tuples as input.
    In our case, prompt is an empty string and the completion is the whole sequence
    whose loglikelihood we want to compute.

    :param requests: list[Request]
        - the list of requests to be processed

    :return: list[tuple[float]]
        - the list of loglikelihoods for each request
    """

    #Pass an empty string as the prompt in each request to self.loglikelihood
    ll_requests = [SimpleNamespace(args=['',req.args[0]]) for req in requests]
    ll_outputs = self.loglikelihood(ll_requests)
    return [ll for ll,_ in ll_outputs]

  def generate_until(self, requests) -> list[str]:
    """
    Generates completions for the requests

    :param requests: list[Request]
        - the list of requests to be processed

    :return: list[str]
        - the list of completions for each request
    """

    outputs = []
    num_batches = int(np.ceil(len(requests)/self.batch_sz))
    self.tokenizer = tokenizer.load_tokenizer(
      tokenizer_path=self.model.config.tokenizer_path,
      add_bos=True,
      add_eos=False)

    for i in range(num_batches):
      start_time = time.time()
      batch_reqs = requests[i*self.batch_sz: (i+1)*self.batch_sz]

      temperature = np.array([req.args[1]['temperature'] for req in batch_reqs])
      assert np.all(temperature == 0), "temperature is non-zero"
      prompts = [req.args[0] for req in batch_reqs]
      until = [req.args[1]['until'] for req in batch_reqs]
      outputs += generate(self.model, {'params':self.model_vars.params},
                          self.tokenizer, self.rng, prompts, until=until)
      end_time = time.time()
      print(f'count: {i},  total: {num_batches}, time: {end_time - start_time}')
      import ipdb; ipdb.set_trace()
    return outputs


def lm_evaluate_helper(lm_wrapper,tasks,task_manager,num_fewshots):
  """
  Helper function to do pre-processing before calling lm_eval.evaluate
  """
  task_metrics = { "scalar" : {}}
  
  task2shots,task_dict = {},{}
  for task_idx,task in enumerate(tasks):
    single_task_dict = lm_eval.tasks.get_task_dict([task], task_manager)
    for k in single_task_dict.keys():
      task2shots[k] = int(num_fewshots[task_idx])
    task_dict.update(single_task_dict)
  for task_name,task_obj in task_dict.items():
    num_fewshot = task2shots[task_name]
    if isinstance(task_obj, tuple):
      _, task_obj = task_obj
      if task_obj is None:
        continue
    task_obj.set_config(key="num_fewshot", value=num_fewshot)
  lm_eval_results = lm_eval.evaluate(lm=lm_wrapper,task_dict=task_dict,limit=80)
  lm_eval_metrics = ['acc,none','acc_norm,none']
  for task_name in lm_eval_results['results']:
    for metric_name in lm_eval_metrics:
      if metric_name in lm_eval_results['results'][task_name]:
        task_metrics['scalar'].update({f"evaluation/lm_eval/{task_name}/{metric_name}":
                                        lm_eval_results['results'][task_name][metric_name]})
 
    
  return lm_eval_results,task_metrics


def run_lm_eval(argv : Sequence[str]) -> None:
  """
  Entrypoint for lm_eval_harness
  """
  config = entrypoint_helper.read_cmd_line_get_config(argv)
  validate_config(config)

  start_time = time.time()

  lm_obj = MaxTextWrapperLMEvalBatched(config)
  task_manager = lm_eval.tasks.TaskManager(
    include_path=os.path.join(str(pathlib.Path(__file__).parent.parent),'lm_eval_harness_tasks'))
  results,_ = lm_evaluate_helper(lm_obj,['random_task'],task_manager,config.eval_n_shots.split())

  metrics = results['results']
  print(f'Results: {metrics}')

  end_time = time.time()
  print(f'Total time: {end_time - start_time}')
  wandb_logger = WandbLogger()  # or empty if wandb.init(...) already called before
  wandb_logger.post_init(results)
  wandb_logger.log_eval_result()
  wandb_logger.log_eval_samples(results["samples"])  # if log_samples

if __name__ == "__main__":
  app.run(run_lm_eval)
