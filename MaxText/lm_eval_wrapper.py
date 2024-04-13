"""
Entrypoint for lm_eval harness
"""
import os
import time
import pathlib
from typing import Sequence, List
import numpy as np
import jax
from flax.linen import partitioning as nn_partitioning
import lm_eval
import pyconfig
from types import SimpleNamespace
from absl import app
from lm_eval.logging_utils import WandbLogger
from lm_eval.api.model import LM
import batch_maxengine as maxengine
from loglikelihood_helper import get_dataloader, model_ll
from jetstream.engine import token_utils
import tokenizer



""" Contains the MaxTextWrapperLMEvalBatched class, which acts a wrapper
    for a MaxText model to be evaluated on lm_eval tasks.
    The function run_lm_eval in this file is used to evaluate the model on lm_eval tasks."""

class MaxTextWrapperLMEvalBatched(LM):
  """
  The class is a wrapper for a MaxText model for lm_eval tasks.
  The class is used to evaluate the model on lm_eval tasks.
  """
  def __init__(self, config):
    """
    The function initializes the model, model variables, tokenizer and random
    number generator for lm_eval tasks
    """
    super().__init__()
    self.config = config
    #Note: max_prefill_predict_length should be greater than largest possible
    #prompt fed to the model in the eval
    self.max_len = config.max_prefill_predict_length
    engine = maxengine.MaxEngine(config)
    self.params = engine.load_params()
    metadata = engine.get_tokenizer()
    self.vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)
    self.tokenizer = self.vocab.tokenizer
    self.sp_tokenizer = tokenizer.load_tokenizer(tokenizer_path=config.tokenizer_path,
                                          add_bos=True,
                                          add_eos=True)
    self.engine = engine
    self.model = self.engine.model
    self.rng = self.engine.rng
    self.batch_sz = int(self.config.per_device_batch_size * jax.device_count())

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

    # this will contain the entire dataset
    num_requests = len(requests)

    # Pad up requests to be an integral multiple of global batch size
    if num_requests%self.batch_sz != 0:
      add_requests = self.batch_sz - num_requests % self.batch_sz
      requests += requests[:add_requests]

    # convert requests: List into a sharded dataloader that divides up the dataset
    # between the various devices
    eval_iter = get_dataloader(self.model, self.sp_tokenizer, requests, self.batch_sz, self.max_len)
    for batch in eval_iter:
      with self.model.mesh, nn_partitioning.axis_rules(self.model.config.logical_axis_rules):
        ll, is_greedy = model_ll(self.params, batch, self.rng,self.model)
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
    num_requests = len(requests)
    batch_sz = int(self.config.per_device_batch_size * jax.device_count())
    num_batches = int(np.ceil(len(requests)/batch_sz))
    if num_requests%batch_sz != 0:
      add_requests = batch_sz - num_requests % batch_sz
      requests += requests[:add_requests]
    
    prompts = [req.args[0] for req in requests]
    until = [req.args[1]['until'] for req in requests]
    all_tokens,all_true_lengths,sampled_tokens_list,outputs = [],[],[],[]

    token_start_time = time.time()
    for t in prompts:
      tokens, true_length = token_utils.tokenize_and_pad(t, self.vocab, is_bos=True,
                                                      prefill_lengths=[self.config.max_prefill_predict_length])
      all_tokens.append(tokens)
      all_true_lengths.append(true_length)
    token_end_time = time.time()
    print(f"Time taken for tokenization: {token_end_time - token_start_time:.2f}s")

    for i in range(num_batches):
      print(f"Batch {i+1}/{num_batches}")
      start_time = time.time()
      batch_tokens = all_tokens[i*batch_sz:(i+1)*batch_sz]
      true_lengths = all_true_lengths[i*batch_sz:(i+1)*batch_sz]
      sampled_tokens_list.append(self.generate_helper(batch_tokens,true_lengths))
      end_time = time.time()
      print(f"Time taken for batch: {end_time - start_time:.2f}s")

    
    get_device_time = time.time()
    all_results = []
    for sampled_token_batch in sampled_tokens_list:
      sampled_tokens_array = []
      for seq_idx in range(len(sampled_token_batch)):
        sampled_tokens_array.append(jax.device_get(sampled_token_batch[seq_idx].data)[:,0:1])
      sampled_tokens_array = np.concatenate(sampled_tokens_array,axis=1)
      all_results.append(sampled_tokens_array)
    
    all_results = np.concatenate(all_results,axis=0)
    
    end_device_time = time.time()
    print(f"Time taken for get_device: {end_device_time - get_device_time:.2f}s")
    
    detoken_start_time = time.time()
    for res_idx in range(len(all_results)):
        output = self.tokenizer.detokenize(all_results[res_idx].tolist())
        output = stop_sequence(output,until[res_idx])
        outputs.append(prompts[res_idx] + output)
    detoken_end_time = time.time()
    print(f"Time taken for detokenize: {detoken_end_time - detoken_start_time:.2f}s")
   
    return outputs[:num_requests]

  def generate_helper(self,batch_tokens,true_lengths):
    start_time = time.time()

    input_tokens, positions, segment_ids, true_length = self.engine.get_data(batch_tokens, true_lengths)

    prefill_result = self.engine.prefill(
    params=self.params, input_tokens=input_tokens, positions=positions, segment_ids=segment_ids, true_length=true_length
    )
    slot=0
    decode_state = self.engine.init_decode_state()
    decode_state = self.engine.insert(
        prefill_result, decode_state, slot=slot
    )

    prefill_end_time = time.time()
    print(f"Time taken for prefill: {prefill_end_time - start_time:.2f}s")
    steps = range(self.config.max_prefill_predict_length, self.config.max_target_length)
    sampled_tokens_list = []
    count = 0
    for _ in steps:
      count += 1
      step_start_time = time.time()
      decode_state, sampled_tokens = self.engine.generate(
        self.params, decode_state
      )
      sampled_tokens_list.append(sampled_tokens)
      step_end_time = time.time()
      #print(f"Time taken for one step: {step_end_time - step_start_time:.2f}s count: {count}")
    
    return sampled_tokens_list


def stop_sequence(text, until):
  """
  Function to clip the generated sequence at the first occurence of a stop string

  :param text: str
    - the generated sequence
  :param until: str
    - the stop string

  :return: str
    - the generated sequence clipped at the first occurence of the stop string
  """
  #TODO: Optimize by checking for "stop strings" in the decoding loop directly
  #- currently we generate a sequence of max_token_length, and then remove all
  # tokens after a stop string is found. Note that since we are batching,
  # the decoding loop must only stop generating for entries in the batch that
  # have hit a "stop string" but not others.
  for stop_str in until:
    if stop_str in text:
      text = text.split(stop_str)[0]

  return text


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
  lm_eval_results = lm_eval.evaluate(lm=lm_wrapper,task_dict=task_dict)
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
  pyconfig.initialize(argv)
  config = pyconfig.config

  start_time = time.time()

  lm_obj = MaxTextWrapperLMEvalBatched(config)
  task_manager = lm_eval.tasks.TaskManager(
    include_path=os.path.join(str(pathlib.Path(__file__).parent.parent),'lm_eval_harness_tasks'))
  results,_ = lm_evaluate_helper(lm_obj,['hellaswag'],task_manager,['0'])

  metrics = results['results']
  print(f'Results: {metrics}')

  end_time = time.time()
  print(f'Total time: {end_time - start_time}')
  wandb_logger = WandbLogger()  # or empty if wandb.init(...) already called before
  wandb_logger.post_init(results)
  wandb_logger.log_eval_result()
  wandb_logger.log_eval_samples(results["samples"])  # if log_samples

if __name__ == "__main__":
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  app.run(run_lm_eval)
