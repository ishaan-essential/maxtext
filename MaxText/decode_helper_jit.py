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
"""Training loop and Decoding of the model."""


import functools
from typing import Sequence
import datetime
from flax.linen import partitioning as nn_partitioning
import flax
import orbax
import time
import os
from absl import app
import numpy as np
import pyconfig
import max_utils
import inference_utils
from layers import models, quantizations
import common_types
import jax
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh
from jax.experimental.compilation_cache import compilation_cache as cc
import max_logging
import tokenizer
from multihost_dataloading import  _form_global_array
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
cc.initialize_cache(os.path.expanduser("~/jax_cache"))

Transformer = models.Transformer

def replicate_globally(np_array, mesh):
  return  _form_global_array(None,np_array,mesh)

def match_input_and_output_stream(prompt, outputs, sp_tokenizer):
  for i in range(len(prompt)):
    prompt_mini = prompt[0:i+1]
    prompt_mini_arr = np.array(prompt_mini, dtype=np.int32)
    prompt_mini_str = decode_tokens(prompt_mini_arr, sp_tokenizer)
    output_mini = outputs[i:i+1]
    output_mini_arr = np.array(output_mini, dtype=np.int32)
    output_mini_str = decode_tokens(output_mini_arr, sp_tokenizer)
    print(f"{prompt_mini_str} -> {output_mini_str}")

def decode_tokens(toks, sp_tokenizer):
  return sp_tokenizer.detokenize(toks).numpy().decode("utf-8"), len(toks)

def default_prompts(config):
  return [config.prompt] * int(config.per_device_batch_size * jax.device_count())


def encode_strings(strs, max_len, sp_tokenizer, mesh):
  """Pack prefill prompts into Jax.Array. The prompts are `right-aligned`, i.e. padded with zeros and all ending on the same
     index."""
  tokenized_batch = np.zeros((len(strs), max_len), np.int32)
  positions = np.zeros((len(strs), max_len), np.int32)
  segment_ids = np.zeros((len(strs), max_len), np.int32)

  for i, s in enumerate(strs):
    toks = sp_tokenizer.tokenize(s).numpy()
    assert toks.shape[0] <= max_len, f"We aren't able to tokenize input {i}, it is too long"
    prompt = toks
    start_index = max_len - prompt.shape[0]
    tokenized_batch[i, start_index:] = prompt
    padded_start_index = start_index
    segment_ids[i, padded_start_index:] = common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR
    positions[i, padded_start_index:] = np.arange(len(prompt))
  return replicate_globally(tokenized_batch, mesh), \
      replicate_globally(positions, mesh), \
      replicate_globally(segment_ids, mesh)



@functools.partial(jax.jit, static_argnums=(5,))
def prefill_predict_step(inputs, input_positions, decoder_segment_ids,
                         model_vars, rngkey, model=None):
  """Prefill KV Cache and output logits"""
  flat_logits, new_vars = model.apply(
    model_vars,
    inputs,
    input_positions,
    decoder_segment_ids=decoder_segment_ids,
    enable_dropout=False,
    model_mode=common_types.MODEL_MODE_PREFILL,
    rngs={'params': rngkey},
    mutable=True
  )
  prefill_cache = new_vars['cache']
  return flat_logits, prefill_cache, None


def compute_prefill(config, model, model_vars, prompts, rng, sp_tokenizer, mesh):
  """Compute the necessary prefill state."""

  # Encode the demo prompt -- to measure performance we encode it multiple times.
  tokenized_prompts, prompt_decoder_positions, prompt_decoder_segment_ids = encode_strings(
    prompts,config.max_prefill_predict_length, sp_tokenizer, mesh)
  prefill_output, prefill_cache, aqt_vars = prefill_predict_step(
    tokenized_prompts, prompt_decoder_positions, prompt_decoder_segment_ids, model_vars, rng, model=model)
  with jax.spmd_mode('allow_all'):
    updated_prompt_decoder_positions = prompt_decoder_positions[:, -1:] + 1

  return prefill_cache, prefill_output[:, -1:], updated_prompt_decoder_positions, aqt_vars

def prefill_or_load(config, model, model_vars, prompts, rng, sp_tokenizer, mesh):
  """We either load the necessary prefill state or generate it.  """
  cache, last_logit, pos, _ = compute_prefill(config, model, model_vars,
                                              prompts, rng, sp_tokenizer, mesh)
  max_logging.log(f"Computed prefill cache {config.prefill_cache_dir}")
  return cache, last_logit, pos


def init_decode(config):
  """Initialize decode model, vars and tokennizer."""
  rng = random.PRNGKey(0)
  # Mesh definition
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  # Model definition
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, mesh = mesh, quant=quant)
  # Tokenizer
  sp_tokenizer = tokenizer.load_tokenizer(tokenizer_path=config.tokenizer_path,
                                          add_bos=True,
                                          add_eos=False)
  # Load model vars
  params = max_utils.load_decode_model_vars(model, config, rng, mesh)
  num_params, bytes_params, bytes_per_param = max_utils.summarize_size_from_pytree(params)
  max_logging.log(f"Number of model params loaded ={num_params/10**9:.3f} billion, memory usage={bytes_params/2**30:.3f}GB, "
                  f"bytes per param={bytes_per_param:.3f}")
  # Update aqt state
  return model, params, sp_tokenizer, rng



def validate_config(config):
  assert config.load_full_state_path == "", "Decode doesn't operate on full states! Convert to parameter checkpoint first."\
                                            "Using generate_param_only_checkpoint."



class DecodeHelper:

  def __init__(self, model, rng):
    mesh = model.mesh
    config = model.config
    self.count = 0
    self.replicated_sharding = jax.sharding.NamedSharding(mesh, P(None)) #jax.sharding.NamedSharding(mesh, P(None))
    kv_cache_annotations = max_utils.get_kv_cache_annotations(model, config, rng, mesh)
    self.kv_cache_mesh_shardings = jax.tree_map(
        lambda p: jax.sharding.NamedSharding(mesh, p), kv_cache_annotations)
    

    

  def generate(self,model, model_vars, sp_tokenizer, rng, prompt, max_gen_toks=2048, until=None):
    """

    :param model: layers.models.Transformer
    :param model_vars: dict
      - dictionary of model parameters
    :param tokenizer: Tokenizer
    :param rng: jax.random.PRNGKey
    :param prompt: list[str]
    :param max_gen_toks: int
      - maximum number of tokens to output for each prompt
    :param until: list[str]
      - list of stop strings for each prompt

    :return: list[str]
      - list of generated sequences (inclusive of the the input prompt)
    """

    completion = self.decode_loop(model, model_vars, sp_tokenizer, rng, prompt)
    decoded = []
    batch_sz = len(prompt)

    for i in range(batch_sz):
      seq = [int(x[i, 0]) for x in completion]
      clipped_seq = seq[:max_gen_toks]
      new_text, _ = decode_tokens(clipped_seq, sp_tokenizer)
      if until is not None:
        new_text = stop_sequence(new_text, until[i])
      decoded.append(prompt[i] + new_text)

    return decoded

  @functools.partial(jax.jit, static_argnums=(0,1,2))
  def decode_ar_one_step(self, config, model, model_vars, new_cache, pos, logits, rng):
    """Compute the necessary prefill state."""
    new_token = inference_utils.sampling(logits, rng, config.decode_sampling_strategy,\
                                        topk=config.decode_sampling_top_k, nucleus_topp=config.decode_sampling_nucleus_p,
                                        temperature=config.decode_sampling_temperature)
    
    
    flat_logits, new_vars = model.apply(
      model_vars | new_cache,
      new_token,
      pos,
      enable_dropout=False,
      model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
      rngs={'params': rng},
      mutable=['cache'])

    #new_cache = jax.lax.with_sharding_constraint(new_vars["cache"], self.kv_cache_mesh_shardings)
    return pos+1, new_vars, flat_logits, new_token


  def decode_loop(self, model, model_vars, sp_tokenizer, rng, prompts):
    """Decoding loop for the Transformer model."""
    mesh = model.mesh
    config = model.config

    num_params, bytes_params, bytes_per_param = max_utils.summarize_size_from_pytree(model_vars['params'])
    max_logging.log(f"Number of model params={num_params/10**9:.3f} billion, memory usage={bytes_params/2**30:.3f}GB, "
                    f"bytes per param={bytes_per_param:.3f}")

    bytes_aqt_params = 0
    if model.quant:
      num_params, bytes_params, bytes_per_param = max_utils.summarize_size_from_pytree(model_vars['params'])
      max_logging.log(f"Number of model params after quantization={num_params/10**9:.3f} billion, "
                      f"memory usage={bytes_params/2**30:.3f}GB, "
                      f"bytes per param={bytes_per_param:.3f}")

      num_aqt_params, bytes_aqt_params, bytes_per_aqt_param = max_utils.summarize_size_from_pytree(model_vars['aqt'])
      max_logging.log(f"Number of aqt params={num_aqt_params/10**9:.3f} billion, "
                      f"memory usage={bytes_aqt_params/2**30:.3f}GB, "
                      f"bytes per aqt param={bytes_per_aqt_param:.3f}")

    # Compute shardings
    prefill_start = time.time()
    prefill_cache, next_logit, new_position = prefill_or_load(
      config, model, model_vars, prompts, rng, sp_tokenizer, mesh)
    prefill_end = time.time()
    print(f"Time taken for prefill: {prefill_end-prefill_start}s")
    
    num_cache, bytes_cache, bytes_per_cache = max_utils.summarize_size_from_pytree(prefill_cache)
    max_logging.log(f"Number of cache entries={num_cache/10**9:.3f} billion, memory usage={bytes_cache/2**30:.3f}GB, "
                    f"bytes per cache={bytes_per_cache:.3f}")

    total_memory_GB = (bytes_params + bytes_aqt_params + bytes_cache)/2**30
    max_logging.log(f"Total memory for cache and params (and any quantization state) {total_memory_GB:.3f} GB")

    new_cache = prefill_cache
    first_profiling_step = config.max_prefill_predict_length + config.skip_first_n_steps_for_profiler
    last_profiling_step = np.clip(first_profiling_step + config.profiler_steps - 1,
                                  first_profiling_step, config.max_target_length - 1)

    outputs = []
    max_logging.log("Generate first predicted token")
    first_token_start = time.time()
    new_position, new_cache, next_logit, selected_id = self.decode_ar_one_step(
      config, model, model_vars, {'cache':new_cache}, new_position, next_logit, rng)
    outputs.append(selected_id)
    first_token_end = time.time()
    print(f"Time taken for first token: {first_token_end-first_token_start}s")
    jax.block_until_ready(new_cache)

    starttime = datetime.datetime.now()
    steps = range(config.max_prefill_predict_length + 1, config.max_target_length)
    rngs = [random.PRNGKey(i) for i in range(config.max_target_length)]
    max_logging.log(f"Generate remaining {len(steps)} predicted tokens")

    
    count = 0
    for step in steps:
      count += 1
      start_time = time.time()
      if step == first_profiling_step:
        max_utils.activate_profiler(config)
      new_position, new_cache, next_logit, selected_id = self.decode_ar_one_step(
      config, model, model_vars, new_cache, new_position, next_logit, rngs[step])
      outputs.append(jax.device_get(selected_id))
      if step == last_profiling_step:
        jax.block_until_ready(outputs)
        max_utils.deactivate_profiler(config)
      end_time = time.time()
      print(f"Time taken for {count}: {end_time-start_time}s")


    return outputs



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


