# Copyright 2024 Google LLC
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

'''CLI Utility for Running Inference on a Single Stream'''

import jax
import jax.numpy as jnp

import time
import batch_maxengine as maxengine
from jetstream.engine import token_utils

import os
import pyconfig
import sys
import numpy as np



def main(config):
  engine = maxengine.MaxEngine(config)
  params = engine.load_params()

  text = ['Roger Federer is ']
  metadata = engine.get_tokenizer()
  vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)
  tokenizer = vocab.tokenizer
  batch_tokens,true_lengths = [],[]
  for t in text:
    tokens, true_length = token_utils.tokenize_and_pad(t, vocab, is_bos=True,
                                                     prefill_lengths=[config.max_prefill_predict_length])
    assert tokens.size <= config.max_prefill_predict_length, "can't take too many tokens"
    batch_tokens.append(tokens)
    true_lengths.append(true_length)
  
  prefill_result = engine.prefill(
      params=params, padded_tokens=batch_tokens, true_length=true_lengths
  )
  slot=0

  decode_state = engine.init_decode_state()
  decode_state = engine.insert(
      prefill_result, decode_state, slot=slot
  )

  steps = range(config.max_prefill_predict_length, config.max_target_length)
  sampled_tokens_list = []
  count = 0
  for _ in steps:
    count += 1
    start_time = time.time()
    decode_state, sampled_tokens = engine.generate(
      params, decode_state
    )
    
    sampled_tokens_list.append(sampled_tokens)
    end_time = time.time()
    print(f"Time taken for one step: {end_time - start_time:.2f}s count: {count}")

  for text_idx,prompt in enumerate(text):
    results = [sampled_tokens.get_result_at_slot(text_idx).tokens.item() for sampled_tokens in sampled_tokens_list]
    output = tokenizer.detokenize(results)
    print(f"Input `{prompt}` -> `{output}`")

  if config.autoregressive_decode_assert != "":
    assert output==config.autoregressive_decode_assert, \
    f"generated text mismatch {output=} {config.autoregressive_decode_assert=}"

def validate_config(config):
  assert config.load_full_state_path == "", "Decode doesn't operate on full states! Convert to parameter checkpoint first."\
                                            "Using generate_param_only_checkpoint."

if __name__ == "__main__":
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(sys.argv)
  cfg = pyconfig.config
  validate_config(cfg)
  main(cfg)