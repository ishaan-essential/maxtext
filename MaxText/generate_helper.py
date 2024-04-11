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

import time
from decode_utils_lmeval import decode_loop_lmeval
from decode import decode_tokens


"""
Contains functions to generate sequences from a MaxText model.
Used in the MaxTextWrapperLMEvalBatched class in lm_eval_wrapper.py to
evaluate a MaxText model on LM Evaluaton Harness tasks.
"""

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

def generate(model, model_vars, tokenizer, rng, prompt, max_gen_toks=2048, until=None):
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

  completion = decode_loop_lmeval(model, model_vars, tokenizer, rng, prompt)
  decoded = []
  batch_sz = len(prompt)

  for i in range(batch_sz):
    seq = [int(x[i, 0]) for x in completion]
    clipped_seq = seq[:max_gen_toks]
    new_text, _ = decode_tokens(clipped_seq, tokenizer)
    if until is not None:
      new_text = stop_sequence(new_text, until[i])
    decoded.append(prompt[i] + new_text)

  return decoded
