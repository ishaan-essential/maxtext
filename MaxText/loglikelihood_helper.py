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

from typing import Any, Dict, Iterable
import dataclasses
import functools
import tensorflow as tf
import max_utils
import common_types
import jax
import jax.numpy as jnp
import multihost_dataloading
from jetstream.engine import token_utils

AUTOTUNE = tf.data.experimental.AUTOTUNE


"""Contains helper functions to calculate the loglikelihood of the completions
   given the context and the loglikelihood of the entire sequence.
   Used in the MaxTextWrapperLMEvalBatched class in lm_eval_wrapper.py to
   evaluate a MaxText model on LM Evaluaton Harness tasks."""

@functools.partial(jax.jit, static_argnums=(3,))
def model_ll(state, batch, rng, model):
  """
  Passes the tokenized sentences through the model to calculate the
  loglikelihood of the completion given the context.

  Args:
    :param model: layers.models.Transformer
    :param model_vars: dict
      - dictionary of model parameters
    :param tokenized_sentences: jnp.array
      - tokenized sentences
    :param sentence_decoder_positions: jnp.array
      - positions of the tokens
    :param sentence_decoder_segment_ids: jnp.array
      - segment ids of the tokens (0 for padding, 1 for active sequence)
    :param completion_mask: jnp.array
      - mask for the completion (1 for completion, 0 for padding or context)
    :param rng: jax.random.PRNGKey

  Returns:
    :return loglikelihood: jnp.array
      - loglikelihood of the completion given the context
    :return is_greedy: jnp.array
      - boolean array indicating if the completion was the most-likely response of the model
  """

  tokenized_sentences = batch['inputs']
  sentence_decoder_positions = batch['decoder_positions']
  sentence_decoder_segment_ids = batch['decoder_segment_ids']
  completion_mask = batch['mask']

  flat_logits, _ = model.apply(
    state,
    tokenized_sentences,
    sentence_decoder_positions,
    decoder_segment_ids=sentence_decoder_segment_ids,
    enable_dropout=False,
    model_mode=common_types.MODEL_MODE_TRAIN,
    rngs={'params': rng},
    mutable=True
  )

  input_logits = flat_logits[:,:-1]
  targets = tokenized_sentences[:,1:]

  one_hot_targets = jax.nn.one_hot(targets, model.config.vocab_size)
  xent, _ = max_utils.cross_entropy_with_logits(input_logits, one_hot_targets, 0.0)
  #xent = xent * (sentence_decoder_segment_ids[:,:-1] != 0)


  xent = xent * completion_mask
  xent = jnp.sum(xent, axis=1)
  ll = -xent # loglikelihood


  # is greedy calculation - determine if the completion was the most-likely response of the model
  preds = jnp.argmax(input_logits, axis=-1)
  eqs = jnp.sum((preds == targets)*(completion_mask != 0),axis=1)
  is_greedy = (eqs/jnp.sum(completion_mask, axis=-1)) == 1


  return ll, is_greedy


Features = Dict[str, tf.Tensor]

@dataclasses.dataclass
class TokenizeLLOp:
  """
  Tokenize the contexts and full_sentences efficiently within dataset.map() 
  """

  sp_tokenizer: Any
  data_keys: Iterable[str] = ('contexts', 'full_sentences')

  def __call__(self, features: Features) -> Features:
    for k in self.data_keys:
      features[k] = self.sp_tokenizer.tokenize(features[k])
    return features

@dataclasses.dataclass
class PadTensorOp:
  """
  Compute and pad the tokenized sequences, position and segment ids to max_len 
  """

  max_len: int

  def __call__(self, features: Features) -> Features:
    start_index = self.max_len - tf.shape(features['full_sentences'])[0]
    sentence_length = tf.shape(features['full_sentences'])[0]
    context_length = tf.shape(features['contexts'])[0]
    padding_constant = tf.constant(0, features['full_sentences'].dtype)

    #pad to the left with 0s until start_index, no padding on the right
    padded = tf.pad(features['full_sentences'],[[start_index,0]],mode='constant',
              constant_values=padding_constant)
    segment_ids = tf.pad(tf.fill([sentence_length],
                          common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR),
                          [[start_index,0]],mode='constant',
                          constant_values=padding_constant)
    positions = tf.pad(tf.range(sentence_length),[[start_index,0]],
            mode='constant',constant_values=padding_constant)
    features['inputs'] = padded
    features['decoder_positions'] = positions
    features['decoder_segment_ids'] = segment_ids

    #TODO: +1 shouldn't be added if EOS token is not added
    completion_size = sentence_length - context_length + 1
    features['mask'] = tf.pad(tf.fill([completion_size],1.0),[[self.max_len - completion_size - 1,0]],
                              mode='constant',constant_values=tf.constant(0.0))
    del features['contexts']
    del features['full_sentences']
    return features
  
def encode_pair(context, continuation):
  """
  The function encodes the context and continuation into a single sequence of tokens
  Args:
  :param context: str
    - the context
  :param continuation: str
    - the continuation to the context

  Returns:
  :return: tuple
    - context: str
      - the context
    - whole: str
      - the context and continuation encoded into a single sequence of tokens
  """
  n_spaces = len(context) - len(context.rstrip())
  if n_spaces > 0:
    continuation = context[-n_spaces:] + continuation
    context = context[:-n_spaces]

  whole = context + continuation

  return context, whole

def get_dataloader(model,sp_tokenizer,requests,batch_sz,max_len):
  """ Created a sharded dataloader from the generator """
  cc_tuples = [req.args for req in requests]
  sequences = [encode_pair(*cc_tup) for cc_tup in cc_tuples]
  contexts = [s[0] for s in sequences]
  full_sentences = [s[1] for s in sequences]
  dataset = tf.data.Dataset.from_tensor_slices({'contexts': contexts, 'full_sentences':full_sentences})
  dataset = dataset.map(TokenizeLLOp(sp_tokenizer), num_parallel_calls=AUTOTUNE)
  tf_dataset = dataset.map(PadTensorOp(max_len),num_parallel_calls=AUTOTUNE)
  tf_dataset = tf_dataset.shard(num_shards = jax.process_count(), index = jax.process_index())
  tf_dataset = tf_dataset.batch(batch_sz, drop_remainder=False)
  tf_dataset = tf_dataset.prefetch(AUTOTUNE)
  return multihost_dataloading.MultiHostDataLoadIterator(tf_dataset, model.mesh)
