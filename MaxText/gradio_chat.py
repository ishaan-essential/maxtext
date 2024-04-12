"""
Alternative to the collab notebook to interact with and generate decodings from MaxText models 
"""

import os
import sys
import asyncio
import gradio as gr
import jax
import jax.numpy as jnp
import time
import numpy as np
import pyconfig
import batch_maxengine as maxengine
from jetstream.engine import token_utils



def get_config():
    argv = sys.argv
    pyconfig.initialize(argv)
    config = pyconfig.config
    return config


def get_model(config):
    engine = maxengine.MaxEngine(config)
    params = engine.load_params()
    metadata = engine.get_tokenizer()
    vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)
    tokenizer = vocab.tokenizer
    return engine, params, vocab, tokenizer


config = get_config()
engine, params, vocab, tokenizer = get_model(config)

def generate_helper(text):
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

    all_outputs = []
    for text_idx,prompt in enumerate(text):
        results = [sampled_tokens.get_result_at_slot(text_idx).tokens.item() for sampled_tokens in sampled_tokens_list]
        output = tokenizer.detokenize(results)
        all_outputs.append(output)

    return all_outputs


async def generate_gradio(message,history):
    """Method passed to gradio interface to generate response for the given message"""
    out = generate_helper([message])
    return out[0]


async def main():
    """Main method that creates and launches the gradio interface"""
    demo = gr.ChatInterface(fn=generate_gradio, examples=["hello"],
                            title="MaxText Inference Chatbot")
    demo.launch(server_name='0.0.0.0')

if __name__ == '__main__':
    asyncio.run(main())
