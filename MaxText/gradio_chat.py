"""
Alternative to the collab notebook to interact with and generate decodings from MaxText models 
"""

import os
import sys
import asyncio
from pathlib import Path
import gradio as gr
current_path = os.path.abspath('')
maxtext_path = os.path.join(Path(current_path).parent.parent,'MaxText')
root_path = Path(current_path).parent.parent
sys.path.append(str(maxtext_path))
from generate_helper import generate
import pyconfig
from generate_helper import generate
from decode import init_decode, validate_config
import tokenizer as tok

def get_model():
    """
    The function initializes the model, model variables, tokenizer and random
    """
    argv = ["",
            str(os.path.join(root_path,'MaxText/notebooks/inference.yml'))]

    pyconfig.initialize(argv)
    os.environ["TFDS_DATA_DIR"] = pyconfig.config.dataset_path
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    config = pyconfig.config
    validate_config(config)
    model, state, _, rng, _ = init_decode(config)

    model_vars = {'params': state.params}
    tokenizer = tok.load_tokenizer(tokenizer_path=model.config.tokenizer_path,
                                            add_bos=True,
                                            add_eos=False)
    return model, model_vars, tokenizer, rng

model, model_vars, tokenizer, rng = get_model()


async def generate_gradio(message,history):
    """Method passed to gradio interface to generate response for the given message"""
    out = generate(model, model_vars, tokenizer, rng,[message]*8)
    return out[0].replace(message,"",1)


async def main():
    """Main method that creates and launches the gradio interface"""
    demo = gr.ChatInterface(fn=generate_gradio, examples=["hello"],
                            title="MaxText Inference Chatbot")
    demo.launch(server_name='0.0.0.0')

if __name__ == '__main__':
    asyncio.run(main())
