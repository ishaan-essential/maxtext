import json
import math
import re
import warnings
from collections import defaultdict
from typing import List, Optional

import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm
import numpy as np
import essential_wrapper




INFILL_MODE = False
INSTRUCTION_MODE = False




class TokenizedDataset:
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially. See compute_code for more details.
    The prompt can either be:
    - one prompt: normal code completion
    - two prompts: for infilling mode (prefix, suffix) or instructin-tuning mode (instruction, context)
    """

    def __init__(
        self,
        task,
        dataset,
        tokenizer,
        num_devices,
        max_length,
        limit_start=0,
        n_tasks=None,
        n_copies=1,
        prefix="",
        has_encoder=False,
        instruction_tokens=None,
    ):
        self.task = task
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_devices = num_devices
        self.max_length = max_length
        self.limit_start = limit_start
        self.n_tasks = n_tasks
        self.n_copies = n_copies
        self.prefix = prefix
        self.has_encoder = has_encoder
        self.instruction_tokens = instruction_tokens

    def get_data(self):
        prompts = []
        prompts_encoder = []
        infill = []
        instruction = []
        for sample in range(self.limit_start, self.limit_start + self.n_tasks):
            prompt_contents = self.task.get_prompt(self.dataset[sample])
            if isinstance(prompt_contents, str):
                # Normal code completion mode
                infill.append(False)
                instruction.append(False)
                prompt = self.prefix + prompt_contents
            elif isinstance(prompt_contents, dict):
                if set(prompt_contents.keys()) == {"prefix", "suffix"}:
                    # Infilling mode
                    infill.append(True)
                    instruction.append(False)
                    prompt = self._make_infill_prompt(
                        **prompt_contents, preprefix=self.prefix
                    )
                elif set(prompt_contents.keys()) == {"instruction", "context"}:
                    # Instruction-tuning mode
                    instruction.append(True)
                    infill.append(False)
                    prompt = self._make_instruction_prompt(
                        **prompt_contents, prefix=self.prefix
                    )
            else:
                raise ValueError(f"Unsupported prompt format: {type(prompt_contents)}")
            prompts.append(prompt)
            if self.has_encoder:
                prompt_encoder = self.task.get_prompt_encoder(self.dataset[sample])
                if isinstance(prompt_encoder, str):
                    prompt_encoder = self.prefix + prompt_encoder
                prompts_encoder.append(prompt_encoder)

        if not len(set(infill)) == 1 or not len(set(instruction)) == 1:
            raise ValueError(
                "Mixed infill/instruction and completion prompts are not supported."
            )
        global INFILL_MODE
        global INSTRUCTION_MODE
        INFILL_MODE = infill[0]
        INSTRUCTION_MODE = instruction[0]
        if INFILL_MODE:
            return_token_type_ids = False
        else:
            return_token_type_ids = None  # default
        
        inputs = []


        """
        if self.n_copies == 1 and self.n_tasks % self.num_devices != 0:
            self.n_copies = 2
            warnings.warn(
                "n_copies (n_samples/batch_size) was changed from 1 to 2 because n_tasks isn't proportional to num devices"
            )
        """

        assert self.n_copies == 1, 'Num copies should be 1 in the Tokenized Dataset'

        for sample in range(self.n_tasks):
            for _ in range(self.n_copies):
                inputs.append({
                    "prompt": prompts[sample],
                    "task_id": sample
                })
        return inputs

    def _make_infill_prompt(self, prefix, suffix, preprefix=""):
        """Make a prompt for infilling.
        Currently supported only for official InCoder and SantaCoder implementations.
        """
        model_id = self.tokenizer.name_or_path
        if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            return f"{preprefix}{prefix}<|mask:0|>{suffix}<|mask:0|>"
        elif model_id in ["bigcode/santacoder"]:
            return f"<fim-prefix>{preprefix}{prefix}<fim-suffix>{suffix}<fim-middle>"
        elif model_id in ["bigcode/starcoder", "bigcode/starcoderbase"]:
            return f"<fim_prefix>{preprefix}{prefix}<fim_suffix>{suffix}<fim_middle>"
        else:
            raise ValueError(f"Infilling not yet supported for: {model_id}")

    def _make_instruction_prompt(self, instruction, context, prefix=""):
        """Make a prompt for instruction-tuning. Delimit instruction and context with specific tokens if provided."""
        if not self.instruction_tokens:
            warnings.warn(
                "Instruction-tuning tokens are not provided for an instruction-tuning task, we will leave them empty."
            )
            user_token, end_token, assistant_token = "", "", "\n"
        else:
            user_token, end_token, assistant_token = self.instruction_tokens
            if not user_token or not assistant_token or not end_token:
                warnings.warn(
                    "Instruction-tuning tokens provided but one or more are empty. Ignore warning if this was intended"
                )
        prompt = (
            prefix + user_token + instruction + end_token + assistant_token + context
        )

        return prompt


def _parse_infill(code, tokenizer):
    """Reorder infill code and remove remaining special tokens."""
    model_id = tokenizer.name_or_path
    if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
        prefix, suffix, infill = code.split("<|mask:0|>", 2)
        infill = infill.split("<|endofmask|>")[0]
    elif model_id in ["bigcode/santacoder"]:
        prefix, rest = code.split("<fim-suffix>", 1)
        suffix, infill = rest.split("<fim-middle>", 1)
        infill = infill.split("<|endoftext|>")[0]
    elif model_id in ["bigcode/starcoder", "bigcode/starcoderbase"]:
        prefix, rest = code.split("<fim_suffix>", 1)
        suffix, infill = rest.split("<fim_middle>", 1)
        infill = infill.split("<|endoftext|>")[0]
    else:
        raise ValueError(f"Infilling not yet supported for: {model_id}")
    for k, v in tokenizer.special_tokens_map.items():
        if k == "additional_special_tokens":
            for t in v:
                infill = infill.replace(t, "")
        else:
            infill = infill.replace(v, "")
    return infill


def _parse_instruction(code, instruction_tokens):
    """Return code block after assistant_token/end_token"""
    _, end_token, assistant_token = instruction_tokens
    if not assistant_token and end_token:
        assistant_token = end_token
    elif not assistant_token and not end_token:
        return code

    idx = code.find(assistant_token)
    shift = len(assistant_token)
    if idx == -1:
        warnings.warn(
            "The assistant token was not detected in the generation, this might disrupt the post-processing and lead to lower evaluation scores"
        )
        return code

    if "```python" in assistant_token:
        idx = code.find("```python", idx)
        shift = len("```python")
    return code[idx + shift :]


def complete_code(
    task,
    accelerator,
    model,
    tokenizer,
    dataset,
    n_tasks,
    limit_start=0,
    num_samples=20,
    prefix="",
    instruction_tokens=None,
    postprocess=True,
    is_wrapped=False,
    save_every_k_tasks: int = -1,
    intermediate_generations: Optional[List[Optional[List[Optional[str]]]]] = None,
    intermediate_save_generations_path: Optional[str] = None,
    **gen_kwargs,
):
    """Generate multiple codes for each task in the dataset using multiple GPUs with accelerate.
    dataloader sends all the prompts from the evalution dataset to the model as the following:
    [p_0_0, p_0_1, ..., p_0_nc-1, p_1_0, ..., p_nt-1_nc-1] where nc is the number of copies of the prompt,
    and nt is the number of tasks. nc is such that num_samples(for each task)= nc * batch_size
    """
    # keep track of the list of generated codes
    # where len(code_gens) = n_tasks and len(code_gens[0]) = number of generated code samples
    code_gens: List[List[Optional[str]]] = [[] for _ in range(n_tasks)]
    generations = [] if not intermediate_generations else intermediate_generations
    gen_code_dict = defaultdict(list)  # dict of list of generated tokens

    print(f'len(dataset): {len(dataset)}')

   
    
    generated_code = essential_wrapper.eai_model.generate(
        dataset,
        num_return_sequences=num_samples,
        **gen_kwargs,
    )

    for task_index in range(len(dataset)):
        task_id = dataset[task_index]['task_id']
        gen_code_dict[task_id] = generated_code[task_index]

    

    code_gens = update_code_gens(
        task,
        tokenizer,
        limit_start,
        prefix,
        instruction_tokens,
        postprocess,
        code_gens,
        gen_code_dict,
    )

    

    generations.extend(code_gens)
    return generations


def update_code_gens(
    task,
    tokenizer,
    limit_start,
    prefix,
    instruction_tokens,
    postprocess,
    code_gens,
    gen_code_dict,
):  
    for sample, generated_code in gen_code_dict.items():
        for gen_code in generated_code:
            gen_code = gen_code[len(prefix) :]

            if postprocess:
                code_gens[sample].append(
                    task.postprocess_generation(gen_code, int(sample) + limit_start)
                )
            else:
                warnings.warn(
                    "model output is not postprocessed, this might lower evaluation scores"
                )
                code_gens[sample].append(gen_code)
    return code_gens


def remove_after_return(code):
    """
    Takes as input a code, and removes everything that is after the return.
    That is, the first line that does not start with a space character
    """
    pattern = r"[^\n]+(\n|$)"
    end_last_match = None
    # Go trough the regex to match any sequence of characters ending with a \n
    for match in re.finditer(pattern, code):
        start_match, end_match = match.span()
        # Search for the first line which does not start by a space character
        if (
            end_last_match is not None
            and start_match < len(code)
            and code[start_match].strip() != ""
        ):
            return code[0: start_match]
        end_last_match = end_match
    return code
