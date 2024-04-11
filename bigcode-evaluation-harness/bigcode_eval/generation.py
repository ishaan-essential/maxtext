import json
from math import ceil

from typing import List, Optional

from accelerate.utils import set_seed
from torch.utils.data.dataloader import DataLoader
from transformers import StoppingCriteria, StoppingCriteriaList

from bigcode_eval.utils import TokenizedDataset, complete_code


def parallel_generations(
        task,
        dataset,
        accelerator,
        model,
        tokenizer,
        n_tasks,
        args,
        curr_sample_idx: int = 0,
        save_every_k_tasks: int = -1,
        intermediate_generations: Optional[List[Optional[List[Optional[str]]]]] = None,
        intermediate_save_generations_path: Optional[str] = None,
):
    if args.load_generations_path:
        # load generated code
        with open(args.load_generations_path) as fp:
            generations = json.load(fp)
            if accelerator.is_main_process:
                print(
                    f"generations loaded, {n_tasks} selected from {len(generations)} with {len(generations[0])} candidates"
                )
        return generations[:n_tasks]

    set_seed(args.seed, device_specific=True)

    # Setup generation settings
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_length": args.max_length_generation,
    }
    stopping_criteria = []
    # The input_length / start_length set to 0 for now will be adjusted later
    # Check if the task has a custom check_fn method for the stopping criteria
    if task.stop_words and tokenizer.eos_token: #TODO: Essential - change to our eos token
        task.stop_words.append(tokenizer.eos_token)    
    
    


    if args.instruction_tokens:
        instruction_tokens = args.instruction_tokens.split(",")
        if len(instruction_tokens) != 3:
            raise ValueError(
                "Instruction tokens should contain exactly 3 tokens separated by a comma. If a token is empty, represent it as ''"
            )
        for token in instruction_tokens:
            if token.strip() != "":
                task.stop_words.append(token)
    else:
        instruction_tokens = None

    gen_kwargs["stopping_criteria"] = task.stop_words 
   
    if accelerator.is_main_process:
        print(f"number of problems for this task is {n_tasks}")


    n_copies = 1 #ceil(args.n_samples / args.batch_size)

    ds_tokenized = TokenizedDataset(
        task,
        dataset,
        tokenizer,
        num_devices=accelerator.state.num_processes,
        max_length=args.max_length_generation,
        limit_start=args.limit_start + curr_sample_idx,
        n_tasks=n_tasks,
        n_copies=n_copies,
        prefix=args.prefix,
        has_encoder=args.modeltype == "seq2seq",
        instruction_tokens=instruction_tokens,
    )

    generations = complete_code(
        task,
        accelerator,
        model,
        tokenizer,
        ds_tokenized.get_data(),
        n_tasks=n_tasks,
        limit_start=args.limit_start + curr_sample_idx,
        num_samples=args.n_samples,
        prefix=args.prefix,
        instruction_tokens=instruction_tokens,
        postprocess=args.postprocess,
        is_wrapped=False,
        save_every_k_tasks=save_every_k_tasks,
        intermediate_generations=intermediate_generations,
        intermediate_save_generations_path=intermediate_save_generations_path,
        **gen_kwargs,
    )
    return generations
