import os
import fnmatch
import json
import warnings

import datasets
import torch
import transformers
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
)

from bigcode_eval.arguments import EvalArguments
from bigcode_eval.evaluator import Evaluator
from bigcode_eval.tasks import ALL_TASKS
import essential_wrapper


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def get_arguments(eai_wrapper,tasks,n_samples,huggingface_dummy_model='codeparrot/codeparrot-small',modeltype='causal',peft_model=None,revision=None,use_auth_token=False,
                trust_remote_code=False, instruction_tokens=None,batch_size=1,max_length_generation=512,precision='fp32',load_in_8bit=False,load_in_4bit=False,left_padding=False,
                limit=None,limit_start=0,save_every_k_tasks=-1,postprocess=True,allow_code_execution=True,generation_only=False,load_generations_path=None,load_data_path=None,
                metric_output_path='evaluation_results.json',save_generations=True,load_generations_intermediate_paths=None,save_generations_path='generations.json',
                save_references=False,save_references_path='references.json',prompt='prompt',max_memory_per_gpu=None,check_references=False,temperature=0.2,top_k=0,top_p=0.95,
                eos="<|endoftext|>",seed=0,do_sample=True,prefix=""):

    args = EvalArguments()
    essential_wrapper.eai_model = eai_wrapper
    args.model = huggingface_dummy_model
    args.modeltype = modeltype
    args.peft_model = peft_model
    args.revision = revision
    args.use_auth_token = use_auth_token
    args.trust_remote_code = trust_remote_code
    args.tasks = tasks
    args.instruction_tokens = instruction_tokens
    args.batch_size = batch_size
    args.max_length_generation = max_length_generation
    args.precision = precision
    args.load_in_8bit = load_in_8bit
    args.load_in_4bit = load_in_4bit
    args.left_padding = left_padding
    args.limit = limit
    args.limit_start = limit_start
    args.save_every_k_tasks = save_every_k_tasks
    args.postprocess = postprocess
    args.allow_code_execution = allow_code_execution
    args.generation_only = generation_only
    args.load_generations_path = load_generations_path
    args.load_data_path = load_data_path
    args.metric_output_path = metric_output_path
    args.save_generations = save_generations
    args.load_generations_intermediate_paths = load_generations_intermediate_paths
    args.save_generations_path = save_generations_path
    args.save_references = save_references
    args.save_references_path = save_references_path
    args.prompt = prompt
    args.max_memory_per_gpu = max_memory_per_gpu
    args.check_references = check_references
    args.temperature = temperature
    args.top_k = top_k
    args.top_p = top_p
    args.n_samples = n_samples
    args.eos = eos
    args.seed = seed
    args.do_sample = do_sample
    args.prefix = prefix

    return args


def pattern_match(patterns, source_list):
    """Returns a list containing all values of the source_list that
    match at least one of the patterns"""
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def get_gpus_max_memory(max_memory, num_gpus):
    max_memory = {i: max_memory for i in range(num_gpus)}
    print("Loading model via these GPUs & max memories: ", max_memory)
    return max_memory


def evaluate_code(eai_wrapper,tasks,n_samples):
    args = get_arguments(eai_wrapper,tasks,n_samples)
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()

    if args.tasks is None:
        task_names = ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), ALL_TASKS)

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Selected Tasks: {task_names}")

    results = {}
    if args.load_generations_path:
        # here we don't generate code but only evaluate previously computed generations
        if accelerator.is_main_process:
            print("evaluation only mode")
        evaluator = Evaluator(accelerator, None, None, args)
        for task in task_names:
            results[task] = evaluator.evaluate(task)
    else:
        # here we generate code and save it (evaluation is optional but True by default)
        dict_precisions = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if args.precision not in dict_precisions:
            raise ValueError(
                f"Non valid precision {args.precision}, choose from: fp16, fp32, bf16"
            )

        model_kwargs = {
            "revision": args.revision,
            "trust_remote_code": args.trust_remote_code,
            "use_auth_token": args.use_auth_token,
        }
        if args.load_in_8bit:
            print("Loading model in 8bit")
            model_kwargs["load_in_8bit"] = args.load_in_8bit
            model_kwargs["device_map"] = {"": accelerator.process_index}
        elif args.load_in_4bit:
            print("Loading model in 4bit")
            model_kwargs["load_in_4bit"] = args.load_in_4bit
            model_kwargs["device_map"] = {"": accelerator.process_index}
        else:
            print(f"Loading model in {args.precision}")
            model_kwargs["torch_dtype"] = dict_precisions[args.precision]

            if args.max_memory_per_gpu:
                if args.max_memory_per_gpu != "auto":
                    model_kwargs["max_memory"] = get_gpus_max_memory(
                        args.max_memory_per_gpu, accelerator.num_processes
                    )
                    model_kwargs["offload_folder"] = "offload"
                else:
                    model_kwargs["device_map"] = "auto"
                    print("Loading model in auto mode")

        if args.modeltype == "causal":
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                **model_kwargs,
            )
        elif args.modeltype == "seq2seq":
            warnings.warn(
                "Seq2Seq models have only been tested for HumanEvalPack & CodeT5+ models."
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model,
                **model_kwargs,
            )
        else:
            raise ValueError(
                f"Non valid modeltype {args.modeltype}, choose from: causal, seq2seq"
            )

        if args.peft_model:
            from peft import PeftModel  # dynamic import to avoid dependency on peft

            model = PeftModel.from_pretrained(model, args.peft_model)
            print("Loaded PEFT model. Merging...")
            model.merge_and_unload()
            print("Merge complete.")

        if args.left_padding:
            # left padding is required for some models like chatglm3-6b
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                revision=args.revision,
                trust_remote_code=args.trust_remote_code,
                use_auth_token=args.use_auth_token,
                padding_side="left",  
            )
        else:
            # used by default for most models
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                revision=args.revision,
                trust_remote_code=args.trust_remote_code,
                use_auth_token=args.use_auth_token,
                truncation_side="left",
                padding_side="right",  
            )
        if not tokenizer.eos_token:
            if tokenizer.bos_token:
                tokenizer.eos_token = tokenizer.bos_token
                print("bos_token used as eos_token")
            else:
                raise ValueError("No eos_token or bos_token found")
        try:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Some models like CodeGeeX2 have pad_token as a read-only property
        except AttributeError:
            print("Not setting pad_token to eos_token")
            pass
        WIZARD_LLAMA_MODELS = [
            "WizardLM/WizardCoder-Python-34B-V1.0",
            "WizardLM/WizardCoder-34B-V1.0",
            "WizardLM/WizardCoder-Python-13B-V1.0"
        ]
        if args.model in WIZARD_LLAMA_MODELS:
            tokenizer.bos_token = "<s>"
            tokenizer.bos_token_id = 1
            print("Changing bos_token to <s>")

        evaluator = Evaluator(accelerator, model, tokenizer, args)

        if (
            args.load_generations_intermediate_paths
            and len(args.load_generations_intermediate_paths) != len(task_names)
        ):
            raise ValueError(
                "If passing --load_generations_intermediate_paths, \
                must pass equal number of files as number of tasks"
            )

        for idx, task in enumerate(task_names):
            intermediate_generations = None
            if args.load_generations_intermediate_paths:
                with open(args.load_generations_intermediate_paths[idx], "r") as f_in:
                    # intermediate_generations: list[list[str | None]] of len n_tasks
                    # where list[i] = generated codes or empty
                    intermediate_generations = json.load(f_in)

            if args.generation_only:
                if accelerator.is_main_process:
                    print("generation mode only")
                generations, references = evaluator.generate_text(
                    task, intermediate_generations=intermediate_generations
                )
                if accelerator.is_main_process:
                    save_generations_path = f"{os.path.splitext(args.save_generations_path)[0]}_{task}.json"
                    save_references_path = f"references_{task}.json"
                    evaluator.save_json_files(
                        generations,
                        references,
                        save_generations_path,
                        save_references_path,
                    )
            else:
                results[task] = evaluator.evaluate(
                    task, intermediate_generations=intermediate_generations
                )

    # Save all args to config
    results["config"] = vars(args)
    if not args.generation_only:
        dumped = json.dumps(results, indent=2)
        if accelerator.is_main_process:
            print(dumped)

        with open(args.metric_output_path, "w") as f:
            f.write(dumped)
    return results



