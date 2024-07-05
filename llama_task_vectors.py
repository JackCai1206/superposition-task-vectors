from dataclasses import dataclass

from tqdm import tqdm
from tasks import Dataset
import torch
from transformer_lens import HookedTransformer
from transformer_lens.utilities.devices import get_device_for_block_index
from transformers import AutoTokenizer, HfArgumentParser, set_seed
import os
from matplotlib import pyplot as plt
from utils import get_task_vectors_from_dataset, mixed_dataset_residual_main, task_vec_PCA_main, task_vec_interpolation_main

@dataclass
class ScriptArguments:
    model_id: str
    num_examples: int
    prompt_size: int
    task1: str
    task2: str
    task3: str = None
    task1_alias: str = None
    task2_alias: str = None
    task3_alias: str = None
    average_over: int = 1
    use_task_vec_cache: bool = True
    use_results_cache: bool = True
    do_interpolation: bool = False
    do_residual: bool = False
    do_tv_PCA: bool = False
    full_range_operands: bool = True
    num_operands: int = 2
    skip_title: bool = False
    out_dir: str = 'out'

if __name__ == '__main__':
    set_seed(42)
    
    args: ScriptArguments = HfArgumentParser((ScriptArguments)).parse_args()

    # Specify the model ID
    # model_id = "meta-llama/Llama-2-7b-hf" 
    # model_id = "meta-llama/Meta-Llama-3-8B"
    # model_id = 'Qwen/Qwen1.5-7B'

    # Load the tokenizer from the Hugging Face library
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Load the model with TransformerLens, specifying the torch_dtype as torch.float16 to reduce memory usage
    model = HookedTransformer.from_pretrained_no_processing(args.model_id, torch_dtype=torch.bfloat16)
    model.eval()
    device = get_device_for_block_index(0, model.cfg)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    layers = list(range(model.cfg.n_layers))
    dataset_1 = Dataset({args.task1: 1}, args.num_examples, args.prompt_size, args, reseed=42)
    tv_file_1 = get_task_vectors_from_dataset(model, tokenizer, device, dataset_1, layers, args)

    dataset_2 = Dataset({args.task2: 1}, args.num_examples, args.prompt_size, args, reseed=42)
    tv_file_2 = get_task_vectors_from_dataset(model, tokenizer, device, dataset_2, layers, args)
    
    if args.task3:
        dataset_3 = Dataset({args.task3: 1}, args.num_examples, args.prompt_size, args, reseed=42)
        tv_file_3 = get_task_vectors_from_dataset(model, tokenizer, device, dataset_3, layers, args)

    # Interpolate between the task vectors
    if args.do_interpolation: task_vec_interpolation_main(model, tokenizer, device, tv_file_1, tv_file_2, tv_file_3, args)
    
    # Look at residual of mixed dataset
    if args.do_residual: mixed_dataset_residual_main(model, tokenizer, device, tv_file_1, tv_file_2, tv_file_3, args)

    if args.do_tv_PCA: task_vec_PCA_main(model, tokenizer, device, tv_file_1, tv_file_2, tv_file_3, args)
