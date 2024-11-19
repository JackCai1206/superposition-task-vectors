from dataclasses import dataclass, field
from typing import List

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
    task4: str = None
    # task1_alias: str = None
    # task2_alias: str = None
    # task3_alias: str = None
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
    layers: List[int] = None
    separator: str = '->'
    fit_PCA_all_classes: bool = False
    use_alt_dataset_impl: bool = False
    dim_reduc_method: str = 'LDA'
    
    def __post_init__(self):
        self.layers = list(range(*self.layers)) if self.layers else None

if __name__ == '__main__':
    set_seed(44)
    
    args: ScriptArguments = HfArgumentParser((ScriptArguments)).parse_args_into_dataclasses()[0]

    # Specify the model ID
    # model_id = "meta-llama/Llama-2-7b-hf" 
    # model_id = "meta-llama/Meta-Llama-3-8B"
    # model_id = 'Qwen/Qwen1.5-7B'

    dataset_1 = Dataset({args.task1: 1}, args.num_examples, args.prompt_size, args, reseed=0)
    tv_file_1 = get_task_vectors_from_dataset(dataset_1, args)

    dataset_2 = Dataset({args.task2: 1}, args.num_examples, args.prompt_size, args, reseed=0)
    tv_file_2 = get_task_vectors_from_dataset(dataset_2, args)

    if args.task3:
        dataset_3 = Dataset({args.task3: 1}, args.num_examples, args.prompt_size, args, reseed=0)
        tv_file_3 = get_task_vectors_from_dataset(dataset_3, args)

    if args.task4:
        dataset_4 = Dataset({args.task4: 1}, args.num_examples, args.prompt_size, args, reseed=0)
        tv_file_4 = get_task_vectors_from_dataset(dataset_4, args)

    # Interpolate between the task vectors
    if args.do_interpolation: task_vec_interpolation_main(tv_file_1, tv_file_2, tv_file_3, args)

    # Look at residual of mixed dataset
    if args.do_residual: mixed_dataset_residual_main(tv_file_1, tv_file_2, tv_file_3, args)

    if args.do_tv_PCA:
        if args.task3 and args.task4:
            task_vec_PCA_main([tv_file_1, tv_file_2, tv_file_3, tv_file_4], args)
        elif args.task3:
            task_vec_PCA_main([tv_file_1, tv_file_2, tv_file_3], args)
