from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from transformers import AutoTokenizer

from . import tasks

@dataclass
class PromptConfig:
    num_exper : int
    num_examples : int
    num_tasks: int
    num_tokens: int
    max_tokens: Optional[int]
    num_beams: Optional[int]
    task_funcs_dict: dict
    question_name: str
    question_kwargs: dict
    task_examples: List[int]
    actual_distribution: List[float]
    order: str
    prompt_template: str
    device: str
    seed: int

def parse_raw_config_for_prompt(raw_config : dict):
    # parse raw_config
    num_exper = raw_config['num_exper']
    num_examples = raw_config['num_examples']
    num_tasks = raw_config['num_tasks']
    num_tokens = raw_config['num_tokens']
    max_tokens = raw_config['max_tokens']
    num_beams = raw_config['num_beams']
    task_funcs_dict = raw_config['task_funcs_dict']
    question_name = raw_config['question']['name']
    question_kwargs = raw_config['question']['kwargs']
    distribution = [eval(num) if type(num)==str else num for num in raw_config['distribution']]
    if len(distribution) != num_tasks:
        raise ValueError("length of distribution is not the same as the length of task_func_names")
    if 1 - sum(distribution) > 1e-5 or sum(distribution) - 1 > 1e-5:
        raise ValueError("distribution does not sum to 1")
    task_examples = [int(frac * num_examples) for frac in distribution]
    if sum(task_examples) > num_examples:
        raise ValueError("distribution sums to more than 1")
    elif sum(task_examples) < num_examples:
        num_diff = num_examples - sum(task_examples)
        i = 0
        while num_diff > 0 and i < len(task_examples):
            if not (num_examples * distribution[i]).is_integer():
                task_examples[i] += 1
                num_diff -= 1
            i += 1
        if sum(task_examples) < num_examples:
            raise ValueError("distribution sums to less than 1")
    order = raw_config['order']
    device = raw_config['device']
    seed = raw_config['seed']
    
    prompt_template_name = raw_config['prompt_template_name']
    with open(f"alt_dataset_impl/prompt/{prompt_template_name}.prompt", "r") as f:
        prompt_template = f.read()

    config = PromptConfig(num_exper=num_exper, 
                          num_examples=num_examples,
                          num_tasks=num_tasks,
                          num_tokens=num_tokens,
                          max_tokens=max_tokens,
                          num_beams=num_beams,
                          task_funcs_dict=task_funcs_dict,
                          question_name=question_name,
                          question_kwargs=question_kwargs,
                          task_examples=task_examples,
                          actual_distribution=[num_examples / sum(task_examples) for num_examples in task_examples],
                          order=order,
                          prompt_template=prompt_template,
                          device=device,
                          seed=seed)
    return config

def generate_prompt(prompt_config: PromptConfig, tokenizer: AutoTokenizer):
    examples = []
    task_funcs = [getattr(tasks, f"generate_task_{prompt_config.task_funcs_dict[i]['name']}") for i in range(prompt_config.num_tasks)]
    task_indices = np.array([999] * prompt_config.num_examples)
    num_tasks = len(task_funcs)
    if prompt_config.order == 'random':
        indices = np.arange(prompt_config.num_examples)
        for i in range(num_tasks):
            indices_select = np.random.choice(indices, prompt_config.task_examples[i], replace=False)
            task_indices[indices_select] = i
            indices = np.setdiff1d(indices, indices_select, assume_unique=True)
    else:
        raise NotImplementedError
    for idx in task_indices:
        task_func = task_funcs[idx]
        task_kwargs = prompt_config.task_funcs_dict[idx]['kwargs']
        example = task_func(examples, tokenizer, **task_kwargs)
        examples.append(example)
    examples_str = "\n".join(examples)

    question_func = getattr(tasks, f"question_{prompt_config.question_name}")
    question_kwargs = prompt_config.question_kwargs
    
    question_str, output_choices = question_func(examples, tokenizer, **question_kwargs)
    prompt = prompt_config.prompt_template.format(examples_str=examples_str, question_str=question_str)
    return prompt, output_choices
