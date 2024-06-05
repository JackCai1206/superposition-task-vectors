from dataclasses import dataclass

from tqdm import tqdm
from tasks import get_dataset
from utils import get_task_vectors_from_cfg, get_task_vec_interpolation, eval_task_vectors
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, HfArgumentParser, set_seed
import os
from matplotlib import pyplot as plt

@dataclass
class ScriptArguments:
    model_id: str
    num_examples: int
    prompt_size: int
    task1: str
    task2: str
    interpolation_average_over: int
    use_task_vec_cache: bool = True
    use_results_cache: bool = True

if __name__ == '__main__':
    set_seed(42)
    
    args:ScriptArguments = HfArgumentParser((ScriptArguments)).parse_args_into_dataclasses()[0]
    
    # Check if a GPU is available and set the device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Specify the model ID
    # model_id = "meta-llama/Llama-2-7b-hf" 
    # model_id = "meta-llama/Meta-Llama-3-8B"
    # model_id = 'Qwen/Qwen1.5-7B'

    # Load the tokenizer from the Hugging Face library
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Load the model with TransformerLens, specifying the torch_dtype as torch.float16 to reduce memory usage
    model = HookedTransformer.from_pretrained_no_processing(args.model_id, torch_dtype=torch.bfloat16, device=device)
    model.eval()

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    layers = list(range(model.cfg.n_layers))
    dataset_1, task_vecs_1, best_layer_1 = get_task_vectors_from_cfg(model, tokenizer, device, {args.task1: 1}, layers, args)
    print(f'Best layer for {args.task1}: {best_layer_1}')
    dataset_2, task_vecs_2, best_layer_2 = get_task_vectors_from_cfg(model, tokenizer, device, {args.task2: 1}, layers, args)
    print(f'Best layer for {args.task2}: {best_layer_2}')

    # # create mixed dataset
    cfg_dict = {args.task1: 0.5, args.task2: 0.5}
    dataset = get_dataset(cfg_dict, args.num_examples, args.prompt_size)

    # # Interpolate between the task vectors
    result_loc = f'out/task_vector_interpolation/{args.model_id}/{args.task1.replace("/", "-")}_{args.task2.replace("/", "-")}_results.pt'
    lambs = torch.linspace(0, 1, 30)
    layers = list(range(min(best_layer_1, best_layer_2), max(best_layer_1, best_layer_2) + 1))
    if os.path.exists(result_loc) and args.use_results_cache:
        print(f'Loading results from {result_loc}')
        result = torch.load(result_loc)
    else:
        os.makedirs(os.path.dirname(result_loc), exist_ok=True)
        result_list = []
        for example in tqdm(dataset[:args.interpolation_average_over]):
            test_q = example[1]
            test_ans = example[2]
            result, lambs = get_task_vec_interpolation(model, tokenizer, device, task_vecs_1, task_vecs_2, lambs, layers, test_q, test_ans)
            result_list.append(result)

        result = torch.tensor(result_list).mean(0)
        torch.save(result, result_loc)

    for i, layer in enumerate(layers):
        save_loc = f'out/task_vector_interpolation/{args.model_id}/{args.task1.replace("/", "-")}_{args.task2.replace("/", "-")}_layer-{layer}.pdf'
        os.makedirs(os.path.dirname(save_loc), exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.title(f'{args.task1} to {args.task2}\n average over {args.interpolation_average_over} examples\nLayer {layer}')
        for task_num, prob_list in enumerate(result[i]):
            label = f'Task {task_num}' if task_num <= 1 else 'Other'
            plt.plot(lambs, prob_list, label=label)
        # plt.title(title + '\nQuestion: ' + repr(question)[1:-1] + '(' + '|'.join(target_tokens) + ')')
        plt.xlabel('lambda')
        plt.ylabel('P(ans)')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig(save_loc)
