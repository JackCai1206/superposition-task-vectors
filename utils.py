import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name
from functools import partial
from tqdm import tqdm
from collections import defaultdict
import os
from tasks import Dataset
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

# Function to generate text
def get_task_vecs(model: HookedTransformer, layers, dataset, tokenizer, device):
    print('Getting task vectors')
    prompts = [d[0] + d[1] for d in dataset]
    task_vecs = defaultdict(list)
    # THis hook will collect the activations at the specified layer
    def hook(task_vecs, value, hook: HookPoint):
        task_vecs[hook.layer()].append(value[:, -1, :].cpu().detach()) # -1 is the last token '->'
    with torch.no_grad():
        task_vecs = defaultdict(list)
        all_inputs = tokenizer(prompts, return_tensors="pt", padding='longest')
        B = 8
        for i in tqdm(range(0, len(prompts), B)):
            inputs = all_inputs.input_ids[i:i+B].to(device)
            attention_mask = all_inputs.attention_mask[i:i+B].to(device)
            fwd_hooks = [(get_act_name('resid_post', layer), partial(hook, task_vecs)) for layer in layers]
            model.run_with_hooks(inputs, attention_mask=attention_mask, fwd_hooks=fwd_hooks)

    torch.cuda.empty_cache()
    # breakpoint()
    return {k: torch.cat(x) for k, x in task_vecs.items()}

def patch_hook(tv, pos, value, hook: HookPoint):
    tv = tv.to(value.dtype).to(value.device)
    value[torch.arange(value.shape[0]), pos, :] = tv
    return value

def align_sentences(sent, sep_id, pad_id):
    # align ragged tensors so that sep is at the same position
    sent_ids = sent.input_ids
    B, L = sent_ids.shape
    sep_pos = torch.nonzero(sent_ids == sep_id)[:, 1]
    assert sep_pos.shape[0] == B, 'All inputs must have exactly one separator token'
    sent_start = torch.max(sent_ids != pad_id, dim=1).indices
    sent_len_temp = sep_pos.max() - sep_pos.min() + L
    sent_len = (sep_pos - sent_start).max() + (L - sep_pos).max()
    aligned_ids = torch.full((B, sent_len_temp), pad_id)
    start_pos = sep_pos.max() - sep_pos
    aligned_ids[:, :L] = sent_ids
    roll_indices = (torch.arange(aligned_ids.shape[1])[None, :] - start_pos[:, None]) % aligned_ids.shape[1]
    aligned_ids = torch.gather(aligned_ids, 1, roll_indices)
    if sent_len_temp != sent_len: assert (aligned_ids[:, 0] == pad_id).all()
    aligned_ids = aligned_ids[:, -sent_len:]
    attention_mask = (aligned_ids != pad_id).long()
    sep_pos = sep_pos.max() - (sent_len_temp - sent_len)

    return aligned_ids, attention_mask, sep_pos

def patch_get_output_prob(model, tokenizer, device, question, ans, patch_layer, tv_mix):
    if type(question) == str:
        question = [question]
    inputs = tokenizer([q+a for q,a in zip(question, ans)], return_tensors="pt", padding=True)
    input_ids, attention_mask, sep_pos = align_sentences(inputs, tokenizer.convert_tokens_to_ids('->'), tokenizer.pad_token_id)
    with torch.no_grad():
        output = model.run_with_hooks(input_ids.to(device), attention_mask=attention_mask.to(device), fwd_hooks=[(get_act_name('resid_post', patch_layer), partial(patch_hook, tv_mix, sep_pos))]).cpu().detach()
    probs = output.softmax(dim=-1)
    probs[..., tokenizer.pad_token_id] = 1
    ans_prob = torch.gather(probs[:, sep_pos:-1], -1, input_ids[:, sep_pos+1:].unsqueeze(-1)).squeeze(-1).prod(dim=-1)
    return ans_prob

def collate(batch):
    prompt, question, answers = list(zip(*batch))
    answers = list(zip(*answers)) # (n_task, Bz)
    return prompt, question, answers

def eval_task_vectors(model, tokenizer, device, dataset, task_vecs):
    print('Evaluating task vectors')
    acc_A = []
    layers = range(model.cfg.n_layers)
    for patch_layer in tqdm(layers):
        tv_A = task_vecs[patch_layer].mean(0)
        # tv_A = task_vecs[layer][:, -1, :][0]
        # num_correct = 0
        ans_probs = []
        with torch.no_grad():
            B = 100 # Bz 1 will max out the GPU memory
            for i in range(0, len(dataset), B):
                prompt, question, answers = collate(dataset[i:i+B])
                assert len(answers) == 1
                ans = answers[0]
                ans_prob = patch_get_output_prob(model, tokenizer, device, question, ans, patch_layer, tv_A)
                ans_probs += ans_prob.tolist()
        # acc_A.append(num_correct / len(eval_prompts))
        acc_A.append(sum(ans_probs) / len(ans_probs))
    # breakpoint()
    return acc_A

def get_task_vectors_from_dataset(model, tokenizer, device, dataset: Dataset, layers, args):
    # Get the task vectors
    task_vecs, best_layer = None, None
    save_loc = f'out/task_vectors/{args.model_id}/{dataset.seed}/{repr(dataset)}.pt'
    if os.path.exists(save_loc) and args.use_task_vec_cache:
        print(f'Loading task vectors from {save_loc}')
        file = torch.load(save_loc)
        if len(file['task_vecs']) == len(layers) and file['dataset'] == dataset:
            return file

    print(f'Generating task vectors for {dataset.cfg_dict} with dataset seed {dataset.seed}')
    os.makedirs(os.path.dirname(save_loc), exist_ok=True)
    task_vecs = get_task_vecs(model, layers, dataset, tokenizer, device)
    if len(dataset.cfg_dict) == 1:
        accs = eval_task_vectors(model, tokenizer, device, dataset, task_vecs)
        print(accs)
        best_layer = np.argmax(accs)
        print(f'Best layer for {list(dataset.cfg_dict.keys())[0]}: {best_layer}')
    else:
        best_layer = 0
    # Save the task vectors
    file = {
        'cfg_dict': dataset.cfg_dict,
        'task_vecs': task_vecs,
        'dataset': dataset,
        'model_id': args.model_id,
        'best_layer': best_layer
    }
    torch.save(file, save_loc)
    
    return file

def get_task_vec_interpolation(model, tokenizer, device, task_vecs_A, task_vecs_B, lambs, layers, questions, answers):
    # assert len(answers) == 2
    # plt.figure(figsize=(10, 4))
    probs_by_layer = []
    for patch_layer in layers:

        # tv_5050 = task_vecs_mixed[target_layer][:, -1, :].mean(0)
        tv_A = task_vecs_A[patch_layer].mean(0)
        tv_B = task_vecs_B[patch_layer].mean(0)

        batch_ans = list(zip(*answers))
        prob_dict = [[] for _ in range(len(batch_ans))] # probability sweep for each task
        for lamb in lambs:
            tv_mix = (lamb * tv_B + (1-lamb) * tv_A)
            # multiply the conditional probabilities of the tokens in the string
            for task_num, ans in enumerate(batch_ans):
                ans_prob = patch_get_output_prob(model, tokenizer, device, questions, ans, patch_layer, tv_mix)
                prob_dict[task_num].append(ans_prob.tolist())
        probs_by_layer.append(prob_dict)
    result = torch.tensor(probs_by_layer).permute(3, 0, 1, 2) # (bz, n_layer, n_task, n_lamb)
    return result, lambs

def task_vec_interpolation_main(model, tokenizer, device, tv_file_1, tv_file_2, args):
    # # create mixed dataset
    cfg_dict = {args.task1: 0.5, args.task2: 0.5}
    dataset = Dataset(cfg_dict, args.num_examples, args.prompt_size)
    best_layer_1 = tv_file_1['best_layer']
    best_layer_2 = tv_file_2['best_layer']
    task_vecs_1 = tv_file_1['task_vecs']
    task_vecs_2 = tv_file_2['task_vecs']
    dataset_1 = tv_file_1['dataset']
    dataset_2 = tv_file_2['dataset']

    # # Interpolate between the task vectors
    result_loc = f'out/task_vector_interpolation/{args.model_id}/{repr(dataset_1)}_{repr(dataset_2)}_results.pt'
    lambs = torch.linspace(0, 1, 30)
    layers = list(range(min(best_layer_1, best_layer_2), max(best_layer_1, best_layer_2) + 1))
    if os.path.exists(result_loc) and args.use_results_cache:
        print(f'Loading results from {result_loc}')
        result = torch.load(result_loc)
    else:
        os.makedirs(os.path.dirname(result_loc), exist_ok=True)
        result_list = []
        bz = 8
        for i in tqdm(range(0, args.average_over, bz)):
            batch = dataset[i:i+bz]
            test_q = [example[1] for example in batch]
            test_ans = [example[2] for example in batch]
            result, lambs = get_task_vec_interpolation(model, tokenizer, device, task_vecs_1, task_vecs_2, lambs, layers, test_q, test_ans)
            result_list.append(result)

        result = torch.cat(result_list).mean(0)
        torch.save(result, result_loc)

    for i, layer in enumerate(layers):
        save_loc = f'out/task_vector_interpolation/{args.model_id}/{args.task1.replace("/", "-")}_{args.task2.replace("/", "-")}/layer-{layer}.pdf'
        os.makedirs(os.path.dirname(save_loc), exist_ok=True)
        plt.figure(figsize=(6, 5))
        plt.title(f'{args.task1} to {args.task2}\n average over {args.average_over} examples\nLayer {layer}')
        task_names = dataset.all_tasks + ['other']
        # for task_num, prob_list in enumerate(result[i]):
        #     label = task_names[task_num]
        #     plt.plot(lambs, prob_list, label=label)
        blues = sns.color_palette("flare", len(dataset.given_tasks))
        reds = sns.color_palette("crest", len(dataset.extra_tasks))
        grey = [sns.colors.crayons['Gray']]
        colors = blues + reds + grey
        other = 1 - result[i].sum(0)
        plt.stackplot(lambs, *result[i], other, labels=task_names, colors=colors)
        # plt.title(title + '\nQuestion: ' + repr(question)[1:-1] + '(' + '|'.join(target_tokens) + ')')
        plt.xlabel('lambda')
        plt.ylabel('P(ans)')
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        plt.legend()
        plt.savefig(save_loc)

from sklearn.linear_model import LinearRegression

def mixed_dataset_residual_main(model, tokenizer, device, tv_file_1, tv_file_2, args):
    best_layer_1 = tv_file_1['best_layer']
    best_layer_2 = tv_file_2['best_layer']
    task_vecs_1 = tv_file_1['task_vecs']
    task_vecs_2 = tv_file_2['task_vecs']
    dataset_1 = tv_file_1['dataset']
    dataset_2 = tv_file_2['dataset']
    
    # mix_fracs = [0.2, 0.4, 0.6, 0.8]
    mix_fracs = [0.5]
    layers = range(model.cfg.n_layers)

    result_loc = f'out/mixed_dataset_residual/{args.model_id}/{repr(dataset_1)}_{repr(dataset_2)}_results.pt'
    results = None
    if os.path.exists(result_loc) and args.use_results_cache:
        print(f'Loading results from {result_loc}')
        result = torch.load(result_loc)
    
    if results is None or len(result) != len(mix_fracs):
        os.makedirs(os.path.dirname(result_loc), exist_ok=True)
        lr = LinearRegression()
        result_list = []

        for trial in range(args.average_over):
            result_list.append([])
            datasets = []
            for task_1_frac in mix_fracs:
                cfg_dict = {args.task1: np.round(task_1_frac, 2), args.task2: np.round(1 - task_1_frac, 2)}
                dataset_mixed = Dataset(cfg_dict, args.num_examples, args.prompt_size, reseed=42+trial)
                datasets.append(dataset_mixed)
                dataset_random = Dataset(cfg_dict, args.num_examples, args.prompt_size, reseed=42+trial, random_ans=True)
                datasets.append(dataset_random)
            for ds in datasets:
                tv_file_mixed = get_task_vectors_from_dataset(model, tokenizer, device, ds, layers, args)
                task_vecs_mixed = tv_file_mixed['task_vecs']
                resid_list = []
                for layer in layers:
                    tv_1 = task_vecs_1[layer].mean(0)
                    tv_1 /= tv_1.norm()
                    tv_2 = task_vecs_2[layer].mean(0)
                    tv_2 /= tv_2.norm()
                    tv_mixed = task_vecs_mixed[layer].mean(0)
                    tv_mixed /= tv_mixed.norm()
                    X = torch.stack([tv_1, tv_2]).T.cpu().float().numpy()
                    y = tv_mixed.cpu().float().numpy()
                    lr.fit(X, y)
                    resid = np.linalg.norm(y - lr.predict(X))
                    # resid = 0
                    # for i in range(args.num_examples):
                    #     tv_1 = task_vecs_1[layer][i]
                    #     tv_1 /= tv_1.norm()
                    #     tv_2 = task_vecs_2[layer][i]
                    #     tv_2 /= tv_2.norm()
                    #     tv_mixed = task_vecs_mixed[layer].mean(0)
                    #     tv_mixed /= tv_mixed.norm()
                    #     X = torch.stack([tv_1, tv_2]).T.cpu().float().numpy()
                    #     y = tv_mixed.cpu().float().numpy()
                    #     lr.fit(X, y)
                    #     resid += np.linalg.norm(y - lr.predict(X))
                    # resid /= args.num_examples
                    resid_list.append(resid)
                result_list[-1].append(resid_list)
        result = torch.tensor(result_list).mean(0)
        torch.save(result, result_loc)

    plt.figure(figsize=(6, 4))
    plt.title(f'Mixed dataset residual\n average over {args.average_over}')
    for i, frac in enumerate(mix_fracs):
        save_loc = f'out/mixed_dataset_residual/{args.model_id}/{args.task1.replace("/", "-")}_{args.task2.replace("/", "-")}.pdf'
        os.makedirs(os.path.dirname(save_loc), exist_ok=True)
        label = f'{args.task1.split("/")[0]} frac {frac}'
        plt.plot(layers, result[i], label=label)
        # plt.title(title + '\nQuestion: ' + repr(question)[1:-1] + '(' + '|'.join(target_tokens) + ')')
    plt.xlabel('layer')
    plt.ylabel('linear fit residual')
    plt.legend()
    plt.savefig(save_loc)
