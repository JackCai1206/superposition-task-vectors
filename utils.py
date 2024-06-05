import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name
from functools import partial
from tqdm import tqdm
from collections import defaultdict
import os
from tasks import get_dataset

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

def get_task_vectors_from_cfg(model, tokenizer, device, cfg_dict, layers, args):
    # Get the task vectors
    task_vecs, dataset, best_layer = None, None, None
    cfg_dict_str = str(cfg_dict).replace('/', '-').replace(' ', '').replace('{', '').replace('}', '').replace(':', '-').replace('\'', '')
    save_loc = f'out/task_vectors/{args.model_id}/{cfg_dict_str}.pt'
    if os.path.exists(save_loc) and args.use_task_vec_cache:
        print(f'Loading task vectors from {save_loc}')
        file = torch.load(save_loc)
        cfg_dict = file['cfg_dict']
        task_vecs = file['task_vecs']
        dataset = file['dataset']
        best_layer = file['best_layer']
    else:
        print(f'Generating task vectors for {cfg_dict}')
        os.makedirs(os.path.dirname(save_loc), exist_ok=True)
        dataset = get_dataset(cfg_dict, args.num_examples, args.prompt_size)
        task_vecs = get_task_vecs(model, layers, dataset, tokenizer, device)
        accs = eval_task_vectors(model, tokenizer, device, dataset, task_vecs)
        print(accs)
        best_layer = np.argmax(accs)
        # Save the task vectors
        torch.save({
            'cfg_dict': cfg_dict,
            'task_vecs': task_vecs,
            'dataset': dataset,
            'model_id': args.model_id,
            'best_layer': best_layer
        }, save_loc)
    
    return dataset, task_vecs, best_layer

import numpy as np 

def get_task_vec_interpolation(model, tokenizer, device, task_vecs_A, task_vecs_B, lambs, layers, question, answers):
    assert len(answers) == 2
    # plt.figure(figsize=(10, 4))
    probs_by_layer = []
    for patch_layer in layers:

        # tv_5050 = task_vecs_mixed[target_layer][:, -1, :].mean(0)
        tv_A = task_vecs_A[patch_layer].mean(0)
        tv_B = task_vecs_B[patch_layer].mean(0)

        prob_dict = [[] for _ in range(3)]
        for lamb in lambs:
            tv_mix = (lamb * tv_A + (1-lamb) * tv_B)
            # multiply the conditional probabilities of the tokens in the string
            for task_num, ans in enumerate(answers):
                ans_prob = patch_get_output_prob(model, tokenizer, device, question, ans, patch_layer, tv_mix)
                prob_dict[task_num].append(ans_prob.item())
            prob_dict[2].append(1 - sum([prob_dict[i][-1] for i, ans in enumerate(answers)]))
        probs_by_layer.append(prob_dict)
    return probs_by_layer, lambs
