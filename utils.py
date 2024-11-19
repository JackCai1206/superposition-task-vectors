from hashlib import sha1
import json
import string
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.utilities.devices import get_device_for_block_index
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name
from transformer_lens.loading_from_pretrained import get_pretrained_model_config, convert_llama_weights, convert_hf_model_config
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from functools import partial
from tqdm import tqdm
from collections import defaultdict
import os
from charactertokenizer.core import CharacterTokenizer
from tasks import Dataset
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from safetensors import torch as safetensors

memoized_results = {}

def memoize(func):
    def wrapper(args):
        if args.model_id in memoized_results:
            return memoized_results[args.model_id]
        result = func(args)
        memoized_results[args.model_id] = result
        return result
    return wrapper

@memoize
def get_model_tokenizer(args):
    if os.path.exists(args.model_id):
        orig_cfg = json.load(open(f'{args.model_id}/config.json'))
        
        vocab = string.ascii_letters + string.digits + string.punctuation + " \n"
        tokenizer = CharacterTokenizer(vocab, orig_cfg['max_position_embeddings'])
        tokenizer.padding_side = 'left'
        tokenizer.sep_token = '->'
        tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        
        model_config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            num_hidden_layers=orig_cfg['num_hidden_layers'],
            hidden_size=orig_cfg['hidden_size'],
            intermediate_size=orig_cfg['hidden_size'] * 4,
            num_attention_heads=orig_cfg['num_attention_heads'],
            max_position_embeddings=orig_cfg['max_position_embeddings'],
            attn_implementation='sdpa'
        )
        model = LlamaForCausalLM(model_config)
        model.load_state_dict(safetensors.load_file(f'{args.model_id}/model.safetensors', device='cpu'))
        print('Number of parameters:', model.num_parameters())
        cfg = HookedTransformerConfig.from_dict({
            "d_model": orig_cfg['hidden_size'],
            "d_head": orig_cfg['hidden_size'] // orig_cfg['num_attention_heads'],
            "n_heads": orig_cfg['num_attention_heads'],
            "d_mlp": orig_cfg['hidden_size'] * 4,
            "n_layers": orig_cfg['num_hidden_layers'],
            "n_ctx": orig_cfg['max_position_embeddings'],
            "eps": 1e-5,
            "d_vocab": orig_cfg['vocab_size'],
            "act_fn": "silu",
            "n_key_value_heads": orig_cfg['num_attention_heads'],
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 64,
            "final_rms": True,
            "gated_mlp": True,
        })
        state_dict = convert_llama_weights(model, cfg)
        model = HookedTransformer(cfg)
        model.load_state_dict(model.fill_missing_keys(state_dict))
        args.model_id = args.model_id.replace('/', '-')
    else:
        # Load the tokenizer from the Hugging Face library
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)

        # Load the model with TransformerLens, specifying the torch_dtype as torch.float16 to reduce memory usage
        model = HookedTransformer.from_pretrained_no_processing(args.model_id, torch_dtype=torch.bfloat16)

    model.eval()
    device = get_device_for_block_index(0, model.cfg)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    return model, tokenizer, device

def get_args_hash(args):
    d = args.__dict__.copy()
    # d.pop('task1_alias')
    # d.pop('task2_alias')
    # d.pop('task3_alias')
    return hash(json.dumps(d, sort_keys=True))

# Function to generate text
def get_task_vecs(model: HookedTransformer, dataset, tokenizer, device, layers=None):
    if layers is None:
        layers = range(model.cfg.n_layers)
    print('Getting task vectors')
    prompts = [d[0] + d[1] for d in dataset]
    task_vecs = defaultdict(list)
    # THis hook will collect the activations at the specified layer
    def hook(task_vecs, value, hook: HookPoint):
        task_vecs[hook.layer()].append(value[:, -1, :].cpu().detach()) # -1 is the last token '->'
    with torch.no_grad():
        task_vecs = defaultdict(list)
        all_inputs = tokenizer(prompts, return_tensors="pt", padding='longest')
        # breakpoint()
        B = 5
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
    sep_pos = torch.argmax((sent_ids == sep_id).cumsum(1), 1) # find the last separator token
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

def patch_get_output_prob(model, tokenizer, device, question, ans, patch=True, patch_layer=None, tv_mix=None, return_pred=False, separator='->'):
    if type(question) == str:
        question = [question]
    inputs = tokenizer([q+a for q,a in zip(question, ans)], return_tensors="pt", padding=True)
    input_ids, attention_mask, sep_pos = align_sentences(inputs, tokenizer.convert_tokens_to_ids(separator), tokenizer.pad_token_id)
    with torch.no_grad():
        if patch:
            output = model.run_with_hooks(input_ids.to(device), attention_mask=attention_mask.to(device), fwd_hooks=[(get_act_name('resid_post', patch_layer), partial(patch_hook, tv_mix, sep_pos))]).cpu().detach()
        else:
            output = model(input_ids.to(device), attention_mask=attention_mask.to(device), return_type='logits').cpu().detach()
    probs = output.softmax(dim=-1)
    pred = tokenizer.batch_decode(torch.argmax(probs[:, sep_pos:-1], dim=-1))
    probs[..., tokenizer.pad_token_id] = 1
    ans_prob = torch.gather(probs[:, sep_pos:-1], -1, input_ids[:, sep_pos+1:].unsqueeze(-1)).squeeze(-1).prod(dim=-1)
    if return_pred:
        return ans_prob, pred
    return ans_prob

def collate(batch):
    prompt, question, answers = list(zip(*batch))
    answers = list(zip(*answers)) # (n_task, Bz)
    return prompt, question, answers

def eval_task_vectors(model, tokenizer, device, args, dataset, task_vecs, use_prob=True):
    print('Evaluating task vectors')
    acc_A = []
    prob_A = []
    layers = range(model.cfg.n_layers)
    for patch_layer in tqdm(layers):
        tv_A = task_vecs[patch_layer].mean(0)
        # tv_A = task_vecs[layer][:, -1, :][0]
        num_correct = 0
        ans_probs = []
        with torch.no_grad():
            B = 100 # Bz 1 will max out the GPU memory
            for i in range(0, len(dataset), B):
                prompt, question, answers = collate(dataset[i:i+B])
                assert len(answers) == 1
                ans = answers[0]
                ans_prob, pred = patch_get_output_prob(model, tokenizer, device, question, ans, patch=True, patch_layer=patch_layer, tv_mix=tv_A, return_pred=True, separator=args.separator)
                ans_probs += ans_prob.tolist()
                num_correct += sum([a == p for a, p in zip(ans, pred)])
        acc_A.append(num_correct / len(dataset))
        prob_A.append(sum(ans_probs) / len(dataset))
    # breakpoint()
    if use_prob:
        return prob_A
    else:
        return acc_A

def get_task_vectors_from_dataset(dataset: Dataset, args, layers=None):
    # Get the task vectors
    task_vecs, best_layer = None, None
    save_loc = f'{args.out_dir}/task_vectors/{args.model_id}/{dataset.seed}/{dataset.get_hash()}.pt'
    if os.path.exists(save_loc) and args.use_task_vec_cache:
        print(f'Loading task vectors from {save_loc}')
        file = torch.load(save_loc)
        # if all([l in file['task_vecs'] for l in layers]) and file['dataset'] == dataset:
            # task_vecs = file['task_vecs']
            # accs = eval_task_vectors(model, tokenizer, device, dataset, task_vecs, use_prob=False)
            # print(accs)
            # best_layer = np.argmax(accs)
            # print(f'Best layer for {list(dataset.cfg_dict.keys())[0]}: {best_layer}')
        return file

    print(f'Generating task vectors for {dataset.cfg_dict} with dataset seed {dataset.seed}')
    os.makedirs(os.path.dirname(save_loc), exist_ok=True)
    model, tokenizer, device = get_model_tokenizer(args)
    task_vecs = get_task_vecs(model, dataset, tokenizer, device, layers=layers)
    if len(dataset.cfg_dict) == 1:
        accs = eval_task_vectors(model, tokenizer, device, args, dataset, task_vecs)
        print(accs)
        best_layer = np.argmax(accs)
        print(f'Best layer for {list(dataset.cfg_dict.keys())[0]}: {best_layer}')
    else:
        best_layer = None
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
                ans_prob = patch_get_output_prob(model, tokenizer, device, questions, ans, patch=True, patch_layer=patch_layer, tv_mix=tv_mix)
                prob_dict[task_num].append(ans_prob.tolist())
        probs_by_layer.append(prob_dict)
    result = torch.tensor(probs_by_layer).permute(3, 0, 1, 2) # (bz, n_layer, n_task, n_lamb)
    return result, lambs

def get_task_vec_interpolation_v2(model, tokenizer, device, args, task_vecs_A, task_vecs_B, task_vecs_C, lambs, layers, questions, answers):
    # assert len(answers) == 2
    # plt.figure(figsize=(10, 4))
    probs_by_layer = []
    for patch_layer in layers:

        # tv_5050 = task_vecs_mixed[target_layer][:, -1, :].mean(0)
        tv_A = task_vecs_A[patch_layer].mean(0)
        tv_B = task_vecs_B[patch_layer].mean(0)
        tv_C = task_vecs_C[patch_layer].mean(0)

        batch_ans = list(zip(*answers))
        prob_dict = [[] for _ in range(len(batch_ans))] # probability sweep for each task
        
        initial_dist = torch.tensor([1, 0, 0])
        final_dist = torch.tensor([0, 1, 0])
        for lamb in lambs:
            tv_mix = (initial_dist * (1-lamb) + final_dist * lamb) @ torch.stack([tv_A, tv_B, tv_C]).to(torch.float)
            # multiply the conditional probabilities of the tokens in the string
            for task_num, ans in enumerate(batch_ans):
                ans_prob = patch_get_output_prob(model, tokenizer, device, questions, ans, patch=True, patch_layer=patch_layer, tv_mix=tv_mix, separator=args.separator)
                prob_dict[task_num].append(ans_prob.tolist())
        probs_by_layer.append(prob_dict)
    result = torch.tensor(probs_by_layer).permute(3, 0, 1, 2) # (bz, n_layer, n_task, n_lamb)
    return result, lambs

def get_task_mixing(model, tokenizer, device, task1, task2, task3, lambs, layers, args):
    num_tasks = 9 # hardcode for now, return the first two tasks
    bz = 8
    prob_dict = torch.zeros(args.num_examples, 9, len(lambs)) # probability sweep for each task

    initial_dist = torch.tensor([1, 0, 0])
    final_dist = torch.tensor([0, 1, 0])
    for lamb_i, lamb in enumerate(lambs):
        dist = (initial_dist * (1-lamb) + final_dist * lamb)
        ds_cfg = {t: d.item() for i, (t, d) in enumerate(zip([task1, task2, task3][:num_tasks], dist))}
        mixed_dataset = Dataset(ds_cfg, args.num_examples, args.prompt_size, args)
        for i in tqdm(range(0, len(mixed_dataset), bz)):
            batch = mixed_dataset[i:i+bz]
            questions = [example[0] + example[1] for example in batch]
            answers = [example[2] for example in batch]
            batch_ans = list(zip(*answers))
            # multiply the conditional probabilities of the tokens in the string
            for task_num, ans in enumerate(batch_ans[:num_tasks]):
                ans_prob = patch_get_output_prob(model, tokenizer, device, questions, ans, patch=False)
                prob_dict[i:i+bz, task_num, lamb_i] = ans_prob
    result = prob_dict
    return result, lambs

def task_vec_interpolation_main(tv_file_1, tv_file_2, tv_file_3, args):
    # # create mixed dataset
    cfg_dict = {args.task1: 1/3, args.task2: 1/3, args.task3: 1/3}
    dataset = Dataset(cfg_dict, args.num_examples, args.prompt_size, args)
    best_layer_1 = tv_file_1['best_layer']
    best_layer_2 = tv_file_2['best_layer']
    best_layer_3 = tv_file_3['best_layer']
    task_vecs_1 = tv_file_1['task_vecs']
    task_vecs_2 = tv_file_2['task_vecs']
    task_vecs_3 = tv_file_3['task_vecs']
    dataset_1 = tv_file_1['dataset']
    dataset_2 = tv_file_2['dataset']
    dataset_3 = tv_file_3['dataset']

    # # Interpolate between the task vectors
    result_loc = f'{args.out_dir}/task_vector_interpolation/{args.model_id}/{get_args_hash(args)}_results.pt'
    lambs = torch.linspace(0, 1, 30)
    lambs2 = torch.linspace(0, 1, 5)
    layers = list(range(min(best_layer_1, best_layer_2, best_layer_3), max(best_layer_1, best_layer_2, best_layer_3) + 1))
    if os.path.exists(result_loc) and args.use_results_cache:
        print(f'Loading results from {result_loc}')
        result = torch.load(result_loc)
    else:
        model, tokenizer, device = get_model_tokenizer(args)
        result = {}
        os.makedirs(os.path.dirname(result_loc), exist_ok=True)
        result_list = []
        bz = 8
        for i in tqdm(range(0, args.average_over + 1, bz)):
            batch = dataset[i:i+bz]
            test_q = [example[1] for example in batch]
            test_ans = [example[2] for example in batch]
            interp_result, lambs = get_task_vec_interpolation_v2(model, tokenizer, device, args, task_vecs_1, task_vecs_2, task_vecs_3, lambs, layers, test_q, test_ans)
            result_list.append(interp_result)

        result['interpolation'] = torch.cat(result_list).mean(0)
        result['task_mixing'], lambs2 = get_task_mixing(model, tokenizer, device, args.task1, args.task2, args.task3, lambs2, layers, args)
        result['task_mixing'] = result['task_mixing'].mean(0)

        torch.save(result, result_loc)

    for i, layer in enumerate(layers):
        save_loc = f'{args.out_dir}/task_vector_interpolation/{args.model_id}/{args.task1.replace("/", "-")}_{args.task2.replace("/", "-")}_{args.task3.replace("/", "-")}/layer-{layer}.pdf'
        os.makedirs(os.path.dirname(save_loc), exist_ok=True)
        fig = plt.figure(figsize=(4.2, 4.2))
        # plt.title(f'{args.task1} to {args.task2}\n average over {args.average_over} examples\nLayer {layer}')
        task_names = dataset.given_tasks[:2] + dataset.extra_tasks[1:3]
        # if args.task1_alias:
        #     task_names[0] = args.task1_alias
        # if args.task2_alias:
        #     task_names[1] = args.task2_alias
        # if args.task3_alias:
        #     task_names[2] = args.task3_alias
        task_names = [dataset.simple_name_mapping['/'.join(t)] for t in task_names] + ['other']
        
        # for task_num, prob_list in enumerate(result[i]):
        #     label = task_names[task_num]
        #     plt.plot(lambs, prob_list, label=label)
        # blues = sns.color_palette("flare", len(dataset.given_tasks))
        # reds = sns.color_palette("crest", len(dataset.extra_tasks))
        # grey = [sns.colors.crayons['Gray']]
        # colors = blues + reds + grey

        # colors = sns.color_palette('crest', 2) + sns.color_palette('flare', 2) + [sns.colors.xkcd_rgb['light grey blue']]
        # other = 1 - result['interpolation'][i][:2].sum(0) - result['interpolation'][i][3:5].sum(0)
        # ax1 = fig.add_subplot(2, 1, 1)
        # ax1.stackplot(lambs, *result['interpolation'][i][:2], *result['interpolation'][i][3:5], other, labels=task_names, colors=colors)
        # other = 1 - result['task_mixing'][:2].sum(0) - result['task_mixing'][3:5].sum(0)
        # # ax1.set_xlabel('lambda')
        # ax1.set_ylabel('P(ans)')
        # ax1.set_xlim(0, 1)
        # ax1.set_ylim(0, 1)
        # ax1.set_xticks([])
        # ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        # ax2.stackplot(lambs2, *result['task_mixing'][:2], result['task_mixing'][3:5], other, colors=colors, linestyle='--')
        # ax2.set_xlabel('lambda')
        # ax2.set_ylabel('P(ans)')
        # ax2.set_xlim(0, 1)
        # ax2.set_ylim(0, 1)
        # ax2.set_xticks([0, 0.25, 0.5, 0.75, 1])
        # ax1.legend(bbox_to_anchor=(0., 1.05, 1., .105), loc='lower left', mode='expand', ncol=2, borderaxespad=0.)
        # fig.set_layout_engine('tight')
        # fig.subplots_adjust(hspace=0.03)
        # plt.savefig(save_loc)

        colors = sns.color_palette('crest', 2) + [sns.colors.xkcd_rgb['light grey blue']]
        other = 1 - result['interpolation'][i][:2].sum(0)
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.stackplot(lambs, *result['interpolation'][i][:2], other, labels=task_names, colors=colors)
        other = 1 - result['task_mixing'][:2].sum(0)
        # ax1.set_xlabel('lambda')
        ax1.set_ylabel('P(ans)')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_xticks([])
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        ax2.stackplot(lambs2, *result['task_mixing'][:2], other, colors=colors, linestyle='--')
        ax2.set_xlabel('lambda')
        ax2.set_ylabel('P(ans)')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax1.legend(bbox_to_anchor=(0., 1.05, 1., .105), loc='lower left', mode='expand', ncol=3, borderaxespad=0.)
        fig.set_layout_engine('tight')
        fig.subplots_adjust(hspace=0.03)
        plt.savefig(save_loc)

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def task_vec_PCA_main(tv_files, args=None):
    """
    Analyze task vectors using PCA or LDA for 3 or 4 tasks.
    
    Args:
        tv_files: List of task vector files (3 or 4 files)
        args: Arguments object
    """
    # Get number of tasks
    num_tasks = len(tv_files)
    if num_tasks not in [3, 4]:
        raise ValueError("Number of task vectors must be either 3 or 4")
    
    # Get best layers
    best_layers = [tv_file['best_layer'] for tv_file in tv_files]
    print(f'Best layers: {best_layers}')
    
    # Set layers to analyze
    if args.layers:
        layers = args.layers
    else:
        layers = list(range(min(best_layers), max(best_layers) + 1))
    
    # Get task names
    task_names = [f'task{i+1}' for i in range(num_tasks)]
    task_values = [getattr(args, name) for name in task_names]
    
    # Create mixing configurations
    mixed_cfg_dicts = []
    
    # Single task configurations (already included in tv_files)
    
    if num_tasks == 3:
        # Pairwise mixes for 3 tasks with lambda=0.5
        lamb = 0.5
        pairs = [
            [lamb, 1-lamb, 0],
            [lamb, 0, 1-lamb],
            [0, lamb, 1-lamb]
        ]
        for dist in pairs:
            cfg_dict = {task: weight for task, weight in zip(task_values, dist)}
            mixed_cfg_dicts.append(cfg_dict)
        
        # Equal mix of all 3 tasks
        equal_mix = {task: 1/3 for task in task_values}
        mixed_cfg_dicts.append(equal_mix)
        
    else:  # num_tasks == 4
        # Pairwise mixes for 4 tasks with lambda=0.5
        lamb = 0.5
        pairs = [
            [lamb, lamb, 0, 0],
            [lamb, 0, lamb, 0],
            [lamb, 0, 0, lamb],
            [0, lamb, lamb, 0],
            [0, lamb, 0, lamb],
            [0, 0, lamb, lamb]
        ]
        for dist in pairs:
            cfg_dict = {task: weight for task, weight in zip(task_values, dist)}
            mixed_cfg_dicts.append(cfg_dict)
        
        # Equal mix of all 4 tasks
        equal_mix = {task: 1/4 for task in task_values}
        mixed_cfg_dicts.append(equal_mix)
    
    # Generate mixed datasets and their task vectors
    mixed_datasets = [Dataset(cfg_dict, args.num_examples, args.prompt_size, args) 
                     for cfg_dict in mixed_cfg_dicts]
    tv_file_mixed_list = [get_task_vectors_from_dataset(ds, args, layers) 
                         for ds in mixed_datasets]
    all_tv_files = tv_files + tv_file_mixed_list

    # Compute PCA/LDA results
    method = 'LDA'
    result_loc = f'{args.out_dir}/task_vector_{method}/{args.model_id}/{get_args_hash(args)}_results.pt'

    if os.path.exists(result_loc) and args.use_results_cache:
        print(f'Loading results from {result_loc}')
        result_list = torch.load(result_loc)
    else:
        os.makedirs(os.path.dirname(result_loc), exist_ok=True)
        result_list = []
        
        for layer in layers:
            all_tv = [tv_file['task_vecs'][layer] for tv_file in all_tv_files]
            if not args.fit_PCA_all_classes:
                fit_tv = all_tv[:num_tasks]
            else:
                fit_tv = all_tv
            num_classes = len(fit_tv)
            fit_tv = torch.cat(fit_tv).float()
            
            if method == 'PCA':
                pca = PCA(n_components=None, whiten=True)
                pca.fit(fit_tv)
                result = np.stack([pca.transform(tv.float()) for tv in all_tv])
            elif method == 'LDA':
                lda = LinearDiscriminantAnalysis(n_components=None, solver='svd')
                fit_tv = (fit_tv - fit_tv.mean(0))
                lda.fit(fit_tv, torch.arange(num_classes).repeat(args.num_examples, 1).T.flatten())
                result = np.stack([lda.transform(tv.float()) for tv in all_tv])
                
            result_list.append(result)
        torch.save(result_list, result_loc)
    
    # Plotting and analysis
    for i, layer in enumerate(layers):
        result = result_list[i]
        tasks_str = '_'.join([task.replace("/", "-") for task in task_values])
        save_loc = f'{args.out_dir}/task_vector_{method}/{args.model_id}/{tasks_str}/layer-{layer}.pdf'
        os.makedirs(os.path.dirname(save_loc), exist_ok=True)
        
        plt.figure(figsize=(6, 5))
        if not args.skip_title:
            title = ' to '.join(task_values)
            plt.title(f'{title}\n average over {args.average_over} examples')
        
        # Create task names for plotting
        base_names = ['/'.join(str(x) if x != 0 else '0' for x in [1 if j == i else 0 for j in range(num_tasks)])
                     for i in range(num_tasks)]
        task_names = base_names + [tv_file['dataset'].get_simple_name() 
                                 for tv_file in all_tv_files[num_tasks:]]
        colors = sns.color_palette('husl', len(task_names))
        
        # Calculate distances between task vectors
        for i, (task_vecs, label) in enumerate(zip(result, task_names)):
            for j, (task_vecs2, label2) in enumerate(zip(result, task_names)):
                if i == j:
                    continue
                for mode in ['distance-full', 'distance-subspace']:
                    if mode == 'distance-full':
                        dist = np.linalg.norm(task_vecs.mean(0) - task_vecs2.mean(0), axis=-1)
                    elif mode == 'distance-subspace':
                        dist = np.linalg.norm(task_vecs.mean(0)[:2] - task_vecs2.mean(0)[:2], axis=-1)
                    with open(save_loc.replace('.pdf', f'-{mode}.csv'), 'a+') as f:
                        f.write(f'{label}, {label2}, {dist}\n')
        
        # Plot task vectors
        for i, (task_vecs, label) in enumerate(zip(result, task_names)):
            plt.scatter(task_vecs[:, 0], task_vecs[:, 1], label=label, color=colors[i])
        
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(save_loc)
        plt.close()


from sklearn.linear_model import LinearRegression

def mixed_dataset_residual_main(tv_file_1, tv_file_2, tv_file_3, args):
    task_vecs_1 = tv_file_1['task_vecs']
    task_vecs_2 = tv_file_2['task_vecs']
    dataset_1 = tv_file_1['dataset']
    dataset_2 = tv_file_2['dataset']
    best_layer_1 = tv_file_1['best_layer']
    best_layer_2 = tv_file_2['best_layer']
    
    # mix_fracs = [0.2, 0.4, 0.6, 0.8]
    mix_fracs = [0.5]
    layers = range(args.n_layers)

    result_loc = f'{args.out_dir}/mixed_dataset_residual/{args.model_id}/{get_args_hash(args)}_results.pt'
    results = None
    if os.path.exists(result_loc) and args.use_results_cache:
        print(f'Loading results from {result_loc}')
        result = torch.load(result_loc)
    
    if results is None or len(result) != len(mix_fracs):
        model, tokenizer, device = get_model_tokenizer(args)
        os.makedirs(os.path.dirname(result_loc), exist_ok=True)
        lr = LinearRegression()
        result_list = []

        for trial in range(args.average_over):
            result_list.append([])
            datasets = []
            for task_1_frac in mix_fracs:
                cfg_dict = {args.task1: np.round(task_1_frac, 2), args.task2: np.round(1 - task_1_frac, 2)}
                dataset_mixed = Dataset(cfg_dict, args.num_examples, args.prompt_size, args, reseed=42+trial)
                datasets.append(dataset_mixed)
                dataset_rand_num = Dataset(cfg_dict, args.num_examples, args.prompt_size, args, reseed=42+trial, random_ans='random-numeric-answers')
                datasets.append(dataset_rand_num)
                # dataset_shuffle = Dataset(cfg_dict, args.num_examples, args.prompt_size, args, reseed=42+trial, random_ans='random-answers')
                # datasets.append(dataset_shuffle)
                dataset_rand = Dataset(cfg_dict, args.num_examples, args.prompt_size, args, reseed=42+trial, random_ans='random-question-answers')
                datasets.append(dataset_rand)
            
            tv_file_mixed_list = [get_task_vectors_from_dataset(ds, args, layers) for ds in datasets]
            tv_file_mixed_list.append(tv_file_3)
            task_vecs_mixed_list = [tv_file_mixed['task_vecs'] for tv_file_mixed in tv_file_mixed_list]
            for task_vecs_mixed in task_vecs_mixed_list:
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

    plt.figure(figsize=(4.5, 4))
    # plt.title(f'Mixed dataset residual {args.task1.replace("/", "-")}_{args.task2.replace("/", "-")}\n average over {args.average_over}')
    for i, frac in enumerate(mix_fracs):
        save_loc = f'{args.out_dir}/mixed_dataset_residual/{args.model_id}/{args.task1.replace("/", "-")}_{args.task2.replace("/", "-")}.pdf'
        os.makedirs(os.path.dirname(save_loc), exist_ok=True)
        label = f'0.5/0.5 mixed'
        num_compare = len(task_vecs_mixed_list)
        colors = sns.color_palette('husl', num_compare)
        labels = [label, '+ rand ans', '+ rand str', 'other task']
        for j in range(num_compare):
            plt.plot(layers, result[i*num_compare+j], label=labels[j], color=colors[j], linestyle='--' if j > 0 else '-')
    plt.axvline(x=best_layer_1, color='grey', linestyle='-', alpha=0.5)
    plt.axvline(x=best_layer_2, color='grey', linestyle='-', alpha=0.5)
    plt.xlabel('layer')
    plt.ylabel('linear fit residual')
    plt.legend(bbox_to_anchor=(0., 1.05, 1., .105), loc='lower left', mode='expand', ncol=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(save_loc)
