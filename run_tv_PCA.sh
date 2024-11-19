set -e

# "meta-llama/Meta-Llama-3-8B"
# "meta-llama/Llama-2-7b-hf"
        # "COPY_A/NOP/NOP/NOP" "COPY_B/NOP/NOP/NOP" "ADD/NOP/NOP/NOP" 2 30 "copy(op1)" "copy(op2)" "op1 + op2" \
        # "COPY_A/SUB_1/NOP/NOP" "COPY_B/NOP/NOP/NOP" "COPY_C/SUB_5/NOP/NOP" 3 30 "op1 - 1" "copy(op2)" "op3 - 5" \
        # "COPY_A/NOP/TO_FR/NOP" "COPY_A/NOP/TO_DE/NOP" "COPY_A/NOP/TO_IT/NOP" 1 60 "to_fr(op1)" "to_de(op1)" "to_it(op1)" \
    
        # "COPY_A/NOP/NOP/NOP" "COPY_B/NOP/NOP/NOP" "ADD/NOP/NOP/NOP" 2 30 \
        # "COPY_A/SUB_1/NOP/NOP" "COPY_B/NOP/NOP/NOP" "COPY_C/SUB_5/NOP/NOP" 3 30 \
        # "COPY_A/NOP/TO_FR/NOP" "COPY_A/NOP/TO_DE/NOP" "COPY_A/NOP/TO_IT/NOP" 1 60 \

        # "meta-llama/Meta-Llama-3-8B" "CAPITAL/NOP/NOP/NOP" "CONTINENT/NOP/NOP/NOP" "CAPITALIZE/NOP/NOP/NOP" 1 15 \
        # "meta-llama/Meta-Llama-3-8B" "ADD/NOP/TO_ES/NOP" "ADD/NOP/TO_ENG/NOP" "ADD/NOP/TO_FR/NOP" 2 15 \
        # "../icl-pretraining/out3/_copy_6_384_6_mix_1/checkpoint-5000" "COPY_LETTER_2/NOP/NOP/NOP" "COPY_LETTER_5/NOP/NOP/NOP" "COPY_LETTER_8/NOP/NOP/NOP" 8 30 \
        # "meta-llama/Meta-Llama-3-8B" "COPY_A/NOP/NOP/NOP" "COPY_B/NOP/NOP/NOP" "ADD/NOP/NOP/NOP" 2 30 \
        # "../icl-pretraining/out3/_bias_add_simple_6_384_6_mix_1/checkpoint-5000" "ADD_SIMPLE_2/NOP/NOP/NOP" "ADD_SIMPLE_5/NOP/NOP/NOP" "ADD_SIMPLE_8/NOP/NOP/NOP" 1 30 \
        # "meta-llama/Meta-Llama-3-8B" "ADD/NOP/TO_ES/NOP" "ADD/NOP/TO_ENG/NOP" "ADD/NOP/TO_FR/NOP" "ADD/NOP/NOP/NOP" 2 20 \

for model_id opA opB opC num_operands prompt_size in \
    "../icl-pretraining/out3/_copy_6_384_6_mix_1/checkpoint-5000" "COPY_LETTER_2/NOP/NOP/NOP" "COPY_LETTER_5/NOP/NOP/NOP" "COPY_LETTER_8/NOP/NOP/NOP" 8 30 \
    "../icl-pretraining/out3/_bias_add_simple_6_384_6_mix_1/checkpoint-5000" "ADD_SIMPLE_2/NOP/NOP/NOP" "ADD_SIMPLE_5/NOP/NOP/NOP" "ADD_SIMPLE_8/NOP/NOP/NOP" 1 30 \
; do
    CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 python llama_task_vectors.py \
        --model_id=$model_id \
        --num_examples=100 \
        --prompt_size=$prompt_size \
        --task1=$opA \
        --task2=$opB \
        --task3=$opC \
        --do_tv_PCA=True \
        --use_task_vec_cache=True \
        --use_results_cache=True \
        --num_operands=$num_operands \
        --skip_title=True \
        --out_dir='out_paper_figures' \
        --layers 0 6 \
        --separator='>' \
        --fit_PCA_all_classes=True
done

# for model_id opA opB opC num_operands prompt_size in \
#     "meta-llama/Meta-Llama-3-8B" "CAPITAL/NOP/NOP/NOP" "CONTINENT/NOP/NOP/NOP" "CAPITALIZE/NOP/NOP/NOP" 1 15 \
#     # "meta-llama/Meta-Llama-3-8B" "ADD/NOP/TO_ES/NOP" "ADD/NOP/TO_ENG/NOP" "ADD/NOP/TO_FR/NOP" 2 15 \
#     # "meta-llama/Meta-Llama-3-8B" "COPY_A/NOP/NOP/NOP" "COPY_B/NOP/NOP/NOP" "ADD/NOP/NOP/NOP" 2 30 \
# ; do
#     CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 python llama_task_vectors.py \
#         --model_id=$model_id \
#         --num_examples=100 \
#         --prompt_size=$prompt_size \
#         --task1=$opA \
#         --task2=$opB \
#         --task3=$opC \
#         --do_tv_PCA=True \
#         --use_task_vec_cache=True \
#         --use_results_cache=True \
#         --num_operands=$num_operands \
#         --skip_title=True \
#         --out_dir='out_paper_figures' \
#         --use_alt_dataset_impl=True
# done

# for model_id opA opB opC opD num_operands prompt_size in \
#     "meta-llama/Meta-Llama-3-8B" "ADD/NOP/TO_ES/NOP" "ADD/NOP/TO_ENG/NOP" "ADD/NOP/TO_FR/NOP" "ADD/NOP/NOP/NOP" 2 20 \
# ; do
#     CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 python llama_task_vectors.py \
#         --model_id=$model_id \
#         --num_examples=100 \
#         --prompt_size=$prompt_size \
#         --task1=$opA \
#         --task2=$opB \
#         --task3=$opC \
#         --task4=$opD \
#         --do_tv_PCA=True \
#         --use_task_vec_cache=True \
#         --use_results_cache=False \
#         --num_operands=$num_operands \
#         --skip_title=True \
#         --out_dir='out_paper_figures' \
#         --fit_PCA_all_classes=True
# done
