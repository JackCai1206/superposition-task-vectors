set -e

# "meta-llama/Meta-Llama-3-8B"
# "meta-llama/Llama-2-7b-hf"
# "COPY_A/SUB_1/NOP/NOP" "COPY_B/ADD_5/NOP/NOP" "COPY_C/ADD_3/NOP/NOP" 3 \
        # "COPY_A/NOP/NOP/NOP" "COPY_B/NOP/NOP/NOP" "ADD/NOP/NOP/NOP" 2 30 "copy(op1)" "copy(op2)" "op1+op2" \
        # "COPY_A/SUB_1/NOP/NOP" "COPY_B/NOP/NOP/NOP" "COPY_C/SUB_5/NOP/NOP" 3 30 "op1-1" "copy(op2)" "op3-5" \
        # "COPY_A/NOP/TO_FR/NOP" "COPY_A/NOP/TO_DE/NOP" "COPY_A/NOP/TO_IT/NOP" 1 60 "to_fr(op1)" "to_de(op1)" "to_it(op1)" \
        # "ADD/NOP/NOP/NOP" "COPY_A/NOP/NOP/NOP" "COPY_B/NOP/NOP/NOP" 2 30 "op1+op2" "copy(op1)" "copy(op2)" \
        # "COPY_C/SUB_5/NOP/NOP" "COPY_A/SUB_1/NOP/NOP" "COPY_B/NOP/NOP/NOP" 3 30 "op3-5" "op1-1" "copy(op3)" \
        # "COPY_A/NOP/TO_IT/NOP" "COPY_A/NOP/TO_FR/NOP" "COPY_A/NOP/TO_DE/NOP" 1 60 "to_it(op1)" "to_fr(op1)" "to_de(op1)" \
        # "COPY_B/NOP/NOP/NOP" "ADD/NOP/NOP/NOP" "COPY_A/NOP/NOP/NOP" 2 30 "copy(op2)" "op1+op2" "copy(op1)" \
        # "COPY_B/NOP/NOP/NOP" "COPY_C/SUB_5/NOP/NOP" "COPY_A/SUB_1/NOP/NOP" 3 30 "copy(op2)" "op3-5" "op1-1" \
        # "COPY_A/NOP/TO_DE/NOP" "COPY_A/NOP/TO_IT/NOP" "COPY_A/NOP/TO_FR/NOP"  1 60 "to_de(op1)" "to_it(op1)" "to_fr(op1)" \
        # "COPY_A/SUB_1/NOP/NOP" "COPY_B/NOP/NOP/NOP" "COPY_C/SUB_5/NOP/NOP" 3 30 "op1-1" "copy(op2)" "op3-5" \
# for model_id in "meta-llama/Meta-Llama-3-8B"; do
#     for opA opB opC num_operands prompt_size in \
#         "COPY_A/NOP/TO_DE/NOP" "COPY_A/NOP/TO_IT/NOP" "COPY_A/NOP/TO_FR/NOP" 1 60 \
#     ; do
#         CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 python llama_task_vectors.py \
#             --model_id=$model_id \
#             --num_examples=100 \
#             --prompt_size=$prompt_size \
#             --task1=$opA \
#             --task2=$opB \
#             --task3=$opC \
#             --average_over=2 \
#             --do_interpolation=True \
#             --use_task_vec_cache=True \
#             --use_results_cache=True \
#             --num_operands=$num_operands \
#             --out_dir='out_paper_figures_final'
#     done
# done

    # "../icl-pretraining/out3/_copy_6_384_2_mix_1/checkpoint-5000" "COPY_LETTER_1/NOP/NOP/NOP" "COPY_LETTER_7/NOP/NOP/NOP" "COPY_LETTER_4/NOP/NOP/NOP" 8 45 '' \
    # "../icl-pretraining/out3/_copy_6_384_6_mix_1/checkpoint-5001" "COPY_LETTER_1/NOP/NOP/NOP" "COPY_LETTER_7/NOP/NOP/NOP" "COPY_LETTER_4/NOP/NOP/NOP" 8 45 '' \
for model_id opA opB opC num_operands prompt_size sep_2 in \
    "../icl-pretraining/out3/_bias_add_simple_6_384_2_mix_1/checkpoint-5000" "ADD_SIMPLE_2/NOP/NOP/NOP" "ADD_SIMPLE_5/NOP/NOP/NOP" "ADD_SIMPLE_3/NOP/NOP/NOP" 1 90 '+' \
    "../icl-pretraining/out3/_bias_add_simple_6_384_6_mix_1/checkpoint-5007" "ADD_SIMPLE_2/NOP/NOP/NOP" "ADD_SIMPLE_5/NOP/NOP/NOP" "ADD_SIMPLE_3/NOP/NOP/NOP" 1 90 '+' \
; do
    CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 python llama_task_vectors.py \
        --model_id=$model_id \
        --num_examples=100 \
        --prompt_size=$prompt_size \
        --task1=$opA \
        --task2=$opB \
        --task3=$opC \
        --do_interpolation=True \
        --use_task_vec_cache=True \
        --use_results_cache=False \
        --num_operands=$num_operands \
        --skip_title=True \
        --out_dir='out_paper_figures' \
        --layers 0 6 \
        --separator='>' \
        --separator2=$sep_2
done
