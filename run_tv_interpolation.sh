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
for model_id in "meta-llama/Meta-Llama-3-8B"; do
    for opA opB opC num_operands prompt_size opA_alias opB_alias opC_alias in \
        "COPY_A/NOP/TO_DE/NOP" "COPY_A/NOP/TO_IT/NOP" "COPY_A/NOP/TO_FR/NOP"  1 60 "to_de" "to_it" "to_fr" \
    ; do
        CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 python llama_task_vectors.py \
            --model_id=$model_id \
            --num_examples=100 \
            --prompt_size=$prompt_size \
            --task1=$opA \
            --task2=$opB \
            --task3=$opC \
            --task1_alias=$opA_alias \
            --task2_alias=$opB_alias \
            --task3_alias=$opC_alias \
            --average_over=2 \
            --do_interpolation=True \
            --use_task_vec_cache=True \
            --use_results_cache=True \
            --num_operands=$num_operands \
            --out_dir='out_paper_figures_final'
    done
done
