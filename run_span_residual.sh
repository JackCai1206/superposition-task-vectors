set -e

for model_id in "meta-llama/Meta-Llama-3-8B"; do
    for opA opB opC num_operands prompt_size opC_alias in \
            "COPY_A/NOP/NOP/NOP" "COPY_B/NOP/NOP/NOP" "ADD/NOP/NOP/NOP" 2 30 'op1 + op2' \
            "COPY_A/NOP/TO_FR/NOP" "COPY_A/NOP/TO_DE/NOP" "COPY_A/NOP/TO_IT/NOP" 1 30 'to_it(op1)'\
            "COPY_A/SUB_1/NOP/NOP" "COPY_B/NOP/NOP/NOP" "COPY_C/SUB_5/NOP/NOP" 3 30 'op3 - 5'\
        ; do
        CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python llama_task_vectors.py \
            --model_id=$model_id \
            --num_examples=100 \
            --prompt_size=$prompt_size \
            --task1=$opA \
            --task2=$opB \
            --task3=$opC \
            --task3_alias=$opC_alias \
            --average_over=2 \
            --num_operands=$num_operands \
            --do_residual=True \
            --use_task_vec_cache=True \
            --use_results_cache=True \
            --out_dir='out_paper_figures'
    done
done
