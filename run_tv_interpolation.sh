set -e

# "meta-llama/Meta-Llama-3-8B" 
for model_id in "meta-llama/Llama-2-7b-hf"; do
    for opA1 opB1 in "ADD" "COPY_A"; do
        python -m pdb -c continue llama_task_vectors.py \
            --model_id=$model_id \
            --num_examples=100 \
            --prompt_size=100 \
            --task1="$opA1/NOP/NOP/NOP" \
            --task2="$opB1/NOP/NOP/NOP" \
            --average_over=5 \
            --do_interpolation=True \
            --do_residual=False
    done
done
