set -e

# "meta-llama/Meta-Llama-3-8B" 
for model_id in "meta-llama/Llama-2-7b-hf"; do
    python -m pdb -c continue llama_task_vectors.py \
        --model_id=$model_id \
        --num_examples=100 \
        --prompt_size=100 \
        --task1="COPY_A/NOP/NOP/NOP" \
        --task2="COPY_B/NOP/NOP/NOP" \
        --interpolation_average_over=10
done
