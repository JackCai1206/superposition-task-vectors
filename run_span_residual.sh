for model_id in "meta-llama/Llama-2-7b-hf"; do
    python -m pdb -c continue llama_task_vectors.py \
        --model_id=$model_id \
        --num_examples=100 \
        --prompt_size=100 \
        --task1="COPY_A/NOP/NOP/NOP" \
        --task2="COPY_B/NOP/NOP/NOP" \
        --average_over=5 \
        --do_interpolation=False \
        --do_residual=True
done
