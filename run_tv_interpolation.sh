set -e

# "meta-llama/Meta-Llama-3-8B"
# "meta-llama/Llama-2-7b-hf"
for model_id in "meta-llama/Llama-2-7b-hf"; do
    for opA in "ADD/NOP/NOP/NOP"; do
        for opB in "COPY_B/NOP/NOP/NOP"; do 
            python llama_task_vectors.py \
                --model_id=$model_id \
                --num_examples=100 \
                --prompt_size=200 \
                --task1=$opA \
                --task2=$opB \
                --average_over=10 \
                --do_interpolation=True \
                --do_residual=False \
                --use_results_cache=False
        done
    done
done
