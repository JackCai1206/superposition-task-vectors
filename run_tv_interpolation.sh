set -e

# "meta-llama/Meta-Llama-3-8B"
# "meta-llama/Llama-2-7b-hf"
for model_id in "meta-llama/Meta-Llama-3-8B"; do
    for opA in "ADD/NOP/NOP/NOP"; do
        for opB in "SUB/NOP/NOP/NOP"; do 
            python llama_task_vectors.py \
                --model_id=$model_id \
                --num_examples=100 \
                --prompt_size=100 \
                --task1=$opA \
                --task2=$opB \
                --average_over=20 \
                --do_interpolation=True \
                --do_residual=False
        done
    done
done
