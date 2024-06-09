set -e

# "meta-llama/Meta-Llama-3-8B" 
for model_id in "meta-llama/Llama-2-7b-hf"; do
    for opA in "COPY_A/NOP/TO_ENG/NOP"; do
        for opB in "COPY_A/ADD_1/NOP/NOP"; do 
            python llama_task_vectors.py \
                --model_id=$model_id \
                --num_examples=100 \
                --prompt_size=100 \
                --task1=$opA \
                --task2=$opB \
                --average_over=5 \
                --do_interpolation=True \
                --do_residual=False 
        done
    done
done
