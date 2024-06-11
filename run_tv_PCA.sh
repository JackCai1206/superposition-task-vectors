set -e

# "meta-llama/Meta-Llama-3-8B"
# "meta-llama/Llama-2-7b-hf"
for model_id in "meta-llama/Meta-Llama-3-8B"; do
    for opA in "COPY_A/NOP/NOP/NOP"; do
        for opB in "COPY_B/NOP/NOP/NOP"; do 
            for opC in "ADD/NOP/NOP/NOP"; do
                CUDA_VISIBLE_DEVICES=0 python llama_task_vectors.py \
                    --model_id=$model_id \
                    --num_examples=100 \
                    --prompt_size=100 \
                    --task1=$opA \
                    --task2=$opB \
                    --task3=$opC \
                    --do_tv_PCA=True
            done
        done
    done
done
