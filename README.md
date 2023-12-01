# ReLM (Rephrasing Language Model)
use Rephrasing Language Model (ReLM) to do Chinese Spelling Correction.

```
CUDA_VISIBLE_DEVICES=0 python run_gpt.py \
    --do_train \
    --do_eval \
    --kl_regu \
    --lambd 0.002 \
    --mft \
    --task_name "ecspell" \
    --train_on "law" \
    --eval_on "law" \
    --save_step 100 \
    --learning_rate 5e-5 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --max_train_steps 5000 \
    --output_dir "model/model_law" \
    --load_model_path "../../cache/gpt2-chinese" \
    --fp16 

CUDA_VISIBLE_DEVICES=0 python run_gpt.py \
    --do_test \
    --task_name "ecspell" \
    --eval_on "law" \
    --max_seq_length 128 \
    --eval_batch_size 32 \
    --output_dir "model/model_law" \
    --load_model_path "../../cache/gpt2-chinese" \
    --load_state_dict "model/model_law/step-700_f1-54.20.bin" \
    --fp16 

CUDA_VISIBLE_DEVICES=0 python run_relm.py \
 --do_train \
 --do_eval \
 --mft \
 --mask_mode "noerror" \
 --mask_rate 0.3 \
 --prompt_length 1 \
 --task_name "ecspell" \
 --train_on "law"  \
 --eval_on 'law' \
 --save_steps 100  \
 --learning_rate 5e-5 \
 --max_train_steps 5000 \
 --train_batch_size 128 \
 --eval_batch_size 64 \
 --fp16 \
 --output_dir "model/model_law" 

CUDA_VISIBLE_DEVICES=0 python run_multi.py \
    --do_train \
    --mft \
    --do_eval \
    --task_name "ecspell tnews afqmc"  \
    --train_on "law base base"  \
    --eval_on 'law' \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 20.0 \
    --train_batch_size 128 \
    --eval_batch_size 64 \
    --fp16 \
    --output_dir "model/model_multi" \
    --load_model_path "bert-base-chinese"
    
CUDA_VISIBLE_DEVICES=0 python run_relm_multi.py \
    --do_train \
    --do_eval \
    --mft \
    --mask_mode "noerror" \
    --mask_rate 0.3 \
    --task_name "ecspell tnews afqmc"  \
    --train_on "law base base"  \
    --eval_on 'law' \
    --csc_prompt_length 10 \
    --sent_prompt_length 3 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 20.0 \
    --train_batch_size 128 \
    --eval_batch_size 64 \
    --fp16 \
    --output_dir "model/model_multi" 
```


