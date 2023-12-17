

# Chinese Spelling Correction as Rephrasing Language Model

This is the repo for AAAI 2024 paper [Chinese Spelling Correction as Rephrasing Language Model](https://arxiv.org/abs/2308.08796).



## ReLM

Rephrasing Language Model (ReLM) is trained to rephrase the entire sentence by infilling additional slots, instead of character-to-character tagging.  We also evaluate the transferability of ReLM in multi-task settings. You can find some architectures in recent CSC papers and *ReLM* pre-trained model which is a rephrasing language model trained based on bert-base-chinese and 34 million monolingual data from [the repo of Rethinking Masked Language Modeling for Chinese Spelling Correction](https://github.com/gingasan/lemon).



## Experiments

**ReLM on ECSpell** 

```
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
```

**GPT2-rephrasing on ECSpell**

```
CUDA_VISIBLE_DEVICES=0 python run_gpt.py \
 --do_train \
 --do_eval \
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
```

**ReLM in multi-task setting**

train the multi-task model on three distinct tasks, ECSpell for CSC, AFQMC for semantic similarity, and TNEWS for news classification.

```
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

**linear probing on Tnews**

you should replace the --load_state_dict with the path of your model trained on ECSpell.

```
CUDA_VISIBLE_DEVICES=0 python run_relm_multi.py \
 --do_train\
 --do_eval \
 --load_state_dict model_path \
 --task_name "tnews" \
 --train_on "base" \
 --eval_on 'base' \
 --csc_prompt_length 10 \
 --sent_prompt_length 3 \
 --save_steps 1000 \
 --learning_rate 5e-5 \ 
 --max_train_steps 5000 \
 --train_batch_size 128 \
 --eval_batch_size 64 \
 --fp16 \
 --output_dir "model/model_multi" \
 --freeze_lm \
 --not_apply_prompt \
 --linear_prob
```




**query chatGPT with 10-shot setting**

```
cd utils
sh ./query.sh
```
