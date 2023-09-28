#### mask strategy

##dynamic mask

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "all" --mask_rate 0.3 --prompt_length 10 --task_name "ecspell"  --train_on "law"  --eval_on 'law' --save_steps 100  --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_law"

step-900_f1-90.53.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "all" --prompt_length 3 --task_name "ecspell"  --train_on "med"  --eval_on 'med' --save_steps 100 --output_dir "model/model_med" --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 32 --eval_batch_size 64

step-3700_f1-82.87.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "all" --mask_rate 0.3 --prompt_length 10 --task_name "ecspell"  --train_on "odw"  --eval_on 'odw' --save_steps 100 --output_dir "model/model_odw" --learning_rate 5e-5 --max_train_steps 1500 --train_batch_size 128 --eval_batch_size 64

step-1400_f1-84.01.bin



#### false positive rate

law BERT-fine-tune

|           | f1   | p    | r    | fpr  | wpr  |
| --------- | ---- | ---- | ---- | ---- | ---- |
| 50/       | 39.9 | 46.5 | 34.9 | 10.0 |      |
| 100/      | 39.6 | 45.6 | 34.9 | 9.0  |      |
| 150/      | 39.0 | 44.3 | 34.9 | 10.0 |      |
| 200/      | 38.1 | 42.2 | 34.9 | 12.5 |      |
| 245(all)/ | 37.9 | 41.4 | 34.9 | 11.8 | 38.1 |
| 300       | 37.3 | 40.0 | 34.9 | 12.0 |      |
| 400       | 36.5 | 38.2 | 34.9 | 11.8 |      |

law BERT-fine-tune

|           | f1   | p    | r    | fpr  | wpr  |
| --------- | ---- | ---- | ---- | ---- | ---- |
| 50/       | 55.4 | 58.8 | 54.7 |      |      |
| 100/      |      |      | 54.7 |      |      |
| 150/      |      |      | 54.7 |      |      |
| 200/      |      |      | 54.7 |      |      |
| 245(all)/ | 57.3 | 60.0 | 54.7 |      | 34.3 |
| 300       |      |      | 54.7 |      |      |
| 400       |      |      | 54.7 |      |      |





#### linear probing

|                 | LAW   | ->TNEWS | ->AFQMC |
| --------------- | ----- | ------- | ------- |
| BERT            | 39.14 | 13.15   | 69.00   |
| BERT+mft        | 71.04 | 16.1    | \       |
| LMCorrector     | 53.41 | 51.04   | \       |
| LMCorrector+mft | 90.81 | 50.41   | \       |

|                 | MED   | ->TNEWS | ->AFQMC |
| --------------- | ----- | ------- | ------- |
| BERT            | 23.20 | 14.79   | \       |
| BERT+mft        | 40.49 | 17.6    | \       |
| LMCorrector     | 61.36 | 33.72   | \       |
| LMCorrector+mft | 83.85 | 49.66   | \       |

|                 | ODW   | ->TNEWS | ->AFQMC |
| --------------- | ----- | ------- | ------- |
| BERT            | 25.93 | 15.67   | \       |
| BERT+mft        | 58.62 | 18.48   | \       |
| LMCorrector     | 56.76 | 48.77   | \       |
| LMCorrector+mft | 84.52 | 50.08   | \       |

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --do_eval --task_name "ecspell"  --train_on "med"  --eval_on 'med' --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --print_para_names

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --task_name "ecspell"  --train_on "med"  --eval_on 'med' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --print_para_names

#### BERT



```
###law

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --do_eval --task_name "ecspell"  --train_on "law"  --eval_on 'law' --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"
step-5000_f1-39.14.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --do_eval --mft --task_name "ecspell"  --train_on "law"  --eval_on 'law' --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"
step-4000_f1-51.70.bin

#####dynamic######
CUDA_VISIBLE_DEVICES=1 python run_multi.py --do_train --do_eval --mft --mask_rate 0.3 --task_name "ecspell"  --train_on "law"  --eval_on 'law' --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
step-5000_f1-71.04.bin

#linear probing tnews:

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --load_checkpoint "model/model_multi/step-5000_f1-39.14.bin" --do_eval --task_name "tnews"  --train_on "base"  --eval_on 'base' --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --freeze_lm
step-3000_f1-13.15.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --load_checkpoint "model/model_multi/step-4000_f1-51.70.bin" --do_eval --task_name "tnews"  --train_on "base"  --eval_on 'base' --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --freeze_lm
step-5000_f1-16.07.bin

CUDA_VISIBLE_DEVICES=1 python run_multi.py --do_train --load_checkpoint "model/model_multi/step-5000_f1-71.04.bin" --do_eval --task_name "tnews"  --train_on "base"  --eval_on 'base' --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --freeze_lm --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"

#linear probing afqmc:

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --load_checkpoint "model/model_multi/step-5000_f1-39.14.bin" --do_eval --task_name "afqmc"  --train_on "base"  --eval_on 'base' --save_steps 100 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --freeze_lm
```

```
###med

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --do_eval --task_name "ecspell"  --train_on "med"  --eval_on 'med' --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"

step-5000_f1-23.20.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --do_eval --mft --task_name "ecspell"  --train_on "med"  --eval_on 'med' --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"

step-3000_f1-40.49.bin

#linear probing tnews:

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --load_checkpoint "model/model_multi/step-5000_f1-23.20.bin" --do_eval --task_name "tnews"  --train_on "base"  --eval_on 'base' --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --freeze_lm

step-5000_f1-14.79.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --load_checkpoint "model/model_multi/step-3000_f1-40.49.bin" --do_eval --task_name "tnews"  --train_on "base"  --eval_on 'base' --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --freeze_lm

step-5000_f1-17.63.bin
```

```
#odw

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --do_eval --task_name "ecspell"  --train_on "odw"  --eval_on 'odw' --save_steps 100 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
step-4500_f1-25.93.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --do_eval --mft --mask_rate 0.3 --task_name "ecspell"  --train_on "odw"  --eval_on 'odw' --save_steps 100 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
step-3700_f1-58.62.bin

#linear probing tnews:

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --load_checkpoint "model/model_multi/step-4500_f1-25.93.bin" --do_eval --task_name "tnews"  --train_on "base"  --eval_on 'base' --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --freeze_lm  --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
step-5000_f1-15.67.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --load_checkpoint "model/model_multi/step-3700_f1-58.62.bin" --do_eval --task_name "tnews"  --train_on "base"  --eval_on 'base' --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --freeze_lm --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
step-5000_f1-18.48.bin
```



#### LMCorrector



```
#law

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --load_state_dict "model/model_multi/step-2000_f1-52.61.bin"  --task_name "ecspell"  --train_on "law"  --eval_on 'law' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"
step-3000_f1-53.41.bin

python run_multi_ptuning.py --do_train --do_eval --mft --task_name "ecspell"  --train_on "law"  --eval_on 'law' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"
step-1000_f1-71.05.bin

###dynamic#####
python run_multi_ptuning.py --do_train --do_eval --mft --mask_rate 0.3 --task_name "ecspell"  --train_on "law"  --eval_on 'law' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"
step-3000_f1-90.81.bin

#linear probing tnews:
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --load_state_dict "model/model_multi/step-3000_f1-53.41.bin" --task_name "tnews"  --train_on "base"  --eval_on 'base' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 100 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --freeze_lm
step-4500_f1-51.04.bin

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --load_state_dict "model/model_multi/step-1000_f1-71.05.bin" --task_name "tnews"  --train_on "base"  --eval_on 'base' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --freeze_lm

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --load_state_dict "model/model_multi/step-3000_f1-90.81.bin" --task_name "tnews"  --train_on "base"  --eval_on 'base' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --freeze_lm
step-5000_f1-50.41.bin

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --load_state_dict "model/model_multi/step-3000_f1-90.81.bin" --task_name "tnews"  --train_on "base"  --eval_on 'base' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --freeze_lm --not_apply_prompt --linear_prob
```

```
#med

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --task_name "ecspell"  --train_on "med"  --eval_on 'med' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 100 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"
step-5000_f1-61.36.bin

CUDA_VISIBLE_DEVICES=1 python run_multi_ptuning.py --do_train --do_eval --mft --task_name "ecspell"  --train_on "med"  --eval_on 'med' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"
step-2000_f1-74.36.bin

###dynamic#####
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --mft --mask_rate 0.3 --task_name "ecspell"  --train_on "med"  --eval_on 'med' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
step-2000_f1-83.85.bin
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --mft --mask_rate 0.3 --task_name "ecspell"  --train_on "med"  --eval_on 'med' --csc_prompt_length 1 --sent_prompt_length 1 --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 32 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --not_apply_prompt
step-5000_f1-81.53.bin

#linear probing tnews:

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --load_state_dict "model/model_multi/step-5000_f1-61.36.bin" --task_name "tnews"  --train_on "base"  --eval_on 'base' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --max_train_steps 3000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --freeze_lm
step-3000_f1-33.72.bin

CUDA_VISIBLE_DEVICES=1 python run_multi_ptuning.py --do_train --do_eval --load_state_dict "model/model_multi/step-2000_f1-74.36.bin" --task_name "tnews"  --train_on "base"  --eval_on 'base' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --max_train_steps 2000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --freeze_lm
step-2000_f1-49.74.bin

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --load_state_dict "model/model_multi/step-2000_f1-83.85.bin" --task_name "tnews"  --train_on "base"  --eval_on 'base' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --max_train_steps 2000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --freeze_lm --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
step-2000_f1-49.67.bin
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --load_state_dict "model/model_multi/step-5000_f1-81.53.bin" --task_name "tnews"  --train_on "base"  --eval_on 'base' --csc_prompt_length 1 --sent_prompt_length 1 --save_steps 1000 --learning_rate 5e-5 --max_train_steps 2000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --freeze_lm --linear_prob --not_apply_prompt
```

```
#odw

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --task_name "ecspell"  --train_on "odw"  --eval_on 'odw' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 100 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"
step-3200_f1-56.76.bin

python run_multi_ptuning.py --do_train --do_eval --mft --mask_rate 0.3 --task_name "ecspell"  --train_on "odw"  --eval_on 'odw' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 100 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"
step-3300_f1-84.52.bin

#linear probing tnews:
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --load_state_dict "model/model_multi/step-3300_f1-84.52.bin" --task_name "tnews"  --train_on "base"  --eval_on 'base' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --freeze_lm
step-5000_f1-50.08.bin
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --load_state_dict "model/model_multi/step-3300_f1-84.52.bin" --task_name "tnews"  --train_on "base"  --eval_on 'base' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --freeze_lm --not_apply_prompt --linear_prob
step-3000_f1-53.65.bin

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --load_state_dict "model/model_multi/step-3200_f1-56.76.bin" --task_name "tnews"  --train_on "base"  --eval_on 'base' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --freeze_lm
step-5000_f1-48.77.bin
```



#### chatGPT

law - 34.8

med - 22.3

odw - 41.0



### dynamic mask

| mask_rate | law   | med   | odw   |
| --------- | ----- | ----- | ----- |
| 0.1       | 89.96 | 84.17 | 82.50 |
| 0.2       | 91.33 | 84.82 | 86.93 |
| 0.3       | 92.17 | 85.40 | 86.71 |
| 0.4       | 91.25 | 82.82 | 84.89 |
| 0.6       | 86.73 | 81.79 | 78.81 |

ecspell-single

```
test_law_adv_2_use_ner_ratio0.0_0.txt
### law
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.1 --prompt_length 10 --task_name "ecspell"  --train_on "law"  --eval_on 'law' --save_steps 100  --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_law"
step-3800_f1-89.96.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.2 --prompt_length 10 --task_name "ecspell"  --train_on "law"  --eval_on 'law' --save_steps 100  --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_law"
step-3200_f1-91.33.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.3 --prompt_length 10 --task_name "ecspell"  --train_on "law"  --eval_on 'law' --save_steps 100  --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_law"
step-1600_f1-92.17.bin
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --load_state_dict "model/model_law/step-1600_f1-92.17.bin"  --prompt_length 10 --task_name "ecspell" --output_dir "model/model"  --test_on "law" 

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.4 --prompt_length 10 --task_name "ecspell"  --train_on "law"  --eval_on 'law' --save_steps 100  --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_law"
step-1700_f1-91.25.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.6 --prompt_length 10 --task_name "ecspell"  --train_on "law"  --eval_on 'law' --save_steps 100  --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_law"
step-1100_f1-86.73.bin


CUDA_VISIBLE_DEVICES=0 python run_ginga.py --do_test --test_on "ecspell/test_law.txt" --data_dir "data/" --load_state_dict "../csc/eclaw/step-5000_f1-76.08.bin"
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --load_state_dict "model/model_law/step-1600_f1-92.17.bin"  --prompt_length 10 --task_name "ecspell"  --test_on "law_adv_2_use_ner_ratio0.0_0" 
89.13
CUDA_VISIBLE_DEVICES=0 python run_ginga.py --do_test --test_on "ecspell/test_law_adv_2_use_ner_ratio0.0_0.txt" --data_dir "data/" --load_state_dict "../csc/eclaw/step-5000_f1-76.08.bin"
74.44
CUDA_VISIBLE_DEVICES=0 python run.py  --do_test --task_name "ecspell" --load_checkpoint "model/model_law/step-4100_f1-37.87.bin" --mask_mode "noerror"  --test_on "law_adv_2_use_ner_ratio0.0_0"
7.03


```

```
## odw

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.1 --prompt_length 10 --task_name "ecspell"  --train_on "odw"  --eval_on 'odw' --save_steps 100 --output_dir "model/model_odw" --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
step-2900_f1-82.50.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.2 --prompt_length 10 --task_name "ecspell"  --train_on "odw"  --eval_on 'odw' --save_steps 100 --output_dir "model/model_odw" --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
step-5000_f1-86.93.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.3 --prompt_length 10 --task_name "ecspell"  --train_on "odw"  --eval_on 'odw' --save_steps 100 --output_dir "model/model_odw" --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64
step-4800_f1-86.71.bin
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --load_state_dict "model/model_odw/step-4800_f1-86.71.bin"  --prompt_length 10 --task_name "ecspell" --test_on "odw" 

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.4  --prompt_length 10 --task_name "ecspell"  --train_on "odw"  --eval_on 'odw' --save_steps 100 --output_dir "model/model_odw" --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64
step-2000_f1-84.89.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.6 --prompt_length 10 --task_name "ecspell"  --train_on "odw"  --eval_on 'odw' --save_steps 100 --output_dir "model/model_odw" --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
step-1300_f1-78.81.bin


CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --load_state_dict "model/model_odw/step-4800_f1-86.71.bin"  --prompt_length 10 --task_name "ecspell"  --test_on "odw_adv_2_use_ner_ratio0.0_0" 
82.31
CUDA_VISIBLE_DEVICES=0 python run.py  --do_test --task_name "ecspell" --load_checkpoint "model/model_odw/step-2800_f1-59.23.bin"  --test_on 'odw_adv_2_use_ner_ratio0.0_0' 
64.7
CUDA_VISIBLE_DEVICES=0 python run.py  --do_test --task_name "ecspell" --load_checkpoint "model/model_odw/step-4100_f1-25.00.bin" --mask_mode "noerror"  --test_on 'odw_adv_2_use_ner_ratio0.0_0' 
4.74

```

```
## med

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.1 --prompt_length 10 --task_name "ecspell"  --train_on "med"  --eval_on 'med' --save_steps 100  --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 32 --eval_batch_size 64 --fp16 --output_dir "model/model_med"

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.2 --prompt_length 10 --task_name "ecspell"  --train_on "med"  --eval_on 'med' --save_steps 100  --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 32 --eval_batch_size 64 --fp16 --output_dir "model/model_med"
step-4600_f1-84.82.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.3 --prompt_length 10 --task_name "ecspell"  --train_on "med"  --eval_on 'med' --save_steps 100  --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 32 --eval_batch_size 64 --fp16 --output_dir "model/model_med"
step-2600_f1-85.40.bin
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --load_state_dict "model/model_med/step-2600_f1-85.40.bin"  --prompt_length 10 --task_name "ecspell"  --test_on "med" 
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.3 --prompt_length 10 --task_name "ecspell"  --train_on "med"  --eval_on 'med' --save_steps 100  --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 32 --eval_batch_size 64 --fp16 --output_dir "model/model_med" --load_model_path "../cache/bert_chinese_mc_base"
step-3200_f1-85.01.bin
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.3 --prompt_length 10 --task_name "ecspell"  --train_on "med"  --eval_on 'med' --save_steps 100  --learning_rate 2e-5 --max_train_steps 5000 --train_batch_size 32 --eval_batch_size 64 --fp16 --output_dir "model/model_med" --load_model_path "../cache/bert_chinese_mc_base"
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.3 --prompt_length 10 --task_name "ecspell"  --train_on "med"  --eval_on 'med' --save_steps 100  --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_med" --load_model_path "../cache/bert_chinese_mc_base"


CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.4 --prompt_length 10 --task_name "ecspell"  --train_on "med"  --eval_on 'med' --save_steps 100  --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 32 --eval_batch_size 64 --fp16 --output_dir "model/model_med"
step-3800_f1-82.82.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.6 --prompt_length 10 --task_name "ecspell"  --train_on "med"  --eval_on 'med' --save_steps 100  --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 32 --eval_batch_size 64 --fp16 --output_dir "model/model_med" --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --load_state_dict "model/model_med/step-2600_f1-85.40.bin"  --prompt_length 10 --task_name "ecspell"  --test_on "med_adv_2_use_ner_ratio0.0_0" 
73.01
CUDA_VISIBLE_DEVICES=0 python run_ginga.py  --do_test --test_on "ecspell/test_med_adv_2_use_ner_ratio0.0_0.txt" --data_dir "data/" --load_state_dict "model/model_med/step-4900_f1-58.00.bin" 
52.83
CUDA_VISIBLE_DEVICES=0 python run.py --do_test --load_checkpoint "../promptCSC/model/model_med/step-4600_f1-22.34.bin" --task_name "ecspell"  --test_on "med_adv_2_use_ner_ratio0.0_0" 
4.09
```

| test_on adv_2_use_ner_ratio0.0 | odw  | med  | law  |
| ------------------------------ | ---- | ---- | ---- |
| bert-tag                       | 4.7  | 4.1  | 7.0  |
| bert-tag-mft                   | 64.7 | 52.8 | 74.4 |
| mdcspell-mft                   | 71.3 | 60.1 | 79.2 |
| relm-bert                      | 82.3 | 73.0 | 89.1 |



### no p-tuning ablation

```
## law
#use mft
CUDA_VISIBLE_DEVICES=0 python run_discrete.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.3 --task_name "ecspell"  --train_on "law"  --eval_on 'law' --save_steps 100 --learning_rate 5e-5 --output_dir "model/model_law" --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
step-2100_f1-90.52.bin
#not use mft
CUDA_VISIBLE_DEVICES=0 python run_discrete.py --do_train --do_eval --task_name "ecspell"  --train_on "law"  --eval_on 'law' --save_steps 100 --learning_rate 5e-5 --output_dir "model/model_law" --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64
step-3600_f1-53.28.bin

## med
#not use mft
CUDA_VISIBLE_DEVICES=0 python run_discrete.py --do_train --do_eval --task_name "ecspell"  --train_on "med"  --eval_on 'med' --save_steps 100 --learning_rate 5e-5 --output_dir "model/model_med" --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64
step-1300_f1-55.04.bin
#use mft
CUDA_VISIBLE_DEVICES=0 python run_discrete.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.3 --task_name "ecspell"  --train_on "med"  --eval_on 'med' --save_steps 100 --learning_rate 5e-5 --output_dir "model/model_med" --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
step-2200_f1-81.79.bin

##odw
#use mft
CUDA_VISIBLE_DEVICES=0 python run_discrete.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.3 --task_name "ecspell"  --train_on "odw"  --eval_on 'odw' --save_steps 100 --learning_rate 5e-5 --output_dir "model/model_odw" --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 
step-4600_f1-83.28.bin
```



#### gpt-2 baseline

gpt2-chinese-cluecorpussmall

|                          | law                       | med                       | odw                       |
| ------------------------ | ------------------------- | ------------------------- | ------------------------- |
| gpt2-base                | 41.52  /11.83/45.16/38.43 | 20.87  /30.29/19.76/22.12 | 34.93  /23.93/35.68/34.21 |
| gpt2-mft                 | 71.19  /39.59/61.60/84.31 | 35.62  /49.27/29.61/44.69 | 53.77  /47.00/46.22/64.28 |
| gpt2-mft(control length) | 72.33                     | 46.01                     | 56.02                     |
| above+prefix(10)         | 72.06/ 39.6               |                           |                           |

f1/fpr/p/r

```
## no mft
CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_train --do_eval --task_name "ecspell" --train_on "law" --eval_on "law" --save_step 100 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_dir "model/model_law" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --fp16 --max_train_steps 5000
step-3600_f1-42.83.bin step-2100_f1-44.73.bin
CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_test --task_name "ecspell" --eval_on "law" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --load_state_dict "model/model_law/step-2100_f1-44.73.bin"

CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_train --do_eval --task_name "ecspell" --train_on "med" --eval_on "med" --save_step 100 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_dir "model/model_med" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --fp16 --max_train_steps 5000
CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_test --task_name "ecspell" --eval_on "med" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --load_state_dict "model/model_med/step-2300_f1-32.83.bin"

CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_train --do_eval --task_name "ecspell" --train_on "odw" --eval_on "odw" --save_step 100 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_dir "model/model_odw" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --fp16 --max_train_steps 5000
step-2000_f1-41.42.bin
CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_test --task_name "ecspell" --eval_on "odw" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --load_state_dict "model/model_odw/step-2000_f1-41.42.bin"

## mft
CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_train --do_eval --mft --task_name "ecspell" --train_on "law" --eval_on "law" --save_step 100 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_dir "model/model_law" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --fp16 --max_train_steps 5000
step-2800_f1-79.46.bin
CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_test --task_name "ecspell" --eval_on "law" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --load_state_dict "model/model_law/step-2800_f1-79.46.bin"
# add arrow
CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_train --do_eval --mft --add_arrow --task_name "ecspell" --train_on "law" --eval_on "law" --save_step 100 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_dir "model/model_law" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --fp16 --max_train_steps 5000
step-4600_f1-46.04.bin
CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_test --add_arrow --task_name "ecspell" --eval_on "law" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --load_state_dict "model/model_law/step-4600_f1-46.04.bin"
f1-38.21, fpr-38.75

CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_train --do_eval --mft --task_name "ecspell" --train_on "med" --eval_on "med" --save_step 100 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_dir "model/model_med" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --fp16 --max_train_steps 5000
step-4000_f1-62.12.bin
CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_test --task_name "ecspell" --eval_on "med" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --load_state_dict "model/model_med/step-4000_f1-62.12.bin"

CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_train --mft --do_eval --task_name "ecspell" --train_on "odw" --eval_on "odw" --save_step 100 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_dir "model/model_odw" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --fp16 --max_train_steps 5000
step-3700_f1-67.99.bin
CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_test --task_name "ecspell" --eval_on "odw" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --load_state_dict "model/model_odw/step-3700_f1-67.99.bin"



## mft+kl_divergence
CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_train --kl_regu --lambd 0.002 --do_eval --mft --task_name "ecspell" --train_on "law" --eval_on "law" --save_step 100 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_dir "model/model_law" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --fp16 --max_train_steps 5000
step-4100_f1-79.20.bin
CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_test --kl_regu --task_name "ecspell" --eval_on "law" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --load_state_dict "model/model_law/step-4100_f1-79.20.bin"
71.82/40.4

CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_train --kl_regu --lambd 0.0005 --do_eval --mft --task_name "ecspell" --train_on "law" --eval_on "law" --save_step 100 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_dir "model/model_law" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --fp16 --max_train_steps 5000
step-2800_f1-79.66.bin
CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_test --kl_regu --task_name "ecspell" --eval_on "law" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --load_state_dict "model/model_law/step-2800_f1-79.66.bin"
72.42/38.36

CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_train --kl_regu --lambd 0.05 --do_eval --mft --task_name "ecspell" --train_on "law" --eval_on "law" --save_step 100 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_dir "model/model_law" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --fp16 --max_train_steps 5000
step-3100_f1-78.33.bin
CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_test --kl_regu --task_name "ecspell" --eval_on "law" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --load_state_dict "model/model_law/step-3100_f1-78.33.bin"
70.62/40.0



CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_train --add_prefix --do_eval --mft --task_name "ecspell" --train_on "law" --eval_on "law" --save_step 100 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_dir "model/model_law" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --fp16 --max_train_steps 5000
step-4500_f1-79.93.bin
CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_test --add_prefix --task_name "ecspell" --eval_on "law" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --load_state_dict "model/model_law/step-4500_f1-79.93.bin"
CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_test --add_prefix --beam 3 --task_name "ecspell" --eval_on "law" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --load_state_dict "model/model_law/step-4500_f1-79.93.bin"
----beam search is not a good idea here.

CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_train --add_prefix --do_eval --mft --task_name "ecspell" --train_on "med" --eval_on "med" --save_step 100 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_dir "model/model_med" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --fp16 --max_train_steps 5000
step-3700_f1-68.20.bin
CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_test --add_prefix --task_name "ecspell" --eval_on "med" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --load_state_dict "model/model_med/step-3700_f1-68.20.bin"


##to do
CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_train --add_prefix --do_eval --mft --task_name "ecspell" --train_on "med" --eval_on "med" --save_step 100 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_dir "model/model_med" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --fp16 --max_train_steps 5000

##to do
CUDA_VISIBLE_DEVICES=0 python run_gpt.py --do_train --add_prefix --do_eval --mft --task_name "ecspell" --train_on "odw" --eval_on "odw" --save_step 100 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_dir "model/model_law" --load_model_path "../cache/gpt2-chinese" --load_tokenizer_path "bert-base-chinese" --fp16 --max_train_steps 5000
```

| lamda | f1   | fpr  |
| ----- | ---- | ---- |
| 0     | 72.3 | 37.9 |
| 0.05  | 70.6 | 40.0 |
| 0.002 | 71.8 | 40.4 |
| 0.005 | 72.4 | 38.3 |

gpt2-tagging

```
# no mask
CUDA_VISIBLE_DEVICES=0 python run.py --do_train --model_type "gpt2" --do_eval --task_name "ecspell" --train_on "law" --eval_on "law" --save_step 100 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_dir "model/model_law" --load_model_path "../cache/gpt2-chinese" --fp16 --max_train_steps 5000
step-1800_f1-34.95.bin 15.1/37.7/32.5

CUDA_VISIBLE_DEVICES=0 python run.py --do_train --model_type "gpt2" --do_eval --task_name "ecspell" --train_on "med" --eval_on "med" --save_step 100 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_dir "model/model_med" --load_model_path "../cache/gpt2-chinese" --fp16 --max_train_steps 5000
step-3900_f1-19.41.bin 9.8/23.1/16.7

CUDA_VISIBLE_DEVICES=0 python run.py --do_train --model_type "gpt2" --do_eval --task_name "ecspell" --train_on "odw" --eval_on "odw" --save_step 100 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_dir "model/model_odw" --load_model_path "../cache/gpt2-chinese" --fp16 --max_train_steps 5000
step-3700_f1-22.81.bin 15.5/26.8/19.8

```



### mdcspell

LISE

```
CUDA_VISIBLE_DEVICES=0 python run.py --do_train --model_type "mdcspell" --do_eval --mft --train_on "train_syn_random_use_ner_ratio0.05.txt"  --eval_on 'test_syn_random_use_ner_ratio0.05.txt' --data_dir "../liulf/data/sghspell" --save_steps 1000 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_lise"
step-8000_f1-76.42.bin
CUDA_VISIBLE_DEVICES=0 python run.py  --do_test --model_type "mdcspell"  --test_on "test_adv_2_use_ner_ratio0.0_0.txt" --data_dir "../liulf/data/sghspell" --load_state_dict "model/model_lise/step-8000_f1-76.42.bin" 
f1-67.58
```

ecspell

```
CUDA_VISIBLE_DEVICES=0 python run.py --do_train --model_type "mdcspell" --do_eval --mft --train_on "train_law.txt"  --eval_on 'test_law.txt' --data_dir "../liulf/data/ecspell" --save_steps 100 --learning_rate 5e-5 --max_train_step 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_ecspell"
CUDA_VISIBLE_DEVICES=0 python run.py --do_test --model_type "mdcspell" --test_on "ecspell/test_law_adv_2_use_ner_ratio0.0_0.txt" --data_dir "../liulf/data/" --load_state_dict "model/model_ecspell/step-4600_f1-80.60.bin"
79.16

CUDA_VISIBLE_DEVICES=0 python run.py --do_train --model_type "mdcspell" --do_eval --mft --train_on "train_med.txt"  --eval_on 'test_med.txt' --data_dir "../liulf/data/ecspell" --save_steps 100 --learning_rate 5e-5 --max_train_step 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_ecspell"
CUDA_VISIBLE_DEVICES=0 python run.py --do_test --model_type "mdcspell" --test_on "ecspell/test_med_adv_2_use_ner_ratio0.0_0.txt" --data_dir "../liulf/data/" --load_state_dict "model/model_ecspell/step-5000_f1-69.63.bin"
60.29
#######--load_model_path "../cache/bert_chinese_mc_base"
CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --mft --train_on "train_med.txt"  --eval_on 'test_med.txt' --data_dir "../liulf/data/ecspell" --save_steps 100 --learning_rate 5e-5 --max_train_step 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_ecspell" --load_model_path "../cache/bert_chinese_mc_base"
step-4800_f1-56.12.bin
CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --mft --train_on "train_med.txt"  --eval_on 'test_med.txt' --data_dir "../liulf/data/ecspell" --save_steps 100 --learning_rate 5e-5 --max_train_step 5000 --train_batch_size 32 --eval_batch_size 64 --fp16 --output_dir "model/model_ecspell" --load_model_path "../cache/bert_chinese_mc_base"
CUDA_VISIBLE_DEVICES=0 python run.py --do_train --model_type "mdcspell" --do_eval --mft --train_on "train_med.txt"  --eval_on 'test_med.txt' --data_dir "../liulf/data/ecspell" --save_steps 100 --learning_rate 5e-5 --max_train_step 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_ecspell" --load_model_path "../cache/bert_chinese_mc_base"
step-4800_f1-67.76.bin

CUDA_VISIBLE_DEVICES=0 python run.py --do_train --model_type "mdcspell" --do_eval --mft --train_on "train_odw.txt"  --eval_on 'test_odw.txt' --data_dir "../data/csc/ecspell" --save_steps 100 --learning_rate 2e-5 --max_train_step 5000 --train_batch_size 32 --eval_batch_size 64 --fp16 --output_dir "model/model_ecspell"
CUDA_VISIBLE_DEVICES=0 python run.py --do_test --model_type "mdcspell" --test_on "ecspell/test_odw_adv_2_use_ner_ratio0.0_0.txt" --data_dir "../liulf/data/" --load_state_dict "model/model_ecspell/step-4600_f1-66.92.bin"
71.36
CUDA_VISIBLE_DEVICES=0 python run.py --do_test --model_type "mdcspell" --test_on "ecspell/test_odw.txt" --data_dir "../liulf/data/" --load_state_dict "model/model_ecspell/step-4600_f1-66.92.bin"

```

```
BERT-Tagging
CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --train_on "train_law.txt"  --eval_on 'test_law.txt' --data_dir "../liulf/data/ecspell" --save_steps 100 --learning_rate 5e-5 --max_train_step 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_ecspell"
f1-39.8
CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --mft --train_on "train_odw.txt"  --eval_on 'test_odw.txt' --data_dir "../data/csc/ecspell" --save_steps 100 --learning_rate 5e-5 --max_train_step 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_ecspell"
```



### LISE transfer to Ecspell

```
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_rate 0.3 --prompt_length 10 --mask_mode "noerror" --task_name "ecspell"  --train_on "sgh"  --eval_on 'law' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_sghspell"

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --load_state_dict "model/model_sghspell/step-1000_f1-28.35.bin" --do_eval --mft --mask_mode "noerror" --mask_rate 0.3 --prompt_length 10 --task_name "ecspell"  --train_on "law"  --eval_on 'law' --save_steps 100  --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_law"

step-1400_f1-91.84.bin
```



### case study

tagging-mft->relm

1. fn->tp

公民在法律面前一律平等主要指司法平等和(手->守)法平等，不包括立法平等

货物的风(线->险)在装运港越过船舷时由公司转移给公司

**怕它(像->向)肌肉和骨头方向发展。** semantics

2. fp->tn

依据我国宪法和有关法律的规定，下列哪(些->一)表述是错误的?

法系划分的主要依据是法(赖->所)以存在的经济基础的性质

下列关于徇私(枉->执)法罪与(包->保)庇罪的区别说法正确是哪些?

**大兴安(岭->山)地区人民医院院长办公室主任** over-correct place name

3. wp->tp

   1. correct in a wrong way

      **(实->执)法权不是一种决策权、执(征->行)权，而是一种判断权** inference based on context and professional knowledge

      

   2. partially correct when there are multiple errors

      我国法律对妇女的权益给予特殊的保(户->护)，这从本职上来说是违背正义原则的

      政务工开是依法行政的(毕->必)然要求

      **责(券->权)人在提出破产申请时可以选择适用重整程序或者清算程序



### use bert-chinese-mc-base

|                  | med   |      |      |
| ---------------- | ----- | ---- | ---- |
| bert-tagging mft | 56.12 |      |      |
| mdcspell mft     | 67.76 |      |      |
| bert-relm        | 85.01 |      |      |





KL divergence

