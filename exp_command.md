## questions

1. pad_id(ignore_index): 0 or -100
2. loss only includes mask tokens
3. discard the LSTM head

## result

#### P-tuning 

--save_steps 100 --learning_rate 5e-5 --num_train_epochs 100.0 --train_batch_size 32

|          | CPT  | CPT+MLM | FT   | MFT  |
| -------- | ---- | ------- | ---- | ---- |
| law      |      | 71.62   |      |      |
| med      |      |         |      |      |
| sighan15 |      |         |      |      |

#### discrete prompt 

--save_steps 100 --learning_rate 5e-5 --num_train_epochs 100.0 --train_batch_size 32

|      | CPT  | CPT+MLM | FT   | MFT  |
| ---- | ---- | ------- | ---- | ---- |
| P    |      |         |      |      |
| R    |      |         |      |      |
| f1   |      |         |      |      |



## command

### train using prompt

```
### use mft
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --prompt_length 3 --task_name "ecspell"  --train_on "law"  --eval_on 'law' --save_steps 100 --learning_rate 2e-5 --num_train_epochs 10.0 --train_batch_size 32
## f1=39.79

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --prompt_length 3 --task_name "ecspell"  --train_on "law"  --eval_on 'law' --save_steps 100 --learning_rate 5e-5 --num_train_epochs 80.0 --train_batch_size 32

CUDA_VISIBLE_DEVICES=0 python run_err_corr_continuous_prompt.py --do_train --do_eval --mft --task_name "ecspell"  --train_on "law"  --eval_on 'law' --save_steps 100 --learning_rate 2e-5 --num_train_epochs 10.0 --train_batch_size 32
```

