## questions

1. discard the LSTM head

## result

#### P-tuning 

--save_steps 100 --learning_rate 5e-5 --num_train_epochs 100.0 --train_batch_size 32

the model evaluated on sighan15 is trained on train_all.txt

|          | CPT   | CPT+MLM | FT    | MFT   |
| -------- | ----- | ------- | ----- | ----- |
| law      | 55.51 | 72.31   | 37.87 | 66.27 |
| med      | 60.03 | 71.53   | 22.34 | 53.11 |
| sighan15 |       | 52.51   | 43.37 | 52.63 |

#### discrete prompt 

--save_steps 100 --learning_rate 5e-5 --num_train_epochs 100.0 --train_batch_size 32

the model evaluated on sighan15 is trained on train_all.txt

|          | PT    | PT+MLM | FT    | MFT   |
| -------- | ----- | ------ | ----- | ----- |
| law      | 52.70 | 69.90  | 37.87 | 66.27 |
| med      | 55.54 | 69.76  | 22.34 | 53.11 |
| sighan15 |       |        | 43.37 | 52.63 |

43.37

## command

### train using prompt

```
############use p-tuning#################
## use mft
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --prompt_length 3 --task_name "sighan"  --train_on "all"  --eval_on '15' --save_steps 100 --output_dir "model/model_15" --learning_rate 5e-5 --num_train_epochs 60.0 --train_batch_size 32 --eval_batch_size 64
## not use mft
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --prompt_length 3 --task_name "sighan"  --train_on "all"  --eval_on '15' --save_steps 100 --output_dir "model/model_15" --learning_rate 5e-5 --num_train_epochs 60.0 --train_batch_size 32 --eval_batch_size 64
#########use discret prompt################
## use mft
CUDA_VISIBLE_DEVICES=0 python run_discrete.py --do_train --do_eval --mft --task_name "sighan"  --train_on "all"  --eval_on '15' --save_steps 100 --learning_rate 5e-5 --output_dir "model/model_15" --num_train_epochs 60.0 --train_batch_size 32
## not use mft
CUDA_VISIBLE_DEVICES=0 python run_discrete.py --do_train --do_eval --task_name "sighan"  --train_on "all"  --eval_on '15' --save_steps 100 --learning_rate 5e-5 --output_dir "model/model_15" --num_train_epochs 60.0 --train_batch_size 32
######### fine tune##########################
## use mft
CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --mft --task_name "sighan"  --train_on "all"  --eval_on '15' --output_dir "model/model_15" --save_steps 100 --learning_rate 5e-5 --num_train_epochs 60.0 --train_batch_size 32
## not use mft
CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --task_name "sighan"  --train_on "all"  --eval_on '15' --output_dir "model/model_15" --save_steps 100 --learning_rate 5e-5 --num_train_epochs 60.0 --train_batch_size 32
```

