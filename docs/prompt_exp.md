## questions

1. discard the LSTM head
1. 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

## result

#### P-tuning 

--save_steps 100 --learning_rate 5e-5 --num_train_epochs 80.0 --train_batch_size 32

the model evaluated on sighan15 is trained on train_all.txt

sighan15(1) fine-tuned on train_all.txt

sighan15(2) fine-tuned on train_hybrid.txt and train_all.txt

|                        | CPT（l=3） | CPT+MLM(l=3) | FT    | MFT       | chatGPT |
| ---------------------- | ---------- | ------------ | ----- | --------- | ------- |
| law                    | **55.51**  | **72.31**    | 37.87 | 66.27     | 34.82   |
| med                    | **60.03**  | **71.53**    | 22.34 | 53.11     | 19.12   |
| sighan15 (1)           | **51.95**  | 52.51        | 43.37 | **52.63** |         |
| sighan15 (2)           |            |              |       |           |         |
| sghspell (random)      |            |              |       |           |         |
| sghspell (max entropy) |            |              |       |           |         |

#### influence of the length of prompt

|      | CPT（l=2） | CPT+MLM(l=2) | CPT（l=4） | CPT+MLM(l=4) | CPT（l=3） | CPT+MLM(l=3) |
| ---- | ---------- | ------------ | ---------- | ------------ | ---------- | ------------ |
| law  | 54.14      | 70.88        | 54.04      | 71.54        | **55.51**  | **72.31**    |
| med  | 57.19      | **72.79**    | **60.42**  | 72.30        | 60.03      | 71.53        |

l=3 is the best for law, and the result is relatively stable. (all the results are better than fine-tuning)

#### influence of anchor

##### CPT

|          | no anchor | [SEP]纠错 | [SEP]请纠错 |
| -------- | --------- | --------- | ----------- |
| law(l=3) | 55.51     | 53.97     | **56.54**   |
| med(l=4) | 60.42     |           |             |

##### CPT+MLM

|          | no anchor | [SEP]纠错 | [SEP]请纠错 |
| -------- | --------- | --------- | ----------- |
| law(l=3) | **72.31** | 70.38     | 69.62       |
| med(l=2) | 72.79     |           |             |

#### discrete prompt 

--save_steps 100 --learning_rate 5e-5 --num_train_epochs 80.0 --train_batch_size 32

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
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --prompt_length 2 --task_name "ecspell"  --train_on "med"  --eval_on 'med' --save_steps 100 --output_dir "model/model_med" --learning_rate 5e-5 --num_train_epochs 80.0 --train_batch_size 32 --eval_batch_size 64
## not use mft
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --prompt_length 2 --task_name "ecspell"  --train_on "med"  --eval_on 'med' --save_steps 100 --output_dir "model/model_med" --learning_rate 5e-5 --num_train_epochs 80.0 --train_batch_size 32 --eval_batch_size 64
## use anchor
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --anchor "请纠错" --prompt_length 3 --task_name "ecspell"  --train_on "med"  --eval_on 'med' --save_steps 100 --output_dir "model/model_med" --learning_rate 5e-5 --num_train_epochs 80.0 --train_batch_size 32 --eval_batch_size 64
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --anchor "请纠错" --prompt_length 3 --task_name "ecspell" --train_on "med" --eval_on 'med' --save_steps 100 --output_dir "model/model_med" --learning_rate 5e-5 --num_train_epochs 80.0 --train_batch_size 32 --eval_batch_size 64
## load_checkpoints

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

#####
数据集标点规范（，。）
异常处理
########### use chatgpt########################
python run_chatgpt.py --task_name "ecspell" --test_on "law" --begin 456
python run_chatgpt.py --task_name "ecspell" --test_on "med"

```

