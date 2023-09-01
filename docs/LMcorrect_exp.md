experiment plan

four methods:

1. BERT

2. BERT-mft

3. P-tuning (LMcorrect)

4. P-tuning-mft (LMcorrect)

   

5. single task
   1. BERT-finetune Vs LMcorrect (P-tuning) dataset: sighan ecspell sghspell
   2. exclusive errors, inclusive errors

6. mleeg errors
   1. how to improve the data quality
   2. total performance

7. multi task
   1. P-tuning-multi Vs BERT-multi:  total performance
   2. case study

8. ablation:

   1. mask mode
   2. prompt length





to do:

1. prompt=10 mask all sighan15
2. prompt=10 mask noerror sghspell



### sighan

substitution strategy:

1. 0% mlm: use candidates of mlm to replace the token chosen 
2. 100% plome:
   1. pinyin-> if candidates in confusion: mlm
   2. jinyin-> if candidates in confusion: mlm
   3. stroke-> if candidates in confusion: mlm
   4. random-> if candidates in confusion: mlm

#### train

1. train_all.txt ##sighan 13+sighan 14+sighan 15

2. train_hybrid.txt ##wang

3. train_hybrid_long.txt ##**l>20,num=200000**

4. train_hybrid_random_plome_ner_200000.txt ## **2+**random position+plome substitution strategy

5. train_hybrid_adv_max_50_mlm_plome_window5_long.txt ## **max entropy position(old)**+plome substitution strategy

6. train_hybrid_random_adv_plome_long_200000.txt ## 2+5(long_extract, l>20,num=200000)

7. train_hybrid_adv_max_50_mlm_plome_use_ner_window5_long.txt ## **max entropy position(old)**(**use ner**)+plome substitution strategy

8. train_hybrid_random_adv_plome_ner_long_200000.txt ## 2+7(long_extract, l>20,num=200000)

9. train_hybrid_adv_mlm_ner_window5_200000.txt ##2+**max entropy position(use mlm candidates)**(**use ner**)+plome substitution strategy(based on 3)

10. train_hybrid_adv_mlm_ner_window5_ratio0.05_200000.txt ##2+**max entropy position(use mlm candidates)**(**use ner**)+plome substitution strategy+**multiple_errors**(based on 3)

#### test

1. test_15.txt
2. test_15_adv_5_use_ner_0.txt ##**max entropy position(use mlm candidates)**(**use ner**)+plome substitution strategy
3. test_14.txt
4. test_14_adv_5_use_ner_0.txt ##**max entropy position(use mlm candidates)**(**use ner**)+plome substitution strategy
5. test_13.txt
6. test_13_adv_5_use_ner_0.txt ##**max entropy position(use mlm candidates)**(**use ner**)+plome substitution strategy

#### Prompt

mft is not always

|      | BERT fine-tune | Prompt(BERT) | BERT+mft1 | BERT+mft | Prompt(BERT)+mft+p10(static) | Prompt(BERT)+mft+p10(dynamic) |
| ---- | -------------- | ------------ | --------- | -------- | ---------------------------- | ----------------------------- |
| p    | 70.8           | 75.99        | 74.08     | \        | \                            |                               |
| r    | 75.0           | 76.38        | 78.88     | \        | \                            |                               |
| f1   | 72.84          | 76.19        | 76.41     | 75.79    | 73.22                        | 76.08                         |

#BERT-finetune

```
CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --task_name "sighan"  --train_on "hybrid"  --eval_on '15' --save_steps 1000 --learning_rate 5e-5 --max_train_steps 30000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_sighan"

step-29000_f1-50.65.bin

CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --load_checkpoint "model/model_sighan/step-29000_f1-50.65.bin" --task_name "sighan"  --train_on "all"  --eval_on '15' --save_steps 100 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 32 --eval_batch_size 64

step-2000_f1-72.84.bin

CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --task_name "sighan"  --train_on "hybrid"  --eval_on '15' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64

step-14000_f1-50.16.bin

CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --load_checkpoint "model/step-14000_f1-50.16.bin" --task_name "sighan"  --train_on "hybrid"  --eval_on '15' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 40.0 --train_batch_size 128 --eval_batch_size 64

step-70000_f1-50.28.bin

CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --load_checkpoint "model/step-70000_f1-50.28.bin" --task_name "sighan"  --train_on "all"  --eval_on '15' --save_steps 100 --learning_rate 5e-5 --num_train_epochs 50.0 --train_batch_size 32 --eval_batch_size 64

step-2900_f1-74.23.bin

CUDA_VISIBLE_DEVICES=0 python run.py --do_test --load_checkpoint "model/step-2900_f1-74.23.bin" --task_name "sighan"  --test_on '15' --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run.py --do_test --load_checkpoint "model/model_sighan/step-2000_f1-72.84.bin" --task_name "sighan"  --test_on '15_adv_5_use_ner_0' --eval_batch_size 64
```



#bert mft

```
CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --mft --mask_mode "noerror" --task_name "sighan"  --train_on "hybrid"  --eval_on '15' --save_steps 1000 --learning_rate 5e-5 --max_train_steps 30000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_sighan"

step-18000_f1-54.61.bin

CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --mft --load_checkpoint "model/model_sighan/step-18000_f1-54.61.bin" --task_name "sighan"  --train_on "all"  --eval_on '15' --save_steps 100 --learning_rate 5e-5 --num_train_epochs 50.0 --train_batch_size 32 --eval_batch_size 64 --fp16 --output_dir "model/model_sighan"

step-800_f1-75.79.bin

CUDA_VISIBLE_DEVICES=0 python run.py --load_checkpoint "model/model_sighan/step-800_f1-75.79.bin"  --do_test --task_name "sighan" --test_on "15"

CUDA_VISIBLE_DEVICES=0 python run.py --do_test --load_checkpoint "model/model_sighan/step-800_f1-75.79.bin" --task_name "sighan"  --test_on '15_adv_5_use_ner_0' --eval_batch_size 64

62.08,66.66,64.29



CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --mft --mask_mode "noerror" --task_name "sighan"  --train_on "hybrid"  --eval_on '15' --save_steps 1000 --learning_rate 5e-5 --max_train_steps 30000 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_sighan"

CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --mft --load_checkpoint "model/model_sighan/step-18000_f1-54.61.bin" --task_name "sighan"  --train_on "all"  --eval_on '15' --save_steps 100 --learning_rate 5e-5 --num_train_epochs 50.0 --train_batch_size 32 --eval_batch_size 64 --fp16 --output_dir "model/model_sighan"



CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --mft --task_name "sighan"  --train_on "hybrid"  --eval_on '15' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 30.0 --train_batch_size 128 --eval_batch_size 64

step-41000_f1-53.30.bin

CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --mft --load_checkpoint "model/step-41000_f1-53.30.bin" --task_name "sighan"  --train_on "all"  --eval_on '15' --save_steps 100 --learning_rate 5e-5 --num_train_epochs 50.0 --train_batch_size 32 --eval_batch_size 64

step-1300_f1-76.41.bin 


```

#prompt+mlm

```
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval  --prompt_length 3 --task_name "sighan"  --train_on "hybrid"  --eval_on '15' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64

step-9000_f1-64.40.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --load_state_dict "model/step-9000_f1-64.40.bin"  --prompt_length 3 --task_name "sighan"  --train_on "all"  --eval_on '15' --save_steps 100 --learning_rate 5e-5 --num_train_epochs 50.0 --train_batch_size 32 --eval_batch_size 64

step-400_f1-74.64.bin



CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval  --prompt_length 10 --task_name "sighan"  --train_on "hybrid"  --eval_on '15' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_sighan"

step-14000_f1-66.50.bin step-10000_f1-66.67.bin step-20000_f1-65.13.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --load_state_dict "model/model_sighan/step-10000_f1-66.67.bin"  --prompt_length 10 --task_name "sighan"  --train_on "all"  --eval_on '15' --save_steps 100 --learning_rate 5e-5 --num_train_epochs 50.0 --train_batch_size 32 --eval_batch_size 64 --output_dir "model/model_sighan"

step-600_f1-76.19.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --load_state_dict "model/model_sighan/step-20000_f1-65.13.bin"  --prompt_length 10 --task_name "sighan"  --train_on "all"  --eval_on '15' --save_steps 100 --learning_rate 5e-5 --num_train_epochs 50.0 --train_batch_size 32 --eval_batch_size 64 --output_dir "model/model_sighan"

step-500_f1-76.72.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10  --task_name "sighan" --load_state_dict "model/model_sighan/step-500_f1-76.72.bin" --test_on "15"

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10  --task_name "sighan" --load_state_dict "model/model_sighan/step-500_f1-76.72.bin" --test_on "15_adv_5_use_ner_0"



CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval  --prompt_length 7 --task_name "sighan"  --train_on "hybrid"  --eval_on '15' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_sighan"

step-10000_f1-63.12.bin
```



#prompt+mlm+mft

```
#########prompt length=10################

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --prompt_length 10 --task_name "sighan"  --train_on "hybrid"  --eval_on '15' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64 --output_dir "model/model_sighan"

step-10000_f1-58.07.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --load_state_dict "model/model_sighan/step-10000_f1-58.07.bin"  --prompt_length 10 --task_name "sighan"  --train_on "all"  --eval_on '15' --save_steps 100 --learning_rate 5e-5 --num_train_epochs 50.0 --train_batch_size 32 --eval_batch_size 64 --output_dir "model/model_sighan"

step-700_f1-73.22.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10  --task_name "sighan" --load_state_dict "model/model_sighan/step-700_f1-73.22.bin" --eval_on "15"

#dynamic mask#
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.3 --prompt_length 10 --task_name "sighan"  --train_on "hybrid"  --eval_on '15' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_sighan"
step-20000_f1-55.05.bin
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_rate 0.3 --load_state_dict "model/model_sighan/step-20000_f1-55.05.bin"  --prompt_length 10 --task_name "sighan"  --train_on "all"  --eval_on '15' --save_steps 100 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 32 --eval_batch_size 64 --output_dir "model/model_sighan"
step-1200_f1-76.08.bin
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_rate 0.3 --load_state_dict "model/model_sighan/step-20000_f1-65.13.bin"  --prompt_length 10 --task_name "sighan"  --train_on "all"  --eval_on '15' --save_steps 100 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 32 --eval_batch_size 64 --fp16 --output_dir "model/model_sighan"
step-1000_f1-76.14.bin

###mask all##########

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "all" --prompt_length 10 --task_name "sighan"  --train_on "hybrid"  --eval_on '15' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64 --output_dir "model/model_sighan"

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "all" --load_state_dict "model/model_sighan/step-11000_f1-55.09.bin"  --prompt_length 10 --task_name "sighan"  --train_on "all"  --eval_on '15' --save_steps 100 --learning_rate 5e-5 --num_train_epochs 50.0 --train_batch_size 32 --eval_batch_size 64 --output_dir "model/model_sighan"

###prompt length=3######################

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft  --prompt_length 3 --task_name "sighan"  --train_on "hybrid"  --eval_on '15' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64

step-11000_f1-56.94.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --load_state_dict "model/step-11000_f1-56.94.bin"  --prompt_length 3 --task_name "sighan"  --train_on "all"  --eval_on '15' --save_steps 100 --learning_rate 5e-5 --num_train_epochs 50.0 --train_batch_size 32 --eval_batch_size 64

step-700_f1-71.44.bin
```



#### sighan15

|      | train4->test1 | train4->test2 | train9->test1 | train9->test2 |
| ---- | ------------- | ------------- | ------------- | ------------- |
| p    | 68.68         | 44.44         | 66.94         | 53.71         |
| r    | 72.69         | 33.85         | 73.88         | 44.78         |
| f1   | 70.63         | 38.43         | 70.24         | 48.85         |

#### sighan14

|      | train4->test3 | train4->test4 | train9->test3 | train9->test4 |
| ---- | ------------- | ------------- | ------------- | ------------- |
| p    | 56.67         | 40.81         | 55.53         | 46.74         |
| r    | 61.87         | 31.55         | 64.07         | 40.72         |
| f1   | 59.16         | 35.59         | 59.49         | 43.52         |

#### sighan13

|      | train4->test5 | train4->test6 | train9->test5 | train9->test6 |
| ---- | ------------- | ------------- | ------------- | ------------- |
| p    | 83.40         | 37.05         | 84.58         | 48.57         |
| r    | 71.53         | 18.55         | 74.72         | 27.96         |
| f1   | 77.01         | 24.72         | 79.34         | 35.49         |

CUDA_VISIBLE_DEVICES=0 python run_err_corr.py --do_train --do_eval --task_name "sighan"  --train_on 'hybrid_random_plome_ner_200000'  --eval_on '15' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 32 --eval_batch_size 64 --max_seq_length 180 --seed 17

step-147000_f1-51.50.bin

python run_err_corr.py --do_train --do_eval --load_checkpoint "model/step-147000_f1-51.50.bin" --task_name "sighan"  --train_on 'all'  --eval_on '15' --save_steps 300 --learning_rate 5e-5 --num_train_epochs 30.0 --train_batch_size 64

step-1200_f1-70.63.bin

python run.py --do_test --load_checkpoint "model/step-1200_f1-70.63.bin" --task_name "sighan"  --test_on "15_adv_5_use_ner_0" 

f1=38.43 p=44.44 r=33.85

python run.py --do_test --load_checkpoint "model/step-2000_f1-72.84.bin" --task_name "sighan"  --test_on "15_adv_5_use_ner_0"



#################sighan14######################

python run.py --do_test --load_checkpoint "model/step-1200_f1-70.63.bin" --task_name "sighan"  --test_on "14" 

python run.py --do_test --load_checkpoint "model/step-1200_f1-70.63.bin" --task_name "sighan"  --test_on "14_adv_5_use_ner_0" 

####################sighan13######################## 

########### we do fine-tuning on sighan13 instead of "all" #########

python run.py --do_train --do_eval --load_checkpoint "model/step-147000_f1-51.50.bin" --task_name "sighan"  --train_on '13'  --eval_on '13' --save_steps 300 --learning_rate 5e-5 --num_train_epochs 30.0 --train_batch_size 64

step-300_f1-77.01.bin

python run.py --do_test --load_checkpoint "model/step-300_f1-77.01.bin" --task_name "sighan"  --test_on "13_adv_5_use_ner_0" 



CUDA_VISIBLE_DEVICES=0 python run_err_corr.py --do_train --do_eval --task_name "sighan"  --train_on **'hybrid_random_adv_plome_ner_long_200000'**  --eval_on '15' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 32 --eval_batch_size 64 --max_seq_length 180 --seed 17

step-206000_f1-49.46.bin

python run_err_corr.py --do_train --do_eval --load_checkpoint "model/step-206000_f1-49.46.bin" --task_name "sighan"  --train_on 'all'  --eval_on '15' --save_steps 300 --learning_rate 5e-5 --num_train_epochs 30.0 --train_batch_size 64

step-2400_f1-71.35.bin



CUDA_VISIBLE_DEVICES=0 python **run.py** --do_train --do_eval --task_name "sighan"  --train_on 'hybrid_adv_mlm_ner_window5_200000'  --eval_on '15' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 32 --eval_batch_size 64 --max_seq_length 180 --seed 17

step-212000_f1-51.89.bin

python run.py --do_train --do_eval --load_checkpoint "model/step-212000_f1-51.89.bin" --task_name "sighan" train_on "all" --eval_on "15_adv_5_use_ner_0" --save_steps 300 --learning_rate 5e-5 --num_train_epochs 30.0 --train_batch_size 64

step-600_f1-48.85.bin

python run.py --do_test --load_checkpoint "model/step-600_f1-48.85.bin" --task_name "sighan"  --test_on "15" 

f1 70.24 p 66.94 r 73.88

python run.py --do_test --load_checkpoint "model/step-600_f1-48.85.bin" --task_name "sighan"  --test_on "14" 

python run.py --do_test --load_checkpoint "model/step-600_f1-48.85.bin" --task_name "sighan"  --test_on "14_adv_5_use_ner_0"

########### we do fine-tuning on sighan13 instead of "all" #########

 python run.py --do_train --do_eval --load_checkpoint "model/step-212000_f1-51.89.bin" --task_name "sighan" --train_on "13" --eval_on "13" --save_steps 300 --learning_rate 5e-5 --num_train_epochs 30.0 --train_batch_size 64

step-300_f1-79.35.bin

python run.py --do_test --load_checkpoint "model/step-300_f1-79.35.bin" --task_name "sighan"  --test_on "13_adv_5_use_ner_0"

#### 

### sghspell

sghspell: train: hupu(50 000)+gamersky(50 000)+sophera(10 000)

​				test: hupu(500)+gamersky(500)+sophera(100)

满足：

1. 长度(10<=l<=64)	

2. 只包含汉字(标点)和阿拉伯数字

#### train

1. train_sgh.txt ## fake data(trf==src)

2. train_syn_random_use_ner.txt 

   ####random position+plome substitution strategy len=110,000

3. train_syn_random_use_ner_long.txt

   ####different from **2**, random position+plome substitution strategy, l>=20

4. train_syn_random_use_ner_ratio0.05.txt

   ####random position+plome substitution strategy,**multiple errors**, len=110,000

5. train_syn_random_use_ner_ratio0.05_long.txt

   ####different from **4**, random position+plome substitution strategy, **multiple errors**, l>=20

6. train_adv_5_use_ner_long.txt 

   ####**max entropy position(use mlm candidates)**(**use ner**)+plome substitution strategy(based on 1) we **extract** the examples of **length >=20**.   num_examples=61,277

7. train_adv_5_use_ner_ratio0.05_long.txt

   ####**max entropy position(use mlm candidates)**(**use ner**)+plome substitution strategy(based on 1) we **extract** the examples of **length >=20**.   num_examples=61,277  **multiple errors**

8. train_sgh_adv_mlm_ner_window5_60000.txt 

   ####**6+2**

9. train_sgh_random_ner_60000.txt ##  **2+3**

10. train_sgh_random_ner_ratio0.05_60000.txt ### **4+5**

11. train_sgh_adv_mlm_ner_window5_ratio0.05_60000.txt ### **4+7**

#### test

1. test_sgh.txt ##fake data(src==trg)
2. test_syn_random_use_ner.txt ##random position+plome substitution strategy
3. test_syn_random_use_ner_ratio0.05.txt ##random position+plome substitution strategy, 5%of the examples have two errors
4. test_adv_5_use_ner_ratio0.0_0.txt ##**max entropy position(use mlm candidates)**(**use ner**)+plome substitution strategy
5. test_adv_5_use_ner_ratio0.05_0.txt ##**max entropy position(use mlm candidates)**(**use ner**)+plome substitution strategy, 5%of the examples have two errors

#### experiment

###BERT no mft

CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --task_name "sghspell"  --train_on "sgh_random_ner_ratio0.05_60000"  --eval_on 'syn_random_use_ner_ratio0.05' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64

step-6000_f1-61.50.bin

CUDA_VISIBLE_DEVICES=0 python run.py --do_test  --load_checkpoint "model/model_sghspell/step-6000_f1-61.50.bin" --task_name "sghspell"   --test_on 'adv_5_use_ner_ratio0.05_0'  --eval_batch_size 32

CUDA_VISIBLE_DEVICES=0 python run.py --do_test  --load_checkpoint "model/model_sghspell/step-6000_f1-61.50.bin" --task_name "sghspell"   --test_on 'syn_random_use_ner_ratio0.05'  --eval_batch_size 32

CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --mask_mode "noerror" --task_name "sghspell"   --train_on "syn_random_use_ner_ratio0.05"  --eval_on 'syn_random_use_ner_ratio0.05' --save_steps 1000 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_sghspell"

step-8000_f1-53.68.bin

CUDA_VISIBLE_DEVICES=0 python run.py --do_test  --load_checkpoint "model/model_sghspell/step-8000_f1-53.68.bin" --task_name "sghspell"   --test_on 'syn_random_use_ner_ratio0.05'  --eval_batch_size 32

CUDA_VISIBLE_DEVICES=0 python run.py --do_test  --load_checkpoint "model/model_sghspell/step-8000_f1-53.68.bin" --task_name "sghspell"   --test_on 'adv_5_use_ner_ratio0.0_0'  --eval_batch_size 32

CUDA_VISIBLE_DEVICES=0 python run.py --do_test  --load_checkpoint "model/model_sghspell/step-8000_f1-53.68.bin" --task_name "sghspell"   --test_on 'adv_2_use_ner_ratio0.0_0'  --eval_batch_size 32

CUDA_VISIBLE_DEVICES=0 python run.py --do_test  --load_checkpoint "model/model_sghspell/step-8000_f1-53.68.bin" --task_name "sghspell"   --test_on 'adv_nomlm_use_ner_ratio0.0'  --eval_batch_size 32

##adv

CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --task_name "sghspell"  --train_on "sgh_adv_mlm_ner_window5_ratio0.05_60000"  --eval_on 'syn_random_use_ner_ratio0.05' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64

step-13000_f1-62.06.bin

CUDA_VISIBLE_DEVICES=0 python run.py --do_test  --load_checkpoint "model/step-13000_f1-62.06.bin" --task_name "sghspell"   --test_on "syn_random_use_ner_ratio0.05"  --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run.py --do_test  --load_checkpoint "model/model_step-13000_f1-62.06.bin" --task_name "sghspell"   --test_on 'adv_5_use_ner_ratio0.05_0'  --eval_batch_size 64



###BERT mft

train_on sgh_random_ner_ratio0.05_60000

```
CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --mft --task_name "sghspell"  --train_on "sgh_random_ner_ratio0.05_60000"  --eval_on 'syn_random_use_ner_ratio0.05' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64

step-12000_f1-66.90.bin

CUDA_VISIBLE_DEVICES=0 python run.py --do_test  --load_checkpoint "model/step-12000_f1-66.90.bin" --task_name "sghspell"   --test_on 'adv_5_use_ner_ratio0.05_0'  --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --mft --mask_mode "noerror" --task_name "sghspell"  --train_on "sgh_random_ner_ratio0.05_60000"  --eval_on 'syn_random_use_ner_ratio0.05' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_sghspell"

step-13000_f1-74.19.bin

CUDA_VISIBLE_DEVICES=0 python run.py --do_test  --load_checkpoint "model/model_sghspell/step-13000_f1-74.19.bin" --task_name "sghspell"   --test_on 'adv_5_use_ner_ratio0.05_0'  --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --mft --mask_mode "noerror" --task_name "sghspell"   --train_on "sgh_random_ner_ratio0.05_60000"  --eval_on 'syn_random_use_ner_ratio0.05' --save_steps 1000 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_sghspell"

step-12000_f1-68.17.bin

CUDA_VISIBLE_DEVICES=0 python run.py --do_test  --load_checkpoint "model/model_sghspell/step-12000_f1-68.17.bin" --task_name "sghspell"   --test_on 'adv_5_use_ner_ratio0.05_0'  --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run.py --do_test  --load_checkpoint "model/model_sghspell/step-12000_f1-68.17.bin" --task_name "sghspell"   --test_on 'syn_random_use_ner_ratio0.05'  --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run.py --do_test  --load_checkpoint "model/model_sghspell/step-12000_f1-68.17.bin" --task_name "sghspell"   --test_on 'sgh'  --eval_batch_size 64
```



train_on syn_random_use_ner_ratio0.05

```
CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --mft --mask_mode "noerror" --task_name "sghspell"   --train_on "syn_random_use_ner_ratio0.05"  --eval_on 'syn_random_use_ner_ratio0.05' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_sghspell"

step-8000_f1-60.36.bin

CUDA_VISIBLE_DEVICES=0 python run.py --do_test  --load_checkpoint "model/model_sghspell/step-8000_f1-60.36.bin" --task_name "sghspell"   --test_on 'syn_random_use_ner_ratio0.05'  --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run.py --do_test  --load_checkpoint "model/model_sghspell/step-8000_f1-60.36.bin" --task_name "sghspell"   --test_on 'adv_5_use_ner_ratio0.0_0'  --eval_batch_size 32

CUDA_VISIBLE_DEVICES=0 python run.py --do_test  --load_checkpoint "model/model_sghspell/step-8000_f1-60.36.bin" --task_name "sghspell"   --test_on 'adv_2_use_ner_ratio0.0_0'  --eval_batch_size 32

CUDA_VISIBLE_DEVICES=0 python run.py --do_test  --load_checkpoint "model/model_sghspell/step-8000_f1-60.36.bin" --task_name "sghspell"   --test_on 'adv_nomlm_use_ner_ratio0.0'  --eval_batch_size 32
```



#############################prompt#########################

###Prompt not use mft

train_on sgh_random_ner_ratio0.05_60000

```
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --prompt_length 3 --task_name "sghspell"  --train_on "sgh_random_ner_ratio0.05_60000"  --eval_on 'syn_random_use_ner_ratio0.05' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64

step-13000_f1-64.34.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 3 --task_name "sghspell"  --test_on 'syn_random_use_ner_ratio0.05' --load_state_dict "model/step-13000_f1-64.34.bin" --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 3 --task_name "sghspell"  --test_on 'adv_5_use_ner_ratio0.05_0' --load_state_dict "model/step-13000_f1-64.34.bin" --eval_batch_size 64



CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --prompt_length 10 --task_name "sghspell"  --train_on "sgh_random_ner_ratio0.05_60000"  --eval_on 'syn_random_use_ner_ratio0.05' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10 --task_name "sghspell"  --test_on 'adv_5_use_ner_ratio0.05_0' --load_state_dict "model/model_sghspell/step-13000_f1-64.19.bin" --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --prompt_length 10 --task_name "sghspell"  --train_on "sgh_random_ner_ratio0.05_60000"  --eval_on 'syn_random_use_ner_ratio0.05' --save_steps 1000 --learning_rate 2e-5 --num_train_epochs 20.0 --train_batch_size 64 --eval_batch_size 64

step-18000_f1-62.30.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --prompt_length 10 --task_name "sghspell"  --train_on "sgh_random_ner_ratio0.05_60000"  --eval_on 'syn_random_use_ner_ratio0.05' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 32 --eval_batch_size 64 --fp16 --output_dir "model/model_sghspell"

step-26000_f1-64.58.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10 --task_name "sghspell"  --test_on 'adv_5_use_ner_ratio0.05_0' --load_state_dict "model/model_sghspell/step-26000_f1-64.58.bin" --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10 --task_name "sghspell"  --test_on 'syn_random_use_ner_ratio0.05' --load_state_dict "model/model_sghspell/step-26000_f1-64.58.bin" --eval_batch_size 64
```



train_on syn_random_use_ner_ratio0.05

```
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --prompt_length 10 --mask_mode "noerror" --task_name "sghspell"  --train_on "syn_random_use_ner_ratio0.05"  --eval_on 'syn_random_use_ner_ratio0.05' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_sghspell"

step-5000_f1-62.63.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10 --task_name "sghspell"  --test_on 'adv_5_use_ner_ratio0.0_0' --load_state_dict "model/model_sghspell/step-5000_f1-62.63.bin" --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10 --task_name "sghspell"  --test_on 'adv_2_use_ner_ratio0.0_0' --load_state_dict "model/model_sghspell/step-5000_f1-62.63.bin" --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10 --task_name "sghspell"  --test_on 'syn_random_use_ner_ratio0.05' --load_state_dict "model/model_sghspell/step-5000_f1-62.63.bin" --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10 --task_name "sghspell"  --test_on 'adv_nomlm_use_ner_ratio0.0' --load_state_dict "model/model_sghspell/step-5000_f1-62.63.bin" --eval_batch_size 64
```



###Prompt use_mft

--tain_on syn_random_use_ner_ratio0.05

```
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --prompt_length 10 --mask_mode "noerror" --task_name "sghspell"  --train_on "syn_random_use_ner_ratio0.05"  --eval_on 'syn_random_use_ner_ratio0.05' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_sghspell"

step-6000_f1-64.80.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10 --task_name "sghspell"  --test_on 'syn_random_use_ner_ratio0.05' --load_state_dict "model/model_sghspell/step-6000_f1-64.80.bin" --eval_batch_size 64
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10 --task_name "sghspell"  --test_on 'adv_5_use_ner_ratio0.0_0' --load_state_dict "model/model_sghspell/step-6000_f1-64.80.bin" --eval_batch_size 64
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10 --task_name "sghspell"  --test_on 'adv_nomlm_use_ner_ratio0.0' --load_state_dict "model/model_sghspell/step-6000_f1-64.80.bin" --eval_batch_size 64


CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_rate 0.4 --prompt_length 10 --mask_mode "noerror" --task_name "sghspell"  --train_on "syn_random_use_ner_ratio0.05"  --eval_on 'syn_random_use_ner_ratio0.05' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_sghspell"
step-7000_f1-67.09.bin
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10 --task_name "sghspell"  --test_on 'adv_5_use_ner_ratio0.0_0' --load_state_dict "model/model_sghspell/step-7000_f1-67.09.bin" --eval_batch_size 64
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10 --task_name "sghspell"  --test_on 'adv_nomlm_use_ner_ratio0.0' --load_state_dict "model/model_sghspell/step-7000_f1-67.09.bin" --eval_batch_size 64

#######dynamic mask############
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_rate 0.3 --prompt_length 5 --mask_mode "noerror" --task_name "sghspell"  --train_on "syn_random_use_ner_ratio0.05"  --eval_on 'syn_random_use_ner_ratio0.05' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_sghspell"

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_rate 0.3 --prompt_length 10 --mask_mode "noerror" --task_name "sghspell"  --train_on "syn_random_use_ner_ratio0.05"  --eval_on 'syn_random_use_ner_ratio0.05' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_sghspell"
step-8000_f1-69.48.bin
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10 --task_name "sghspell"  --test_on 'syn_random_use_ner_ratio0.05' --load_state_dict "model/model_sghspell/step-8000_f1-69.48.bin" --eval_batch_size 64
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10 --task_name "sghspell"  --test_on 'adv_5_use_ner_ratio0.0_0' --load_state_dict "model/model_sghspell/step-8000_f1-69.48.bin" --eval_batch_size 64
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10 --task_name "sghspell"  --test_on 'adv_nomlm_use_ner_ratio0.0' --load_state_dict "model/model_sghspell/step-8000_f1-69.48.bin" --eval_batch_size 64
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10 --task_name "sghspell"  --test_on 'adv_2_use_ner_ratio0.0_0' --load_state_dict "model/model_sghspell/step-8000_f1-69.48.bin" --eval_batch_size 64
```



--tain_on sgh_random_ner_ratio0.05_60000

```
CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --prompt_length 3 --task_name "sghspell"  --train_on "sgh_random_ner_ratio0.05_60000"  --eval_on 'syn_random_use_ner_ratio0.05' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64

step-11000_f1-67.53.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 3 --task_name "sghspell"  --test_on 'syn_random_use_ner_ratio0.05' --load_state_dict "model/step-11000_f1-67.53.bin" --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 3 --task_name "sghspell"  --test_on 'adv_5_use_ner_ratio0.05_0' --load_state_dict "model/step-11000_f1-67.53.bin" --eval_batch_size 64

##prompt=10,mask_mode=noerror

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --prompt_length 10 --mask_mode "noerror" --task_name "sghspell"  --train_on "sgh_random_ner_ratio0.05_60000"  --eval_on 'syn_random_use_ner_ratio0.05' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 32 --eval_batch_size 64 --fp16 --output_dir "model/model/sghspell"

69.69

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --prompt_length 10 --mask_mode "noerror" --task_name "sghspell"  --train_on "sgh_random_ner_ratio0.05_60000"  --eval_on 'syn_random_use_ner_ratio0.05' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 20.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_sghspell"

step-16000_f1-69.93.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10 --task_name "sghspell"  --test_on 'adv_5_use_ner_ratio0.05_0' --load_state_dict "model/model_sghspell/step-16000_f1-69.93.bin" --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10 --task_name "sghspell"  --test_on 'syn_random_use_ner_ratio0.05' --load_state_dict "model/model_sghspell/step-16000_f1-69.93.bin" --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 10 --task_name "sghspell"  --test_on 'sgh' --load_state_dict "model/model_sghspell/step-16000_f1-69.93.bin" --eval_batch_size 64
```



##adv

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --prompt_length 3 --task_name "sghspell"  --train_on "sgh_adv_mlm_ner_window5_ratio0.05_60000"  --eval_on 'syn_random_use_ner_ratio0.05' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64

step-13000_f1-67.49.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 3 --task_name "sghspell"  --test_on 'syn_random_use_ner_ratio0.05' --load_state_dict "model/step-13000_f1-67.49.bin" --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --prompt_length 3 --task_name "sghspell"  --test_on 'syn_random_use_ner_ratio0.05' --load_state_dict "model/step-13000_f1-67.49.bin" --eval_batch_size 64

#### sghspell

##### single error

|      | train5->test2 | train5->test4 | train6->test2 | train6->test4 |
| ---- | ------------- | ------------- | ------------- | ------------- |
| p    | 63.58         | 53.72         | 64.82         | 60.19         |
| r    | 58.84         | 40.60         | 59.34         | 48.64         |
| f1   | 61.13         | 46.25         | 61.96         | 53.08         |

##### train:random

test:

1. normal errors, multiple

	 2. semantic-mlm: test_adv_5_use_ner_ratio0.0_0.txt test_adv_2_use_ner_ratio0.0_0.txt
	
	 				  single, top-50 entropy, top-5 mlm candidates have priority.
	 2. semantic: test_adv_nomlm_use_ner_ratio0.0.txt 
	single, top-50 entropy, do not use mlm candidates to do substitution.

|                     | BERT  | BERT-mft | LMCorrector | LMCorrector-mft-0.2(static) | LMCorrector-mft-0.4(static) | LMCorrector-mft-0.3(dynamic) |
| ------------------- | ----- | -------- | ----------- | --------------------------- | --------------------------- | ---------------------------- |
| f1                  | 53.68 | 60.36    | 62.63       | 64.80                       | 67.09                       | 69.48                        |
| f1-semantic-mlm(w5) | 36.84 | 39.86    | 35.28       | 40.02                       | 42.16                       | 47.05                        |
| f1-semantic(w5)     | 39.71 | 42.91    | 38.81       | 45.61                       | 46.49                       | 50.80                        |
| f1-semantic-mlm(w2) | 45.94 | 53.16    | 49.58       | /                           | /                           | 55.65                        |





### ecspell

|                     | med-f1 | law-f1 | odw-f1 |
| ------------------- | ------ | ------ | ------ |
| BERT+mft            | 58.0   | 76.1   | 59.2   |
| prompt+mft(dynamic) | 85.40  | 92.17  | 86.71  |
| prompt              | 56.9   | 57.3   | 59.0   |

|            | non  | multi | inc  | exc   |
| ---------- | ---- | ----- | ---- | ----- |
| BERT+mft   | 14.7 | 58.7  | 97.5 | 87.02 |
| prompt+mft | 5.31 | 57.5  | 93.5 | 83.0  |



#### med

```
#BERT-finetune mft

step-4600_f1-47.20.bin

CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --mft --task_name "ecspell" --mask_mode "noerror"  --train_on "med"  --eval_on 'med' --save_steps 100 --learning_rate 5e-5 --max_train_steps 5000 --num_train_epochs 3.0 --train_batch_size 128 --eval_batch_size 64 --output_dir "model/model_med" --fp16



CUDA_VISIBLE_DEVICES=0 python run_ginga.py --do_train  --do_eval --mft --train_on "ecspell/train_med.txt" --eval_on "ecspell/test_med.txt" --data_dir "data/" --save_steps 100 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --output_dir "model/model_med" --fp16
step-4900_f1-58.00.bin

CUDA_VISIBLE_DEVICES=0 python run_ginga.py  --do_test --test_on "ecspell/test_med_adv_5_use_ner_ratio0.05_0.txt" --data_dir "data/" --load_state_dict "model/model_med/step-4900_f1-58.00.bin" 
CUDA_VISIBLE_DEVICES=0 python run_ginga.py  --do_test --test_on "ecspell/test_med.txt" --data_dir "data/" --load_state_dict "model/model_med/step-4900_f1-58.00.bin" 


#prompt mft

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --prompt_length 3 --task_name "ecspell"  --train_on "med"  --eval_on 'med' --save_steps 100 --output_dir "model/model_med" --learning_rate 5e-5 --num_train_epochs 80.0 --train_batch_size 32 --eval_batch_size 64

step-7400_f1-74.62.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --prompt_length 10 --task_name "ecspell"  --train_on "med"  --eval_on 'med' --save_steps 100 --output_dir "model/model_med" --learning_rate 5e-5 --train_batch_size 128 --eval_batch_size 64 --max_train_steps 5000 --fp16

step-3900_f1-74.00.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --load_state_dict "model/model_med/step-7400_f1-74.62.bin"  --prompt_length 3 --task_name "ecspell"  --test_on "med" 

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --load_state_dict "model/model_med/step-7400_f1-74.62.bin"  --prompt_length 3 --task_name "ecspell"  --test_on "med_adv_5_use_ner_ratio0.05_0" 

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --load_state_dict "../promptCSC/model/model_med/step-3000_f1-60.03.bin"  --prompt_length 3 --task_name "ecspell"  --test_on "med" 



CUDA_VISIBLE_DEVICES=0 python run.py --do_test --load_checkpoint "../promptCSC/model/model_med/step-4600_f1-22.34.bin" --task_name "ecspell"  --test_on "law" 
```



#### law

```
#BERT fine-tune

step-4100_f1-37.87.bin

CUDA_VISIBLE_DEVICES=0 python run.py  --do_test --task_name "ecspell" --load_checkpoint "model/model_law/step-4100_f1-37.87.bin" --mask_mode "noerror"  --test_on "law"

CUDA_VISIBLE_DEVICES=0 python run.py  --do_test --task_name "ecspell" --load_checkpoint "model/model_law/step-4100_f1-37.87.bin" --mask_mode "noerror"  --test_on "law_adv_5_use_ner_ratio0.05_0"

#BERT-finetune mft

CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --mft --task_name "ecspell" --mask_mode "noerror"  --train_on "law"  --eval_on 'law' --save_steps 100 --learning_rate 5e-5 --max_train_steps 5000 --num_train_epochs 3.0 --train_batch_size 128 --eval_batch_size 64 --output_dir "model/model_law" --fp16

step-4800_f1-72.23.bin

../csc/eclaw/step-5000_f1-76.08.bin

CUDA_VISIBLE_DEVICES=0 python run.py  --do_test --task_name "ecspell" --load_checkpoint "model/model_law/step-4800_f1-72.23.bin" --mask_mode "noerror"  --test_on "law_adv_5_use_ner_ratio0.05_0"

#prompt mft

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --prompt_length 10 --task_name "ecspell"  --train_on "law"  --eval_on 'law' --save_steps 100 --output_dir "model/model_law" --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 32 --eval_batch_size 64 --output_dir "model/model_law" --fp16

f1-76

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --prompt_length 10 --task_name "ecspell"  --train_on "law"  --eval_on 'law' --save_steps 100 --output_dir "model/model_law" --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --output_dir "model/model_law" --fp16

step-3100_f1-78.48.bin

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --prompt_length 10 --task_name "ecspell"  --train_on "law"  --eval_on 'law' --save_steps 100 --output_dir "model/model_law" --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64  --output_dir "model/model_law" --fp16

#prompt  no mft

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval  --prompt_length 10 --task_name "ecspell"  --train_on "law"  --eval_on 'law' --save_steps 100 --output_dir "model/model_law" --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --output_dir "model/model_law" --fp16

step-3900_f1-57.25.bin

#EXC INC

CUDA_VISIBLE_DEVICES=0 python run_ginga.py  --do_test --test_on "ecspell/test_law_inc.txt" --data_dir "data/" --load_state_dict "../csc/eclaw/step-5000_f1-76.08.bin" 

CUDA_VISIBLE_DEVICES=0 python run_ginga.py  --do_test --test_on "ecspell/test_law_non.txt" --data_dir "data/" --load_state_dict "../csc/eclaw/step-5000_f1-76.08.bin" 

CUDA_VISIBLE_DEVICES=0 python run_ginga.py  --do_test --test_on "ecspell/test_law_exc.txt" --data_dir "data/" --load_state_dict "../csc/eclaw/step-5000_f1-76.08.bin" 

CUDA_VISIBLE_DEVICES=0 python run_ginga.py  --do_test --test_on "ecspell/test_law_multi.txt" --data_dir "data/" --load_state_dict "../csc/eclaw/step-5000_f1-76.08.bin" 

CUDA_VISIBLE_DEVICES=0 python run_ginga.py  --do_test --test_on "ecspell/test_law.txt" --data_dir "data/" --load_state_dict "../csc/eclaw/step-5000_f1-76.08.bin" 



CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --load_state_dict "model/model_law/step-3100_f1-78.48.bin"  --prompt_length 10 --task_name "ecspell"  --test_on "law_exc" 

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --load_state_dict "model/model_law/step-3100_f1-78.48.bin"  --prompt_length 10 --task_name "ecspell"  --test_on "law_non" 

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --load_state_dict "model/model_law/step-3100_f1-78.48.bin"  --prompt_length 10 --task_name "ecspell"  --test_on "law_multi" 

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --load_state_dict "model/model_law/step-3100_f1-78.48.bin"  --prompt_length 10 --task_name "ecspell"  --test_on "law_inc" 

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --load_state_dict "model/model_law/step-3100_f1-78.48.bin"  --prompt_length 10 --task_name "ecspell"  --test_on "law" 

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --load_state_dict "model/model_law/step-3900_f1-57.25.bin"  --prompt_length 10 --task_name "ecspell"  --test_on "law" 



CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --load_state_dict "model/model_law/step-3100_f1-78.48.bin"  --prompt_length 10 --task_name "ecspell"  --test_on "law_adv_5_use_ner_ratio0.05_0" 

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --load_state_dict "model/model_law/step-3900_f1-57.25.bin"  --prompt_length 10 --task_name "ecspell"  --test_on "law_adv_5_use_ner_ratio0.05_0" 
```



#### odw

```
test_odw_adv_5_use_ner_ratio0.05_0.txt

#bert mft

CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval --mft --task_name "ecspell" --mask_mode "noerror"  --train_on "odw"  --eval_on 'odw' --save_steps 100 --learning_rate 5e-5 --max_train_steps 5000 --num_train_epochs 3.0 --train_batch_size 128 --eval_batch_size 64 --output_dir "model/model_law" --fp16

step-2800_f1-59.23.bin

CUDA_VISIBLE_DEVICES=0 python run.py  --do_test --task_name "ecspell" --load_checkpoint "model/model_odw/step-2800_f1-59.23.bin"  --test_on 'odw' 

#bert



step-4600_f1-24.61.bin

CUDA_VISIBLE_DEVICES=0 python run.py --do_train --do_eval  --task_name "ecspell"  --train_on "odw"  --eval_on 'odw' --save_steps 100 --learning_rate 5e-5 --max_train_steps 5000 --num_train_epochs 3.0 --train_batch_size 32 --eval_batch_size 64 --output_dir "model/model_odw" --fp16
step-4100_f1-25.00.bin

CUDA_VISIBLE_DEVICES=0 python run.py  --do_test --task_name "ecspell" --load_checkpoint "model/model_odw/step-4100_f1-25.00.bin" --mask_mode "noerror"  --test_on 'odw' 



CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --prompt_length 10 --task_name "ecspell"  --train_on "odw"  --eval_on 'odw' --save_steps 100 --output_dir "model/model_odw" --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 128 --eval_batch_size 64 --fp16

f1-72.5

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --load_state_dict "model/model_odw/step-5000_f1-72.52.bin"  --prompt_length 10 --task_name "ecspell"  --test_on "odw" 

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_test --load_state_dict "model/model_odw/step-5000_f1-72.52.bin"  --prompt_length 10 --task_name "ecspell"  --test_on "odw_adv_5_use_ner_ratio0.05_0" 

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --prompt_length 10 --task_name "ecspell"  --train_on "odw"  --eval_on 'odw' --save_steps 100 --learning_rate 5e-5 --max_train_steps 8000 --train_batch_size 64 --eval_batch_size 64 --max_seq_length 256 --fp16 --output_dir "model/model_odw"

CUDA_VISIBLE_DEVICES=0 python run_continuous.py --do_train --do_eval --mft --mask_mode "noerror" --prompt_length 5 --task_name "ecspell"  --train_on "odw"  --eval_on 'odw' --save_steps 100 --learning_rate 5e-5 --max_train_steps 8000 --train_batch_size 32 --eval_batch_size 64 --max_seq_length 256 --fp16 --output_dir "model/model_odw"
```



### multi-task

#### afqmc and tnews

step-800_f1-73.96.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/step-800_f1-73.96.bin" --do_test --task_name "afqmc" --test_on 'base'



#### BERT-multi

####BERT no mft#############################################

sighan

```
CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --do_eval --task_name "sighan tnews afqmc"  --train_on "hybrid base base"  --eval_on '15' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64
step-28000_f1-32.85.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --do_eval --load_checkpoint "model/step-28000_f1-32.85.bin"  --task_name "sighan tnews afqmc"  --train_on "all base base"  --eval_on '15' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64
step-6000_f1-56.56.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_test --load_checkpoint "model/step-6000_f1-56.56.bin"  --task_name "sighan" --eval_on '15' --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_test --load_checkpoint "model/step-6000_f1-56.56.bin"  --task_name "tnews" --eval_on 'base' --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_test --load_checkpoint "model/step-6000_f1-56.56.bin"  --task_name "afqmc" --eval_on 'base' --eval_batch_size 64
```

law

```
CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --do_eval --task_name "ecspell tnews afqmc"  --train_on "law base base"  --eval_on 'law' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"

step-6000_f1-26.11.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --load_checkpoint "model/model_multi/step-6000_f1-26.11.bin" --do_eval --task_name "ecspell tnews afqmc"  --train_on "law base base"  --eval_on 'law' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"

step-6000_f1-34.54.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-6000_f1-34.54.bin" --do_test --task_name "tnews" --test_on 'base'

f1-56.2

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-6000_f1-34.54.bin" --do_test --task_name "afqmc" --test_on 'base'

f1-72.6
```

med

```
CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --do_eval --task_name "ecspell tnews afqmc"  --train_on "med base base"  --eval_on 'med' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 20.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"

step-13000_f1-15.08.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-13000_f1-15.08.bin" --do_test --task_name "tnews" --test_on 'base'

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-13000_f1-15.08.bin" --do_test --task_name "afqmc" --test_on 'base'


```

odw

```
CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --do_eval --task_name "ecspell tnews afqmc"  --train_on "odw base base"  --eval_on 'odw' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 20.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"

step-13000_f1-16.81.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-13000_f1-16.81.bin" --do_test --task_name "ecspell" --test_on 'odw'

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-13000_f1-16.81.bin" --do_test --task_name "tnews" --test_on 'base'

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-13000_f1-16.81.bin" --do_test --task_name "afqmc" --test_on 'base'
```



#### BERT mft

sighan

```
CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --do_eval --mft --task_name "sighan tnews afqmc"  --train_on "hybrid base base"  --eval_on '15' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 20.0 --train_batch_size 128 --eval_batch_size 64 --output_dir "model/model_multi" 

step-33000_f1-38.83.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --do_eval --mft --load_checkpoint "model/model_multi/step-33000_f1-38.83.bin"  --task_name "sighan"  --train_on "all"  --eval_on '15' --save_steps 100 --learning_rate 5e-5 --num_train_epochs 30.0 --train_batch_size 32 --eval_batch_size 64 --output_dir "model/model_multi"

step-4300_f1-64.98.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-4300_f1-64.98.bin" --do_test --task_name "sighan" --test_on '15'

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-4300_f1-64.98.bin" --do_test --task_name "tnews" --test_on 'base'
f1=54.82

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-4300_f1-64.98.bin" --do_test --task_name "afqmc" --test_on 'base'
f1=70.63

######dynamic mask###########
CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --do_eval --mft --mask_mode "noerror" --task_name "sighan tnews afqmc"  --train_on "hybrid base base"  --eval_on '15' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
step-34000_f1-37.12.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --do_eval --mft --mask_rate 0.3 --load_checkpoint "model/model_multi/step-34000_f1-37.12.bin"  --task_name "sighan"  --train_on "all"  --eval_on '15' --save_steps 100 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 32 --eval_batch_size 64 --output_dir "model/model_multi" --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
step-1700_f1-68.52.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --do_eval --mft --mask_rate 0.3 --load_checkpoint "model/model_multi/step-34000_f1-37.12.bin"  --task_name "sighan tnews afqmc"  --train_on "all base base"  --eval_on '15' --save_steps 100 --learning_rate 5e-5 --max_train_steps 10000 --train_batch_size 32 --eval_batch_size 64 --output_dir "model/model_multi" --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
step-3100_f1-50.83.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-3100_f1-50.83.bin" --do_test --task_name "sighan" --test_on '15' --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-3100_f1-50.83.bin" --do_test --task_name "tnews" --test_on 'base' --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-3100_f1-50.83.bin" --do_test --task_name "afqmc" --test_on 'base' --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
```



law

```
CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --mft --do_eval --task_name "ecspell tnews afqmc"  --train_on "law base base"  --eval_on 'law' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"

step-7000_f1-45.25.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --mft --load_checkpoint "model/model_multi/step-7000_f1-45.25.bin" --do_eval --task_name "ecspell tnews afqmc"  --train_on "law base base"  --eval_on 'law' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"

step-7000_f1-54.65.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --mft --load_checkpoint "model/model_multi/step-7000_f1-54.65.bin" --do_eval --task_name "ecspell tnews afqmc"  --train_on "law base base"  --eval_on 'law' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-7000_f1-54.65.bin" --do_test --task_name "tnews" --test_on 'base'

f1-56.5

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-7000_f1-54.65.bin" --do_test --task_name "afqmc" --test_on 'base'

f1-72.3

####### dynamic mask##########
CUDA_VISIBLE_DEVICES=1 python run_multi.py --do_train --mft --do_eval --task_name "ecspell tnews afqmc"  --train_on "law base base"  --eval_on 'law' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 20.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
step-14000_f1-61.99.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-14000_f1-61.99.bin" --do_test --task_name "ecspell" --test_on 'law' --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-14000_f1-61.99.bin" --do_test --task_name "tnews" --test_on 'base' --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-14000_f1-61.99.bin" --do_test --task_name "afqmc" --test_on 'base' --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
```



med

```
CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --do_eval --mft --task_name "ecspell tnews afqmc"  --train_on "med base base"  --eval_on 'med' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 20.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"

step-14000_f1-37.44.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-14000_f1-37.44.bin" --do_test --task_name "tnews" --test_on 'base'

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-14000_f1-37.44.bin" --do_test --task_name "afqmc" --test_on 'base'

####### dynamic mask##########
CUDA_VISIBLE_DEVICES=1 python run_multi.py --do_train --mft --do_eval --task_name "ecspell tnews afqmc"  --train_on "med base base"  --eval_on 'med' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 20.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
step-13000_f1-48.82.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-13000_f1-48.82.bin" --do_test --task_name "ecspell" --test_on 'med' --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-13000_f1-48.82.bin" --do_test --task_name "tnews" --test_on 'base' --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-13000_f1-48.82.bin" --do_test --task_name "afqmc" --test_on 'base' --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
```



odw

```
CUDA_VISIBLE_DEVICES=0 python run_multi.py --do_train --do_eval --mft --task_name "ecspell tnews afqmc"  --train_on "odw base base"  --eval_on 'odw' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 20.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"

step-13000_f1-40.08.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-13000_f1-40.08.bin" --do_test --task_name "ecspell" --test_on 'odw'

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-13000_f1-40.08.bin" --do_test --task_name "tnews" --test_on 'base'

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-13000_f1-40.08.bin" --do_test --task_name "afqmc" --test_on 'base'

####### dynamic mask##########
CUDA_VISIBLE_DEVICES=1 python run_multi.py --do_train --mft --do_eval --task_name "ecspell tnews afqmc"  --train_on "odw base base"  --eval_on 'odw' --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 20.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
step-12000_f1-52.35.bin

CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-12000_f1-52.35.bin" --do_test --task_name "ecspell" --test_on 'odw' --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-12000_f1-52.35.bin" --do_test --task_name "tnews" --test_on 'base' --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
CUDA_VISIBLE_DEVICES=0 python run_multi.py --load_checkpoint "model/model_multi/step-12000_f1-52.35.bin" --do_test --task_name "afqmc" --test_on 'base' --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
```



#### Prompt-mlm-multi

sighan

```
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --task_name "sighan tnews afqmc"  --train_on "hybrid base base"  --eval_on '15' --csc_prompt_length 3 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 64 --eval_batch_size 64

step-31000_f1-56.95.bin

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --load_state_dict "model/step-31000_f1-56.95.bin" --task_name "sighan"  --train_on "all"  --eval_on '15' --csc_prompt_length 3 --sent_prompt_length 3 --save_steps 100 --learning_rate 5e-5 --num_train_epochs 10.0 --train_batch_size 64 --eval_batch_size 64

step-900_f1-66.73.bin

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/step-900_f1-66.73.bin" --task_name "tnews"  --test_on "base" --sent_prompt_length 3 --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/step-900_f1-66.73.bin" --task_name "afqmc"  --test_on "base" --sent_prompt_length 3 --eval_batch_size 64



CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --task_name "sighan tnews afqmc"  --train_on "hybrid base base"  --eval_on '15' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"
step-41000_f1-59.30.bin

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --load_state_dict "model/model_multi/step-41000_f1-59.30.bin" --task_name "sighan"  --train_on "all"  --eval_on '15' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 100 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 64 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"
step-100_f1-67.66.bin

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --load_state_dict "model/model_multi/step-41000_f1-59.30.bin" --task_name "sighan"  --train_on "all"  --eval_on '15' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 100 --learning_rate 2e-5 --max_train_steps 5000 --train_batch_size 32 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"
step-400_f1-69.70.bin

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --load_state_dict "model/model_multi/step-41000_f1-59.30.bin" --task_name "sighan tnews afqmc"  --train_on "all base base"  --eval_on '15' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 100 --learning_rate 2e-5 --max_train_steps 5000 --train_batch_size 32 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"
step-4500_f1-69.00.bin

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-4500_f1-69.00.bin" --task_name "sighan"  --test_on "15" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-4500_f1-69.00.bin" --task_name "tnews"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-4500_f1-69.00.bin" --task_name "afqmc"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64
```

law

```
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --task_name "ecspell tnews afqmc"  --train_on "law base base"  --eval_on 'law' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"

step-7000_f1-54.09.bin

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-7000_f1-54.09.bin" --task_name "ecspell"  --test_on "law" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-7000_f1-54.09.bin" --task_name "tnews"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-7000_f1-54.09.bin" --task_name "afqmc"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64

```



med

```
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --task_name "ecspell tnews afqmc"  --train_on "med base base"  --eval_on 'med' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"

step-8000_f1-59.00.bin

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-8000_f1-59.00.bin" --task_name "tnews"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-8000_f1-59.00.bin" --task_name "afqmc"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64
```



odw

```
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --task_name "ecspell tnews afqmc"  --train_on "odw base base"  --eval_on 'odw' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"

step-9000_f1-52.29.bin

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-9000_f1-52.29.bin" --task_name "ecspell"  --test_on "odw" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-9000_f1-52.29.bin" --task_name "tnews"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-9000_f1-52.29.bin" --task_name "afqmc"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64
```



#### Prompt-mft-multi

##no prompt multi-task relm

|       | law       | med       | odw       |
| ----- | --------- | --------- | --------- |
| csc   | **84.17** | 76.06     | 74.96     |
| tnews | **56.92** | **57.07** | **56.90** |
| afqmc | 71.66     | 70.69     | 71.84     |



```
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --load_state_dict "model/model_multi/step-10000_f1-42.03.bin"  --task_name "sighan tnews afqmc" --mft --mask_mode "noerror" --train_on "hybrid base base"  --eval_on '15' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 7.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"
step-19000_f1-51.26.bin

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --load_state_dict "model/model_multi/step-19000_f1-51.26.bin"  --task_name "sighan" --mft --mask_mode "noerror" --train_on "all"  --eval_on '15' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 100 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 64 --eval_batch_size 32 --fp16 --output_dir "model/model_multi"
step-4300_f1-64.60.bin
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-4300_f1-64.60.bin" --task_name "sighan"  --test_on "15" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-4300_f1-64.60.bin" --task_name "tnews"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-4300_f1-64.60.bin" --task_name "afqmc"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64

########dynamic mask###########
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --task_name "sighan tnews afqmc" --mft --mask_mode "noerror" --mask_rate 0.3 --train_on "hybrid base base"  --eval_on '15' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"
step-36000_f1-50.05.bin

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --load_state_dict "model/model_multi/step-36000_f1-50.05.bin"  --task_name "sighan" --mft --mask_mode "noerror" --mask_rate 0.3 --train_on "all"  --eval_on '15' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 100 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 64 --eval_batch_size 32 --fp16 --output_dir "model/model_multi"
step-4500_f1-69.00.bin

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --load_state_dict "model/model_multi/step-36000_f1-50.05.bin"  --task_name "sighan tnews afqmc" --mft --mask_mode "noerror" --mask_rate 0.3 --train_on "all base base"  --eval_on '15' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 100 --learning_rate 5e-5 --max_train_steps 5000 --train_batch_size 64 --eval_batch_size 32 --fp16 --output_dir "model/model_multi"
step-5000_f1-65.17.bin
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-5000_f1-65.17.bin" --task_name "sighan"  --test_on "15" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-5000_f1-65.17.bin" --task_name "tnews"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-5000_f1-65.17.bin" --task_name "afqmc"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64

###law
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --mft --mask_mode "noerror" --task_name "ecspell tnews afqmc"  --train_on "law base base"  --eval_on 'law' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"
step-9000_f1-70.57.bin
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-9000_f1-70.57.bin" --task_name "ecspell"  --test_on "law" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-9000_f1-70.57.bin" --task_name "tnews"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-9000_f1-70.57.bin" --task_name "afqmc"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64

########dynamic mask###########
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.3 --task_name "ecspell tnews afqmc"  --train_on "law base base"  --eval_on 'law' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" 
step-10000_f1-83.27.bin
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-10000_f1-83.27.bin" --task_name "ecspell"  --test_on "law" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-10000_f1-83.27.bin" --task_name "tnews"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-10000_f1-83.27.bin" --task_name "afqmc"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.3 --task_name "ecspell tnews afqmc"  --train_on "law base base"  --eval_on 'law' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 6e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" 
step-10000_f1-83.85.bin
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.3 --task_name "ecspell tnews afqmc"  --train_on "law base base"  --eval_on 'law' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 7e-5 --num_train_epochs 20.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" 
step-14000_f1-87.59.bin
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-14000_f1-87.59.bin" --task_name "ecspell"  --test_on "law" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-14000_f1-87.59.bin" --task_name "tnews"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-14000_f1-87.59.bin" --task_name "afqmc"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64
## no prompt
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.3 --task_name "ecspell tnews afqmc"  --train_on "law base base"  --eval_on 'law' --csc_prompt_length 1 --sent_prompt_length 1 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 20.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --not_apply_prompt
step-10000_f1-84.17.bin
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-10000_f1-84.17.bin" --task_name "tnews"  --test_on "base" --csc_prompt_length 1 --sent_prompt_length 1 --eval_batch_size 64 --not_apply_prompt
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-10000_f1-84.17.bin" --task_name "afqmc"  --test_on "base" --csc_prompt_length 1 --sent_prompt_length 1 --eval_batch_size 64 --not_apply_prompt

###med
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --mft --task_name "ecspell tnews afqmc"  --train_on "med base base"  --eval_on 'med' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"
step-8000_f1-72.50.bin
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-8000_f1-72.50.bin" --task_name "tnews"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-8000_f1-72.50.bin" --task_name "afqmc"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64


########dynamic mask###########
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.3 --task_name "ecspell tnews afqmc"  --train_on "med base base"  --eval_on 'med' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
step-9000_f1-80.82.bin
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-9000_f1-80.82.bin" --task_name "ecspell"  --test_on "med" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64 --output_dir "model/model_multi" --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-9000_f1-80.82.bin" --task_name "tnews"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64 --output_dir "model/model_multi" --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-9000_f1-80.82.bin" --task_name "afqmc"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64 --output_dir "model/model_multi" --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.3 --task_name "ecspell tnews afqmc"  --train_on "med base base"  --eval_on 'med' --csc_prompt_length 1 --sent_prompt_length 1 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --not_apply_prompt
step-10000_f1-76.06.bin
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-10000_f1-76.06.bin" --task_name "tnews"  --test_on "base" --csc_prompt_length 1 --sent_prompt_length 1 --eval_batch_size 64 --output_dir "model/model_multi" --not_apply_prompt
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-10000_f1-76.06.bin" --task_name "afqmc"  --test_on "base" --csc_prompt_length 1 --sent_prompt_length 1 --eval_batch_size 64 --output_dir "model/model_multi" --not_apply_prompt

###odw
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --mft --task_name "ecspell tnews afqmc"  --train_on "odw base base"  --eval_on 'odw' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi"
step-10000_f1-68.20.bin
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-10000_f1-68.20.bin" --task_name "ecspell"  --test_on "odw" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-10000_f1-68.20.bin" --task_name "tnews"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-10000_f1-68.20.bin" --task_name "afqmc"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64

########dynamic mask###########
CUDA_VISIBLE_DEVICES=1 python run_multi_ptuning.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.3 --task_name "ecspell tnews afqmc"  --train_on "odw base base"  --eval_on 'odw' --csc_prompt_length 10 --sent_prompt_length 3 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
step-10000_f1-77.96.bin

CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-10000_f1-77.96.bin" --task_name "ecspell"  --test_on "odw" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64 --output_dir "model/model_multi" --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-10000_f1-77.96.bin" --task_name "tnews"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64 --output_dir "model/model_multi" --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-10000_f1-77.96.bin" --task_name "afqmc"  --test_on "base" --csc_prompt_length 10 --sent_prompt_length 3 --eval_batch_size 64 --output_dir "model/model_multi" --load_model_path "../cache/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55"


CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_train --do_eval --mft --mask_mode "noerror" --mask_rate 0.3 --task_name "ecspell tnews afqmc"  --train_on "odw base base"  --eval_on 'odw' --csc_prompt_length 1 --sent_prompt_length 1 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 15.0 --train_batch_size 128 --eval_batch_size 64 --fp16 --output_dir "model/model_multi" --not_apply_prompt
step-10000_f1-74.96.bin
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-10000_f1-74.96.bin" --task_name "tnews"  --test_on "base" --csc_prompt_length 1 --sent_prompt_length 1 --eval_batch_size 64 --output_dir "model/model_multi" --not_apply_prompt
CUDA_VISIBLE_DEVICES=0 python run_multi_ptuning.py --do_test --load_state_dict "model/model_multi/step-10000_f1-74.96.bin" --task_name "afqmc"  --test_on "base" --csc_prompt_length 1 --sent_prompt_length 1 --eval_batch_size 64 --output_dir "model/model_multi" --not_apply_prompt
```



mft1: mask_all

mft: mask noerror

|                                | law/med/odw       | tnews | afqmc | sighan      | tnews | afqmc |
| ------------------------------ | ----------------- | ----- | ----- | ----------- | ----- | ----- |
| BERT-single                    | 37.9/22.3/25.0    | \     | \     |             | \     | \     |
| Prompt-mlm-single              | 57.3/56.9/59.0    | \     | \     |             | \     | \     |
| BERT+mft-single(dynamic)       | 76.1/58.0/59.2    | \     | \     |             | \     | \     |
| mdcspell+mft(dynamic))         | 81.1/72.4/72.0    | \     | \     |             | \     | \     |
| Prompt-mlm+mft-single(dynamic) | 92.17/85.40/86.71 | \     | \     |             | \     | \     |
| ***                            | ***               | ***   | ***   | ***         | ***   | ***   |
| Bert-multi                     |                   |       |       |             |       |       |
| Prompt-mlm-multi               |                   |       |       | 69.00(12.9) | 56.3  | 70.9  |
| Bert+mft-multi(static)         | \                 | \     | \     | \           | \     | \     |
| Bert+mft-multi(dynamic)        | 61.99/48.82/52.35 |       |       | 50.83(22.2) | 55.4  | 70.2  |
| Prompt-mlm+mft-multi(static)   | \                 | \     | \     | \           | \     | \     |
| Prompt-mlm+mft-multi(dynamic)  | 83.27/            |       |       | 65.17(11.2) | 56.7  | 71.5  |
