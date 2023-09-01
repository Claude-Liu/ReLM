### 

#### abstract

问题：

learning a better language model   Vs  alleviating the degeneration of language model during pre-training

state-of-the-art



#### motivation and introduction

1. what is a CSC task: error patterns and semantic information are both important
2. sequence tagging: reverse causality
3. drawbacks of sequence tagging: over-correction, semantic error and transfer ability
4. error language model
5. examples: bert tagging and error language model
6. contribution

问题： sequence tagging



#### related work

1. BERT fine-tuning based CSC models

   softmask

   spellgcn

   mdcspell,

2. rethinking

3. our method



####  analysis of BERT-Tagging

1. the performance on semantic errors

   (问题：sghspell 结果不理想 1. 在semantic errors上训练后没有提升 2. 合成同义句和歧义句

   解决： 1. big entropy, 2. 在sighan基础上合成semantic errors)

2. the performance in multi-task scenario

   multi-task evaluation

   linear probing

3.  over-correction and influence of fpr 



#### method

1. single task
2. multi-task

问题：

1. 有点短， 图示添加多任务？

2. 是否提及 masked fine-tuning LMCorrector



#### experiment

问题： 加入数据集的statistics: negative examples and positive examples

1. single-task

   问题：

   1. 解释sighan15上 mft+LMCorrector 不如 LMCorrector

   2. 是否加入MSCspell

   3. 如何体现 cross-domain transfer ability: 

      exc-f1: LMCorrector 不如 Masked-FT BERT, few-shot?

2. multi-task

   问题：训练方法改进：在csc上单独训练更多步，freeze BERT mlm?



#### further analysis

1. mask strategy
2. linear probing
3. model scaling





##### 优先级：

已完成

1. further analysis: false positive rate
2. further analysis: mask strategy
3. semantic errors: lise
4. odw 多任务结果
5. method: 重新画图
6. sighan 多任务实验方法：multi-task in two stages
7. multi-task preliminary: linear probing (how to freeze): why degenerate
8. multi-task preliminary: vanilla BERT 
9. LISE semantic error--window size small(2+2+1=5) 
10. p-tuning消融：纯lmcorrect e.g. x,x,...,x [seq], m,m,...,m
11. gpt-2代码
12. gpt-2->ecspell实验 作为baseline
13. 修改introduction+related+preliminary+method(故事，lmcorrector->error lm)
14. LMCorrector 和 BERT 名字问题 
15. multi-task results ok
16. false positive rates 插图 ok
17. linear probing ok
18. prefix ok
19. mask strategy ok
20. mask rate ok



待完成 

1. LEMON: 数据集+实验结果

2. multi-task模型描述

   

   

问题：

1. dropout: BERTFormlm没有dropout ， BERTForTokenClassification 有

2. 加和不加 token_type_ids 结果不影响 relm(no prefix)

   


GPT: detector-corrector network, 多任务

GPT prefix-learning







#### 写作修改

1. 名词规范
   1. masked fine-tuning
   2. LMcorrector Vs BERT fine-tuning
   3. BERT fine-tuning based CSC models
2. aaai 格式：
   1. appendix solely for reference
