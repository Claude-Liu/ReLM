## Prompt Learning CSC

**To construct prompts**

1. MLM-based LMs

No detection learning

```txt
今天天气不戳。[SEP]今天天气不[MASK]。 -> 今天天气不戳。[SEP]今天天气不错。
```

Regular fine-tuning BERT models

It is unnecessary to predict so many slots and many of them are copies.

```txt
今天天气不戳。[SEP][MASK][MASK][MASK][MASK][MASK][MASK][MASK] -> 今天天气不戳。[SEP]今天天气不错。
```

Randomly mask

```txt
今天天气不戳。[SEP]今天[MASK]气不[MASK]。
```

2. Autoregressive LMs / prefix LMs

T5

```txt
CSC:今天天气不戳。answer:_______
```



**Training prompts**

1. Froze the LM weights
2. Train the LM weights
3. When learning domain specific, better to train the LM



**Prompt tuning**

```txt
p1,p2,p3,...,x1,x2,x3,...
```

**P-tuning**

```txt
p1,p2,p3,x1,x2,x3,...p4,p5,p6,y1,y2,y3,...
```



## Experiments

**MLM-based LMs**

Prompt

```txt
Source: 今天天气不戳。[SEP]Target: [MASK][MASK][MASK][MASK][MASK][MASK][MASK]
```

Prompt with MLM

```txt
今天天[MASK]不[MASK]。[SEP][MASK][MASK][MASK][MASK][MASK][MASK][MASK]
```

For FT, msl=64, for prompt, msl=128

Using BERT-base-Chinese, step=3000

P-tuning

```txt
p1,p2,p3,x1,x2,...,p4,p5,p6,m,m,...
```

l = 3

|        | FT   | MFT  | PT   | PT+MLM   |
| ------ | ---- | ---- | ---- | -------- |
| EC-LAW | 37.2 | 60.4 | 52.7 | **70.0** |
| EC-MED | 18.9 | 38.9 | 42.9 | **57.5** |
| SIGHAN | 42.7 | 52.5 | 49.6 | 50.6     |



**P-tuning**

|        | FT   | MFT  | CPT  | CPT+MLM  |
| ------ | ---- | ---- | ---- | -------- |
| EC-LAW | 37.2 | 60.4 | 55.4 | **73.0** |
| EC-MED | 18.9 | 38.9 | 37.0 | **57.8** |
| SIGHAN | 42.7 | 52.5 |      |          |



**Scaling**

On EC-LAW

|               | FT   | MFT  | PT+MLM | CPT+MLM |
| ------------- | ---- | ---- | ------ | ------- |
| BERT-base     | 37.2 | 60.4 | 70.0   | 73.0    |
| MacBERT-large | 43.5 | 80.8 | 70.3   |         |
