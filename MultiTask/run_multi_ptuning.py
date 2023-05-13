from __future__ import absolute_import, division, print_function
import argparse
from curses import raw
import logging
import os
import random
import math
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import *

import sklearn.metrics as mtc
from scipy.stats import spearmanr
from transformers import AutoTokenizer
from transformers import SchedulerType, get_scheduler
from transformers import BertForMaskedLM

from MultiTask.MultiTaskDataset import SighanProcessor, EcspellProcessor, TnewsProcessor, AfqmcProcessor
from MultiTask.MultiTaskDataset import csc_convert_examples_to_features, seq_convert_examples_to_features
from MultiTask import MultiTaskDataset

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


'''
task_csc = Task(1,'csc','task_classification')
task_tnews = Task(2, 'tnews', 'seq_classification')
task_qmc = Task(3,'afqmc','question-similarity')
'''


class PTuningWrapper(nn.Module):

    def __init__(self, model, tokenizer, verbalizer_tnews, verbalizer_afqmc, prompt_length_sent, prompt_length_csc):
        super().__init__()
        self.config = model.config
        self.tokenizer = tokenizer
        self.prompt_length_csc = prompt_length_csc
        self.prompt_length_sent = prompt_length_sent
        self.verbalizer_tnews = verbalizer_tnews
        self.verbalizer_afqmc = verbalizer_afqmc
        self.tnews_label_words_ids = verbalizer_tnews.label_words_ids
        self.afqmc_label_words_ids = verbalizer_afqmc.label_words_ids

        self.csc_num_labels = self.config.vocab_size
        self.tnews_num_labels = verbalizer_tnews.num_labels
        self.afqmc_num_labels = verbalizer_afqmc.num_labels

        self.model = model ## mlm
        ## the embdedding layer of BERT
        self.word_embeddings = getattr(self.model, self.model_type).embeddings.word_embeddings
        ## pronpt embedding for afqmc
        self.afqmc_prompt_embeddings = nn.Embedding(self.prompt_length_sent, self.config.hidden_size)
        self.afqmc_prompt_lstm = nn.LSTM(input_size=self.hidden_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=2,
                                   bidirectional=True,
                                   batch_first=True)
        self.afqmc_prompt_linear = nn.Sequential(nn.Linear(2 * self.config.hidden_size, self.config.hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.config.hidden_size, self.config.hidden_size))

        ## prompt embedding for tnews
        self.tnews_prompt_embeddings = nn.Embedding(self.prompt_length_sent, self.config.hidden_size)
        self.tnews_prompt_lstm = nn.LSTM(input_size=self.hidden_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=2,
                                   bidirectional=True,
                                   batch_first=True)
        self.tnews_prompt_linear = nn.Sequential(nn.Linear(2 * self.config.hidden_size, self.config.hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.config.hidden_size, self.config.hidden_size))
        
        ## prompt embedding for csc
        self.csc_prompt_embeddings = nn.Embedding(2*self.prompt_length, self.hidden_size)
        ## LSTM: input:(batch,seq,input_size)-->output[0]:(batch,seq,2*hidden)
        self.csc_prompt_lstm = nn.LSTM(input_size=self.hidden_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=2,
                                   bidirectional=True,
                                   batch_first=True)
        self.csc_prompt_linear = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_size, self.hidden_size))
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        prompt_mask=None,
        active_bits=None,
        task_id = None,
        labels=None,
        inputs_embeds=None,
        output_hidden_states=None,
        return_dict = True
        ):
        ## get embdding of all the tasks 
        inputs_embeds = self.word_embeddings(input_ids) if inputs_embeds is None else inputs_embeds

        ## afqmc
        replace_embeds_afqmc = self.afqmc_prompt_embeddings(torch.LongTensor(list(range(self.prompt_length_sent))).to(inputs_embeds.device))
        replace_embeds_afqmc = replace_embeds_afqmc.unsqueeze(0)
        replace_embeds_afqmc = self.afqmc_prompt_lstm(replace_embeds_afqmc)[0]##(prompt_length,2*hidden_size)
        replace_embeds_afqmc = self.afqmc_prompt_linear(replace_embeds_afqmc).squeeze() ## (prompt_length,hidden)
        ## tnews
        replace_embeds_tnews = self.tnews_prompt_embeddings(torch.LongTensor(list(range(self.prompt_length_sent))).to(inputs_embeds.device))
        replace_embeds_tnews = replace_embeds_tnews.unsqueeze(0)
        replace_embeds_tnews = self.tnews_prompt_lstm(replace_embeds_tnews)[0]##(prompt_length,2*hidden_size)
        replace_embeds_tnews = self.tnews_prompt_linear(replace_embeds_tnews).squeeze() ## (prompt_length,hidden)
        ## csc
        replace_embeds_csc = self.csc_prompt_embeddings(torch.LongTensor(list(range(2*self.prompt_length_csc))).to(input_ids.device))
        replace_embeds_csc = replace_embeds_csc.unsqueeze(0)##(1,2*prompt_length,hidden_size)
        replace_embeds_csc = self.csc_prompt_lstm(replace_embeds_csc)[0]##(2*prompt_length,2*hidden_size)
        replace_embeds_csc = self.csc_prompt_linear(replace_embeds_csc).squeeze()##(2*prompt_length,hidden_size)


        csc_task_filter = (task_id == 1)
        tnews_task_filter = (task_id == 2)
        afqmc_task_filter = (task_id == 3)
        ## prompt_mask (batch,seq)
        prompt_mask_csc = prompt_mask[csc_task_filter] ## (batch size for csc,seq)
        blocked_indices_csc = (prompt_mask_csc == 1).nonzero().reshape((inputs_embeds.shape[0], 2*self.prompt_length_csc, 2))[:, :, 1] ## (batch size for csc,2*prompt_length_csc)
        prompt_mask_tnews = prompt_mask[tnews_task_filter] ## (batch size for tnews,seq)
        blocked_indices_tnews = (prompt_mask_tnews == 1).nonzero().reshape((inputs_embeds.shape[0], self.prompt_length_sent, 2))[:, :, 1] ## (batch size for tnews,prompt_length_sent)
        prompt_mask_afqmc = prompt_mask[afqmc_task_filter] ## (batch size for afqmc,seq)
        blocked_indices_afqmc = (prompt_mask_afqmc == 1).nonzero().reshape((inputs_embeds.shape[0], self.prompt_length_sent, 2))[:, :, 1] ## (batch size for afqmc,prompt_length_sent)

        ## replace the prompt positions in input_embeds with prompt embeddings correspondingly
        csc_i, tnews_i, afqmc_i=0,0,0
        for i in range(inputs_embeds.shape[0]):
            if task_id[i]==1:
                for j in range(blocked_indices_csc.shape[1]):
                    inputs_embeds[i, blocked_indices_csc[csc_i, j], :] = replace_embeds_csc[j, :]
                csc_i+=1
            if task_id[i]==2:
                for j in range(blocked_indices_tnews.shape[1]):
                    inputs_embeds[i, blocked_indices_tnews[tnews_i, j], :] = replace_embeds_tnews[j, :]
                tnews_i+=1
            else:
                assert task_id[i]==1
                for j in range(blocked_indices_afqmc.shape[1]):
                    inputs_embeds[i, blocked_indices_afqmc[afqmc_i, j], :] = replace_embeds_afqmc[j, :]
                afqmc_i+=1

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states,
            return_dict = return_dict
        )
        logits = outputs.logits ## batch,seq,vocab_size
        loss_all=[]
        ## csc
        if csc_task_filter.any():
            csc_logits = logits[csc_task_filter]
            csc_loss =None
            if labels is not None:
                labels_csc = labels[csc_task_filter]
                input_csc = input_ids[csc_task_filter]
                labels_csc[input_csc==labels_csc]=-100
                loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
                csc_loss = loss_fct(csc_logits.view(-1, self.csc_num_labels), labels_csc.view(-1))
            logits_output = csc_logits
            loss_all.append(csc_loss)
        ## tnews
        if tnews_task_filter.any():
            mask_length = 2
            tnews_logits = logits[tnews_task_filter] ## tnews_batch,seq,vocab
            tnews_active_bits = active_bits[tnews_task_filter] ## tnews_batch,seq
            tnews_logits = tnews_logits[torch.where(tnews_active_bits != -100)]\
                .view(-1,mask_length,self.tokenizer.vocab_size) # tnews_batch,mask_length=2,vocab

            label_words_logits_1 = tnews_logits[:,0, self.tnews_label_words_ids[:,0]] # tnews_batch,num_label
            label_words_logits_2 = tnews_logits[:,1, self.tnews_label_words_ids[:,1]] # tnews_batch,num_label
            label_words_logits = label_words_logits_1 * label_words_logits_2
            assert label_words_logits.shape[-1] == self.tnews_num_labels
            tnews_loss = None
            if labels is not None:
                labels_tnews = labels[tnews_task_filter]
                loss_fct = nn.CrossEntropyLoss()
                tnews_loss = loss_fct(label_words_logits.view(-1, self.tnews_num_labels), labels_tnews.view(-1))
            logits_output = label_words_logits
            loss_all.append(tnews_loss)
        ## afqmc 
        if afqmc_task_filter.any():
            afqmc_logits = logits[afqmc_task_filter] ## afqmc_batch,seq,vocab
            afqmc_active_bits = active_bits[afqmc_task_filter] ## afqmc_batch,seq
            afqmc_logits = afqmc_logits[torch.where(afqmc_active_bits != -100)] # afqmc_batch,vocab

            label_words_logits = afqmc_logits[:, self.afqmc_label_words_ids] # afqmc_batch,num_label,num_label_mapping
            label_words_logits = torch.sum(label_words_logits,dim=-1) # afqmc_batch,num_label
            afqmc_loss = None
            if labels is not None:
                labels_afqmc = labels[afqmc_task_filter]
                loss_fct = nn.CrossEntropyLoss()
                afqmc_loss = loss_fct(label_words_logits.view(-1, self.afqmc_num_labels), labels_afqmc.view(-1))
            logits_output = label_words_logits
            loss_all.append(afqmc_loss)

        if output_hidden_states:
            return loss_all, logits_output, outputs[-1]
        return loss_all, logits_output
    
class Metrics:
    ### metrics for sequence classification
    @staticmethod
    def acc(predictions, labels):
        return mtc.accuracy_score(labels, predictions)

    @staticmethod
    def mcc(predictions, labels):
        return mtc.matthews_corrcoef(labels, predictions)

    @staticmethod
    def spc(predictions, labels):
        return spearmanr(labels, predictions)[0]

    @staticmethod
    def f1(predictions, labels, average="micro"):
        return mtc.f1_score(labels, predictions, average=average)
    ### metrics for csc
    @staticmethod
    def csc_compute(src_sents, trg_sents, prd_sents):
        def difference(src, trg):
            ret = copy.deepcopy(src)
            for i, (src_char, trg_char) in enumerate(zip(src, trg)):
                if src_char!= trg_char:
                    ret[i] = "(" + src_char + "->" + trg_char + ")"

            return "".join(ret)

        pos_sents, neg_sents, tp_sents, fp_sents, fn_sents, prd_pos_sents, prd_neg_sents, wp_sents = [], [], [], [], [], [], [], []
        for s, t, p in zip(src_sents, trg_sents, prd_sents):
            # For positive examples
            if s != t:
                pos_sents.append(difference(s, t))
                if p == t:
                    tp_sents.append(difference(s, t))
                if p == s:
                    fn_sents.append(difference(s, t))
                if (p!=t and p!=s):
                    wp_sents.append(difference(s,t))
            # For negative examples
            else:
                neg_sents.append(difference(s, t))
                if p != t:
                    fp_sents.append(difference(t, p))
            # For predictions
            if s != p:
                prd_pos_sents.append(difference(s, p))
            if s == p:
                prd_neg_sents.append(difference(s, p))

        p = 1.0 * len(tp_sents) / len(prd_pos_sents)
        r = 1.0 * len(tp_sents) / len(pos_sents)
        f1 = 2.0 * (p * r) / (p + r + 1e-12)
        fpr = 1.0 * (len(fp_sents) + 1e-12) / (len(neg_sents) + 1e-12)

        return p, r, f1, fpr, tp_sents, fp_sents, fn_sents, wp_sents
    
def mask_tokens(inputs, tokenizer, noise_probability=0.2):
    inputs = inputs.clone()
    probability_matrix = torch.full(inputs.shape, noise_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs

def main():
    parser = argparse.ArgumentParser()

    # Data config.
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Directory to contain the input data for all tasks.")
    ## mulitple tasks splited by " "
    parser.add_argument("--task_name", type=str, default="SIGHAN tnews afqmc",
                        help="Name of the training task.")
    parser.add_argument("--load_model_path", type=str, default="bert-base-chinese",
                        help="Pre-trained model path to load if needed.")
    parser.add_argument("--cache_dir", type=str, default="../cache/",
                        help="Directory to store the pre-trained language models downloaded from s3.")
    parser.add_argument("--output_dir", type=str, default="model/",
                        help="Directory to output predictions and checkpoints.")
    parser.add_argument("--load_checkpoint", type=str, default="",
                        help="Trained model weights to load for evaluation.")
    
    # Training config.
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to evaluate on the dev set.")
    parser.add_argument("--do_test", action="store_true",
                        help="Whether to evaluate on the test set.")
    ## multiple datasets splited by " "
    parser.add_argument("--train_on", type=str, default="hybrid base base",
                        help="Choose a training set.")
    ## eval and test on only one task
    parser.add_argument("--eval_on", type=str, default="15",
                        help="Choose a dev set.")
    parser.add_argument("--test_on", type=str, default="15",
                        help="Choose a test set.")
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="A slow tokenizer will be used if passed.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum total input sequence length after word-piece tokenization\
                            for the two sentence classification tasks.")
    parser.add_argument("--train_batch_size", type=int, default=128,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=512,
                        help="Total batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides training epochs.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
                        help="Scheduler type for learning rate warmup.")
    parser.add_argument("--warmup_proportion", type=float, default=0.1,
                        help="Proportion of training to perform learning rate warmup for.")
    parser.add_argument("--weight_decay", type=float, default=0.,
                        help="L2 weight decay for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward pass.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use mixed precision.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization.")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="How many steps to save the checkpoint once.")
    parser.add_argument("--mft", action="store_true",
                        help="Training with masked-fine-tuning (not published yet).")

    parser.add_argument("--csc_prompt_length",type=int, default=3,help="the length of the continuous prompt")
    parser.add_argument("--sent_prompt_length",type=int, default=3,help="the length of the continuous prompt")
    

    args = parser.parse_args()  

    processors_all = {
        "sighan": SighanProcessor,
        "ecspell": EcspellProcessor,
        "sghspell": SighanProcessor,## the data format in sghspell is the same as sighan
        "tnews": TnewsProcessor,
        "afqmc": AfqmcProcessor,
    }

    task_class={"csc":["sighan","ecspell","sghspell"],
                "seq":["tnews","afqmc"]}

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, "Unsupported", args.fp16))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_train:
        torch.save(args, os.path.join(args.output_dir, "train_args.bin"))

    task_names = args.task_name.lower().split()
    train_on_list = args.train_on.lower().split()
    for task_name in task_names:
        if task_name not in processors_all:
            raise ValueError("Task not found: %s" % task_name)
    # processors is a map containing all the processors we will use
    processors={}
    train_on_dataset={}
    for task_name in task_names:
        processors[task_name]=processors_all[task_name]()
    for train_on,task_name in zip(train_on_list,task_names):
        train_on_dataset[task_name]=train_on

    cache_dir = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.load_model_path,
                                              do_lower_case=args.do_lower_case,
                                              cache_dir=cache_dir,
                                              use_fast=not args.use_slow_tokenizer,
                                              add_prefix_space=True)
    
    