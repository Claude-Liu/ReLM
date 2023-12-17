from __future__ import absolute_import, division, print_function
import argparse
import json
import logging
import os
import random
import copy
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, SubsetRandomSampler
import sklearn.metrics as mtc
from scipy.stats import spearmanr
from tqdm import tqdm, trange
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2PreTrainedModel
from transformers import SchedulerType, get_scheduler
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import torch.nn as nn
from utils.data_processor import EcspellProcessor, InputExample
from utils.metrics import Metrics
from accelerate import Accelerator

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptEmbeddings(nn.Module):
    def __init__(self, hidden_size, num_virtual_tokens=10):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(num_virtual_tokens, hidden_size)
        self.prompt_lstm = nn.LSTM(input_size=self.hidden_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=2,
                                   bidirectional=True,
                                   batch_first=True)
        self.prompt_linear = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_size, self.hidden_size))

    def forward(self, input_ids):
        input_embeds = self.embedding(input_ids)
        input_embeds = self.prompt_lstm(input_embeds)[0]
        output_embeds = self.prompt_linear(input_embeds)

        return output_embeds

class KLDivRegularization(nn.Module):
    def __init__(self, lambd, num_labels):
        super().__init__()
        self.lambd = lambd
        self.klDiv = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.num_label = num_labels
    def forward(self, shift_inputs, shift_logits, shift_labels):
        src_mask = (shift_inputs!=-100).unsqueeze(-1).repeat(1,1,self.num_label)
        src_logits_ = torch.masked_select(shift_logits,src_mask).reshape(-1,self.num_label)
        src_logits = src_logits_.clone().detach()
        trg_mask = (shift_labels!=-100).unsqueeze(-1).repeat(1,1,self.num_label)
        trg_logits = torch.masked_select(shift_logits,trg_mask).reshape(-1,self.num_label)
        assert src_logits.shape==trg_logits.shape
        kl_penalty = self.klDiv(trg_logits.log_softmax(-1), src_logits.log_softmax(-1))
        return self.lambd*kl_penalty

class RegularizedGPT2LMForCSC(GPT2LMHeadModel):
    '''
    class for rephrasing model based on GPT2
    apply kl-divergence for regularization by passing regu_src in forward()
    add prefix by setting add_prefix as True
    '''
    def __init__(self,config, lambd=0,add_prefix=False):
        super().__init__(config)
        self.lambd = lambd
        self.num_label = config.vocab_size
        self.klDivRegularization = KLDivRegularization(lambd, self.num_label)
        self.pe = PromptEmbeddings(hidden_size=config.hidden_size)
        self.add_prefix = add_prefix

    def forward(self,input_ids,attention_mask,labels=None,
                past_key_values=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                regu_src=None):

        if self.add_prefix: 
            batch_size = input_ids.size(0)
            prefix_attention_mask = torch.ones(batch_size, self.pe.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

            inputs_embeds = self.transformer.wte(input_ids)
            indices = torch.arange(self.pe.num_virtual_tokens).unsqueeze(0).expand(batch_size, -1).to(input_ids.device)
            prompts = self.pe(indices).to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)

            if labels is not None:
                labels = labels.clone()
                labels = torch.cat((torch.full_like(indices, -100).to(indices.dtype), labels), dim=1)
            
            outputs = self.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.transformer(
                input_ids = input_ids,
                attention_mask=attention_mask,
            )

        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)
        if not self.add_prefix and regu_src is not None:
            shift_inputs = regu_src[..., 1:].contiguous()##(batch,max_seq-1)
        shift_logits = logits[..., :-1, :].contiguous()##(batch,max_seq-1,vocab)
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
    
        loss=None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            loss_lm = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # apply kl divergence
            if not self.add_prefix and regu_src is not None: 
                kl_penalty = self.klDivRegularization(shift_inputs, shift_logits, shift_labels)
                loss = loss_lm + self.lambd*kl_penalty
            else:
                loss = loss_lm

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
            
class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, labels, regu_src=None, target_ref=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.regu_src = regu_src
        self.target_ref = target_ref

def convert_examples_to_features(examples, max_seq_length, tokenizer, add_arrow):
    '''
        input_ids: cls + src + sep + trg + sep +pad(0)
        regu_src: -100 + src + sep + ...-100... + pad(-100)
        target_ref: cls + trg + sep + ...-100... + pad(-100)
        labels:    ... -100 ... + trg + sep +pad(-100)
        '''
    features = []
    def truncate(x, max_length):
        return x[: max_length]
    def add_arrow_(source,target):
        new_target=[]
        for st,tt in zip(source,target):
            new_target+=[st,'>',tt]
        return new_target
    for i, example in enumerate(examples):
        #truncate the source and the target and modify the target if needed
        if add_arrow:
            max_length = max_seq_length//4-1
            example.src = truncate(example.src,max_length)
            example.trg = truncate(example.trg,max_length)
            example.trg = add_arrow_(example.src,example.trg)
            assert len(example.trg)%3 == 0
        else:
            max_length=max_seq_length//2-2
            example.src = truncate(example.src,max_length)
            example.trg = truncate(example.trg,max_length)
        
        encoded_inputs = tokenizer(example.src, add_special_tokens=True ,is_split_into_words=True)
        encoded_inputs["labels"] = [-100] * len(encoded_inputs["input_ids"])

        trg_ids= tokenizer(example.trg, add_special_tokens=False, is_split_into_words=True)["input_ids"] + [tokenizer.eos_token_id]
        src_ids= tokenizer(example.src, add_special_tokens=False, is_split_into_words=True)["input_ids"] + [tokenizer.sep_token_id]
        encoded_inputs['regu_src'] = [-100] + src_ids + [-100]*len(trg_ids)
        encoded_inputs['target_ref'] = [tokenizer.cls_token_id] + trg_ids + [-100]*len(trg_ids)
        encoded_inputs["labels"] += trg_ids
        encoded_inputs["input_ids"] += trg_ids
        encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])
    
        offset_length = max_seq_length - len(encoded_inputs["input_ids"])
        # pad left
        encoded_inputs["input_ids"] = [tokenizer.pad_token_id] * offset_length + encoded_inputs["input_ids"]
        encoded_inputs["attention_mask"] = [0] * offset_length + encoded_inputs["attention_mask"] 
        encoded_inputs["labels"] =  [-100] * offset_length + encoded_inputs["labels"]
        encoded_inputs['regu_src'] = [-100] * offset_length + encoded_inputs['regu_src']
        encoded_inputs['target_ref'] = [-100] * offset_length + encoded_inputs['target_ref']
        
        input_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"]
        labels = encoded_inputs["labels"]
        regu_src = encoded_inputs['regu_src']
        target_ref = encoded_inputs['target_ref']
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(labels) == max_seq_length
        
        if i < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join(tokens))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("labels: %s" % " ".join([str(x) for x in labels]))
            logger.info("regu_src: %s" % " ".join([str(x) for x in regu_src]))
            logger.info("target_ref: %s" % " ".join([str(x) for x in target_ref]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels,
                          regu_src=regu_src,
                          target_ref=target_ref,)
        )

    return features
    
def dynamic_mask_token(inputs, targets_ref, tokenizer, device, noise_probability=0.2):
    '''
        the masked-FT proposed in 'Rethinking Masked Language Model for Chinese Spelling Correction'
        input_ids: cls + src + sep + trg + sep +pad(0)
        reg_ref: cls + trg + sep + ...-100... + pad(-100)
    '''
    inputs = inputs.clone()
    probability_matrix = torch.full(inputs.shape, noise_probability).to(device)
    #do not mask sepcail tokens
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool).to(device)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    ## do not mask target part and the error tokens in src part
    probability_matrix.masked_fill_(inputs!=targets_ref, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs

def main():
    parser = argparse.ArgumentParser()

    # Data config
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Directory to contain the input data for all tasks.")
    parser.add_argument("--task_name", type=str, default="SIGHAN",
                        help="Name of the training task.")
    parser.add_argument("--load_model_path", type=str, default="gpt2-chinese/",
                        help="Pre-trained language model to load.")
    parser.add_argument("--cache_dir", type=str, default="../../cache/",
                        help="Directory to store the pre-trained language models downloaded from s3.")
    parser.add_argument("--output_dir", type=str, default="model/",
                        help="Directory to output predictions and checkpoints.")
    parser.add_argument("--load_state_dict", type=str, default="",
                        help="Checkpoint to load for trianing or evaluation.")

    # Training config
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to evaluate on the dev set.")
    parser.add_argument("--do_test", action="store_true",
                        help="Whether to evaluate on the test set.")
    parser.add_argument("--train_on", type=str, default="",
                        help="Choose a training set.")
    parser.add_argument("--eval_on", type=str, default="",
                        help="Choose a dev set.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum total input sequence length after word-piece tokenization.")
    parser.add_argument("--train_batch_size", type=int, default=128,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=256,
                        help="Total batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Peak learning rate for optimization.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform (overrides training epochs).")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
                        help="Scheduler type for learning rate warmup.")
    parser.add_argument("--warmup_proportion", type=float, default=0.06,
                        help="Proportion of training to perform learning rate warmup for.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="L2 weight decay for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward pass.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use mixed precision.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization.")
    parser.add_argument("--lora", action="store_true",
                        help="Whether to use low rank adaption.")
    parser.add_argument("--doask", action="store_true",
                        help="Whether to augment the training data.")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="How many steps to save the checkpoint once.")
    parser.add_argument("--mft", action="store_true",
                        help="Training with masked-fine-tuning (not published yet).")
    parser.add_argument("--mask_mode", type=str, default="noerror", help="noerror,error or all")
    parser.add_argument("--mask_rate", type=float, default=0.2, help="the percentage we mask the source sentence in mask-ft technique")
    parser.add_argument("--kl_regu", action="store_true")
    parser.add_argument("--lambd", type=float, default=0.2, help="the value of lambda when we apply regularization")
    parser.add_argument("--add_prefix", action="store_true")
    parser.add_argument("--beam", type=int, default=1, help="number of beams if we use beam search for generation.")
    parser.add_argument("--add_arrow", action="store_true", help="if we add arrows in target between characters.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, "-accelerate", args.fp16))

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

    processor = EcspellProcessor()

    cache_dir = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.load_model_path,
                                              do_lower_case=args.do_lower_case,
                                              padding_side="left",
                                              cache_dir=cache_dir)
    
    if getattr(tokenizer, "eos_token_id") is None:
        tokenizer.eos_token_id = tokenizer.sep_token_id

    logger.info("tokenizer.eos_token_id: %d", tokenizer.eos_token_id)
    task_name = args.task_name.lower()

    if args.do_train:
        train_examples = processor.get_train_examples(os.path.join(args.data_dir, task_name), args.train_on)
        train_features = convert_examples_to_features(train_examples, args.max_seq_length, tokenizer, args.add_arrow)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
        all_labels = torch.tensor([f.labels for f in train_features], dtype=torch.long)
        all_regu_src = torch.tensor([f.regu_src for f in train_features], dtype=torch.long)
        all_target_ref = torch.tensor([f.target_ref for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_attention_mask, all_labels, all_regu_src, all_target_ref)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        accelerator = Accelerator(cpu=args.no_cuda, mixed_precision="fp16" if args.fp16 else "no")
        device = accelerator.device        
        model = RegularizedGPT2LMForCSC.from_pretrained(args.load_model_path,cache_dir=cache_dir,lambd=args.lambd, add_prefix=args.add_prefix)
 
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        scheduler = get_scheduler(name=args.lr_scheduler_type,
                                  optimizer=optimizer,
                                  num_warmup_steps=args.max_train_steps * args.warmup_proportion,
                                  num_training_steps=args.max_train_steps)

        if args.do_eval:
            eval_examples = processor.get_dev_examples(os.path.join(args.data_dir, task_name), args.eval_on)
            eval_features = convert_examples_to_features(eval_examples, args.max_seq_length, tokenizer, args.add_arrow)

            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
            all_labels = torch.tensor([f.labels for f in eval_features], dtype=torch.long)
            all_regu_src = torch.tensor([f.regu_src for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_attention_mask, all_labels, all_regu_src)
            eval_dataloader = DataLoader(eval_data, shuffle=False, batch_size=args.eval_batch_size)

        model, optimizer, scheduler, train_dataloader = accelerator.prepare(model, optimizer, scheduler, train_dataloader)
        
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", args.max_train_steps)

        global_step = 0
        best_epoch = 0
        best_result = list()
        progress_bar = tqdm(range(args.max_train_steps))
        for epoch in range(int(args.num_train_epochs)):
            train_loss = 0
            num_train_examples = 0
            train_steps = 0
            for step, batch in enumerate(train_dataloader):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, labels, regu_src, target_ref = batch
                
                if args.mft:
                    input_ids = dynamic_mask_token(input_ids, target_ref, tokenizer, device, noise_probability=args.mask_rate)
                if not args.kl_regu:
                    regu_src=None
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                                regu_src=regu_src,
                                )
                loss = outputs["loss"]

                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)

                train_loss += loss.item()
                num_train_examples += input_ids.size(0)
                train_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    progress_bar.update(1)

                if args.do_eval  and global_step % args.save_steps == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)
                    def decode(x):
                        return tokenizer.convert_ids_to_tokens(x, skip_special_tokens=True)
                    model.eval()
                    all_inputs, all_predictions, all_labels = [], [], []
                    
                    for i,batch in enumerate(tqdm(eval_dataloader, desc="Evaluation")):
                        batch = tuple(t.to(device) for t in batch)
                        input_ids, attention_mask, labels, regu_src = batch
                        with torch.no_grad():
                            outputs = model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            labels=labels,
                                            )
                            logits = outputs["logits"]
                            if args.add_prefix:
                                logits = logits[:,model.pe.num_virtual_tokens:]

                            shift_inputs = input_ids[..., 1:].contiguous()
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_attention_mask = attention_mask[...,1:].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                        #(batch,max_seq)
                        prd_ids = shift_logits.argmax(dim=-1)
                        src_ids = shift_inputs.tolist()
                        trg_ids = shift_labels.cpu().numpy().tolist()
                        prd_ids = prd_ids.masked_fill(shift_attention_mask == 0, 0).tolist()
                        if i<3:
                            print("inputs: {}".format(np.array(src_ids).shape))
                            print("predictions: {}".format(np.array(prd_ids).shape))
                            print("labels: {}".format(np.array(trg_ids).shape))
                        for i, (s, t, p) in enumerate(zip(src_ids, trg_ids, prd_ids)):
                            mapped_src = []
                            mapped_trg = []
                            mapped_prd = []
                            flag = False
                            for st, tt, pt in zip(s, t, p):
                                if tt!=-100:
                                    flag=True
                                if not flag:
                                    mapped_src += [st]
                                else:
                                    mapped_trg += [tt if tt!=-100 else 0]
                                    mapped_prd += [pt if tt!=-100 else 0]
                            all_inputs += [decode(mapped_src)]
                            all_labels += [decode(mapped_trg)]
                            all_predictions += [decode(mapped_prd)]
                    print(all_inputs[0])
                    print(all_labels[0])
                    print(all_predictions[0])

                    output_predict_file = os.path.join(args.output_dir, "predict_results.txt")
                    print("all inputs size: {}".format(len(all_inputs)))
                    print("all predictions size: {}".format(len(all_predictions)))
                    print("all labels size: {}".format(len(all_labels)))

                    # recover the predictions and labels when we use add_arrow
                    if args.add_arrow:
                        all_predictions_ = []
                        all_labels_ = []
                        all_inputs_ = []
                        for input, prediction, label in zip(all_inputs, all_predictions, all_labels):
                            label_ = []
                            prediction_ = []
                            if len(label)%3!=0:
                                continue
                            for i in range(len(label)//3):
                                label_.append(label[3*i+2])
                            all_labels_.append(label_)
                            for i in range(len(prediction)//3):
                                prediction_.append(prediction[3*i+2])
                            all_predictions_.append(prediction_)
                            all_inputs_.append(input)
                        all_predictions = all_predictions_
                        all_labels = all_labels_
                        all_inputs = all_inputs_
                        print(all_inputs[0])
                        print(all_labels[0])
                        print(all_predictions[0])
                        print("all inputs size: {}".format(len(all_inputs)))
                        print("all predictions size: {}".format(len(all_predictions)))
                        print("all labels size: {}".format(len(all_labels)))

                    train_epoch_loss = train_loss / len(train_dataloader)
                    try:
                        train_ppl = math.exp(train_epoch_loss)
                    except:
                        train_ppl = math.inf
                    p, r, f1, fpr, wpr, tp, fp, fn, wp = Metrics.csc_compute(all_inputs, all_labels, all_predictions)

                    result = {
                        "global_step": global_step,
                        "train_ppl": train_ppl,
                        "train_loss": train_epoch_loss,
                        "eval_p": p * 100,
                        "eval_r": r * 100,
                        "eval_f1": f1 * 100,
                        "eval_fpr": fpr * 100,
                    }
                    # save model
                    model_to_save = model.module if hasattr(model, "module") else model
                    output_model_file = os.path.join(args.output_dir, "step-%s_f1-%.2f.bin" % (str(global_step), result["eval_f1"]))
                    torch.save(model_to_save.state_dict(), output_model_file)
                    best_result.append((result["eval_f1"], output_model_file))
                    best_result.sort(key=lambda x: x[0], reverse=True)
                    if len(best_result) > 3:
                        _, model_to_remove = best_result.pop()
                        os.remove(model_to_remove)
                    # save eval results
                    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                    with open(output_eval_file, "a") as writer:
                        logger.info("***** Eval results *****")
                        writer.write(
                            "Global step = %s | eval precision = %.2f | eval recall = %.2f | eval f1 = %.2f | eval fp rate = %.2f\n"
                            % (str(result["global_step"]),
                            result["eval_p"],
                            result["eval_r"],
                            result["eval_f1"],
                            result["eval_fpr"]))
                        for key in sorted(result.keys()):
                            logger.info("Global step: %s,  %s = %s", str(global_step), key, str(result[key]))

                if global_step >= args.max_train_steps:
                    break
    
    if args.do_test:
        eval_examples = processor.get_dev_examples(os.path.join(args.data_dir, task_name), args.eval_on)
        all_inputs, all_labels = [], []
        for i, example in enumerate(eval_examples):
            all_inputs+=[example.src]
            all_labels+=[example.trg]

        logger.info("***** Generation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", 1)

        predict_model = RegularizedGPT2LMForCSC.from_pretrained(args.load_model_path,cache_dir=cache_dir,add_prefix=args.add_prefix)
        tokenizer.padding_side = 'left'

        predict_model.to(device)
        if args.load_state_dict:
            predict_model.load_state_dict(torch.load(args.load_state_dict),strict=False)
        predict_model.eval()
        all_predictions = []
        batch_size = args.eval_batch_size
        for i in tqdm(range(0, len(all_inputs), batch_size), desc="Testing"):
            e = min(len(all_inputs)-1, i+batch_size)
            inputs = tokenizer(all_inputs[i: e], return_tensors="pt",is_split_into_words=True, padding=True, max_length=args.max_seq_length)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            if i==0:
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids[0]]))
                logger.info("attention_mask: %s", " ".join([str(x) for x in attention_mask[0]]))
            trg = all_labels[i:e]
            src = all_inputs[i:e]
            with torch.no_grad():
                if args.beam!=1:
                    output_sequences = predict_model.generate(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                max_new_tokens=128,
                                                eos_token_id=tokenizer.eos_token_id,
                                                num_beams = args.beam,
                                                early_stopping = True,
                                                )
                else:
                    output_sequences = predict_model.generate(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                max_new_tokens=128,
                                                eos_token_id=tokenizer.eos_token_id,
                                                )
                                            
                pred = [tokenizer.convert_ids_to_tokens(output_seq, skip_special_tokens=True) for output_seq in output_sequences]
                
                all_inputs+=src
                all_labels+=trg
                all_predictions+=pred
        for i, (input,prediction) in enumerate(zip(all_inputs,all_predictions)):
            all_predictions[i] = prediction[len(input):]
                    
        del predict_model

        print(all_inputs[0])
        print(all_labels[0])
        print(all_predictions[0])
        print("all inputs size: {}".format(len(all_inputs)))
        print("all predictions size: {}".format(len(all_predictions)))
        print("all labels size: {}".format(len(all_labels)))
        p, r, f1, fpr, wpr, tp, fp, fn, wp = Metrics.csc_compute(all_inputs, all_labels, all_predictions) ## no need to decode

        output_tp_file = os.path.join(args.output_dir, "sents.tp")
        with open(output_tp_file, "w") as writer:
            for line in tp:
                writer.write(line + "\n")
        output_fp_file = os.path.join(args.output_dir, "sents.fp")
        with open(output_fp_file, "w") as writer:
            for line in fp:
                writer.write(line + "\n")
        output_fn_file = os.path.join(args.output_dir, "sents.fn")
        with open(output_fn_file, "w") as writer:
            for line in fn:
                writer.write(line + "\n")
        output_wp_file = os.path.join(args.output_dir, "sents.wp")
        with open(output_wp_file, "w") as writer:
            for line in wp:
                writer.write(line + "\n")

        result = {
            "eval_p": p * 100,
            "eval_r": r * 100,
            "eval_f1": f1 * 100,
            "eval_fpr": fpr * 100,
        }
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results *****")
            writer.write(
                "Global step = %s | eval precision = %.2f | eval recall = %.2f | eval f1 = %.2f | eval fp rate = %.2f\n"
                % (str(-1),
                result["eval_p"],
                result["eval_r"],
                result["eval_f1"],
                result["eval_fpr"]))
        for key in sorted(result.keys()):
            logger.info("Global step: %s,  %s = %s", str(-1), key, str(result[key]))


if __name__ == "__main__":
    main()
