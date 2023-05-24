from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import math
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers import SchedulerType, get_scheduler
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BertForErrorCorrection(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.vocab_size

        self.bert = BertModel(config, add_pooling_layer=False)
        ## (num) <=> num (int,float,...)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0) # ignore padding, the default value is -100
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class InputExample(object):
    def __init__(self, guid, src, trg):
        self.guid = guid
        self.src = src
        self.trg = trg


class InputFeatures(object):
    def __init__(self, src_ids, attention_mask, trg_ids):
        self.src_ids = src_ids
        self.attention_mask = attention_mask
        self.trg_ids = trg_ids


class SighanProcessor:
    """Processor for the Sighan data set."""

    def get_train_examples(self, data_dir, division="all"):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train_{}.txt".format(division))), "train")

    def get_dev_examples(self, data_dir, division="15"):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test_{}.txt".format(division))), "dev")

    def get_test_examples(self, data_dir, division="15"):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test_{}.txt".format(division))), "test")

    @staticmethod
    def _read_csv(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                src, trg = line.strip().split("\t")
                lines.append((src.split(), trg.split()))
            return lines

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for i, (src, trg) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if len(src) == len(trg):
                examples.append(InputExample(guid=guid, src=src, trg=trg))
        return examples


class EcspellProcessor:
    """Processor for the ECSpell data set."""

    def get_train_examples(self, data_dir, division="law"):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train_{}.txt".format(division))), "train")

    def get_dev_examples(self, data_dir, division="law"):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test_{}.txt".format(division))), "dev")

    def get_test_examples(self, data_dir, division="law"):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test_{}.txt".format(division))), "test")

    @staticmethod
    def _read_csv(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                src, trg = line.strip().split("\t")
                lines.append((src.split(), trg.split()))
            return lines

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for i, (src, trg) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if len(src) == len(trg):
                if len(src) == len(trg):
                    examples.append(InputExample(guid=guid, src=src, trg=trg))
        return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []
    for i, example in enumerate(examples):
        encoded_inputs = tokenizer(example.src,
                                   max_length=max_seq_length,
                                   padding="max_length",
                                   truncation=True,
                                   return_token_type_ids=True,
                                   is_split_into_words=True)

        trg_ids = tokenizer(example.trg,
                            max_length=max_seq_length,
                            padding="max_length",
                            truncation=True,
                            return_token_type_ids=True,
                            is_split_into_words=True)["input_ids"]

        src_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"]

        assert len(src_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(trg_ids) == max_seq_length

        if i < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("src_tokens: %s" % " ".join(example.src))
            logger.info("trg_tokens: %s" % " ".join(example.trg))
            logger.info("src_ids: %s" % " ".join([str(x) for x in src_ids]))
            logger.info("trg_ids: %s" % " ".join([str(x) for x in trg_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))

        features.append(
                InputFeatures(src_ids=src_ids,
                              attention_mask=attention_mask,
                              trg_ids=trg_ids)
        )
    return features


class Metrics:
    @staticmethod
    def compute(src_sents, trg_sents, prd_sents):
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


def mask_tokens(inputs, targets, tokenizer, device, mask_mode="noerror", noise_probability=0.2):
    ## mask_mode in ["all","error","noerror"]
    inputs = inputs.clone()
    probability_matrix = torch.full(inputs.shape, noise_probability).to(device)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool).to(device)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    if mask_mode == "noerror":
        probability_matrix.masked_fill_(inputs!=targets, value=0.0)
    elif mask_mode == "error":
        probability_matrix.masked_fill_(inputs==targets, value=0.0)
    else:
        assert mask_mode == "all"
    masked_indices = torch.bernoulli(probability_matrix).bool()
    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs


def main():
    parser = argparse.ArgumentParser()

    # Data config.
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Directory to contain the input data for all tasks.")
    parser.add_argument("--task_name", type=str, default="SIGHAN",
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
    parser.add_argument("--train_on", type=str, default="all",
                        help="Choose a training set.")
    parser.add_argument("--eval_on", type=str, default="15",
                        help="Choose a dev set.")
    parser.add_argument("--test_on", type=str, default="15",
                        help="Choose a test set.")
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="A slow tokenizer will be used if passed.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", type=int, default=64,
                        help="Maximum total input sequence length after word-piece tokenization.")
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
    parser.add_argument("--mask_mode", type=str, default="noerror", help="noerror,error or all")

    args = parser.parse_args()

    processors = {
        "sighan": SighanProcessor,
        "ecspell": EcspellProcessor,
        "sghspell": SighanProcessor,## the data format in sghspell is the same as sighan
    }

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

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()

    cache_dir = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.load_model_path,
                                              do_lower_case=args.do_lower_case,
                                              cache_dir=cache_dir,
                                              use_fast=not args.use_slow_tokenizer,
                                              add_prefix_space=True)

    if args.do_train:
        train_examples = processor.get_train_examples(os.path.join(args.data_dir, task_name), args.train_on)
        train_features = convert_examples_to_features(train_examples, args.max_seq_length, tokenizer)

        all_input_ids = torch.tensor([f.src_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.trg_ids for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        model = BertForErrorCorrection.from_pretrained(args.load_model_path,
                                                       return_dict=True,
                                                       cache_dir=cache_dir)
        model.to(device)
        if args.load_checkpoint:
            model.load_state_dict(torch.load(args.load_checkpoint))
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

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

        scaler = None
        if args.fp16:
            from torch.cuda.amp import autocast, GradScaler

            scaler = GradScaler()
        
        if args.do_eval:
            eval_examples = processor.get_dev_examples(os.path.join(args.data_dir, task_name), args.eval_on)
            eval_features = convert_examples_to_features(eval_examples, args.max_seq_length, tokenizer)

            all_input_ids = torch.tensor([f.src_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.trg_ids for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", args.max_train_steps)

        global_step = 0
        best_result = list()
        wrap = False
        progress_bar = tqdm(range(args.max_train_steps))
        for _ in range(int(args.num_train_epochs)):
            train_loss = 0
            num_train_examples = 0
            if wrap: break
            for step, batch in enumerate(train_dataloader):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                src_ids, attention_mask, trg_ids = batch
                if args.mft:
                    src_ids = mask_tokens(src_ids, trg_ids, tokenizer, device, args.mask_mode)## noise probability is 0.2

                if args.fp16:
                    with autocast():
                        outputs = model(input_ids=src_ids,
                                        attention_mask=attention_mask,
                                        labels=trg_ids)
                else:
                    outputs = model(input_ids=src_ids,
                                    attention_mask=attention_mask,
                                    labels=trg_ids)
                loss = outputs.loss

                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                train_loss += loss.item()
                num_train_examples += src_ids.size(0)
                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    progress_bar.update(1)

                if args.do_eval and global_step % args.save_steps == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    def decode(input_ids):
                        return tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)

                    model.eval()
                    eval_loss = 0
                    eval_steps = 0
                    all_inputs, all_labels, all_predictions = [], [], []
                    for batch in tqdm(eval_dataloader, desc="Evaluation"):
                        batch = tuple(t.to(device) for t in batch)
                        src_ids, attention_mask, trg_ids = batch
                        with torch.no_grad():
                            outputs = model(input_ids=src_ids,
                                            attention_mask=attention_mask,
                                            labels=trg_ids)
                            tmp_eval_loss = outputs.loss
                            logits = outputs.logits

                        src_ids = src_ids.tolist()
                        trg_ids = trg_ids.cpu().numpy()
                        eval_loss += tmp_eval_loss.mean().item()
                        _, prd_ids = torch.max(logits, -1)
                        prd_ids = prd_ids.masked_fill(attention_mask == 0, 0).tolist()
                        for s, t, p in zip(src_ids, trg_ids, prd_ids):
                            all_inputs += [decode(s)]
                            all_labels += [decode(t)]
                            all_predictions += [decode(p)]
                        eval_steps += 1
    
                    loss = train_loss / global_step
                    eval_loss = eval_loss / eval_steps
                    p, r, f1, fpr, tp, fp, fn, wp = Metrics.compute(all_inputs, all_labels, all_predictions)
    
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
                        "global_step": global_step,
                        "loss": loss,
                        "eval_loss": eval_loss,
                        "eval_p": p * 100,
                        "eval_r": r * 100,
                        "eval_f1": f1 * 100,
                        "eval_fpr": fpr * 100,
                    }
                    model_to_save = model.module if hasattr(model, "module") else model
                    output_model_file = os.path.join(args.output_dir, "step-%s_f1-%.2f.bin" % (str(global_step), result["eval_f1"]))
                    torch.save(model_to_save.state_dict(), output_model_file)
                    best_result.append((result["eval_f1"], output_model_file))
                    ## sort by f1 and remove model whose f1 is the fourth biggest 
                    best_result.sort(key=lambda x: x[0], reverse=True)
                    if len(best_result) > 3:
                        _, model_to_remove = best_result.pop()
                        os.remove(model_to_remove)

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
                    wrap = True
                    break

    if args.do_test:
        eval_examples = processor.get_test_examples(os.path.join(args.data_dir, task_name), args.test_on)
        eval_features = convert_examples_to_features(eval_examples, args.max_seq_length, tokenizer)

        all_input_ids = torch.tensor([f.src_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.trg_ids for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model = BertForErrorCorrection.from_pretrained(args.load_model_path,
                                                       return_dict=True,
                                                       cache_dir=cache_dir)
        model.to(device)
        if args.load_checkpoint:
            model.load_state_dict(torch.load(args.load_checkpoint))
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        def decode(input_ids):
            return tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)

        model.eval()
        eval_loss = 0
        eval_steps = 0
        all_inputs, all_labels, all_predictions = [], [], []
        for batch in tqdm(eval_dataloader, desc="Evaluation"):
            batch = tuple(t.to(device) for t in batch)
            src_ids, attention_mask, trg_ids = batch
            with torch.no_grad():
                outputs = model(input_ids=src_ids,
                                attention_mask=attention_mask,
                                labels=trg_ids)
                tmp_eval_loss = outputs.loss
                logits = outputs.logits

            src_ids = src_ids.tolist()
            trg_ids = trg_ids.cpu().numpy()
            eval_loss += tmp_eval_loss.mean().item()
            _, prd_ids = torch.max(logits, -1)
            prd_ids = prd_ids.masked_fill(attention_mask == 0, 0).tolist()
            for s, t, p in zip(src_ids, trg_ids, prd_ids):
                all_inputs += [decode(s)]
                all_labels += [decode(t)]
                all_predictions += [decode(p)]
            eval_steps += 1

        eval_loss = eval_loss / eval_steps
        p, r, f1, fpr, tp, fp, fn, wp = Metrics.compute(all_inputs, all_labels, all_predictions)

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
            "eval_loss": eval_loss,
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
