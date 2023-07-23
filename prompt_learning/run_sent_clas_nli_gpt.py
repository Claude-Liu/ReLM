from __future__ import absolute_import, division, print_function
import argparse
import csv
import logging
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import sklearn.metrics as mtc
from scipy.stats import spearmanr
from tqdm import tqdm, trange
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers import SchedulerType, get_scheduler


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class PTuningWrapper(nn.Module):

    def __init__(self, model, verbalizer, prompt_length):
        super().__init__()
        self.config = model.config
        self.prompt_length = prompt_length
        self.verbalizer = verbalizer
        self.label_words_ids = verbalizer.label_words_ids
        self.num_labels = verbalizer.num_labels

        self.model = model
        self.word_embeddings = self.model.transformer.wte

        self.prompt_embeddings = nn.Embedding(self.prompt_length, self.config.hidden_size)
        self.prompt_linear = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.config.hidden_size, self.config.hidden_size))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        prompt_mask=None,
        active_bits=None,
        labels=None,
        inputs_embeds=None,
        output_hidden_states=None
    ):
        inputs_embeds = self.word_embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        replace_embeds = self.prompt_embeddings(torch.LongTensor(list(range(self.prompt_length))).to(inputs_embeds.device))
        replace_embeds = replace_embeds.unsqueeze(0)

        replace_embeds = self.prompt_linear(replace_embeds).squeeze()

        blocked_indices = (prompt_mask == 1).nonzero().reshape((inputs_embeds.shape[0], self.prompt_length, 2))[:, :, 1]

        for i in range(inputs_embeds.shape[0]):
            for j in range(blocked_indices.shape[1]):
                inputs_embeds[i, blocked_indices[i, j], :] = replace_embeds[j, :]

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states
        )
        logits = outputs[0]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_active_bits = active_bits[..., 1:].contiguous()
        logits = shift_logits[torch.where(shift_active_bits != -100)]

        label_words_logits = logits[:, self.label_words_ids]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(label_words_logits.view(-1, self.num_labels), labels.view(-1))

        if output_hidden_states:
            return loss, label_words_logits, outputs[-1]
        return loss, label_words_logits


class Verbalizer(object):
    def __init__(self, label_template, tokenizer):
        self.label_template = label_template
        self.tokenizer = tokenizer
        self.num_labels = len(label_template)
        label_words_ids = []
        for _, words in label_template.items():
            label_words_ids += [[]]
            for w in words:
                label_words_ids[-1] += [tokenizer.convert_tokens_to_ids(w)]
        self.label_words_ids = torch.LongTensor(label_words_ids)


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None, input_template=None, output_template=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.input_template = input_template
        self.output_template = output_template


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, prompt_mask, active_bits, label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.prompt_mask = prompt_mask
        self.active_bits = active_bits
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='\t', quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MnliProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_sick.tsv")), "sick")

    def get_dev_examples(self, data_dir, cat="anli"):
        if cat == "anli":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test_anli.tsv")), "anli")
        elif cat == "sick":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test_sick.tsv")), "sick")
        elif cat == "snli":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test_snli.tsv")), "snli")
        elif cat == "mnli-m":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev_mnli-m.tsv")), "mnli-m")
        elif cat == "mnli-mm":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev_mnli-mm.tsv")), "mnli-mm")
        elif cat == "rte":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev_rte.tsv")), "rte")
        elif cat == "scitail":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test_scitail.tsv")), "scitail")
        elif cat == "mrpc":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev_mrpc.tsv")), "mrpc")
        elif cat == "hans":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test_hans.tsv")), "hans")
        else:
            return NotImplementedError

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    def get_label_template(self):
        return {
            "contradiction": ["no"],
            "entailment": ["yes"],
            "neutral": ["maybe"]
        }

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[-3]
            text_b = line[-2]
            label = line[-1]
            if label == "entailment":
                output_template = ["<pt>", "<text>", "yes"]
            elif label == "contradiction":
                output_template = ["<pt>", "<text>", "no"]
            else:
                output_template = ["<pt>", "<text>", "maybe"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, input_template=output_template, output_template=None))
        return examples


class ColaProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_cola.tsv")), "train-cola")

    def get_dev_examples(self, data_dir, cat="dev-gram"):
        if cat == "cola":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev_cola.tsv")), "dev-cola")
        elif cat == "dev-gram":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev_gram.tsv")), "dev-gram")
        elif cat == "gram":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test_gram.tsv")), "test-gram")
        else:
            return NotImplementedError

    def get_labels(self):
        return ["0", "1"]

    def get_label_template(self):
        return {
            "0": ["no"],
            "1": ["yes"]
        }

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[-2]
            label = line[-1]
            if label == "1":
                output_template = ["<pt>", "<text>", "yes"]
            else:
                output_template = ["<pt>", "<text>", "no"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, input_template=output_template, output_template=None))
        return examples


def convert_examples_to_features(examples, label_list, prompt_length, max_seq_length, tokenizer):
    label_to_id = {label: i for i, label in enumerate(label_list)}
    features = []
    for i, example in enumerate(examples):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        prompt_mask = []
        output_ids = []
        active_bits = []
        label_id = label_to_id[example.label]
        for phi in example.input_template:
            if phi == "<text>":
                if example.text_b:
                    encoded = tokenizer(example.text_a,
                                        example.text_b,
                                        max_length=max_seq_length,
                                        truncation=True,
                                        return_token_type_ids=True)
                else:
                    encoded = tokenizer(example.text_a,
                                        max_length=max_seq_length,
                                        truncation=True,
                                        return_token_type_ids=True)
                attention_mask += encoded["attention_mask"]
                token_type_ids += encoded["token_type_ids"]
                prompt_mask += [0] * len(encoded["input_ids"])
                output_ids += encoded["input_ids"]
                active_bits += [-100] * len(encoded["input_ids"])
            elif phi == "<pt>":
                attention_mask += [1] * prompt_length
                token_type_ids += [0] * prompt_length
                prompt_mask += [1] * prompt_length
                output_ids += [tokenizer.unk_token_id] * prompt_length
                active_bits += [-100] * prompt_length
            elif phi == "<mask>":
                attention_mask += [1]
                token_type_ids += [0]
                prompt_mask += [0]
                output_ids += [tokenizer.convert_tokens_to_ids(phi)]
            else:
                attention_mask += [1]
                token_type_ids += [0]
                prompt_mask += [0]
                output_ids += [tokenizer.convert_tokens_to_ids(phi)]
                active_bits += [tokenizer.convert_tokens_to_ids(phi)]

        max_length = max_seq_length + prompt_length + 2
        if len(attention_mask) < max_length:
            attention_mask += [0] * (max_length - len(attention_mask))
        if len(token_type_ids) < max_length:
            token_type_ids += [0] * (max_length - len(token_type_ids))
        if len(prompt_mask) < max_length:
            prompt_mask += [0] * (max_length - len(prompt_mask))
        if len(output_ids) < max_length:
            output_ids += [tokenizer.eos_token_id] * (max_length - len(output_ids))
        if len(active_bits) < max_length:
            active_bits += [-100] * (max_length - len(active_bits))
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length
        assert len(prompt_mask) == max_length
        assert len(output_ids) == max_length

        if i < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("input_tokens: %s" % " ".join(tokenizer.convert_ids_to_tokens(output_ids)))
            logger.info("input_ids: %s" % " ".join([str(x) for x in output_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("prompt_mask: %s" % " ".join([str(x) for x in prompt_mask]))
            logger.info("active_bits: %s" % " ".join([str(x) for x in active_bits]))
            logger.info("label: %s (id = %s)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=output_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          prompt_mask=prompt_mask,
                          active_bits=active_bits,
                          label_id=label_id)
        )

    return features


class Metrics:
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


def main():
    parser = argparse.ArgumentParser()

    # Data config
    parser.add_argument("--data_dir", type=str, default="nli/",
                        help="Directory to contain the input data for all tasks.")
    parser.add_argument("--task_name", type=str, default="MNLI",
                        help="Name of the training task.")
    parser.add_argument("--load_model_path", type=str, default="gpt2",
                        help="Pre-trained language model to load.")
    parser.add_argument("--cache_dir", type=str, default="../cache/",
                        help="Directory to store the pre-trained language models downloaded from s3.")
    parser.add_argument("--output_dir", type=str, default="model/",
                        help="Directory to output predictions and checkpoints.")
    parser.add_argument("--load_state_dict", type=str, default="",
                        help="Trained model weights to load for evaluation.")

    # Training config
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to evaluate on the dev set.")
    parser.add_argument("--train_on", type=str, default="mnli",
                        help="Choose a training set.")
    parser.add_argument("--eval_on", type=str, default="anli",
                        help="Choose a dev set.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum total input sequence length after word-piece tokenization.")
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=32,
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
    parser.add_argument("--prompt_length", type=int, default=10,
                        help="Maximum total input sequence length after word-piece tokenization.")
    parser.add_argument("--freeze_lm", action="store_true",
                        help="Whether to keep LM parameters frozen.")
    parser.add_argument("--trainer", type=str, default="base",
                        help="Specify a type of training method.")
    parser.add_argument("--adv_steps", type=int, default=2,
                        help="Inner ascent steps for AT.")
    parser.add_argument("--adv_lr", type=float, default=1e-1,
                        help="Step size for AT.")
    parser.add_argument("--adv_max_norm", type=float, default=1e-1,
                        help="Decision boundary for AT.")
    parser.add_argument("--adv_temp", type=float, default=1.0,
                        help="Temperature coefficient for AT.")
    parser.add_argument("--adv_init_var", type=float, default=0.0,
                        help="Temperature coefficient for AT.")
    parser.add_argument("--ppd", type=float, nargs="+", default=[1e-1, 1e-2, 1e-3, 1e-4],
                        help="To do OnceAT if passed.")

    args = parser.parse_args()

    processors = {
        "mnli": MnliProcessor,
        "cola": ColaProcessor,
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
    label_list = processor.get_labels()
    num_labels = len(label_list)

    cache_dir = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.load_model_path,
                                              do_lower_case=args.do_lower_case,
                                              cache_dir=cache_dir)
    verbalizer = Verbalizer(processor.get_label_template(), tokenizer)

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        train_features = convert_examples_to_features(train_examples, label_list, args.prompt_length, args.max_seq_length, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long)
        all_prompt_mask = torch.tensor([f.prompt_mask for f in train_features], dtype=torch.long)
        all_active_bits = torch.tensor([f.active_bits for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_token_type_ids, all_prompt_mask, all_active_bits, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        model = GPT2LMHeadModel.from_pretrained(args.load_model_path,
                                                return_dict=True,
                                                cache_dir=cache_dir)
        model = PTuningWrapper(model, verbalizer, args.prompt_length)
        model.to(device)
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

        if args.freeze_lm:
            prompt_params = ["prompt_"]
            for n, p in model.named_parameters():
                if not any(nd in n for nd in prompt_params):
                    p.requires_grad = False
                    logger.info("Freeze `{}`".format(n))

        if args.do_eval and args.eval_on != "all":
            eval_examples = processor.get_dev_examples(args.data_dir, args.eval_on)
            eval_features = convert_examples_to_features(eval_examples, label_list, args.prompt_length, args.max_seq_length, tokenizer)

            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in eval_features], dtype=torch.long)
            all_prompt_mask = torch.tensor([f.prompt_mask for f in eval_features], dtype=torch.long)
            all_active_bits = torch.tensor([f.active_bits for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_token_type_ids, all_prompt_mask, all_active_bits, all_label_ids)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", args.max_train_steps)

        if args.trainer == "freelb":
            logger.info("  Trainer = %s", "FreeLB")
            from trainer.prompt import FreeLBTrainer
            trainer = FreeLBTrainer(model, optimizer, scheduler, args.max_train_steps, args.gradient_accumulation_steps, args.fp16,
                                    args.adv_steps, args.adv_lr, args.adv_max_norm)
        elif args.trainer == "creat":
            logger.info("  Trainer = %s", "CreAT")
            from trainer.prompt import CreATTrainer
            trainer = CreATTrainer(model, optimizer, scheduler, args.max_train_steps, args.gradient_accumulation_steps, args.fp16,
                                   args.adv_steps, args.adv_lr, args.adv_max_norm, args.adv_temp, args.adv_init_var)
        elif args.trainer == "onceat":
            logger.info("  Trainer = %s", "OnceAT")
            from trainer.freelb_once import FreeLBTrainerForOnce
            trainer = FreeLBTrainerForOnce(model, optimizer, scheduler, args.max_train_steps, args.gradient_accumulation_steps, args.fp16,
                                           args.adv_steps, args.adv_lr, args.ppd)
        elif args.trainer == "smart":
            logger.info("  Trainer = %s", "SMART")
            from trainer.smart import SMARTTrainer
            trainer = SMARTTrainer(model, optimizer, scheduler, args.max_train_steps, args.gradient_accumulation_steps, args.fp16,
                                   args.adv_steps, args.adv_lr, args.adv_max_norm)
        elif args.trainer == "r3f":
            logger.info("  Trainer = %s", "R3F")
            from trainer.prompt import RPLSTrainer
            trainer = RPLSTrainer(model, optimizer, scheduler, args.max_train_steps, args.gradient_accumulation_steps, args.fp16,
                                  args.adv_max_norm)
        elif args.trainer == "rpga":
            logger.info("  Trainer = %s", "RPGA")
            from trainer.prompt import RPGATrainer
            trainer = RPGATrainer(model, optimizer, scheduler, args.max_train_steps, args.gradient_accumulation_steps, args.fp16,
                                  args.adv_max_norm)
        else:
            logger.info("  Trainer = %s", "Base")
            from trainer.prompt import Trainer
            trainer = Trainer(model, optimizer, scheduler, args.max_train_steps, args.gradient_accumulation_steps, args.fp16)

        global_step = 0
        best_epoch = 0
        best_result = 0.0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

            train_loss, train_steps = trainer.step(train_dataloader)
            global_step = trainer.global_step

            model_to_save = model.module if hasattr(model, "module") else model
            output_model_file = os.path.join(args.output_dir, "{}_pytorch_model.bin".format(epoch))
            torch.save(model_to_save.state_dict(), output_model_file)

            if args.do_eval:
                model_state_dict = torch.load(output_model_file)
                predict_model = GPT2LMHeadModel.from_pretrained(args.load_model_path,
                                                                return_dict=True,
                                                                cache_dir=cache_dir)
                predict_model = PTuningWrapper(predict_model, verbalizer, args.prompt_length)
                predict_model.load_state_dict(model_state_dict)
                predict_model.to(device)

                avg_acc = 0.0
                results = {}
                if task_name == "mnli":
                    cats = ["mnli-mm", "rte", "scitail", "hans"] if args.eval_on == "all" else [args.eval_on]
                elif task_name == "cola":
                    cats = ["cola", "gram"] if args.eval_on == "all" else [args.eval_on]
                elif task_name == "sst":
                    cats = ["sst", "yelp", "amazon"] if args.eval_on == "all" else [args.eval_on]
                for eval_on in cats:
                    eval_examples = processor.get_dev_examples(args.data_dir, eval_on)
                    eval_features = convert_examples_to_features(eval_examples, label_list, args.prompt_length, args.max_seq_length, tokenizer)

                    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                    all_input_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
                    all_token_type_ids = torch.tensor([f.token_type_ids for f in eval_features], dtype=torch.long)
                    all_prompt_mask = torch.tensor([f.prompt_mask for f in eval_features], dtype=torch.long)
                    all_active_bits = torch.tensor([f.active_bits for f in eval_features], dtype=torch.long)
                    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

                    eval_data = TensorDataset(all_input_ids, all_input_mask, all_token_type_ids, all_prompt_mask, all_active_bits, all_label_ids)
                    eval_sampler = SequentialSampler(eval_data)
                    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    predict_model.eval()
                    eval_loss = 0
                    num_eval_examples = 0
                    eval_steps = 0
                    all_predictions, all_labels = [], []
                    for batch in tqdm(eval_dataloader, desc="Evaluation"):
                        batch = tuple(t.to(device) for t in batch)
                        input_ids, attention_mask, token_type_ids, prompt_mask, active_bits, labels = batch
                        with torch.no_grad():
                            outputs = predict_model(input_ids=input_ids,
                                                    attention_mask=attention_mask,
                                                    token_type_ids=token_type_ids,
                                                    prompt_mask=prompt_mask,
                                                    active_bits=active_bits,
                                                    labels=labels)
                            tmp_eval_loss = outputs[0]
                            logits = outputs[-1]

                        logits = logits.detach().cpu().numpy()
                        labels = labels.to("cpu").numpy()
                        eval_loss += tmp_eval_loss.mean().item()
                        all_predictions.extend(np.argmax(logits, axis=1).squeeze().tolist())
                        all_labels.extend(labels.squeeze().tolist())
                        num_eval_examples += input_ids.size(0)
                        eval_steps += 1

                    eval_loss = eval_loss / eval_steps
                    if eval_on in ["qnli", "rte", "scitail", "hans", "mrpc"]:
                        all_predictions = [p if p != 2 else 0 for p in all_predictions]
                    if task_name == "cola":
                        eval_acc = Metrics.mcc(all_predictions, all_labels)
                    else:
                        eval_acc = Metrics.acc(all_predictions, all_labels)
                    eval_acc = eval_acc * 100

                    results[eval_on] = eval_acc
                    avg_acc += eval_acc

                logger.info("Epoch: %s,  %s = %s", str(epoch), "avg_acc", str(avg_acc / len(cats)))
                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                with open(output_eval_file, "a") as writer:
                    logger.info("***** Eval results *****")
                    writer.write("Epoch %s:" % str(epoch))
                    for k, v in results.items():
                        writer.write("\tdata = %s | eval score = %.2f" % (k, v))
                    writer.write("\tavg score = %.2f\n" % (avg_acc / len(cats)))

                del predict_model

    if not args.do_train and args.do_eval:
        predict_model = GPT2LMHeadModel.from_pretrained(args.load_model_path,
                                                        return_dict=True,
                                                        cache_dir=cache_dir)
        predict_model = PTuningWrapper(predict_model, verbalizer, args.prompt_length)
        predict_model.load_state_dict(torch.load(args.load_state_dict))
        predict_model.to(device)

        if task_name == "mnli":
            cats = ["mnli-mm", "snli", "rte", "scitail", "hans"] if args.eval_on == "all" else [args.eval_on]
        elif task_name == "cola":
            cats = ["cola", "gram"] if args.eval_on == "all" else [args.eval_on]
        for eval_on in cats:
            eval_examples = processor.get_dev_examples(args.data_dir, eval_on)
            eval_features = convert_examples_to_features(eval_examples, label_list, args.prompt_length, args.max_seq_length, tokenizer)

            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in eval_features], dtype=torch.long)
            all_prompt_mask = torch.tensor([f.prompt_mask for f in eval_features], dtype=torch.long)
            all_active_bits = torch.tensor([f.active_bits for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_token_type_ids, all_prompt_mask, all_active_bits, all_label_ids)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)

            predict_model.eval()
            eval_loss = 0
            num_eval_examples = 0
            eval_steps = 0
            all_predictions, all_labels = [], []
            for batch in tqdm(eval_dataloader, desc="Evaluation"):
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, token_type_ids, prompt_mask, active_bits, labels = batch
                with torch.no_grad():
                    outputs = predict_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids,
                                            prompt_mask=prompt_mask,
                                            active_bits=active_bits,
                                            labels=labels)
                    tmp_eval_loss = outputs[0]
                    logits = outputs[-1]

                logits = logits.detach().cpu().numpy()
                labels = labels.to("cpu").numpy()
                eval_loss += tmp_eval_loss.mean().item()
                all_predictions.extend(np.argmax(logits, axis=1).squeeze().tolist())
                all_labels.extend(labels.squeeze().tolist())
                num_eval_examples += input_ids.size(0)
                eval_steps += 1

            eval_loss = eval_loss / eval_steps
            if eval_on in ["qnli", "rte", "scitail", "hans"]:
                all_predictions = [p if p != 2 else 0 for p in all_predictions]
            if task_name == "cola":
                eval_acc = Metrics.mcc(all_predictions, all_labels)
            else:
                eval_acc = Metrics.acc(all_predictions, all_labels)
            eval_acc = eval_acc * 100

            result = {
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
            }

            for key in sorted(result.keys()):
                logger.info("Epoch: %s,  %s = %s", str(-1), key, str(result[key]))

            output_eval_file = os.path.join(args.output_dir, "test_results.txt")
            with open(output_eval_file, "a") as writer:
                writer.write(
                    "Task %s: eval score = %.2f | eval loss = %.3f\n"
                    % (eval_on,
                       result["eval_acc"],
                       result["eval_loss"]))

        del predict_model


if __name__ == "__main__":
    main()
