from __future__ import absolute_import, division, print_function
import argparse
import json
import logging
import os
import random
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, SubsetRandomSampler
import sklearn.metrics as mtc
from scipy.stats import spearmanr
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import SchedulerType, get_scheduler
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from accelerate import Accelerator


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


with open("data/database.json") as f:
    db = json.load(f)


def genask(cpt):
    starts = []
    ends = []
    move_types = []
    for c in cpt:
        k, v = c.split("_")
        if k == "P":
            q = "Q: What is {}?".format(v)
            a = "A: {}".format(db["pokemon"][v][0])
            starts += [q]
            ends += [a]
        elif k == "I":
            q = "Q: What is {}?".format(v)
            a = "A: {}".format(db["item"][v])
            starts += [q]
            ends += [a]
        elif k == "T":
            target_type = v
        elif k == "M":
            q1 = "Q: What is {}?".format(v)
            a1 = "A: {}".format(db["move"][v][0])
            starts += [q1]
            ends += [a1]
            if len(db["move"][v]) == 2:
                q2 = "Q: What is the effect of {}?".format(v)
                a2 = "A: {}".format(db["move"][v][1])
                starts += [q2]
                ends += [a2]
            if not a1.endswith("no power"):
                move_types += [a1[3:].split()[0]]
    for mt in move_types:
        q = "Q: How effective when {} attacks {}?".format(mt, target_type)
        e = 1
        for tt in target_type.split(" and "):
            e *= db["type"]["{} to {}".format(mt, tt)]
        eft = {0: "Immune", 0.25: "Not effective", 0.5: "Not effective", 1: "Normal", 2: "Effective", 4: "Super effective"}[e]
        a = "A: {}".format(eft)
        starts += [q]
        ends += [a]

    return starts ,ends


class InputExample(object):
    def __init__(self, guid, context, starts=None, ends=None):
        self.guid = guid
        self.context = context
        self.starts = starts
        self.ends = ends


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels


class DataProcessor:

    def get_train_examples(self, input_file, au=False):
        return self._create_examples(
            self._read_jsonl(os.path.join(input_file)), "train", au=au)

    def get_dev_examples(self, input_file):
        return self._create_examples(
            self._read_jsonl(os.path.join(input_file)), "dev", ev=True)
    
    @staticmethod
    def _create_examples(lines, set_type, au=False, ev=False):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            context = line["context"] if "context" in line else ""
            starts = line["start"] if isinstance(line["start"], list) else [line["start"]]
            ends = line["end"] if isinstance(line["end"], list) else [line["end"]]
            au = False if ev else au  
            if au:
                _starts, _ends = genask(line["cpt"])
                random.shuffle(_starts)
                random.shuffle(_ends)
                starts = _starts[:8] + starts
                ends = _ends[:8] + ends
                examples.append(
                    InputExample(guid=guid, context=context, starts=starts, ends=ends))
            examples.append(
                InputExample(guid=guid, context=context, starts=starts[-1:], ends=ends[-1:]))

        return examples

    @classmethod
    def _read_jsonl(cls, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(json.loads(line.strip()))
            return lines


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []
    for i, example in enumerate(examples):
        encoded_inputs = tokenizer(example.context)
        sample_input_ids = encoded_inputs["input_ids"]
        encoded_inputs["labels"] = [-100] * len(sample_input_ids)
        for start, end in zip(example.starts, example.ends):
            start_input_ids = tokenizer(start, add_special_tokens=False)["input_ids"]
            end_input_ids = tokenizer(end, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
            encoded_inputs["input_ids"] += start_input_ids + end_input_ids
            encoded_inputs["labels"] += [-100] * len(start_input_ids) + end_input_ids
        encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])

        offset_length = max_seq_length - len(encoded_inputs["input_ids"])
        encoded_inputs["input_ids"] = [tokenizer.pad_token_id] * offset_length + encoded_inputs["input_ids"]
        encoded_inputs["attention_mask"] = [0] * offset_length + encoded_inputs["attention_mask"]
        encoded_inputs["labels"] = [-100] * offset_length + encoded_inputs["labels"]

        offset_length = min(0, offset_length)
        input_ids = encoded_inputs["input_ids"][-offset_length:]
        attention_mask = encoded_inputs["attention_mask"][-offset_length:]
        labels = encoded_inputs["labels"][-offset_length:]
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
            logger.info("tag: %s" % ("; ".join(example.ends)))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels)
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
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Directory to contain the input data for all tasks.")
    parser.add_argument("--load_model_path", type=str, default="bert-base-uncased",
                        help="Pre-trained language model to load.")
    parser.add_argument("--cache_dir", type=str, default="../cache/",
                        help="Directory to store the pre-trained language models downloaded from s3.")
    parser.add_argument("--output_dir", type=str, default="model/",
                        help="Directory to output predictions and checkpoints.")
    parser.add_argument("--load_ckpt", type=str, default="",
                        help="Checkpoint to load for trianing or evaluation.")

    # Training config
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to evaluate on the dev set.")
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

    processor = DataProcessor()

    cache_dir = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.load_model_path,
                                              do_lower_case=args.do_lower_case,
                                              padding_side="left",
                                              cache_dir=cache_dir)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.do_train:
        train_examples = processor.get_train_examples(os.path.join(args.data_dir, args.train_on), args.doask)
        train_features = convert_examples_to_features(train_examples, args.max_seq_length, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
        all_labels = torch.tensor([f.labels for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_attention_mask, all_labels)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        accelerator = Accelerator(cpu=args.no_cuda, mixed_precision="fp16" if args.fp16 else "no")
        device = accelerator.device

        model = AutoModelForCausalLM.from_pretrained(args.load_model_path,
                                                     cache_dir=cache_dir)
        if args.lora:
            if args.load_ckpt:
                model = PeftModel.from_pretrained(model, args.load_ckpt, is_trainable=True)
            else:
                peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1)
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()

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
            eval_examples = processor.get_dev_examples(os.path.join(args.data_dir, args.eval_on))
            eval_features = convert_examples_to_features(eval_examples, args.max_seq_length, tokenizer)

            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
            all_labels = torch.tensor([f.labels for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_attention_mask, all_labels)
            eval_dataloader = DataLoader(eval_data, shuffle=False, batch_size=args.eval_batch_size)

        model, optimizer, scheduler, train_dataloader = accelerator.prepare(model, optimizer, scheduler, train_dataloader)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", args.max_train_steps)

        global_step = 0
        best_epoch = 0
        best_result = 0.0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            if args.doask:
                train_examples = processor.get_train_examples(os.path.join(args.data_dir, args.train_on), args.doask)
                train_features = convert_examples_to_features(train_examples, args.max_seq_length, tokenizer)

                all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
                all_attention_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
                all_labels = torch.tensor([f.labels for f in train_features], dtype=torch.long)

                train_data = TensorDataset(all_input_ids, all_attention_mask, all_labels)
                train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
                train_dataloader = accelerator.prepare(train_dataloader)

            model.train()
            train_loss = 0
            num_train_examples = 0
            train_steps = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", leave=True)):
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, labels = batch

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss

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

                if global_step >= args.max_train_steps:
                    break

            model_to_save = model.module if hasattr(model, "module") else model
            output_model_file = os.path.join(args.output_dir, "checkpoint_ep-{}".format(epoch + 1))
            if (epoch + 1) % 5 == 0:
                model_to_save.save_pretrained(output_model_file)

            if args.do_eval:
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                model.eval()
                all_predictions, all_labels = [], []
                for batch in tqdm(eval_dataloader, desc="Evaluation"):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, attention_mask, labels = batch
                    with torch.no_grad():
                        outputs = model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        labels=labels)
                        logits = outputs[1]
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        selected_logits = shift_logits[torch.where(shift_labels > 0)]
                        selected_labels = shift_labels[torch.where(shift_labels > 0)]

                    predictions, labels = accelerator.gather_for_metrics((selected_logits.argmax(dim=-1), selected_labels))
                    predictions, labels = predictions.to("cpu").numpy(), labels.to("cpu").numpy()
                    all_predictions.extend(predictions.squeeze().tolist())
                    all_labels.extend(labels.squeeze().tolist())

                output_predict_file = os.path.join(args.output_dir, "predict_results.txt")

                def decode_acc():
                    slacc = tlacc = 0
                    n = m = 0
                    tmp_prediction = []
                    tmp_label = []
                    with open(output_predict_file, "w") as writer:
                        for p, l in zip(all_predictions, all_labels):
                            if l == tokenizer.eos_token_id:
                                writer.write(" -> ".join([tokenizer.decode(tmp_label), tokenizer.decode(tmp_prediction)]) + "\n")
                                if tmp_prediction == tmp_label:
                                    slacc += 1
                                n += 1
                                tlacc += sum([int(c == d) for c, d in zip(tmp_prediction, tmp_label)])
                                m += len(tmp_label)
                                del tmp_prediction[:]
                                del tmp_label[:]
                            else:
                                tmp_prediction += [p]
                                tmp_label += [l]
                    return slacc / n, tlacc / m

                train_epoch_loss = train_loss / len(train_dataloader)
                train_ppl = math.exp(train_epoch_loss)
                acc1, acc2 = decode_acc()

                result = {
                    "global_step": global_step,
                    "train_ppl": train_ppl,
                    "eval_acc": acc1 * 100,
                    "eval_token_acc": acc2 * 100
                }
                if result["eval_acc"] > best_result:
                    best_epoch = epoch
                    best_result = result["eval_acc"]

                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

                def printf():
                    with open(output_eval_file, "a") as writer:
                        writer.write(
                            "Epoch %s: global step = %s | train ppl = %.3f | setence acc = %.2f | token acc = %.2f\n"
                            % (str(epoch),
                               str(result["global_step"]),
                               result["train_ppl"],
                               result["eval_acc"],
                               result["eval_token_acc"]))

                printf()
                for key in sorted(result.keys()):
                    logger.info("Epoch: %s,  %s = %s", str(epoch), key, str(result[key]))
            logger.info("Best epoch: %s, result:  %s", str(best_epoch), str(best_result))

    if not args.do_train and args.do_eval:
        eval_examples = processor.get_dev_examples(os.path.join(args.data_dir, args.eval_on))

        logger.info("***** Generation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", 1)

        predict_model = AutoModelForCausalLM.from_pretrained(args.load_model_path,
                                                             cache_dir=cache_dir)
        predict_model = PeftModel.from_pretrained(predict_model, args.load_ckpt)
        predict_model.print_trainable_parameters()
        predict_model.to(device)

        predict_model.eval()
        output_predict_file = os.path.join(args.output_dir, "test_results.txt")
        acc = n = 0
        with open(output_predict_file, "w") as writer:
            for ex in tqdm(eval_examples, desc="Testing"):
                text = " ".join([ex.context, ex.starts[-1]])
                input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    ot = predict_model.generate(input_ids=input_ids,
                                                max_new_tokens=20,
                                                eos_token_id=tokenizer.eos_token_id)
                                                
                    ot = tokenizer.deocde(ot[0, input_ids.shape[1]:], skip_special_tokens=True)
                    gt = ex.ends[-1]

                    writer.write(" -> ".join([gt, ot]) + "\n")
                    if ot == gt:
                        acc += 1
                    n += 1

        del predict_model

        eval_acc = acc / n
        result = {
            "eval_acc": eval_acc * 100
        }

        for key in sorted(result.keys()):
            logger.info("Epoch: %s,  %s = %s", str(-1), key, str(result[key]))


if __name__ == "__main__":
    main()
