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
from transformers import AutoTokenizer, AutoModelForMaskedLM
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
        self.model_type = self.config.model_type.split("-")[0]
        self.word_embeddings = getattr(self.model, self.model_type).embeddings.word_embeddings

        self.hidden_size = self.config.embedding_size if hasattr(self.config, "embedding_size") else self.config.hidden_size
        self.prompt_embeddings = nn.Embedding(self.prompt_length, self.hidden_size)
        self.prompt_lstm = nn.LSTM(input_size=self.hidden_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=2,
                                   bidirectional=True,
                                   batch_first=True)
        self.prompt_linear = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_size, self.hidden_size))

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        prompt_mask=None,
        active_bits=None,
        labels=None
    ):
        inputs_embeds = self.word_embeddings(input_ids)
        replace_embeds = self.prompt_embeddings(torch.LongTensor(list(range(self.prompt_length))).to(input_ids.device))
        replace_embeds = replace_embeds.unsqueeze(0)

        replace_embeds = self.prompt_lstm(replace_embeds)[0]
        replace_embeds = self.prompt_linear(replace_embeds).squeeze()

        blocked_indices = (prompt_mask == 1).nonzero().reshape((input_ids.shape[0], self.prompt_length, 2))[:, :, 1]
        for i in range(input_ids.shape[0]):
            for j in range(blocked_indices.shape[1]):
                inputs_embeds[i, blocked_indices[i, j], :] = replace_embeds[j, :]

        logits = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]
        logits = logits[torch.where(active_bits != -100)]

        label_words_logits = logits[:, self.label_words_ids]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(label_words_logits.view(-1, self.num_labels), labels.view(-1))

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
        with open(input_file, 'r', encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='\t', quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class SstProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ['0', '1']

    def get_label_template(self):
        return {
            "0": ["bad"],
            "1": ["good"]
        }

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = line[-1]
                label = '0'
            else:
                text_a = line[-2]
                label = line[-1]
            input_template = ["<pt>", "<mask>", "<text>"]
            output_template = ["<pt>", "bad" if label == "0" else "good", "<text>"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, input_template=input_template, output_template=output_template))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ['0', '1']

    def get_label_template(self):
        return {
            "0": ["bad"],
            "1": ["good"]
        }

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if set_type == "test" and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = line[-1]
                label = '0'
            else:
                text_a = line[-1]
                label = line[-3]
            input_template = ["<pt>", "<mask>", "<text>"]
            output_template = ["<pt>", "bad" if label == "0" else "good", "<text>"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, input_template=input_template, output_template=output_template))
        return examples


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ['0', '1']

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
            if set_type == "test":
                text_a = line[-2]
                text_b = line[-1]
                label = '0'
            else:
                text_a = line[-2]
                text_b = line[-1]
                label = line[0]
            input_template = ["<pt>", "<mask>", "<text>"]
            output_template = ["<pt>", "yes" if label == "1" else "no", "<text>"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, input_template=input_template, output_template=output_template))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, cat='m'):
        if cat == 'm':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")
        else:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    def get_test_examples(self, data_dir, cat='m'):
        if cat == 'm':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched")
        else:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test_mismatched")

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    def get_label_template(self):
        return {
            "contradiction": ["no"],
            "entailment": ["yes"],
            "neutral": ["uncertain"]
        }

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if set_type == "test_matched" or set_type == "test_mismatched":
                text_a = line[-2]
                text_b = line[-1]
                label = "contradiction"
            elif set_type == "dev_matched" or set_type == "dev_mismatched":
                text_a = line[-8]
                text_b = line[-7]
                label = line[-1]
            else:
                text_a = line[-4]
                text_b = line[-3]
                label = line[-1]
            input_template = ["<pt>", "<text>", "<mask>"]
            if label == "entailment":
                output_template = ["<pt>", "<text>", "yes"]
            elif label == "contradiction":
                output_template = ["<pt>", "<text>", "no"]
            else:
                output_template = ["<pt>", "<text>", "uncertain"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, input_template=input_template, output_template=output_template))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ['0', '1']

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
            try:
                if set_type == "test":
                    text_a = line[-2]
                    text_b = line[-1]
                    label = '0'
                else:
                    if set_type == "train" and (i == 310122 or i == 362226):
                        continue
                    text_a = line[-3]
                    text_b = line[-2]
                    label = line[-1]
                    if label not in ['0', '1']:
                        continue
            except IndexError:
                continue
            input_template = ["<pt>", "<mask>", "<text>"]
            output_template = ["<pt>", "yes" if label == "1" else "no", "<text>"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, input_template=input_template, output_template=output_template))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["entailment", "not_entailment"]

    def get_label_template(self):
        return {
            "entailment": ["yes"],
            "not_entailment": ["no"]
        }

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = line[-2]
                text_b = line[-1]
                label = "entailment"
            else:
                text_a = line[-3]
                text_b = line[-2]
                label = line[-1]
            input_template = ["<pt>", "<mask>", "<text>"]
            output_template = ["<pt>", "yes" if label == "entailment" else "no", "<text>"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, input_template=input_template, output_template=output_template))
        return examples


class StsProcessor(DataProcessor):
    """Processor for the STS-B data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ['0']

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if set_type == "test":
                text_a = line[-2]
                text_b = line[-1]
                label = 0.
            else:
                text_a = line[-3]
                text_b = line[-2]
                label = float(line[-1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ['0', '1']

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = line[-2]
                text_b = line[-1]
                label = '0'
            else:
                text_a = line[-3]
                text_b = line[-2]
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ImdbProcessor(DataProcessor):
    """Processor for the IMdB data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ['0', '1']

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[-2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class AnliProcessor(DataProcessor):
    """Processor for the Adversarial NLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            try:
                text_a = line[-3]
                text_b = line[-2]
                label = line[-1]
                if label not in ["contradiction", "entailment", "neutral"]:
                    continue
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class PawsqqpProcessor(DataProcessor):
    """Processor for the PAWS-QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_and_test.tsv")), "test")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_and_test.tsv")), "test")

    def get_labels(self):
        return ['0', '1']

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            try:
                text_a = line[-3]
                text_b = line[-2]
                label = line[-1]
                if label not in ['0', '1']:
                    continue
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class PawswikiProcessor(DataProcessor):
    """Processor for the PAWS-WIKI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ['0', '1']

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
            if label not in ['0', '1']:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SnliProcessor(DataProcessor):
    """Processor for the SNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = line[-2]
                text_b = line[-1]
                label = "entailment"
            else:
                text_a = line[-2]
                text_b = line[-1]
                label = line[-3]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SickProcessor(DataProcessor):
    """Processor for the SICK data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

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
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["entailment", "not_entailment"]

    def get_label_template(self):
        return {
            "entailment": ["yes"],
            "not_entailment": ["no"]
        }

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = line[-2]
                text_b = line[-1]
                label = "entailment"
            else:
                text_a = line[-3]
                text_b = line[-2]
                label = line[-1]
            input_template = ["<pt>", "<mask>", "<text>"]
            output_template = ["<pt>", "yes" if label == "entailment" else "no", "<text>"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, input_template=input_template, output_template=output_template))
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
        label_id = label_to_id[example.label]
        for phi, pho in zip(example.input_template, example.output_template):
            if phi == "<text>":
                if example.text_b:
                    encoded = tokenizer(example.text_a,
                                        example.text_b,
                                        max_length=max_seq_length,
                                        padding="max_length",
                                        truncation=True,
                                        return_token_type_ids=True)
                else:
                    encoded = tokenizer(example.text_a,
                                        max_length=max_seq_length,
                                        padding="max_length",
                                        truncation=True,
                                        return_token_type_ids=True)
                input_ids += encoded["input_ids"]
                attention_mask += encoded["attention_mask"]
                token_type_ids += encoded["token_type_ids"]
                prompt_mask += [0] * len(encoded["input_ids"])
                output_ids += encoded["input_ids"]
            elif phi == "<pt>":
                input_ids += [tokenizer.unk_token_id] * prompt_length
                attention_mask += [1] * prompt_length
                token_type_ids += [0] * prompt_length
                prompt_mask += [1] * prompt_length
                output_ids += [tokenizer.unk_token_id] * prompt_length
            elif phi == "<mask>":
                input_ids += [tokenizer.mask_token_id]
                attention_mask += [1]
                token_type_ids += [0]
                prompt_mask += [0]
                output_ids += [tokenizer.convert_tokens_to_ids(pho)]
            else:
                input_ids += [tokenizer.convert_tokens_to_ids(phi)]
                attention_mask += [1]
                token_type_ids += [0]
                prompt_mask += [0]
                output_ids += [tokenizer.convert_tokens_to_ids(phi)]

        max_length = max_seq_length + prompt_length + 2
        if len(input_ids) < max_length:
            input_ids += [0] * (max_length - len(input_ids))
        if len(attention_mask) < max_length:
            attention_mask += [0] * (max_length - len(attention_mask))
        if len(token_type_ids) < max_length:
            token_type_ids += [0] * (max_length - len(token_type_ids))
        if len(prompt_mask) < max_length:
            prompt_mask += [0] * (max_length - len(prompt_mask))
        if len(output_ids) < max_length:
            output_ids += [0] * (max_length - len(output_ids))
        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length
        assert len(prompt_mask) == max_length
        assert len(output_ids) == max_length

        active_bits = [l if _ == tokenizer.mask_token_id else -100 for _, l in zip(input_ids, output_ids)]

        if i < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("input_tokens: %s" % ' '.join(tokenizer.convert_ids_to_tokens(input_ids)))
            logger.info("output_tokens: %s" % ' '.join(tokenizer.convert_ids_to_tokens(output_ids)))
            logger.info("input_ids: %s" % ' '.join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % ' '.join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % ' '.join([str(x) for x in token_type_ids]))
            logger.info("prompt_mask: %s" % ' '.join([str(x) for x in prompt_mask]))
            logger.info("active_bits: %s" % ' '.join([str(x) for x in active_bits]))
            logger.info("label: %s (id = %s)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
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


def mask_tokens(inputs, labels, tokenizer, noise_probability=0.15):
    inputs = inputs.clone()
    labels = labels.clone()
    probability_matrix = torch.full(inputs.shape, noise_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[masked_indices] = inputs[masked_indices]
    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs, labels


def main():
    parser = argparse.ArgumentParser()

    # Data config
    parser.add_argument("--data_dir", type=str, default="../data/glue/",
                        help="Directory to contain the input data for all tasks.")
    parser.add_argument("--task_name", type=str, default="SST-2",
                        help="Name of the training task.")
    parser.add_argument("--load_model_path", type=str, default="bert-base-uncased",
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
    parser.add_argument("--do_test", action="store_true",
                        help="Whether to evaluate on the test set.")
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
    parser.add_argument("--prompt_length", type=int, default=8,
                        help="Maximum total input sequence length after word-piece tokenization.")
    parser.add_argument("--freeze_lm", action="store_true",
                        help="Whether to keep LM parameters frozen.")

    args = parser.parse_args()

    processors = {
        "sst-2": SstProcessor,
        "cola": ColaProcessor,
        "mrpc": MrpcProcessor,
        "mnli": MnliProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "sts-b": StsProcessor,
        "wnli": WnliProcessor,
        "imdb": ImdbProcessor,
        "anli": AnliProcessor,
        "paws-qqp": PawsqqpProcessor,
        "paws-wiki": PawswikiProcessor,
        "snli": SnliProcessor,
        "sick": SickProcessor,
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
        train_examples = processor.get_train_examples(os.path.join(args.data_dir, task_name))
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

        model = AutoModelForMaskedLM.from_pretrained(args.load_model_path,
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

        if args.fp16:
            from torch.cuda.amp import autocast, GradScaler

            scaler = GradScaler()

        if args.do_eval:
            eval_examples = processor.get_dev_examples(os.path.join(args.data_dir, task_name))
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

        global_step = 0
        best_epoch = 0
        best_result = 0.0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            train_loss = 0
            num_train_examples = 0
            train_steps = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, token_type_ids, prompt_mask, active_bits, labels = batch
                
                if args.fp16:
                    with autocast():
                        outputs = model(input_ids=input_ids,
                                        attention_mask=input_mask,
                                        token_type_ids=token_type_ids,
                                        prompt_mask=prompt_mask,
                                        active_bits=active_bits,
                                        labels=labels)
                else:
                    outputs = model(input_ids=input_ids,
                                    attention_mask=input_mask,
                                    token_type_ids=token_type_ids,
                                    prompt_mask=prompt_mask,
                                    active_bits=active_bits,
                                    labels=labels)
                loss = outputs[0]

                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                train_loss += loss.item()
                num_train_examples += input_ids.size(0)
                train_steps += 1
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

                if global_step >= args.max_train_steps:
                    break

            model_to_save = model.module if hasattr(model, "module") else model
            output_model_file = os.path.join(args.output_dir, "{}_pytorch_model.bin".format(epoch))
            torch.save(model_to_save.state_dict(), output_model_file)

            if args.do_eval:
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                model_state_dict = torch.load(output_model_file)
                predict_model = AutoModelForMaskedLM.from_pretrained(args.load_model_path,
                                                                     num_labels=num_labels,
                                                                     return_dict=True,
                                                                     cache_dir=cache_dir)
                predict_model = PTuningWrapper(predict_model, verbalizer, args.prompt_length)
                predict_model.load_state_dict(model_state_dict)
                predict_model.to(device)
                predict_model.eval()
                eval_loss = 0
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
                    if task_name == "sts-b":
                        all_predictions.extend(logits.squeeze().tolist())
                        all_labels.extend(labels.squeeze().tolist())
                    else:
                        all_predictions.extend(np.argmax(logits, axis=1).squeeze().tolist())
                        all_labels.extend(labels.squeeze().tolist())
                    eval_steps += 1
                del predict_model

                loss = train_loss / train_steps
                eval_loss = eval_loss / eval_steps
                if task_name == "cola":
                    eval_mcc = Metrics.mcc(all_predictions, all_labels)
                    eval_acc = eval_mcc * 100
                elif task_name == "sts-b":
                    eval_spc = Metrics.spc(all_predictions, all_labels)
                    eval_acc = eval_spc * 100
                else:
                    eval_acc = Metrics.acc(all_predictions, all_labels)
                    eval_acc = eval_acc * 100

                result = {
                    "global_step": global_step,
                    "loss": loss,
                    "eval_loss": eval_loss,
                    "eval_acc": eval_acc,
                }
                if result["eval_acc"] > best_result:
                    best_epoch = epoch
                    best_result = result["eval_acc"]

                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

                def printf():
                    with open(output_eval_file, 'a') as writer:
                        writer.write(
                            "Epoch %s: global step = %s | train loss = %.3f | eval score = %.2f | eval loss = %.3f\n"
                            % (str(epoch),
                               str(result["global_step"]),
                               result["loss"],
                               result["eval_acc"],
                               result["eval_loss"]))

                printf()
                for key in sorted(result.keys()):
                    logger.info("Epoch: %s,  %s = %s", str(epoch), key, str(result[key]))
            logger.info("Best epoch: %s, result:  %s", str(best_epoch), str(best_result))

    if args.do_test:
        eval_examples = processor.get_test_examples(os.path.join(args.data_dir, task_name))
        eval_features = convert_examples_to_features(eval_examples, label_list, args.prompt_length, args.max_seq_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in eval_features], dtype=torch.long)
        all_prompt_mask = torch.tensor([f.prompt_mask for f in eval_features], dtype=torch.long)
        all_active_bits = torch.tensor([f.active_bits for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_token_type_ids, all_prompt_mask, all_active_bits)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        predict_model = AutoModelForMaskedLM.from_pretrained(args.load_model_path,
                                                             num_labels=num_labels,
                                                             return_dict=True,
                                                             cache_dir=cache_dir)
        predict_model = PTuningWrapper(predict_model, verbalizer, args.prompt_length)
        predict_model.load_state_dict(torch.load(args.load_state_dict))
        predict_model.to(device)
        predict_model.eval()
        predictions = []
        for input_ids, attention_mask, token_type_ids, prompt_mask, active_bits in tqdm(eval_dataloader, desc="Testing"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            prompt_mask = prompt_mask.to(device)
            active_bits = active_bits.to(device)
            with torch.no_grad():
                outputs = predict_model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        prompt_mask=prompt_mask,
                                        active_bits=active_bits)
                logits = outputs[-1]

            logits = logits.detach().cpu().numpy()
            for (i, prediction) in enumerate(logits):
                if task_name == "sts-b":
                    predict_label = prediction[0]
                else:
                    predict_label = np.argmax(prediction)
                predictions.append(predict_label)

        if task_name != "mnli":
            del predict_model
            output_test_file = os.path.join(args.output_dir, "{}.tsv".format(args.task_name))
        else:
            output_test_file = os.path.join(args.output_dir, "{}-m.tsv".format(args.task_name))
        with open(output_test_file, 'w') as writer:
            writer.write("index" + "\t" + "prediction" + "\n")
            for index, pred in enumerate(predictions):
                if task_name == "sts-b":
                    if pred > 5:
                        pred = 5.
                    if pred < 0:
                        pred = 0.
                    writer.write(str(index) + "\t" + str(pred) + "\n")
                else:
                    writer.write(str(index) + "\t" + str(label_list[int(pred)]) + "\n")
        del predictions[:]

        if task_name == "mnli":
            eval_examples = processor.get_test_examples(os.path.join(args.data_dir, task_name), "mm")
            eval_features = convert_examples_to_features(eval_examples, label_list, args.prompt_length, args.max_seq_length, tokenizer)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in eval_features], dtype=torch.long)
            all_prompt_mask = torch.tensor([f.prompt_mask for f in eval_features], dtype=torch.long)
            all_active_bits = torch.tensor([f.active_bits for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_token_type_ids, all_prompt_mask, all_active_bits)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            logger.info("***** Running test *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            for input_ids, attention_mask, token_type_ids, prompt_mask, active_bits in tqdm(eval_dataloader, desc="Testing"):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                prompt_mask = prompt_mask.to(device)
                active_bits = active_bits.to(device)
                with torch.no_grad():
                    outputs = predict_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids,
                                            prompt_mask=prompt_mask,
                                            active_bits=active_bits)
                    logits = outputs[-1]

                logits = logits.detach().cpu().numpy()
                for (i, prediction) in enumerate(logits):
                    predict_label = np.argmax(prediction)
                    predictions.append(predict_label)
            del predict_model

            output_test_file = os.path.join(args.output_dir, "{}-mm.tsv".format(args.task_name))
            with open(output_test_file, 'w') as writer:
                writer.write("index" + "\t" + "prediction" + "\n")
                for index, pred in enumerate(predictions):
                    writer.write(str(index) + "\t" + str(label_list[int(pred)]) + "\n")


if __name__ == "__main__":
    main()
