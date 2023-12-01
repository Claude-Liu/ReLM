import os
import json
import logging
from dataclasses import dataclass
import torch
import random

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

'''define Task class'''
@dataclass
class Task:
    id: int
    name: str
    type: str

task_csc = Task(1,'csc','task_classification')
task_tnews = Task(2, 'tnews', 'seq_classification')
task_qmc = Task(3,'afqmc','question-similarity')

'''convert label_template to label_words_ids'''
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
    def __init__(self, guid, text_a, text_b=None, label=None,task=None,input_template=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.task=task
        self.input_template=input_template
    
class InputFeatures(object):
    def __init__(self, input_ids, input_mask, token_type_ids, label_ids, trg_ref_ids, task_id,prompt_mask,active_bits):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids
        self.trg_ref_ids = trg_ref_ids
        self.task_id=task_id ## to indicate the multi-task model what task it is
        self.prompt_mask=prompt_mask
        self.active_bits=active_bits


'''afqmc data process'''
class AfqmcProcessor(object):

    def get_train_examples(self,data_dir,division='base'):
        return self._create_examples(self._read_json(os.path.join(data_dir,"train_base.json")),"train")
    
    def get_dev_examples(self,data_dir,division='base'):
        return self._create_examples(self._read_json(os.path.join(data_dir,"dev_base.json")),"dev")
    
    def get_test_examples(self,data_dir,division='base'):
        return self._create_examples(self._read_json(os.path.join(data_dir,"dev_base.json")),"test")

    def get_labels(self):
            return ["1", "0"]

    def get_label_template(self):
        return {
            "1": ["是"],
            "0": ["否"]
        }
    @staticmethod
    def _read_json(path):
        lines=[]
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                example=json.loads(line)
                lines.append((example["sentence1"],example['sentence2'],example['label']))
        return lines
    
    @staticmethod
    def _create_examples(lines,set_type):
        examples = []
        for i, (sentence1,sentence2,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            task = task_qmc
            input_template = ["<text>","<pt>","<mask>"]
            # input_template = ["<pt>","<mask>","<text>"]
            examples.append(InputExample(guid=guid, text_a=sentence1, text_b=sentence2, label=label, \
                                         task=task, input_template=input_template))
        return examples

'''tnews and iflytek data process'''

class TnewsProcessor(object):
    '''processor for tnews data'''
    def get_train_examples(self,data_dir,division='base'):
        return self._create_examples(self._read_json(os.path.join(data_dir,"train_base.json")),"train")
    
    def get_dev_examples(self,data_dir,division='base'):
        return self._create_examples(self._read_json(os.path.join(data_dir,"dev_base.json")),"dev")
    
    def get_test_examples(self,data_dir,division='base'):
        return self._create_examples(self._read_json(os.path.join(data_dir,"dev_base.json")),"test")
    def get_labels(self):
            return ['100', '101', '102', '103', '104', '106', '107', '108',\
                     '109', '110', '112', '113', '114', '115', '116']

    def get_label_template(self):
        return {
            "100": ["故","事"],
            "101": ["文","化"],
            "102": ["娱","乐"],
            "103": ["体","育"],
            "104":["金","融"],
            "106":["楼","市"],
            "107":["汽","车"],
            "108":["教","育"],
            "109":["科","技"],
            "110":["军","事"],
            "112":["旅","行"],
            "113":["世","界"],
            "114":["股","票"],
            "115":["农","业"],
            "116":["游","戏"],
        }
    @staticmethod
    def _read_json(path):
        lines=[] ## list of (seentence, laebel, label_desc)
        with open(path,'r',encoding='utf-8')as f:
            for line in f:
                example = json.loads(line)
                lines.append((example["sentence"],example["label"],example["label_desc"]))
        return lines
    @staticmethod
    def _create_examples(lines,set_type):
        examples = []
        for i, (src, label, label_desc) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            task = task_tnews
            input_template = ["<text>","<pt>","<mask>"]
            # input_template = ["<pt>","<mask>","<text>"]
            examples.append(InputExample(guid=guid, text_a=src, label=label, task=task, input_template=input_template))
        return examples
    
'''example-->prompt-->feature'''
def seq_convert_examples_to_features(examples, label_list, prompt_length, mask_length, max_seq_length, tokenizer):
    ## convert label to label id in [0,C[
    label_map = {label: i for i, label in enumerate(label_list)}
    print(label_map)
    features = []
    for i, example in enumerate(examples):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        prompt_mask = []
        active_bits = []
        label_ids = [label_map[example.label]]
        task_id = example.task.id
        for phi in example.input_template:
            if phi == "<text>":
                ## when tokenizer: no padding, input have not been split into words
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
                input_ids += encoded["input_ids"]
                attention_mask += encoded["attention_mask"]
                token_type_ids += encoded["token_type_ids"]
                prompt_mask += [0] * len(encoded["input_ids"]) ## 0 signifies this position not prompt
                active_bits += [-100] * len(encoded["input_ids"]) ## -100 signifies this position is not label position
            elif phi == "<pt>":
                input_ids += [tokenizer.sep_token_id] * prompt_length
                attention_mask += [1] * prompt_length
                token_type_ids += [0] * prompt_length
                prompt_mask += [1] * prompt_length  ## 0 signifies this position not prompt
                active_bits += [-100] * prompt_length
            elif phi == "<mask>":
                input_ids += [tokenizer.mask_token_id] * mask_length
                attention_mask += [1] * mask_length
                token_type_ids += [0] * mask_length
                prompt_mask += [0] * mask_length
                active_bits += [1] * mask_length
            else:
                input_ids += [tokenizer.convert_tokens_to_ids(phi)]
                attention_mask += [1]
                token_type_ids += [0]
                prompt_mask += [0]
                active_bits += [-100]
        ## max_length is the final length of features in the batch
        max_length = max_seq_length + prompt_length + 2 ## mask_length = 1 or 2
        if len(attention_mask) < max_length:
            attention_mask += [0] * (max_length - len(attention_mask))
        if len(token_type_ids) < max_length:
            token_type_ids += [0] * (max_length - len(token_type_ids))
        if len(prompt_mask) < max_length:
            prompt_mask += [0] * (max_length - len(prompt_mask))
        if len(input_ids) < max_length:
            input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
        if len(active_bits) < max_length:
            active_bits += [-100] * (max_length - len(active_bits))
        ## same size with label ids in csc task
        if len(label_ids) < max_length:
            label_ids += [0]*(max_length-len(label_ids))
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length
        assert len(prompt_mask) == max_length
        assert len(input_ids) == max_length
        assert len(active_bits) == max_length
        assert len(label_ids) == max_length

        if i < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("input_tokens: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids)))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("prompt_mask: %s" % " ".join([str(x) for x in prompt_mask]))
            logger.info("active_bits: %s" % " ".join([str(x) for x in active_bits]))
            logger.info("label: %s (id = %s)" % (example.label, label_ids[0]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          prompt_mask=prompt_mask,
                          active_bits=active_bits,
                          task_id = task_id,
                          label_ids=label_ids,
                          trg_ref_ids=label_ids)
        )

    return features

'''csc tasks data process'''

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
            task = task_csc
            if len(src) == len(trg):
                examples.append(InputExample(guid=guid, text_a=src, label=trg, task=task))
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
            task = task_csc
            if len(src) == len(trg):
                examples.append(InputExample(guid=guid, text_a=src, label=trg, task=task))
        return examples

def convert_examples_to_prompts(src, trg, prompt_length, max_seq_length, tokenizer, anchor=None, mask_rate=0.2):
    def truncate(x, max_length):
        return x[: max_length]
    ## here max_seq = tokenizer.max_seq_length//2, we need to truncate
    src = truncate(src, max_seq_length-prompt_length)
    trg = truncate(trg, max_seq_length-prompt_length)
    if anchor is not None:
        ##[CLS]...[CLS],x1,x2,...,xn,[anchor_1],...,[anchor_n],[SEP],...,[SEP],m1,m2,...,mn
        prompt_src = [tokenizer.cls_token] * prompt_length + src + anchor + [tokenizer.sep_token] * prompt_length + [tokenizer.mask_token for _ in trg]
        prompt_trg = [tokenizer.cls_token] * prompt_length + src + anchor + [tokenizer.sep_token] * prompt_length + trg
        trg_ref = [tokenizer.cls_token] * prompt_length + trg + anchor + [tokenizer.sep_token] * prompt_length + trg
        block_flag = [1] * prompt_length + [0 for _ in src] + [0 for _ in anchor] + [1] * prompt_length + [0 for _ in trg]
    else:
        ##[CLS]...[CLS],x1,x2,...,xn,[SEP],...,[SEP],m1,m2,...,mn
        prompt_src = [tokenizer.cls_token] * prompt_length + src + [tokenizer.sep_token] * prompt_length + [tokenizer.mask_token for _ in trg]
        prompt_trg = [tokenizer.cls_token] * prompt_length + src + [tokenizer.sep_token] * prompt_length + trg
        trg_ref = [tokenizer.cls_token] * prompt_length + trg + [tokenizer.sep_token] * prompt_length + trg
        block_flag = [1] * prompt_length + [0 for _ in src] + [1] * prompt_length + [0 for _ in trg]

    return prompt_src, prompt_trg, block_flag, trg_ref

def csc_convert_examples_to_features(examples, max_seq_length, tokenizer, prompt_length, anchor=None,mask_rate=0.2):
    features = []
    for i, example in enumerate(examples):
        ## max_seq_length = max_length in sent_class
        src, trg, block_flag, trg_ref = convert_examples_to_prompts(example.text_a, example.label, prompt_length, max_seq_length // 2, tokenizer, anchor, mask_rate)
        example.text_a = src
        example.label = trg
        encoded_inputs = tokenizer(example.text_a,
                                   max_length=max_seq_length,
                                   padding="max_length",## use padding to ensure the final length is max_seq_length
                                   truncation=True,
                                   return_token_type_ids=True,
                                   is_split_into_words=True)

        trg_ids = tokenizer(example.label,
                            max_length=max_seq_length,
                            padding="max_length",
                            truncation=True,
                            return_token_type_ids=True,
                            is_split_into_words=True)["input_ids"]
        trg_ref_ids = tokenizer(trg_ref,
                            max_length=max_seq_length,
                            padding="max_length",
                            truncation=True,
                            return_token_type_ids=True,
                            is_split_into_words=True)["input_ids"]

        src_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"]
        token_type_ids = encoded_inputs["token_type_ids"]
        active_bits = [0] * max_seq_length ## just for batching, no use
        task_id = example.task.id
        block_flag = ([0] + block_flag)[: max_seq_length]
        ## zero padding
        if len(block_flag) < max_seq_length:
            block_flag = block_flag + [0] * max(0, max_seq_length - len(block_flag))

        assert len(src_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(trg_ids) == max_seq_length
        assert len(block_flag) == max_seq_length
        assert len(active_bits) == max_seq_length

        if i < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("src_tokens: %s" % " ".join(example.text_a))
            logger.info("trg_tokens: %s" % " ".join(example.label))
            logger.info("src_ids: %s" % " ".join([str(x) for x in src_ids]))
            logger.info("trg_ids: %s" % " ".join([str(x) for x in trg_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))

        features.append(
                InputFeatures(input_ids=src_ids,
                              input_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              prompt_mask=block_flag,
                              active_bits=active_bits,
                              label_ids=trg_ids,
                              task_id=task_id,
                              trg_ref_ids=trg_ref_ids)
        )
    return features