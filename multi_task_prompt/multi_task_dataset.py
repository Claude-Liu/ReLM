import os
import json
import logging
from dataclasses import dataclass

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


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None,task=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.task=task
    @staticmethod    
    def get_label_list(examples):
        label_list=[]
        for i,example in enumerate(examples):
            if example.label not in label_list:
                label_list.append(example.label)
        return label_list
    
class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids,task_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.task_id=task_id ## to indicate the multi-task model what task it is


'''afqmc data process'''
class AfqmcProcessor(object):

    def get_train_examples(self,data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir,"train.json")),"train")
    
    def get_dev_examples(self,data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir,"dev.json")),"dev")
    
    def get_test_examples(self,data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir,"dev.json")),"test")

    @staticmethod
    def _read_json(path):
        lines=[]
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                example=json.loads(line)
                lines.append(example["sentence1"],example['sentence2'],example['label'])
        return lines
    
    @staticmethod
    def _create_examples(lines,set_type):
        examples = []
        for i, (sentence1,sentence2,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            task = task_qmc
            examples.append(InputExample(guid=guid, text_a=sentence1, text_b=sentence2, label=label, task=task))
        return examples

'''tnews and iflytek data process'''

class TnewsProcessor(object):
    '''processor for tnews data'''
    def get_train_examples(self,data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir,"train.json")),"train")
    
    def get_dev_examples(self,data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir,"dev.json")),"dev")
    
    def get_test_examples(self,data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir,"dev.json")),"test")
    
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
            examples.append(InputExample(guid=guid, text_a=src, label=label, task=task))
        return examples
    

def seq_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    ## convert label to label id in [0,C[
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for i, example in enumerate(examples):
        if example.text_b:
            encoded_inputs = tokenizer(example.text_a,
                                       example.text_b,
                                       max_length=max_seq_length,
                                       padding="max_length",
                                       truncation=True,
                                       return_token_type_ids=True)
            input_ids = encoded_inputs["input_ids"]
            input_mask = encoded_inputs["attention_mask"]
            segment_ids = encoded_inputs["token_type_ids"]
            task_id=task_id
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
        else:
            encoded_inputs = tokenizer(example.text_a,
                                       max_length=max_seq_length,
                                       padding="max_length",
                                       truncation=True,
                                       return_token_type_ids=True)
            input_ids = encoded_inputs["input_ids"]
            input_mask = encoded_inputs["attention_mask"]
            segment_ids = encoded_inputs["token_type_ids"]
            task_id = example.task.id
            tokens = tokenizer.convert_ids_to_tokens(input_ids)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if len(label_list) == 1:
            label_id = example.label
        else:
            label_id = label_map[example.label]
        if i < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % ' '.join(tokens))
            logger.info("input_ids: %s" % ' '.join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % ' '.join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % ' '.join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=label_id,
                          task_id=task_id)
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
                if len(src) == len(trg):
                    examples.append(InputExample(guid=guid, text_a=src, label=trg, task=task))
        return examples
    
def csc_convert_examples_to_features(examples, max_seq_length, tokenizer):
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
        segment_ids = encoded_inputs["token_type_ids"]
        task_id = example.task.id

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
                InputFeatures(input_ids=src_ids,
                              input_mask=attention_mask,
                              segment_ids=segment_ids,
                              label_ids=trg_ids,
                              task_id=task_id)
        )
    return features