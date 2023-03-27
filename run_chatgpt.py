import json
import os
import argparse
import openai
import copy
import logging
from tqdm import *

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

#################chatGPT API########################
class ChatGPT4CSC(object):
    def __init__(self,key_file,message_file=None):
        self.key_file=key_file
        self.openai_key=None
        self.messages={"role": "system", "content": "你是一个中文拼写错误修改助手"}
        self.message_file=message_file##directory to store the chat messages if neccesary
    
    def get_api_key(self):
        with open(self.key_file, 'r', encoding='utf-8') as f:
            self.openai_key = json.load(f)[0]["api_key"]
            openai.api_key=self.openai_key
    
    def gptCorrect(self,src):
        result=openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": "你是一个中文拼写错误修改助手"},
                        {"role": "user", "content": "请修改句子中的拼写错误，要求修改后的句子和原句长度相同, 如果句子中没有错误，请直接照抄原句：{}".format(src)},
                    ]
                )
        return result.get("choices")[0].get("message").get("content") ## the response of chatgpt
    
####################### data processor###################################
class InputExample(object):
    def __init__(self, guid, src, trg):
        self.guid = guid
        self.src = src
        self.trg = trg

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

class Metrics:
    @staticmethod
    def compute(src_sents, trg_sents, prd_sents):
        def difference(src, trg):
            ret = copy.deepcopy(src)
            for i, (src_char, trg_char) in enumerate(zip(src, trg)):
                if src_char!= trg_char:
                    ret[i] = "(" + src_char + "->" + trg_char + ")"

            return "".join(ret)

        pos_sents, neg_sents, tp_sents, fp_sents, fn_sents, prd_pos_sents, prd_neg_sents, wp_sents  = [], [], [], [], [], [], [], []
        ## wp_sents are the positive examples corrected in a wrong way (s!=t&p!=t&p!=s)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key_file",type=str,default="../envs/openai_key",help="the file containing the api key")
    parser.add_argument("--message_file",type=str,default="model/messages.json",help="the file to store the chat messages")
    parser.add_argument("--data_dir", type=str, default="../data/csc/",
                        help="Directory to contain the input data for all tasks.")
    parser.add_argument("--task_name", type=str, default="ecspell",
                        help="Name of the training task.")
    parser.add_argument("--test_on", type=str, default="law",help="Choose a dev set.")
    parser.add_argument("--output_dir", type=str, default="model/",
                        help="Directory to output predictions and checkpoints.")
    parser.add_argument("--begin",type=int,default=None)
    
    args = parser.parse_args()

    processors = {
        "sighan": SighanProcessor,
        "ecspell": EcspellProcessor,
    }
    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)
    processor = processors[task_name]()
    ### initialize the chatgpt############
    chat=ChatGPT4CSC(key_file=args.key_file)
    chat.get_api_key()
    logger.info("api_key: %s",chat.openai_key)
    ### load the data#####################
    test_examples=processor.get_test_examples(os.path.join(args.data_dir,args.task_name),division=args.test_on)##[example(.src,.trg,.guid)]
    all_preds=[]
    all_srcs=[]
    all_trgs=[]
    messages=[]
    for i,example in enumerate(tqdm(test_examples,desc="Test")):
        if args.begin is not None and i<args.begin:
            continue
        try:
            src=example.src##[t1,t2,...,tn]
            trg=example.trg
            prediction=chat.gptCorrect("".join(src))
            if i%100 == 0 or (i%10==0 and i<100):
                logger.info("src: %s \n prediction: %s", "".join(src),prediction)
            all_preds.append(list(prediction))
            all_srcs.append(src)
            all_trgs.append(trg)
            messages.append({"src":"".join(src),"trg":"".join(trg), "pred":prediction})
        except:
            with open(args.message_file, 'w', encoding='utf-8') as f:
                json.dump(messages,f, ensure_ascii=False,indent=4)
            raise
    with open(args.message_file, 'w', encoding='utf-8') as f:
        json.dump(messages,f, ensure_ascii=False,indent=4)
    p, r, f1, fpr, tp, fp, fn, wp = Metrics.compute(all_srcs, all_trgs, all_preds)
    output_tp_file = os.path.join(args.output_dir, "sents.tp")
    with open(output_tp_file, "w", encoding="utf-8") as writer:
        for line in tp:
            writer.write(line + "\n")
    output_fp_file = os.path.join(args.output_dir, "sents.fp")
    with open(output_fp_file, "w",encoding='utf-8') as writer:
        for line in fp:
            writer.write(line + "\n")
    output_fn_file = os.path.join(args.output_dir, "sents.fn")
    with open(output_fn_file, "w",encoding="utf-8") as writer:
        for line in fn:
            writer.write(line + "\n")
    output_wp_file = os.path.join(args.output_dir, "sents.wp")
    with open(output_wp_file, "w",encoding='utf-8') as writer:
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
            "eval precision = %.2f | eval recall = %.2f | eval f1 = %.2f | eval fp rate = %.2f\n"
            % (result["eval_p"],
            result["eval_r"],
            result["eval_f1"],
            result["eval_fpr"]))
        for key in sorted(result.keys()):
            logger.info("%s = %s", key, str(result[key]))
if __name__ == "__main__":
    main()