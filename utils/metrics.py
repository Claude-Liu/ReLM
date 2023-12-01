import copy
import sklearn.metrics as mtc
from scipy.stats import spearmanr

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
                #print(difference(s, t))
                if p == t:
                    tp_sents.append(difference(s, t))
                if p == s:
                    fn_sents.append(difference(s, t))
                if (p!=t and p!=s):
                    wp_sents.append(difference(s,p))
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
        if len(pos_sents)==0:
            p=0
            r=0
            f1=0
            wpr=0
        else:
            p = 1.0 * len(tp_sents) / len(prd_pos_sents)
            r = 1.0 * len(tp_sents) / len(pos_sents)
            f1 = 2.0 * (p * r) / (p + r + 1e-12)
            wpr = 1.0 * len(wp_sents) / len(pos_sents)
        fpr = 1.0 * (len(fp_sents) + 1e-12) / (len(neg_sents) + 1e-12)

        return p, r, f1, fpr, wpr, tp_sents, fp_sents, fn_sents, wp_sents


