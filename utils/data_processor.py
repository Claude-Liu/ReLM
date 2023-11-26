import os


class InputExample(object):
    def __init__(self, guid, src, trg):
        self.guid = guid
        self.src = src
        self.trg = trg

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
                examples.append(InputExample(guid=guid, src=src, trg=trg))
        return examples
