from collections import Counter


class Multi30kDataset:
    def __init__(self, split='train'):
        self.split = split
        self.samples = []
        self.src_vocab = {"<unk>": 0, "<pad>": 1, "<sos>": 2, "<eos>": 3}
        self.tgt_vocab = {"<unk>": 0, "<pad>": 1, "<sos>": 2, "<eos>": 3}

    def _simple_tokenize(self, text):
        return text.lower().strip().split()

    def build_vocab(self):
        src_counter, tgt_counter = Counter(), Counter()
        for ex in self.samples:
            src_counter.update(self._simple_tokenize(ex["de"]))
            tgt_counter.update(self._simple_tokenize(ex["en"]))

        for w, _ in src_counter.items():
            if w not in self.src_vocab:
                self.src_vocab[w] = len(self.src_vocab)
        for w, _ in tgt_counter.items():
            if w not in self.tgt_vocab:
                self.tgt_vocab[w] = len(self.tgt_vocab)
        return self.src_vocab, self.tgt_vocab

    def process_data(self):
        data = []
        for ex in self.samples:
            src_tokens = ["<sos>"] + self._simple_tokenize(ex["de"]) + ["<eos>"]
            tgt_tokens = ["<sos>"] + self._simple_tokenize(ex["en"]) + ["<eos>"]
            src_ids = [self.src_vocab.get(t, self.src_vocab["<unk>"]) for t in src_tokens]
            tgt_ids = [self.tgt_vocab.get(t, self.tgt_vocab["<unk>"]) for t in tgt_tokens]
            data.append((src_ids, tgt_ids))
        return data
