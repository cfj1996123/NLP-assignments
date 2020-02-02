from os.path import join, exists
import torch


class Dataset:
    def __init__(self, train_path, eval_path, test_path):
        self.word2idx = dict()
        self.idx2word = []
        # tokenize three files, and get its indexing tensors
        self.train = self.tokenize(train_path)
        self.eval = self.tokenize(eval_path)
        self.test = self.tokenize(test_path)
        self.num_words = len(self.idx2word)

    def tokenize(self, path):
        assert exists(path), "path is not valid."
        # construct word2idx and idx2word
        num_tokens = 0
        with open(path, "r") as f:
            for line in f:
                if len(line.strip()) == 0:
                    continue
                words = line.strip().split() + ['<end>']
                num_tokens += len(words)
                for word in words:
                    if word not in self.word2idx:
                        self.idx2word.append(word)
                        self.word2idx[word] = len(self.idx2word)-1
        # create a tensor of indices of all context
        with open(path, "r") as f:
            idx_tensor = torch.empty(num_tokens, dtype=torch.int64)
            token = 0
            for line in f:
                if len(line.strip()) == 0:
                    continue
                words = line.strip().split() + ['<end>']
                for word in words:
                    idx_tensor[token] = self.word2idx[word]
                    token += 1

        return idx_tensor



if __name__ == "__main__":
    data_path = "ptb_lm_small"
    pass