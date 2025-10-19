

from transformers import AutoTokenizer

from pyserini.encode import QueryEncoder


class TokFreqQueryEncoder(QueryEncoder):
    def __init__(self, model_name_or_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if model_name_or_path else None

    def encode(self, text, **kwargs):
        vector = {}
        if self.tokenizer is not None:
            tok_list = self.tokenizer.tokenize(text)
        else:
            tok_list = text.strip().split()
        for tok in tok_list:
            if tok not in vector:
                vector[tok] = 1
            else:
                vector[tok] += 1
        return vector
