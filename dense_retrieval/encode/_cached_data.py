

import json

from pyserini.encode import QueryEncoder


class CachedDataQueryEncoder(QueryEncoder):
    def __init__(self, model_name_or_path):
        self.vectors = self._load_from_jsonl(model_name_or_path)

    @staticmethod
    def _load_from_jsonl(path):
        vectors = {}
        with open(path) as f:
            for line in f:
                info = json.loads(line)
                text = info['contents'].strip()
                vec = info['vector']
                vectors[text] = vec
        return vectors

    def encode(self, text, **kwargs):
        return self.vectors[text.strip()]
