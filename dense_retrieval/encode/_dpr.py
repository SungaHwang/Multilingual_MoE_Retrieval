

from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer

from pyserini.encode import DocumentEncoder, QueryEncoder


class DprDocumentEncoder(DocumentEncoder):
    def __init__(self, model_name, tokenizer_name=None, device='cuda:0'):
        self.device = device
        self.model = DPRContextEncoder.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer = DPRContextEncoderTokenizer.from_pretrained(tokenizer_name or model_name)

    def encode(self, texts, titles=None,  max_length=256, **kwargs):
        if titles:
            inputs = self.tokenizer(
                titles,
                text_pair=texts,
                max_length=max_length,
                padding='longest',
                truncation=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
        else:
            inputs = self.tokenizer(
                texts,
                max_length=max_length,
                padding='longest',
                truncation=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
        inputs.to(self.device)
        return self.model(inputs["input_ids"]).pooler_output.detach().cpu().numpy()


class DprQueryEncoder(QueryEncoder):
    def __init__(self, model_name: str, tokenizer_name: str = None, device: str = 'cpu'):
        self.device = device
        self.model = DPRQuestionEncoder.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(tokenizer_name or model_name)

    def encode(self, query: str, **kwargs):
        input_ids = self.tokenizer(query, return_tensors='pt')
        input_ids.to(self.device)
        embeddings = self.model(input_ids["input_ids"]).pooler_output.detach().cpu().numpy()
        return embeddings.flatten()
