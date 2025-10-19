import math
import os.path
import random
from dataclasses import dataclass
import torch
import numpy as np
import datasets
from pprint import pprint
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, AutoTokenizer
import torch.distributed as dist

from arguments import DataArguments

SPECIAL_LANG_TOKENS = ['[ko]', '[ar]', '[th]', '[bn]', '[sw]', '[id]', '[fi]', '[te]']
SPECIAL_ORDER_TOKENS = ['[SOV]', '[SVO]', '[VSO]']
SPECIAL_DIVERSITY_TOKENS = ['[D_L]', '[D_M]', '[D_H]']
SPECIAL_LENGTH_TOKENS = ['[L_L]', '[L_M]', '[L_H]']

class SameDatasetTrainDataset(Dataset):
    def __init__(self, args: DataArguments, batch_size: int, seed: int, process_index: int = 0, num_processes: int = 1):
        train_datasets = []
        each_data_inxs = []
        batch_size_inxs = []
        pqloss_flag = []
        cur_all_num = 0

        SMALL_THRESHOLD = args.small_threshold
        DROP_THRESHOLD = args.drop_threshold

        context_feat = datasets.Features({
            'query': datasets.Value('string'),
            'pos': datasets.Sequence(datasets.Value('string')),
            'neg': datasets.Sequence(datasets.Value('string')),
            "lang": datasets.Value("string"),
            'order': datasets.Value('string'),
            'diversity': datasets.Value('float64'),
            'length': datasets.Value('int64'),
        })

        assert isinstance(args.train_data, list) and len(args.train_data) >= 1

        for data_dir in args.train_data:
            if not os.path.isdir(data_dir):
                raise FileNotFoundError(f"{data_dir} is a file, not a directionary")

            small_datasets = []
            small_batch_size = math.inf
            flag = 'parallel_' in data_dir

            for file in os.listdir(data_dir):
                if not (file.endswith('.json') or file.endswith('.jsonl')):
                    continue

                file_path = os.path.join(data_dir, file)
                try:
                    temp_dataset = datasets.load_dataset('json', data_files=file_path, split='train',
                                                         cache_dir=args.cache_path, features=context_feat)
                except:
                    continue

                if len(temp_dataset) == 0:
                    continue
                elif len(temp_dataset) < SMALL_THRESHOLD:
                    small_datasets.append(temp_dataset)
                    small_batch_size = min(small_batch_size, self._get_file_batch_size(file, batch_size,
                                                                                      train_group_size=args.train_group_size))
                else:
                    if args.max_example_num_per_dataset is not None and len(temp_dataset) > args.max_example_num_per_dataset:
                        temp_dataset = temp_dataset.select(random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                    train_datasets.append(temp_dataset)
                    each_data_inxs.append(np.arange(len(temp_dataset)) + cur_all_num)
                    cur_all_num += len(temp_dataset)
                    batch_size_inxs.append(self._get_file_batch_size(file, batch_size, train_group_size=args.train_group_size))
                    pqloss_flag.append(flag)

            if len(small_datasets) > 0:
                small_dataset = datasets.concatenate_datasets(small_datasets)
                if len(small_dataset) >= DROP_THRESHOLD:
                    train_datasets.append(small_dataset)
                    each_data_inxs.append(np.arange(len(small_dataset)) + cur_all_num)
                    cur_all_num += len(small_dataset)
                    batch_size_inxs.append(small_batch_size)
                    pqloss_flag.append(flag)

        self.dataset = datasets.concatenate_datasets(train_datasets)
        self.each_data_inxs = each_data_inxs
        self.datasets_inxs = np.arange(len(each_data_inxs))
        self.batch_size_inxs = batch_size_inxs
        self.pqloss_flag = pqloss_flag

        self.process_index = process_index
        self.num_processes = num_processes
        self.args = args
        self.shuffle_ratio = args.shuffle_ratio

        self.deterministic_generator = np.random.default_rng(seed)
        self.step = 0
        self._refresh_epoch()

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.tokenizer.add_tokens(SPECIAL_LANG_TOKENS + SPECIAL_ORDER_TOKENS + SPECIAL_DIVERSITY_TOKENS + SPECIAL_LENGTH_TOKENS, special_tokens=True)

    def _get_file_batch_size(self, file: str, batch_size: int, train_group_size: int):
        return batch_size

    def _refresh_epoch(self):
        print(f'---------------------------*Rank {self.process_index}: refresh data---------------------------')
        self.deterministic_generator.shuffle(self.datasets_inxs)
        batch_datas = []
        for dataset_inx in self.datasets_inxs:
            self.deterministic_generator.shuffle(self.each_data_inxs[dataset_inx])
            cur_batch_size = self.batch_size_inxs[dataset_inx] * self.num_processes
            flag = self.pqloss_flag[dataset_inx]
            for start_index in range(0, len(self.each_data_inxs[dataset_inx]), cur_batch_size):
                if len(self.each_data_inxs[dataset_inx]) - start_index < 2 * self.num_processes:
                    break
                batch_datas.append((self.each_data_inxs[dataset_inx][start_index:start_index + cur_batch_size], flag))
        self.deterministic_generator.shuffle(batch_datas)
        self.batch_datas = batch_datas
        self.step = 0

    def __getitem__(self, _):
        batch_indices, pqloss_flag = self.batch_datas[self.step]
        cur_batch_size = int(len(batch_indices) / self.num_processes)
        batch_indices = batch_indices[self.process_index * cur_batch_size: (self.process_index + 1) * cur_batch_size]
        batch_data = self.dataset[batch_indices]
        self.step += 1
        queries, passages, teacher_scores, neg_queries = self.create_batch_data(batch_raw_data=batch_data)
        return queries, passages, teacher_scores, pqloss_flag, neg_queries

    def shuffle_text(self, text):
        if self.shuffle_ratio > 0 and len(text) > 100 and random.random() < self.shuffle_ratio:
            split_text = []
            chunk_size = len(text) // 3 + 1
            for i in range(0, len(text), chunk_size):
                split_text.append(text[i:i + chunk_size])
            random.shuffle(split_text)
            return " ".join(split_text)
        else:
            return text

    def create_batch_data(self, batch_raw_data):
        queries, passages = [], []
        teacher_scores = []
        neg_queries = []
        for i in range(len(batch_raw_data['query'])):
            if len(batch_raw_data['neg'][i]) == 0:
                continue
            # Token by feature
            div_level = batch_raw_data['diversity'][i]
            len_level = batch_raw_data['length'][i]

            div_token = '[D_H]' if div_level >= 0.998 else '[D_M]' if div_level >= 0.99 else '[D_L]'
            len_token = '[L_H]' if len_level >= 6 else '[L_M]' if len_level >= 4 else '[L_L]'

            lang_token = f"[{batch_raw_data['lang'][i][:2]}]"
            order_token = f"[{batch_raw_data['order'][i]}]"
            queries.append(f"{lang_token} {order_token} {div_token} {len_token} {batch_raw_data['query'][i]}")

            pos_inx = random.choice(list(range(len(batch_raw_data['pos'][i]))))
            passages.append(self.shuffle_text(batch_raw_data['pos'][i][pos_inx]))
            if 'pos_scores' in batch_raw_data and batch_raw_data['pos_scores'][i] is not None:
                teacher_scores.append(batch_raw_data['pos_scores'][i][pos_inx])

            neg_inx_set = list(range(len(batch_raw_data['neg'][i])))
            if len(batch_raw_data['neg'][i]) < self.args.train_group_size - 1:
                num = math.ceil((self.args.train_group_size - 1) / len(batch_raw_data['neg'][i]))
                neg_inxs = random.sample(neg_inx_set * num, self.args.train_group_size - 1)
            else:
                neg_inxs = random.sample(neg_inx_set, self.args.train_group_size - 1)

            if 'neg_scores' in batch_raw_data and batch_raw_data['neg_scores'][i] is not None:
                neg_scores = [(x, batch_raw_data['neg_scores'][i][x]) for x in neg_inxs]
                neg_scores = sorted(neg_scores, key=lambda x: x[1], reverse=True)
                neg_inxs = [x[0] for x in neg_scores]
                teacher_scores.extend([x[1] for x in neg_scores])

            negs = [batch_raw_data['neg'][i][x] for x in neg_inxs]
            passages.extend(negs)

            if 'neg_q' in batch_raw_data and batch_raw_data['neg_q'][i] is not None:
                neg_queries = [batch_raw_data['neg_q'][i][x] for x in neg_inxs]

            if len(teacher_scores) > 0 and len(passages) > 0:
                assert len(teacher_scores) == len(passages)

        if self.args.query_instruction_for_retrieval is not None:
            queries = [self.args.query_instruction_for_retrieval + q for q in queries]
        if self.args.passage_instruction_for_retrieval is not None:
            passages = [self.args.passage_instruction_for_retrieval + p for p in passages]

        if len(teacher_scores) == 0:
            teacher_scores = None
        return queries, passages, teacher_scores, neg_queries


    def __len__(self):
        return len(self.batch_datas) * self.num_processes


@dataclass
class EmbedCollator(DataCollatorWithPadding):

    query_max_len: int = 32
    passage_max_len: int = 128

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]

        teacher_scores = None
        if len(features[0]) > 2:
            teacher_scores = [f[2] for f in features]
            if teacher_scores[0] is None:
                teacher_scores = None
            else:
                teacher_scores = torch.FloatTensor(teacher_scores)

        flag = None
        if len(features[0]) == 4:
            flag = [f[3] for f in features][0]

        neg_q_collated = None
        if len(features[0][4]) > 0:
            neg_q = [f[4] for f in features]
            if isinstance(query[0], list):
                neg_q = sum(neg_q, [])
                neg_q_collated = self.tokenizer(
                    neg_q,
                    # padding='max_length',   
                    padding=True,
                    truncation=True,
                    max_length=self.passage_max_len,
                    return_tensors="pt",
                )

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        q_collated = self.tokenizer(
            query,
            # padding='max_length',  
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            # padding='max_length',   
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        if teacher_scores is not None:
            teacher_scores = teacher_scores.reshape((len(q_collated['input_ids']), -1))
        return {"query": q_collated, "passage": d_collated, "teacher_scores": teacher_scores, "bi_directions": flag,
                "neg_q": neg_q_collated}
