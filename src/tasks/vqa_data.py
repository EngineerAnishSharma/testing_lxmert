# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.
VQA_DATA_ROOT = 'data/vqa/'
BIAS_MITIGATED_ROOT = 'data/vqa_bias_mitigated/'  # Path to your preprocessed data
MSCOCO_IMGFEAT_ROOT = 'data/mscoco_imgfeat/'  # Change as per your requirement
SPLIT2NAME = {
    'train': 'train2014',
    'valid': 'val2014',
    'minival': 'val2014',
    'nominival': 'val2014',
    'test': 'test2015',
}

class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """

    def __init__(self, splits: str, test_only=False):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        if not test_only:
            for split in self.splits:
                # Load your preprocessed data
                with open(os.path.join(BIAS_MITIGATED_ROOT, f"preprocessed_vqa_bias_mitigated_{split}.json"), 'r') as f:
                    data = json.load(f)
                    self.data.extend(data['questions'])
            print("Load %d data from split(s) %s." % (len(self.data), self.name))

            # Convert list to dict (for evaluation)
            self.id2datum = {
                datum['question_id']: datum
                for datum in self.data
            }

        # Answers
        self.ans2label = json.load(open(os.path.join(VQA_DATA_ROOT, "trainval_ans2label.json")))
        self.label2ans = json.load(open(os.path.join(VQA_DATA_ROOT, "trainval_label2ans.json")))
        assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)

"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""

class VQATorchDataset(Dataset):
    def __init__(self, dataset: VQADataset, test_only=False):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        # Loading detection features to img_data
        img_data = []

        if not test_only:
             for split in dataset.splits:
                # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
                # It is saved as the top 5K features in val2014_***.tsv
                load_topk = 5000 if (split == 'minival' and topk is None) else topk
                img_data.extend(load_obj_tsv(
                    os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
                    topk=load_topk))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['image_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['image_id']
        ques_id = datum['question_id']
        ques = datum['question']  # Use 'question' which contains the neutralized question

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1 + 1e-5)
        np.testing.assert_array_less(-boxes, 0 + 1e-5)

        # Provide label (target)
        if 'answers' in datum:  # Use 'answers' which is a list of answers
            target = torch.zeros(self.raw_dataset.num_answers)
            for answer in datum['answers']:
                if answer in self.raw_dataset.ans2label:
                    label = self.raw_dataset.ans2label[answer]
                    target[label] += 1
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques

class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)