from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys
import re

import pandas as pd
import numpy as np

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask

def _read_file(filename):
  with open(filename, 'r') as file:
      tokens, labels = [], []
      for line in file:
        tok, lab = line.strip().split()
        tokens.append(tok)
        labels.append(lab)
  return {'token': tokens, 'label': labels}


def readfile(filename, eos_marks=['PERIOD', 'QMARK', 'EXCLAM']):
  df = pd.DataFrame.from_dict(_read_file(filename))
  if filename.endswith('train.txt'):
    data_dir = os.path.dirname(filename)
    qmark_file = os.path.join(data_dir, 'generated/train_qmark.txt')
    colon_file = os.path.join(data_dir, 'generated/train_colon.txt')
    semicolon_file = os.path.join(data_dir, 'generated/train_semicolon.txt')
    exclam_file = os.path.join(data_dir, 'generated/train_exclam.txt')
    if os.path.exists(os.path.join(qmark_file)):
      df = pd.concat([df, pd.DataFrame.from_dict(_read_file(qmark_file))], ignore_index=True, axis=0)
    if os.path.exists(os.path.join(colon_file)):
      df = pd.concat([df, pd.DataFrame.from_dict(_read_file(colon_file))], ignore_index=True, axis=0)
    if os.path.exists(os.path.join(semicolon_file)):
      df = pd.concat([df, pd.DataFrame.from_dict(_read_file(semicolon_file))], ignore_index=True, axis=0)
    if os.path.exists(os.path.join(exclam_file)):
      df = pd.concat([df, pd.DataFrame.from_dict(_read_file(exclam_file))], ignore_index=True, axis=0)
  # else:
  #   df = pd.DataFrame.from_dict(_read_file(filename))

  idx = 0
  n_tokens = len(df)
  paragraphs = []
  token_labels = []
  while idx < n_tokens and idx >= 0:
    step = 128
    sub_df = df.iloc[idx: min(idx+step, n_tokens)]
    end_idx = sub_df[sub_df.label.isin(eos_marks)].tail(1).index
    while end_idx.empty:
      step += 128
      sub_df = df.iloc[idx: min(idx+step, n_tokens)]
      end_idx = sub_df[sub_df.label.isin(eos_marks)].tail(1).index

    if step > 256:
      end_idx = idx + 256
    else:
      end_idx = end_idx.item() + 1
    paragraph_df = df.iloc[idx: end_idx]

    # numeric_labels = paragraph_df.label.apply(lambda l: punctuation_marks.index(l))
    paragraphs.append(paragraph_df.token.values.tolist())
    token_labels.append(paragraph_df.label.values.tolist())
    idx = end_idx
  return list(zip(paragraphs, token_labels))


s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
def remove_accents(input_str):
	s = ''
	for c in input_str:
		if c in s1:
			s += s0[s1.index(c)]
		else:
			s += c
	return s


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class PuncProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ['O', 'PERIOD', 'COMMA', 'COLON', 'QMARK', 'EXCLAM', 'SEMICOLON', '[CLS]', '[SEP]']

    def _create_examples(self,lines,set_type):
        examples = []
        for i,(sentence,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, noise_prob = 0.3, mode = 'eval', add_noise=True):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list,1)}

    features = []
    loop_times = [0, 1] if mode == 'train' else [0]
    for (ex_index,example) in enumerate(examples):
      for t in loop_times:
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        num_to_noise = noise_prob * len(textlist)
        count_noise = 0
        for i, word in enumerate(textlist):
            if add_noise and t == 1:
              if random.random() < noise_prob and count_noise < num_to_noise:
                word = remove_accents(word)
                count_noise += 1
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0,1)
        label_mask.insert(0,1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask))
    return features