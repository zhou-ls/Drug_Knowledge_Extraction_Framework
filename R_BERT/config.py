# -*- coding: utf-8 -*-


# dataset
DATA_DIR = "./data/clinical-guidance-drug-drug2"
TRAIN_FILE_PATH = "{}/train.xls".format(DATA_DIR)
TEST_FILE_PATH = "{}/test.xls".format(DATA_DIR)

# model
BERT_MODEL_DIR = "bert_wwm/chinese_L-12_H-768_A-12"
BERT_CONFIG_PATH = "{}/bert_config.json".format(BERT_MODEL_DIR)
BERT_CHECKPOINT_PATH = "{}/bert_model.ckpt".format(BERT_MODEL_DIR)
BERT_VOCAT_PATH = "{}/vocab.txt".format(BERT_MODEL_DIR)
MAX_LEN = 200
EPOCH = 50
BATCH_SIZE = 16
