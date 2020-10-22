import pandas as pd
import tensorflow as tf
import torch
import numpy as np
import time
import datetime
import random
import seaborn as sns
import matplotlib.pyplot as plt
import json
import transformers
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('add  your data path')
df.head()

dftest = pd.get_dummies(df['category'])



df['labels'] = list(zip(dftest.hesap.tolist(), dftest.iade.tolist(), dftest.iptal.tolist(), dftest.kredi.tolist(),  dftest['kredi-karti'].tolist(), dftest['musteri-hizmetleri'].tolist()))
df['text'] = df['text'].apply(lambda x: x.replace('\n', ' '))

df.head()

df.to_csv("dataFrame.csv")

from sklearn.model_selection import train_test_split
train_df, eval_df = train_test_split(df, test_size=0.2)
from simpletransformers.classification import MultiLabelClassificationModel
model = MultiLabelClassificationModel('bert', 'bert-base-multilingual-cased', num_labels=6, args={"output_dir": "/outputs3/", "cache_dir": "/cache3/", "best_model_dir": "/outputs3/best_model/",'train_batch_size':2, 'gradient_accumulation_steps':16, 'learning_rate': 3e-5, 'num_train_epochs': 4, 'max_seq_length': 512,"save_model_every_epoch": True})
print(train_df.head())
model.train_model(train_df)
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
print(result)
print(model_outputs)
predictions, raw_outputs = model.predict(['hesabımı kapatabilir misiniz ?'])
print(predictions)
print(raw_outputs)