
import os
import pandas as pd
import pickle
import sys
import random
import numpy as np
import torch
import glob
import configparser
import torch.nn as nn
from cnn_arc import BagOfEmbeddings
from torch.utils.data import SequentialSampler
from collections import OrderedDict
import tokenizer, utils
from torch.utils.data import TensorDataset, DataLoader
os.environ["CUDA_VISIBLE_DEVICES"]='0'
torch.manual_seed(2020)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
random.seed(2020)


cfg = configparser.ConfigParser()
cfg.read(sys.argv[1])

with open(cfg.get('model','config_pickle'), 'rb') as handle:
    config = pickle.load(handle)

model = BagOfEmbeddings(input_vocab_size=config['input_vocab_size'],output_vocab_size=config['output_vocab_size'],embed_dim=config['embed_dim'],hidden_units=config['hidden_units'],dropout_rate=config['dropout_rate'],out_size=config['out_size'],stride=config['out_size'],kernel=config['kernel'],save_config=False)

state_dict = torch.load(cfg.get('model','model_file'))

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)

print("Model Load Complete...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

model.eval()

def getHSP(x):
    x = x.split("/")
    x = x[len(x) - 1]
    return x[:-4]

l = [filename for filename in glob.glob(cfg.get('data','loc') + "*.txt")]

data_df = pd.DataFrame({"FILENAME":l,"CUIS":l})
data_df["FILENAME"] = data_df.FILENAME.apply(lambda x: getHSP(x))
data_df["CUIS"] = data_df.CUIS.apply(lambda x: open(x, "r").read())
print("Reading files complete...")

with open(cfg.get('model','tokenizer_pickle'),'rb') as pickle_file:
    input_tokenizer = pickle.load(pickle_file)
print("Tokenizer loading complete...")

def make_data_loader(input_seqs, model_outputs, batch_size, partition):
    model_inputs = utils.pad_sequences(input_seqs, max_len=12000)

    if type(model_inputs) is tuple:
        tensor_dataset = TensorDataset(*model_inputs, model_outputs)
    else:
        tensor_dataset = TensorDataset(model_inputs, model_outputs)

    if partition == 'train':
        sampler = RandomSampler(tensor_dataset)
    else:
        sampler = SaequentialSampler(tensor_dataset)

    data_loader = DataLoader(
        tensor_dataset,
        sampler=sampler,
        barch_size=batch_size)

    return data_loader


def predict_batch(x):   
    tr_in_seqs = input_tokenizer.texts_to_seqs(x, add_cls_token=True)
    model_inputs = utils.pad_sequences(tr_in_seqs, max_len=12000)
    model.eval()
    if type(model_inputs) is tuple:
        tensor_dataset = TensorDataset(*model_inputs)
    else:
        tensor_dataset = TensorDataset(model_inputs)

    sampler = SequentialSampler(tensor_dataset)
    data_loader = DataLoader(
        tensor_dataset,
        sampler=sampler,
        batch_size=16)

    all_output = torch.tensor([], device=device)
    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        batch_input = batch[0]
        with torch.no_grad():
            logits = model(batch_input)
            all_output = torch.cat((all_output,logits), 0)
    y_pred_prob = torch.sigmoid(all_output).cpu().numpy()
    return y_pred_prob[:,0], y_pred_prob[:,1], y_pred_prob[:,2]


def getPred(x, y):
    if x < float(y):
        return 0
    else:
        return 1

#data_df = data_df.head(5)
data_df_list = data_df["CUIS"].tolist()

ALC, OPI, NOPI = predict_batch(data_df_list)

print("Prediction complete...")

data_df["ALCOHOL_PROB"] = ALC
data_df["OPIOID_PROB"] = OPI
data_df["NON_OPIOID_PROB"] = NOPI

data_df["ALCOHOL_PRED"] = data_df["ALCOHOL_PROB"].apply(lambda x: getPred(x, cfg.get('args','alc_cut')))
data_df["OPIOID_PRED"] = data_df["OPIOID_PROB"].apply(lambda x: getPred(x, cfg.get('args','opi_cut')))

data_df["NON_OPIOID_PRED"] = data_df["NON_OPIOID_PROB"].apply(lambda x: getPred(x, cfg.get('args','nopi_cut')))

data_df = data_df.drop("CUIS", 1)

data_df.to_csv("ML_Substance_Misuse_Prediction_Result.csv", sep=",", index=False)
print("Result output complete...")

print("Thank you for using our multi label substance misuse machine learning model, the output is saved in the same directory")
