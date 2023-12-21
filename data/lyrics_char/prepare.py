import os
import json
import pickle
import numpy as np

# preprocess data
def preprocess_data(data_type=""):
    raw_file_path = os.path.join(os.path.dirname(__file__), 'corpus/lyric_data_for_CL_no_id.jsonl')
    input_file_path = os.path.join(os.path.dirname(__file__), 'corpus/'+data_type+"_input.txt")

    with open(raw_file_path, 'r') as raw_f, open(input_file_path, 'w') as input_f:
        lines = raw_f.readlines()
        if data_type == "":
            for line in lines:
                text = json.loads(line)['lyric']
                input_f.write('\n'.join(text))
                input_f.write('\n')
        else:
            for line in lines:
                data_split = json.loads(line)['dataset_split']
                text = json.loads(line)['lyric']
                if data_split == data_type:
                    input_f.write('\n'.join(text))
                    input_f.write('\n')

for data_type in ["train", "valid", "test", ""]:
    preprocess_data(data_type=data_type)

# load data            
def load_data(data_type=""):
    input_file_path = os.path.join(os.path.dirname(__file__), 'corpus/'+data_type+"_input.txt")
    with open(input_file_path, 'r') as f:
        data = f.read()
    print(f"length of {data_type} dataset in characters: {len(data):,}")
    return data

train_data = load_data(data_type="train")
val_data = load_data(data_type="valid")
test_data = load_data(data_type="test")
data = load_data(data_type="")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
# print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
test_ids = encode(test_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print(f"test has {len(test_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
test_ids.tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)