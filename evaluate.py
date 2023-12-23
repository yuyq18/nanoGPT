"""
Evaluate PPL of a trained model
"""
import os
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
import numpy as np
from tqdm import trange

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-lyrics-char' # ignored if init_from is not 'resume'

dataset = 'lyrics_char'
batch_size = 64
block_size = 256 # context of up to 256 previous characters
stride = 64

seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)

# load model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)


# load test data
data_dir = os.path.join('data', dataset)
test_data = np.memmap(os.path.join(data_dir, 'test.bin'), dtype=np.uint16, mode='r')
test_data = torch.stack([torch.from_numpy(test_data.astype(np.int64))])
if device_type == 'cuda':
    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    test_data = test_data.pin_memory().to(device, non_blocking=True)
else:
    test_data = test_data.to(device)

# calculate ppl
nlls = []
prev_end_loc = 0

max_length = block_size
seq_len = len(test_data[0])-1

print(f'evaluating {seq_len} tokens in stride={stride} chunks of {max_length}...')

for begin_loc in trange(0, seq_len, stride):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = test_data[:, begin_loc:end_loc]
    
    target_ids = test_data.clone()[:, begin_loc+1:end_loc+1]
    target_ids[:, :-trg_len] = -1

    with torch.no_grad():
        with ctx:
            logits, loss = model(input_ids, targets=target_ids)
    nlls.append(loss)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())

print(f'PPL = {ppl}')
