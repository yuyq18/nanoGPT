import os
import torch
import pickle
import tiktoken
import mdtex2html
import gradio as gr
import sentencepiece as spm
from contextlib import nullcontext
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
wandb_run_name = 'unigram-30000'
tokenizer = None
vocab_size = 30000
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

# model
if init_from == 'resume' or init_from.startswith('uer/gpt2'):
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, wandb_run_name+'ckpt.pt')
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

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path) and tokenizer is None
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
elif tokenizer.startswith('bpe') or tokenizer.startswith('unigram'):
    # use the sentencepiece tokenizer
    tokendizer_path = os.path.join('data', checkpoint['config']['dataset'], 'tokenizers/{}/vocab_{}.model'.format(tokenizer, vocab_size))
    sp = spm.SentencePieceProcessor(model_file=tokendizer_path)
    encode = lambda s: sp.encode(s, out_type=int)
    decode = lambda l: sp.decode(l)
elif tokenizer.startswith('uer/gpt2'):
    from transformers import BertTokenizer
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(tokenizer)
    encode = lambda s: tokenizer.encode(s)
    decode = lambda l: tokenizer.decode(l, skip_special_tokens=True, clean_up_tokenization_spaces=True)
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def predict(chatbot, input, max_length, top_k, temperature):
    chatbot.append((parse_text(input), ''))
    start_ids = encode(input)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    with torch.no_grad():
        with ctx:
            for y in model.stream_generate(x, max_length, temperature=temperature, top_k=top_k):
                response = ''.join(decode(y[0].tolist()).split(' '))
                print(parse_text(response))
                chatbot[-1] = (chatbot[-1][0], parse_text(response))
                yield chatbot


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">歌词续写</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear")
            max_length = gr.Slider(0, 512, value=50, step=1.0, label="Maximum length", interactive=True)
            top_k = gr.Slider(1, 21128, value=200, step=1.0, label="Top K", interactive=True)
            temperature = gr.Slider(0, 1, value=0.8, step=0.01, label="Temperature", interactive=True)

    submitBtn.click(predict, [chatbot, user_input, max_length, top_k, temperature],
                    [chatbot], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot], show_progress=True)

demo.queue().launch(share=False, server_name="127.0.0.1", server_port=8501, inbrowser=True)