import argparse
import gradio as gr
import gc
from torch.utils.tensorboard import SummaryWriter
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import math
from typing import Type


# Define the command-line arguments
parser = argparse.ArgumentParser(description='GPT CLI')
parser.add_argument('--gui', action='store_true', help='Enable Gradio UI mode')
parser.add_argument('--config', default='./gpt_config.json',
                    help='Path to the config file')
subparsers = parser.add_subparsers(dest='command', help='Choose a command')

# Define the training command
train_parser = subparsers.add_parser('train', help='Train the model')
train_parser.add_argument('--load-from-restore', action='store_true',
                          help='Load from restore path instead of training from scratch')

# Define the evaluation command
eval_parser = subparsers.add_parser('eval', help='Evaluate the model')
eval_parser.add_argument('--data', default='./data/evaluation_data.txt',
                         help='Path to the evaluation data file')

# Define the inference command
infer_parser = subparsers.add_parser(
    'infer', help='Generate text from the model')
infer_parser.add_argument('--text', type=str, required=True,
                          help='Input text for generating continuation')
infer_parser.add_argument('--length', type=int,
                          default=100, help='Number of characters to generate')

torch.manual_seed(1337)
# Set device to CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GPTConfig:
    def __init__(self, config_file_path):
        with open(config_file_path, 'r') as f:
            config = json.load(f)

        # Architecture configuration
        architecture_config = config['architecture']
        self.embedding_dim = architecture_config['embedding_dim']
        self.vocab_size = architecture_config['vocab_size']
        self.context_size = architecture_config['context_size']
        self.num_heads = architecture_config['num_heads']
        self.num_layers = architecture_config['num_layers']

        # Training configuration
        training_config = config['training']
        self.batch_size = training_config['batch_size']
        self.training_data_path = training_config['training_data_path']
        self.learning_rate = training_config['learning_rate']
        self.num_steps = training_config['num_steps']
        self.val_interval = training_config['val_interval']

        # Checkpoint restore configuration
        self.restore_path = config['restore_path']


def encode_text(text):
    # Simple dumb ASCII character-level "encoding" since all training data is ASCII.
    # Consider better tokenization if moving off character-level
    return ([ord(t) for t in text])


def decode_text(indices):
    return ([chr(x) for x in indices])


class TextDataset(Dataset):
    def __init__(self, data_tensor, context_size):
        self.data_tensor = data_tensor
        self.context_size = context_size

    def __len__(self):
        return len(self.data_tensor) - self.context_size

    def __getitem__(self, index):
        x = self.data_tensor[index:index + self.context_size]
        y = self.data_tensor[index + 1:index + self.context_size + 1]

        return x, y


def load_dataset(data_path, val, context_size):
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Tensorify data, put it in dataset
    data = torch.tensor(encode_text(text), dtype=torch.int32)

    test_split_idx = int(0.8 * len(data))
    val_split_idx = int(0.9 * len(data))
    train_data = data[:test_split_idx]
    test_data = data[test_split_idx:val_split_idx]
    val_data = data[val_split_idx:]
    # print(f"{len(data)} chars of data")

    train_dataset = TextDataset(train_data, context_size)
    test_dataset = TextDataset(test_data, context_size)
    val_dataset = TextDataset(test_data, context_size)
    return ((train_dataset, test_dataset, val_dataset))


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, device=None, dtype=None):
        super(MultiheadAttention, self).__init__()

        # Save variables
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads

        self.Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)

        # Apply linear layers
        q = self.Q(query)  # [B, C, E]
        k = self.K(key)  # [B, C, E]
        v = self.V(value)  # [B, C, E]

        # Mutate dimensions so the attention matmul can get rid of the inner d_k
        # [batch_size, num_heads, C, d_k]
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # [batch_size, num_heads, C, d_k]
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # [batch_size, num_heads, C, d_k]
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Get raw attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / \
            math.sqrt(self.d_k)  # [B, num_heads, C, C]

        # Apply mask, if necessary
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))

        # Scale by sqrt(k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v  # [B, num_heads, C, d_k]

        # Concat and project
        # Swap C and num_heads, force memory to coalesce, then fuse back num_heads and d_k together
        out = out.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim)
        # Project: give attention "time to think". Maybe this should be part of a different module but whatever
        out = self.out_proj(out)
        return ((out, None))


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return (self.net(x))


class Block(nn.Module):
    """Self-attention"""

    def __init__(self, embed_dim, num_heads, mask, dropout=0.2):
        super(Block, self).__init__()
        self.register_buffer("mask", mask)
        self.head = MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        # self.head = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffwd = FeedForward(embed_dim=embed_dim, dropout=dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Residual connections
        x = self.ln1(x)
        attn_output, _ = self.head(x, x, x, attn_mask=self.mask)
        x = x + attn_output
        out = x + self.ffwd(self.ln2(x))
        return out


class GPT(nn.Module):
    def __init__(self, embedding_dim, vocab_size, context_size):
        super(GPT, self).__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = vocab_size
        self.context_size = context_size

        NUM_HEADS = 4
        NUM_LAYERS = 4

        # Initialize layers
        self.tok_embed = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(context_size, embedding_dim)

        mask = torch.tril(torch.ones(
            self.context_size, self.context_size)).bool()
        mask = ~mask
        self.register_buffer("mask", mask)

        self.blocks = nn.Sequential(
            *[Block(embed_dim=embedding_dim, num_heads=NUM_HEADS, mask=mask, dropout=0.2) for _ in range(NUM_LAYERS)]
        )

        self.ln_f = nn.LayerNorm(self.embedding_dim)
        # Final feed-forward layer from embeddings
        self.ffwd = nn.Linear(
            embedding_dim, out_features=vocab_size, bias=False)

    def forward(self, x):
        tok_embed = self.tok_embed(x)
        pos_embed = self.pos_embed(
            torch.arange(0, self.context_size, device="cuda")
        )
        x = tok_embed + pos_embed

        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.ffwd(x)
        return (logits)

    def infer(self, x):
        with torch.no_grad():
            self.eval()
            res = self.forward(x)
            return (res)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_checkpoint(model, optimizer, path):
    """
    Loads a saved checkpoint file into the model and optimizer.

    Args:
        model (nn.Module): The PyTorch model to load the checkpoint into.
        optimizer (torch.optim.Optimizer): The PyTorch optimizer to load the checkpoint into.
        path (str): The path to the saved checkpoint file.

    Returns:
        Tuple[nn.Module, torch.optim.Optimizer]: The model and optimizer, loaded with the checkpoint state.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return (model, optimizer)


def save_checkpoint(model, optimizer, path, steps):
    """
    Saves a checkpoint of the model and optimizer to disk.

    Args:
        model (nn.Module): The PyTorch model to save the checkpoint of.
        optimizer (torch.optim.Optimizer): The PyTorch optimizer to save the checkpoint of.
        path (str): The path to save the checkpoint file.
        steps (int): The number of training steps that have been completed.

    Returns:
        None
    """
    torch.save({
        'steps': steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def compute_loss(model, criterion, x, y):
    logits = model(x)
    B, C, V = logits.shape
    logits = logits.view(B*C, V)
    y = y.view(B*C)
    loss = F.cross_entropy(logits, y.long())
    return loss


def train(model, optimizer, config: Type[GPTConfig]):
    model = model.to(device)
    criterion = F.cross_entropy

    global_step = 0

    train_dataset, val_dataset = load_dataset(
        config.training_data_path, None, model.context_size)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=512, num_workers=4, shuffle=True)

    model.train()

    EPOCHS = 1
    STEPS = config.num_steps
    VAL_INTERVAL = 100

    writer = SummaryWriter()

    step = 0

    for epoch in range(EPOCHS):
        for data, target in train_dataloader:
            data = data.to(device)
            target = target.to(device)

            loss = compute_loss(model, criterion, data, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar(
                "Loss/train", loss.cpu().detach().numpy(), global_step)
            global_step += 1

            if step % VAL_INTERVAL == 0:
                total_loss = 0
                total_samples = 0

                with torch.no_grad():
                    model.eval()
                    for x, y in test_dataloader:
                        x = x.to(device)
                        y = y.to(device)

                        batch_loss = compute_loss(model, criterion, x, y)
                        total_loss += batch_loss.item() * 512
                        total_samples += 512
                        if total_samples > 10:
                            break

                model.train()
                average_loss = total_loss / total_samples

                print(f"Step {step}; loss: {average_loss}")
                writer.add_scalar("Loss/val", average_loss, global_step)

            step += 1
            if step >= STEPS:
                break

    writer.close()


def evaluate_model(model, val_dataset, block_size=512, max_samples=100000):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = F.cross_entropy

    val_dataloader = DataLoader(
        val_dataset, batch_size=block_size, num_workers=4)
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            batch_loss = compute_loss(model, criterion, inputs, targets)
            total_loss += batch_loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            if total_samples > max_samples:
                break

    average_loss = total_loss / total_samples
    return average_loss


def generate(model, config, prompt, gen_length, temp=1, top_k=10, top_p=None):
    g_cuda = torch.Generator(device=device)
    contexts = torch.tensor(encode_text(prompt), dtype=torch.int32).to(device)

    model.eval()
    for i in range(gen_length):
        transform = nn.LogSoftmax(1)
        x = contexts[-config.context_size:]
        if x.size(0) < config.context_size:
            x = F.pad(x, (config.context_size - x.size(0), 0),
                      "constant", 0).unsqueeze(0)  # B*T
        else:
            x = x.unsqueeze(0)

        preds = model.infer(x)
        preds = preds.squeeze(0)
        preds = preds / temp
        probs = F.softmax(preds, dim=-1)

        if top_p is not None:
            # Apply top-p
            sorted_probs, sorted_indices = torch.sort(
                probs[-1, :], descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # find cutoff
            idx_top_p = (cumulative_probs < top_p).sum().item()
            top_probs = sorted_probs[:idx_top_p]
            top_indices = sorted_indices[:idx_top_p]
            # Null case
            if top_probs.size(0) == 0:
                top_probs = sorted_probs[:1]
                top_indices = sorted_indices[:1]

            next_char = torch.multinomial(
                top_probs, num_samples=1, generator=g_cuda)
            next_char = top_indices[next_char]
        elif top_k is not None:
            top_k_probs, top_k_indices = torch.topk(probs[-1, :], k=top_k)
            next_char = torch.multinomial(
                top_k_probs, num_samples=1, generator=g_cuda)
            next_char = top_k_indices[next_char]
        else:
            next_char = torch.multinomial(
                probs, num_samples=1, generator=g_cuda)

        contexts = torch.cat((contexts, next_char), dim=0)
        print(decode_text(next_char.cpu().numpy())[-1], end="")

    return ("".join(decode_text(contexts.cpu().numpy())))


def main():
    # Parse the command-line arguments
    args = parser.parse_args()

    config = GPTConfig(args.config)
    # Create the GPT model
    model = GPT(
        vocab_size=config.vocab_size,
        context_size=config.context_size,
        embedding_dim=config.embedding_dim
    )
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    if args.gui:
        load_checkpoint(model, optimizer, config.restore_path)
        demo = gr.Interface(
            fn=lambda *args: generate(model, config, *args),
            inputs=[
                gr.Textbox(lines=2, placeholder="Prompt here..."),
                gr.Number(precision=0, value=256),
                gr.Number(value=0.8),
                gr.Slider(maximum=128, value=10),
                gr.Slider(maximum=1, value=1)
            ],
            outputs="text",
            title="Shakespeare-GPT",
            description="Putting theater kids out of their nonexistent jobs since 2023"
        )

        demo.launch()
    elif args.command == "train":
        if args.load_from_restore:
            load_checkpoint(model, optimizer, path)

        train(model, config)
    elif args.command == "eval":
        _, _, test_dataset = load_dataset(
            config.training_data_path, None, model.context_size)
        evaluate_model(model, test_dataset)
    elif args.command == "infer":
        prompt = args.text
        generated_text = generate(model, config, prompt, args.length)
        print(generated_text)


if __name__ == "__main__":
    main()
