# Bharani Nammi
###############################################################################
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import random
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, random_split, SubsetRandomSampler

# Set the random seeds for reproducibility
torch.manual_seed(0)
seed = 0
torch.manual_seed(seed)
random.seed(0)
np.random.seed(0)

# Define the SelfAttention module
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        # Linear transformations for values, keys, and queries
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        # Fully connected layer for output
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        self.attention = None
        self.after_attention_layer = None
        self.after_linear_layer = None
        
    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Reshape values, keys, and queries
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        # Linear transformations
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        e = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        # Attention(Q, K, V)
        attention = torch.softmax(e / (self.embed_size ** (1/2)), dim=3)
        self.attention = attention
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        
        self.after_attention_layer = out
        out = self.fc_out(out)
        self.after_linear_layer = out
        return out

# Define the TransformerBlock module
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(nn.Linear(embed_size, forward_expansion * embed_size),
                                          nn.ReLU(),
                                          nn.Linear(forward_expansion * embed_size, embed_size))
        self.dropout = nn.Dropout(dropout)
            
    def forward(self, values, keys, query):
        attention = self.attention(values, keys, query)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# Define the Encoder module for regression
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, embed_size, padding_idx=0)
        
        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc_out1 = nn.Linear(embed_size, 128)
        self.relu = nn.PReLU()
        self.fc_out2 = nn.Linear(128, 1)
        self.result = None
        self.latent_vector = None
        
    def forward(self, x):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out)
            
        embeddings = out.mean(dim=1)
        self.latent_vector = embeddings
        out = self.fc_out1(out.mean(dim=1))
        self.result = self.fc_out2(self.relu(out))
        out = self.fc_out2(self.relu(out))
        return out, embeddings

# Define the Encoder module for multiclass classification
class EncoderMulticlass(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length, batch_size):
        super(EncoderMulticlass, self).__init__()
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, embed_size, padding_idx=0)
        
        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc_out1 = nn.Linear(embed_size, 128)
        self.relu = nn.PReLU()
        self.fc_out2 = nn.Linear(128, 3)  # Assuming 3 classes for multiclass classification
        self.result = None
        
    def forward(self, x):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out)

        out = self.fc_out1(out.mean(dim=1))
        self.result = self.fc_out2(self.relu(out))
        out = torch.softmax(out, dim=1)  # Softmax activation for multiclass classification
        return out
