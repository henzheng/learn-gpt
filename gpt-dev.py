import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

with open('alpaca_data.json', 'r', encoding='utf-8') as r:
    text = r.read()
    print(text[:1000])
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(''.join(chars))

# Tokenization:
# Character-level model means we are tokenizing by character
# e.g. Google uses SentencePiece (sub-word tokenization) OpenAI uses tiktoken

# Create our lookup tables
# String to int and vice versa
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("Hello world!"))
print(decode(encode("Hello world!")))

# Encoding our dataset and storing it in a torch tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# Splitting our data into train and validation datasets
n = int(0.9*(len(data)))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # Generate random integers (points) in the dataset to get chunks from
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Stack creates a tensor with dimensions batch x block
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# KQV are the key, query, and value matrices

# Embeddings: similar words are assigned to similar points in space
# Can be in any number of dimensions
# Words are initially assigned random embeddings which are adjusted through training.

# Based on context, a word is attracted to other words at varying magnitudes (gravity)
# All words pull other words

# Three methods to measure similarity:
# Generally, greater value = more similar
# In extreme cases, a dot product or cosine similarity of 0 means no correlation

# Measure 1: Dot product
# Calculating the dot product of two matrices gives us a measure of similarity
# Where each number in the matrix may represent a different attribute of the word's meaning
# Dot product of 0 indicates no similarity (points are perpendicular)
# Can result in a negative value as well
 
# Measure 2: Cosine similarity
# Calculating the cosine of the angle between the two vectors from the origin
# Angle can be found with arccosine (using our vectors)
# A value of 1 would indicate the word being compared to itself
# Cos(0) = 1 (No distance between the points)

# Dot product and cosine similarity are alike 
# If vectors are scaled down to length 1 and placed on the unit circle, they are the same
# Same up to a scalar

# Measure 3: Scaled dot product
# Dot product divided by the square root of the length of the vector (number of coordinates or values in the matrix)
# Measure used in attention

# Attention: Adjust position for a word vector based on similarity
# New position is the sum of the word's dot product with each other word in the sentence (add all our similarities together)

# Normalization: similarity values should be normalized to smaller values for easier computation
# We want percentages where the coefficients add to 1.
# Achieve this by dividing by the sum of our coefficients.
# However, this does not account for negative values of similarity, which can result in dividing by 0.

# So we use Softmax: where the coefficient in our calculations is replaced by e^(coefficient)
# This method preserves the magnitude of similarity while making everything positive.
# If you simply made every negative value positive, it ruins the measure of similarity.
# However, e^0 is not the same as a coefficient of 0. In practice, however, it comes out as a negligible number. 
# We use our coefficients as a percent to move them that amount towards the word they were being compared to.

# Get new embeddings from existing ones by applying linear transformations. (Multiplying all values in a matrix by a scalar)
# Keys and Queries matrix helps us find the best embeddings (ones that relate words the best when applying attention)
# K and Q modify our embeddings with matrix multiplication, and then we can calculate similarity as usual
# Knows features of the words (color, size, features), more similar to transformers encoder

# Values matrix (more similar to transformers decoder)
# Multiplies our embedding with K and Q (optimized for finding similarities) by matrix V to create an embedding optimized for generation (finding the next word)
# Another linear transformation 
# Knows when two words can appear in the same context (e.g. apple or orange can both work as the next word given a context)

# Self attention: (Scaled Dot-Product Attention)
# Perform a linear transformation by multiplying our embedding by k and q
# Calculate our similarities
# Apply the Softmax function to our values and then multiply by matrix v to apply another linear transformation 
# Results in an embedding optimized for predicting the next word

# Multi-head attention
# Multiple layers of Scaled Dot-Product Attention
# Many heads (many k,q,v matrices)
# The more layers you use, the better but that means more computing power needed
# At the end, we concatenate the outputs of our layers, resulting in an embedding with a high number of dimensions
# Linear step turns our concatenated embeddings into a more manageable, lower dimension 
# Worse embeddings will be scaled down, better ones will be scaled up

# K, Q, and V come from weights trained with the transformer model

# Feed forward neural network attempts to compute the next word or token.