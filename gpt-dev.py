import torch

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

# Never train on entire dataset, train on chunks at a time
block_size = 8
print(train_data[:block_size + 1])

torch.manual_seed(1337)
batch_size = 4 # Number of concurrent sequences being processed
block_size = 8 # Maximum context length for predictions

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # Generate random integers (points) in the dataset to get chunks from
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Stack creates a tensor with dimensions batch x block
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)