import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import re


class SelfAttention(nn.Module):
    def __init__(self, context_size, d_model, query_size, num_heads):
        super().__init__()

        self.context_size = context_size
        self.d_model = d_model
        self.query_size = query_size
        self.num_heads = num_heads

        mask = self._attention_mask(context_size)
        self.register_buffer("mask", mask)

        self.heads = nn.ModuleList()
        # TODO: could probably make this a single matrix multiplication
        for i in range(num_heads):
            self.heads.append(nn.ModuleDict())
            self.heads[i]["WQ"] = nn.Linear(d_model, query_size)
            self.heads[i]["WK"] = nn.Linear(d_model, query_size)
            self.heads[i]["WV"] = nn.Linear(d_model, query_size)
        self.reducing_layer = nn.Linear(query_size*num_heads, d_model)

    def _attention_mask(self, size):
        mask = np.tril(np.ones([size, size]), 0) == 0
        return torch.from_numpy(mask).requires_grad_(False)

    def _calculate_attention(self, Q, K, V):
        scores = (Q@torch.transpose(K, -2, -1)) / np.sqrt(self.d_model)
        p_attn = F.softmax(scores.masked_fill(self.mask, -1e9), dim=-1) @ V
        return F.dropout(p_attn, p=0.1)

    def forward(self, x):
        attention_by_head = []
        for i, head in enumerate(self.heads):
            Q, K, V = head["WQ"](x), head["WK"](x), head["WV"](x)
            attention_by_head.append(self._calculate_attention(Q, K, V))
        attention = torch.cat(attention_by_head, dim=-1)
        return self.reducing_layer(attention)

class Transformer(nn.Module):
    def __init__(self, d_model, context_size, vocab_size, num_layers, num_attention_heads):
        super().__init__()

        pos_encoding = self._get_positional_encoding(context_size, d_model)
        self.register_buffer("pos_encoding", pos_encoding)

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([nn.ModuleDict() for i in range(num_layers)])
        for i in range(num_layers):
            self.layers[i]["attention"] = SelfAttention(context_size, d_model, d_model, num_attention_heads)
            self.layers[i]["norm1"] = nn.LayerNorm(d_model)
            self.layers[i]["feedforward"] = nn.Sequential(
                    nn.Linear(d_model, d_model*4),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model*4, d_model)
                )
            self.layers[i]["norm2"] = nn.LayerNorm(d_model)

        self.final_linear = nn.Linear(d_model, vocab_size)

    def _get_positional_encoding(self, seq_length, dim):
        pos, i = np.mgrid[:seq_length, :((dim+1)//2)]

        angle_rads = pos/(10000**(2*i/dim))

        result = np.empty([seq_length, dim], dtype=np.float32)
        result[:, 0::2] = np.sin(angle_rads)
        if (dim % 2) == 0:
            result[:, 1::2] = np.cos(angle_rads)
        else:
            result[:, 1::2] = np.cos(np.delete(angle_rads, -1, axis=1))

        return torch.from_numpy(result).requires_grad_(False)

    def forward(self, x):
        x = self.embedding(x)

        # this can't be "x += self.pos_encoding" because that's in-place, and pytorch has
        # problems with that
        x = F.dropout(x + self.pos_encoding, p=0.1)

        for layer in self.layers:
            attn_out = F.dropout(layer["attention"](x), p=0.1)
            ff_in = layer["norm1"](attn_out + x)
            ff_out = F.dropout(layer["feedforward"](ff_in), p=0.1)
            x = layer["norm2"](ff_out + ff_in)

        return self.final_linear(x)

# RNN for comparison
class SimpleRNN(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rnn = torch.nn.RNN(d_model, d_model, nonlinearity='relu', batch_first=True)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(self.rnn(self.embedding(x))[0])


def train(data, model, loss_fn, optimizer):
    size = len(data)*len(data[0][0])
    model.train()
    for batch, (X, y) in enumerate(data):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(torch.flatten(pred, 0, 1), torch.flatten(y))

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(data, model, loss_fn):
    num_batches = len(data)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in data:
            pred = model(X)
            test_loss += loss_fn(torch.flatten(pred, 0, 1), torch.flatten(y)).item()
    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")

# load our dataset: the complete works of shakespeare
def load_shakespeare_data(context_size, batch_size, device="cpu"):
    # source: https://www.gutenberg.org/files/100/100-0.txt
    with open("shakespeare.txt") as shakespeare_file:
        raw_text = shakespeare_file.read()

    # tokenize text
    # separate based on spaces and special characters
    # (spaces are removed but special characters are kept as separate tokens)
    words = list(filter(None, re.split(' |(?=\.|\?|\(|\)|,|\.|;|!|\[|\]|")', re.sub("\s+", " ", raw_text))))
    token_map = {word: number for number, word in enumerate(set(words))}
    tokens = torch.tensor([token_map[word] for word in words], dtype=torch.long)

    # trim tokens that can't fit nicely into batches
    data_len = ((len(tokens)-1) // (context_size*batch_size)) * context_size*batch_size
    unshuffled_x_data = torch.reshape(tokens[:data_len], (-1, context_size))
    unshuffled_y_data = torch.reshape(tokens[1:data_len+1], (-1, context_size))

    # shuffle sentences, then batch them
    shuffle_idxs = torch.randperm(len(unshuffled_x_data))
    x_data = torch.reshape(unshuffled_x_data[shuffle_idxs], (-1, batch_size, context_size))
    y_data = torch.reshape(unshuffled_y_data[shuffle_idxs], (-1, batch_size, context_size))

    data = []
    for batch_x, batch_y in zip(x_data, y_data):
        data.append((torch.reshape(batch_x, (batch_size, context_size)).to(device), torch.reshape(batch_y, (batch_size, context_size)).to(device)))

    # TODO: proper k-fold cross validation
    train_cutoff = int(0.8*len(data))
    train_data = data[:train_cutoff]
    test_data = data[train_cutoff:]

    return train_data, test_data, token_map

def greedy_inference(sentence, model, token_map, context_size, device):
    input_tokens = sentence.split()
    encoded_words = torch.zeros(context_size, dtype=torch.long, device=device)

    for i, word in enumerate(input_tokens):
        encoded_words[i] = token_map[word]

    reverse_token_map = {v: k for k, v in token_map.items()}

    model.eval()
    with torch.no_grad():
      for idx in range(len(input_tokens), context_size):
        pred = model(encoded_words)
        encoded_words[idx] = torch.argmax(pred[idx-1]).item()

    decoded_words = [reverse_token_map[token.item()] for token in encoded_words]
    decoded_sentence = ' '.join(decoded_words)

    return decoded_sentence

# reproducible results
torch.manual_seed(1)
np.random.seed(0)

# small values here cause i'm poor
context_size = 10
d_model = 50
batch_size = 32
num_layers = 2
num_attention_heads = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data, test_data, token_map = load_shakespeare_data(context_size, batch_size, device)
vocab_size = len(token_map)

model = Transformer(d_model, context_size, vocab_size, num_layers, num_attention_heads).to(device)
#model = SimpleRNN(d_model, vocab_size).to(device)

# TODO: learning rate schedule
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_data, model, loss_fn, optimizer)
    test(test_data, model, loss_fn)
print("Done!")

print(greedy_inference("Thy heart", model, token_map, context_size, device))

# TODO: save and load weights

