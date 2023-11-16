import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 

# reads in all the words
words = open("names.txt", 'r').read().splitlines()

# building encoder
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

# set parameters
numb_neurons = 300
data_dim = 20
numb_tests = 100000
block_size = 10 # context, number of characters taken in to predict the next
numb_chars = len(itos)


g = torch.Generator().manual_seed(2147483647)
C = torch.randn((numb_chars, data_dim), generator=g) # embed table, 27 rows for 27 characters, crammed into a two dimensional space(each character has a 2d embedding)
W1 = torch.randn((block_size*data_dim, numb_neurons), generator=g) # first number corresponds to the number of embeddings in emb, which for each data point is 3, 2 = 6. Second num is number of neurons
b1 = torch.randn(numb_neurons, generator=g)
W2 = torch.randn((numb_neurons, numb_chars), generator=g) # input is 100 neurons, output is number of characters
b2 = torch.randn(numb_chars, generator=g) # bias
parameters = [C, b1, W2, b2, W1]

# building the data set
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size # padding with [0] = '.'
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context) # X stores characters before each letter, so ..., ..e, .em, emm
            Y.append(ix) # Y store the letters that follow the context in X, so e, m, m, a
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y



 # training split, dev/validation split, test split
 # 80%, 10%, 10%
import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])


emb = C[Xtr] # for each integer in each row of X, we retrieve the embedding of it in the last dimension, and the letters before it in the second last dimension, shape is 32, 3, 2
           # e.g C[X][13, 2] gives a vector containing two random numbers that are the embeddings for each integer in X, essentially embedding all of X
           # e.g the number 0 in X could be represented by [1.00, 0.674], and [0, 0, 0] in X is [[1.00, 0.674], [1.00, 0.674], [1.00, 0.674]] 

                           # one weight for each number in embedding
h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1) # 100 dimensional activations for each of the data points. tanh to limit number to -1 < num < 1, shape = 32, 100
logits = h @ W2 + b2 #output of the logic net, shape = 32, 27 for 27 for probability of each character in each data point
# counts = logits.exp() # turn into prob
# prob = counts/counts.sum(1, keepdim=True) #normalize so sum of prob space = 1
# loss = -prob[torch.arrange(32), Y].log().mean() # current prob for the correct character in the sequence, then calculates the neg log likelihood
loss = F.cross_entropy(logits, Ytr) # same as above commented out code, but more well behaved for large neg and pos numbers in logits, and it achieves that by
                                    # internally subtracting the every number by the largest neg number in array

for p in parameters:
    p.requires_grad = True

# testing learning rates
# lre = torch.linspace(-3, 0, 1000)
# lrs = 10**lre
# lri = []
lossi = []
stepi = []
# backward pass
for i in range(numb_tests):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (32, )) # list of 32 integers that index into the data, e.g their range is between 0 and number of data inputs,
    

    # forward pass
    emb = C[Xtr[ix]] # (X.shape[0], 3, 2)
    h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)
    logits = h @ W2 + b2 # shape = 32, 27
    loss = F.cross_entropy(logits, Ytr[ix])
    for p in parameters:
        p.grad = None

    loss.backward() # updates gradients for each neuron

    # update
    lr = 0.1 if i < (numb_tests/2) else 0.01
    for p in parameters:
        p.data += -lr * p.grad # adjust data according to gradient
    lossi.append(loss.item())
    stepi.append(i)

# visualizes the places of the embedding, only works for 2D embeddings
def visualize_emb(C):
    plt.figure(figsize=(8, 8))
    plt.scatter(C[:, 0].data, C[:, 1].data, s = 200) # sets x and y to 2d vec value of each letter
    for i in range(C.shape[0]):
        plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha="center", va="center", color='white') # plots each character i in embedding table C
    plt.grid('minor')
    plt.show()

#visualize_emb(C)


# evaluating loss on dev set
def eval_dev(Xdev, Ydev):
    emb = C[Xdev]
    h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)
    logits = h @ W2 + b2 # shape = 32, 27
    loss = F.cross_entropy(logits, Ydev)
    print(loss.item())

print(loss.item())
plt.plot(stepi, lossi)
eval_dev(Xdev, Ydev)
plt.show()

#sampling from model
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])] # (1, block_size, d)
        h = torch.tanh(emb.view(1, -1) @W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))
