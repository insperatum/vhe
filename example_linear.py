from builtins import super
import random
import numpy as np

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.distributions.normal import Normal

from vhe import SimpleVHE 


# Model
x_dim = 10
c_dim = 10
h_dim = 10

class Encoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.enc = nn.Linear(x_dim, h_dim)
		self.mu = nn.Linear(h_dim, c_dim)
		self.sigma = nn.Sequential(nn.Linear(h_dim, c_dim), nn.Softplus())

	def sample(self, inputs):	
		embeddings = [self.enc(x) for x in inputs]
		mean_embedding = sum(embeddings)/len(embeddings)
		mu = self.mu(mean_embedding)
		sigma = self.sigma(mean_embedding)
		return Normal(mu, sigma).rsample()

class Decoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.mu = nn.Linear(c_dim, x_dim)
		self.sigma = nn.Sequential(nn.Linear(c_dim, x_dim), nn.Softplus())

	def sample(self, c):
		return Normal(self.mu(c), self.sigma(c)).rsample()

	def score(self, c, x):
		scores = Normal(self.mu(c), self.sigma(c)).log_prob(x)
		return scores.view(scores.size(0), -1).sum(dim=1)

vhe = SimpleVHE(Encoder(), Decoder())

# Generate dataset
n = 0
classes = []
class_idxs = []
for i in range(100):
	mu = torch.randn(1, x_dim)
	sigma = torch.rand(1, x_dim)
	class_size = random.randint(10,20)
	classes.append(mu + sigma*torch.randn(class_size, x_dim))
	class_idxs.append(list(range(n, n+class_size)))
	n += class_size
data = torch.cat(classes)

def makeBatch(batch_size, n_inputs):
	classes = np.random.choice(range(len(class_idxs)), size=batch_size, p=[len(x)/n for x in class_idxs])
	elems = [np.random.choice(class_idxs[j], size=n_inputs+1) for j in classes]
	x_idx = torch.LongTensor([e[0] for e in elems])
	D_idx = [torch.LongTensor([e[k] for e in elems]) for k in range(1, n_inputs+1)]
	x = Variable(torch.index_select(data, 0, x_idx))
	D = [Variable(torch.index_select(data, 0, d_idx)) for d_idx in D_idx]
	return D, x

# Training
batch_size = 500
n_inputs = 5

optimiser = optim.Adam(vhe.parameters(), lr=0.01)
for i in range(1000):
	optimiser.zero_grad()
	D, x = makeBatch(batch_size, n_inputs)
	score = vhe.score(D, 100, x)
	(-score).backward()
	optimiser.step()
	if i%100==0: print("Iteration", i, "Score", score.item())

test_inputs = [-1+0.5*torch.randn(1,x_dim) for _ in range(n_inputs)]
print("Posterior predictive for", test_inputs)
print(vhe.sample(inputs=test_inputs))
