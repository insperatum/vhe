from builtins import super
import random

import torch
from torch import nn, optim
from torch.distributions.normal import Normal

from vhe import VHE, DataLoader, Transform

# This toy example demonstrates how to train a Variational Homoencoder (VHE) for data organised into classes
#
# We generate a synthetic dataset where:
#   Each example x is a pair floats: x = (a,b)
#   Each class contains shifted versions of the same (a,b): [(a+d1, b+d1), (a+d2, b+d2), ...]
#       That is to say, the difference (b-a) is constant for all elements in a class
#
# For the VHE, we use a simple hierarchical linear model:
#  _______
# |       |  c_i ~ Normal(0, 1) for each class i
# |  (c)  |
# | __|__ |
# ||  |  ||
# || (z) ||  z_ij ~ Normal(0, 1) for each element j
# ||  |  ||
# || (x) ||  x_ij ~ Normal( mu = A*[c_i, z_ij] + b,
# ||_____||                 sigma = softplus(C*[c_i, z_ij] + d) )
# |_______|
#
# The encoder network for c is a linear function of the support set D (averaged over elements)
# The encoder network for z is a linear function of x
#
# ---------------------------
#    Writing distributions
# ---------------------------
# Every distribution (encoders, decoders, prior) should be an nn.Module which implements:
#     forward(self, [inputs,] *args, <variable>=None)
# Arguments:
#   First argument is 'inputs' for an encoder (batch * |D| * ...)
#   <variable> is the random variable to be scored/sampled by the distribution
#   args are latent variables which <variable> depends on
# Returns:
#   A tuple: (value, log_prob)
#   If <variable> is None: sample value
#   Otherwise: value = <variable>




## ----- Model definition -----
Dc = 2 # Number of elements in the support set D (i.e. number of elements input to encoder q_c)
x_dim = 2
c_dim = 3
z_dim = 1
h_dim = 10

class Px(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = nn.Linear(c_dim + z_dim, x_dim)
        self.sigma = nn.Sequential(nn.Linear(c_dim + z_dim, x_dim), nn.Softplus())

    def forward(self, c, z, x=None):
        cz = torch.cat([c,z], dim=1)
        dist = Normal(self.mu(cz), self.sigma(cz))
        if x is None: x = dist.rsample()
        return x, dist.log_prob(x).sum(dim=1) # Return value, score

class Qc(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(x_dim, h_dim)
        self.mu_c = nn.Linear(h_dim, c_dim)
        self.sigma_c = nn.Sequential(nn.Linear(h_dim, c_dim), nn.Softplus())

    def forward(self, inputs, c=None):    
        inputs_permuted = inputs.transpose(0,1) # |D| * batch * ... 
        embeddings = [self.enc(x) for x in inputs_permuted]
        mean_embedding = sum(embeddings)/len(embeddings)
        mu_c = self.mu_c(mean_embedding)
        sigma_c = self.sigma_c(mean_embedding)
        dist = Normal(mu_c, sigma_c)
        if c is None: c = dist.rsample()
        return c, dist.log_prob(c).sum(dim=1) # Return value, score

class Qz(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu_z = nn.Linear(x_dim, z_dim)
        self.sigma_z = nn.Sequential(nn.Linear(x_dim, z_dim), nn.Softplus())

    def forward(self, inputs, c, z=None):    
        mu_z = self.mu_z(inputs[:, 0])
        sigma_z = self.sigma_z(inputs[:, 0])
        dist = Normal(mu_z, sigma_z)
        if z is None: z = dist.rsample()
        return z, dist.log_prob(z).sum(dim=1) # Return value, score

vhe = VHE(encoder=[Qc(), Qz()], decoder=Px()) #Use default prior c,z ~ N(0, 1)


## ----- Generate dataset -----
n = 0
def generateClass():
    class_size = random.randint(10,20) # Randomly sample between 10 and 20 examples of each class
    mu = torch.randn(1, x_dim) + 3
    offset = torch.randn(class_size, 1)
    return mu + offset
classes = [generateClass() for i in range(1000)]
data = torch.cat(classes)
class_labels = [i for i in range(len(classes)) for j in range(len(classes[i]))] 


## ----- Data loader creates minibatches of data -----
batch_size = 100
data_loader = DataLoader(
        data = data,
        labels = {"c":class_labels, "z":range(len(data))},
        k_shot = {"c":Dc, "z":1},
        batch_size = batch_size)

## ----- Training -----
nEpochs = 50
optimiser = optim.Adam(vhe.parameters(), lr=1e-3)
for epoch in range(nEpochs):
    for batch in data_loader:
        optimiser.zero_grad()
        score, kl = vhe.score(inputs=batch.inputs, sizes=batch.sizes, x=batch.target, return_kl=True)
        (-score).backward() # Loss = Negative log-likelihood
        optimiser.step()
    print("Epoch %d Score %3.3f KLc %3.3f KLz %3.3f" % (epoch+1, score.item(), kl.c.item(), kl.z.item()))
print("\n" + "-" * 20)

## ----- Testing -----
def pretty(vector): return "[" + ", ".join("%2.2f" % x for x in vector) + "]"
for i in range(3):
    test_class = generateClass()
    test_D = test_class[:Dc].unsqueeze(0) # Batch size 1
    print("\nSupport set D = [" + ", ".join(pretty(x) for x in test_D[0]) + "]" )
    print("Posterior predictive samples:")
    for j in range(3):
        x = vhe.sample(inputs={"c":test_D}).x[0]
        print("  x = " + pretty(x))