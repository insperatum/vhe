**(Note: Code and documentation under development, expect API changes)**

# Variational Homoencoder
This is a simple PyTorch implementation for training a _Variational Homoencoder_, as in the paper:\
[The Variational Homoencoder:
Learning to learn high capacity generative models from few examples](http://auai.org/uai2018/proceedings/papers/351.pdf)

This code is written to be generic, so it should apply easily to different domains and network architectures. It also extends easily to a variety of generative model structures, including the hierarchical and factorial latent variable models shown in the paper. The code covers the stochastic subsampling of data used during training (Algorithm 1), as well as the reweighting of KL terms in the training objective. 

Thanks to [pixel-cnn-pp](https://github.com/pclucas14/pixel-cnn-pp) for the PyTorch PixelCNN++ implementation used in `pixelcnn_vhe.py`

# How to use
`example_czx.py` provides a toy example of a model where data are partitioned into classes.
- Each class will get its own latent variable `c`, with a Gaussian prior
- Each element will get its own latent variable `z`, with a Gaussian prior
- The likelihood `p(x|c,z)` will be a Gaussian distribution, with parameters given by a linear neural network.
- Encoders are `q(z|x)` and `q(c|D)`, where the support-set size is `|D|=5`

## Step 1.
Define an `nn.Module` for each conditional distribution. Its _forward_ function should return a value with associated log probability, wrapped in a `vhe.Result`. Use `<var>=None` to indicate that the random variable should be sampled.

**TODO: is it easier to separate sample and score functions?**
```python
import torch
import vhe

class Px(nn.Module): # p(x|c,z)
    def __init__(self):
        super().__init__()
        ... #Define any params

    def forward(self, c, z, x=None):
        mu, sigma = ...
        dist = torch.distributions.normal.Normal(mu, sigma) 
        
        if x is None: x = dist.rsample() # Sample x if not given
        log_prob = dist.log_prob(x).sum(dim=1) # Should be a 1D vector with nBatch elements
        return vhe.Result(x, log_prob)

class Qc(nn.Module): # q(c|D)
    def __init__(self): ...

    def forward(self, inputs, c=None):    
        # inputs is a (batch * |D| * ...) size tensor, containing the support set D
        ...
        return vhe.Result(c, log_prob)

class Qz(nn.Module): # q(z|x,c)
    def __init__(self): ...

    def forward(self, inputs, c, z=None):
        # inputs is a (batch * 1 * ...) size tensor, containing the input example x
        ...
        return vhe.Result(z, log_prob)
        
px = Px()
qc = Qc()
qz = Qz()
```

## Step 2.
Create a `vhe.VHE` module from the encoder and decoder modules. All variables use an isotroptic Gaussian prior by default, but may also be specified.

**TODO: don't really need kwargs in vhe.Factors**
```python
encoder = vhe.Factors(c=qc, z=qz)
decoder = px
model = vhe.VHE(encoder, decoder)
# or: model = vhe.VHE(encoder, decoder, prior=vhe.Factors(...))
```

## Step 3.
Create a `vhe.DataLoader` to sample data for training.

```python
data_loader = vhe.DataLoader(data=data,
        labels={"c":class_labels, # The class label for each element in data
                "z":range(len(data))},  # A unique label for each element in data
        k_shot={"c":5, "z":1}, # Number of elements given to each encoder
        batch_size=batch_size)
```

## Step 4.
Train using the variational lower bound `model.score(...)`

```python
optimiser = optim.Adam(vhe.parameters(), lr=1e-3)
for epoch in range(...):
    for batch in data_loader:
        optimiser.zero_grad()
        log_prob = model.score(inputs=batch.inputs, sizes=batch.sizes, x=batch.target)
        (-log_prob).backward() # Negative to get loss
        optimiser.step()
```
