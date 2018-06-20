from builtins import super
import pickle

from torch import nn, optim

from pinn import RobustFill
import string

from vhe import VHE, DataLoader, Factors, Result


# Model
x_dim = 5
c_dim = 10
z_dim = 10
h_dim = 10

class Pc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = RobustFill(input_vocabularies=[], target_vocabulary=string.printable[:-4])

    def forward(self, c=None):
        if c is None: c, score = self.net.sampleAndScore()
        else: score = self.net.score([], c, autograd=True)
        return Result(value=c, reinforce_log_prob=score)

class Px(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = RobustFill(input_vocabularies=[string.printable[:-4]], target_vocabulary=string.printable[:-4])

    def forward(self, c, x=None):
        _c = [[example] for example in c] #1 robustfill 'example'
        if x is None: x, score = self.net.sampleAndScore(_c)
        else: score = self.net.score(_c, x, autograd=True)
        return Result(value=x, reinforce_log_prob=score)

class Qc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = RobustFill(input_vocabularies=[string.printable[:-4]], target_vocabulary=string.printable[:-4])

    def forward(self, inputs, c=None):    
        if c is None: c, score = self.net.sampleAndScore(inputs)
        else: score = self.net.score(inputs, c, autograd=True)
        return Result(value=c, reinforce_log_prob=score)

prior = Factors(c=Pc())
encoder = Factors(c=Qc())
decoder = Px()
vhe = VHE(encoder, decoder, prior=prior)


# Generate dataset
n = 0
with open("./csv_900.p", "rb") as f:
    classes = pickle.load(f)[0]
    data = [x for X in classes for x in X]
    class_labels = [i for i,X in enumerate(classes) for x in X]

# Training
batch_size = 100
n_inputs = 1
data_loader = DataLoader(data=data, c=class_labels,
        batch_size=batch_size, n_inputs=n_inputs)

# Training
optimiser = optim.Adam(vhe.parameters(), lr=1e-3)
for epoch in range(1,11):
    print("\n------------\nEpoch %d\n" % epoch)
    for batch in data_loader:
        optimiser.zero_grad()
        score, kl = vhe.score(inputs=batch.inputs, sizes=batch.sizes, x=batch.target, return_kl=True)
        (-score).backward()
        optimiser.step()
        print("Score %3.3f KLc %3.3f" % (score.item(), kl.c.item()), "|", \
              "".join(batch.inputs['c'][0]), "-->", \
              "".join(vhe.sample(inputs=batch.inputs).x[0]))
