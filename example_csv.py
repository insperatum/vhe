from builtins import super
import pickle
import string
import argparse

import torch
from torch import nn, optim

from pinn import RobustFill
import pregex as pre
from vhe import VHE, DataLoader, Factors, Result

from regex_prior import RegexPrior

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, choices=['qc','pc','px','vhe'])
args = parser.parse_args()

regex_prior = RegexPrior()
k_shot = 2
regex_vocab = list(string.printable[:-4]) + \
    [pre.OPEN, pre.CLOSE, pre.String, pre.Concat, pre.Alt, pre.KleeneStar, pre.Plus, pre.Maybe] + \
    regex_prior.character_classes

# Model
class Pc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = RobustFill(input_vocabularies=[], target_vocabulary=regex_vocab)

    def forward(self, c=None):
        if c is None: c, score = self.net.sampleAndScore()
        else: score = self.net.score([], c, autograd=True)
        return Result(value=c, reinforce_log_prob=score)

class Px(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = RobustFill(input_vocabularies=[regex_vocab], target_vocabulary=string.printable[:-4])

    def forward(self, c, x=None):
        _c = [[example] for example in c] #1 robustfill 'example'
        if x is None: x, score = self.net.sampleAndScore(_c)
        else: score = self.net.score(_c, x, autograd=True)
        return Result(value=x, reinforce_log_prob=score)

class Qc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = RobustFill(input_vocabularies=[string.printable[:-4]], target_vocabulary=regex_vocab)

    def forward(self, inputs, c=None):    
        if c is None: c, score = self.net.sampleAndScore(inputs)
        else: score = self.net.score(inputs, c, autograd=True)
        return Result(value=c, reinforce_log_prob=score)

if __name__ == "__main__":
    print("Loading pc", flush=True)
    try: pc=torch.load("./vhe_pc.p")
    except FileNotFoundError: pc=Pc()

    print("Loading px", flush=True)
    try: px=torch.load("./vhe_px.p")
    except FileNotFoundError: px=Px()

    print("Loading qc", flush=True)
    try: qc=torch.load("./vhe_qc.p")
    except FileNotFoundError: qc=Qc()


    ######## Pretraining ########
    batch_size = 500
    max_length = 15

    def getInstance():
        """
        Returns a single problem instance, as input/target strings
        """
        while True:
            r = regex_prior.sampleregex()
            c = r.flatten()
            x = r.sample()
            Dc = [r.sample() for i in range(k_shot)]
            c_input = [c]
            if all(len(x)<max_length for x in Dc + [c, x]): break
        return {'Dc':Dc, 'c':c, 'c_input':c_input, 'x':x}

    def getBatch():
        """
        Create a batch of problem instances, as tensors
        """
        instances = [getInstance() for i in range(batch_size)]
        Dc = [inst['Dc'] for inst in instances]
        c = [inst['c'] for inst in instances]
        c_input = [inst['c_input'] for inst in instances]
        x = [inst['x'] for inst in instances]
        return Dc, c, c_input, x

    if args.train in ['pc', 'px', 'qc']:
        print("Training", args.train, flush=True)
        f = {'pc':pc, 'px':px, 'qc':qc}[args.train]
        if not hasattr(f, 'iteration'):
            f.iteration = 0
            f.scores = []
        for i in range(f.iteration, 20000):
            Dc, c, c_input, x = getBatch()
            if args.train=="qc": score = qc.net.optimiser_step(Dc, c)
            if args.train=="pc": score = pc.net.optimiser_step([], c)
            if args.train=="px": score = px.net.optimiser_step(c_input, x)
            f.scores.append(score)
            f.iteration += 1
            if i%10==0: print(args.train, "iteration", i, "score:", score, flush=True)
            if i%500==0: torch.save(f, './vhe_' + args.train + '.p')


    ######## VHE Training #######
    if args.train == 'vhe':
        try: vhe=torch.load("./vhe.p")
        except FileNotFoundError:
            prior = Factors(c=pc)
            encoder = Factors(c=qc)
            decoder = Px()
            vhe = VHE(encoder, decoder, prior=prior).cuda()
            vhe.iteration = 0
            vhe.scores = []
            vhe.kls = []
        print("Training vhe")

        # Generate dataset
        n = 0
        with open("./csv_900.p", "rb") as f:
            classes = pickle.load(f)[0]
            data = [x for X in classes for x in X]
            class_labels = [i for i,X in enumerate(classes) for x in X]

        # Training
        batch_size = 500
        data_loader = DataLoader(data=data, labels={"c":class_labels}, k_shot={"c":1},
                batch_size=batch_size)

        # Training
        optimiser = optim.Adam(vhe.parameters(), lr=1e-3)
        while vhe.iteration<20000:
            for batch in data_loader:
                optimiser.zero_grad()
                score, kl = vhe.score(inputs=batch.inputs, sizes=batch.sizes, x=batch.target, return_kl=True)
                (-score).backward()
                optimiser.step()
                vhe.iteration += 1
                vhe.scores.append(score)
                vhe.kls.append(kl)
                s = vhe.sample(inputs=batch.inputs)
                if vhe.iteration%10==0:
                    print("Iteration", vhe.iteration, "Score %3.3f KLc %3.3f" % (score.item(), kl.c.item()), "|", \
                          "".join(batch.inputs['c'][0]), "-->", \
                          s.c[0], "-->", \
                          "".join(s.x[0]))
                if vhe.iteration%500==0: torch.save(vhe, './vhe.p')
