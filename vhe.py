from builtins import super

import inspect
from collections import namedtuple, OrderedDict
import numbers

import torch
from torch import nn, distributions
import numpy as np

class VHE(nn.Module):
    def __init__(self, encoder, decoder, prior=None):
        """
        encoder can be any of: nn.Module, list[nn.Module], dict(class_name -> nn.Module)
        decoder is an nn.Module
        prior defaults to Normal(0,1), or can be any of: nn.Module, list[nn.Module], dict(class_name -> nn.Module)
        """
        super(VHE, self).__init__()

        self.encoder = asFactors(encoder)
        self.decoder = createFactorFromModule(decoder)
        if prior is None:
            self.prior = Factors(**{k:NormalPrior() for k in self.encoder.variables})
        else:
            self.prior = asFactors(prior)
            assert set(self.prior.variables) == set(self.encoder.variables)
            
        self.modules = nn.ModuleList(self.prior.modules + self.encoder.modules + [self.decoder.module])
        self.observation = self.decoder.name 
        self.latents = self.prior.variables 

    def score(self, inputs, sizes, return_kl=False, kl_factor=1, **kwargs):
        assert set(inputs.keys()) == set(self.latents)
        assert set(sizes.keys()) == set(self.latents)
        assert set(kwargs.keys()) == set([self.observation])
        
        if isinstance(kl_factor, numbers.Number): kl_factor = {k: kl_factor for k in self.latents}
        else: kl_factor = {k: kl_factor.get(k, 1) for k in self.latents}
            
        # Sample from encoder
        sampled_vars = {}
        sampled_log_probs = {}
        sampled_reinforce_log_probs = {}
        for k, factor in self.encoder.factors.items():
            result = factor(inputs=inputs[k], **sampled_vars)
            #result = factor(inputs=inputs[k], **{k:v for k,v in sampled_vars.items() if k in factor.variables})
            sampled_vars[k], sampled_log_probs[k] = result.value, result.log_prob
            if result.reinforce_log_prob is not None: sampled_reinforce_log_probs[k] = result.reinforce_log_prob
        sampled_vars[self.observation] = kwargs[self.observation]

        # Score under prior
        priors = {}
        for k, factor in self.prior.factors.items():
            args = {k2:v for k2,v in sampled_vars.items() if k2==k or k2 in factor.args}
            priors[k] = factor(**args).log_prob

        # KL Divergence
        kl = {k:sampled_log_probs[k]-priors[k] for k in self.latents}

        # Log likelihood
        args = {k:v for k,v in sampled_vars.items() if k==self.decoder.name or k in self.decoder.args}
        ll = self.decoder(**args).log_prob

        # VHE Objective
        t = ll
        lowerbound = ll - sum(kl_factor[k]*kl[k]/t.new(sizes[k]) for k in self.latents) 

        # Reinforce objective
        objective = lowerbound
        for k,v in sampled_reinforce_log_probs.items():
            downstream_kl = sum(kl_factor[k]*kl[k]/t.new(sizes[k]) for k2 in self.latents if k in self.encoder.dependencies[k2] or k2 == k)
            objective += v * (ll - downstream_kl).data

        if return_kl:
            return objective.mean(), KL(**{k:v.mean() for k,v in kl.items()})
        else:
            return objective.mean()

    def sample(self, inputs=None, batch_size=None):
        if inputs is None:
            samplers = self.prior
        else:
            batch_size = len(list(inputs.values())[0]) #First latent 
            samplers = {k:self.encoder.factors[k] if k in inputs else self.prior.factors[k] for k in self.latents}
            samplers[self.observation] = self.decoder

        sampled_vars = {}
        try:
            while len(samplers) > 0:
                k = next(k for k,v in samplers.items() if v.args <= set(sampled_vars.keys()))
                kwargs = {k2:sampled_vars[k2] for k2 in samplers[k].args}
                if k in inputs: kwargs['inputs'] = inputs[k]
                if len(kwargs)==0: kwargs['batch_size'] = batch_size
                sampled_vars[k] = samplers[k](**kwargs).value
                del samplers[k]
        except StopIteration:
            raise Exception("Can't find a valid sampling path")

        return Vars(**sampled_vars)


def assert_msg(condition, message):
    if not condition: raise Exception(message)

class CustomDistribution():
    def make(self, name): raise NotImplementedError()

class NormalPrior(CustomDistribution):
    class F(nn.Module):
        def __init__(self, name, size):
            super().__init__()
            self.name = name
            self.size = size

        def forward(self, batch_size=None, **kwargs):
            assert (batch_size != None) != (self.name in kwargs)

            if self.name in kwargs:  #Don't sample
                x = kwargs[self.name]
                if self.size is None:
                    self.size = list(x.size()[1:])
                    self.t = x.new_zeros(1,1)
                dist = distributions.normal.Normal(x.new_zeros(1), x.new_ones(1))
                batch_size = x.size(0)
                return Result(kwargs[self.name], dist.log_prob(x).view(batch_size,-1).sum(dim=1))
            else:
                size = [batch_size] + self.size
                dist = distributions.normal.Normal(self.t.new_zeros(size), self.t.new_ones(size))
                x = dist.rsample()
                return Result(x, dist.log_prob(x))
    
    def __init__(self, sizes=None):
        self.size = None

    def make(self, name):
        module = self.F(name, self.size)
        args = set()
        return Factor(module, name, args)

class Result:
    def __init__(self, value, reparam_log_prob=0, reinforce_log_prob=None):
        self.value = value
        self.reparam_log_prob = reparam_log_prob
        self.reinforce_log_prob = reinforce_log_prob
        self.log_prob = self.reparam_log_prob
        if self.reinforce_log_prob is not None: self.log_prob += self.reinforce_log_prob


VHEBatch = namedtuple("VHEBatch", ["inputs", "sizes", "target"])
class DataLoader():
    def __init__(self, data, batch_size, labels, k_shot, transforms=[]):
        self.data = data
        self.mode = "tensor" if torch.is_tensor(data) else "list"
        if self.mode == "tensor":
            self.select_data = lambda x_idx: torch.index_select(self.data, 0, x_idx)
        else:
            self.select_data = lambda x_idx: [self.data[i] for i in x_idx]
        self.labels = {}     # For each label type, a LongTensor assigning elements to labels
        self.label_idxs = {} # For each label type, for each label, a list of indices
        for k,v in labels.items():
            v = list(v)
            if torch.is_tensor(v[0]): v = [x.item() for x in v]
            unique_oldlabels = list(set(v))
            map_label = {oldlabel:label for label, oldlabel in enumerate(unique_oldlabels)}
            self.labels[k] = torch.LongTensor([map_label[oldlabel] for oldlabel in v])
            self.label_idxs[k] = {j:[] for j in range(len(unique_oldlabels))} 
            for i,j in enumerate(self.labels[k]):
                self.label_idxs[k][j.item()].append(i)
            for j in range(len(unique_oldlabels)):
                self.label_idxs[k][j]=torch.LongTensor(self.label_idxs[k][j])
        self.batch_size = batch_size
        self.k_shot = k_shot
        self.transforms = transforms

    def __iter__(self):
        self.next_i = 0
        self.x_idx = torch.randperm(len(self.data))
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.next_i+self.batch_size > len(self.data):
            raise StopIteration()
        else:
            x_idx = self.x_idx[self.next_i:self.next_i+self.batch_size]
            self.next_i += self.batch_size

        labels = {k: torch.index_select(self.labels[k], 0, x_idx) for k in self.labels}
        x = self.select_data(x_idx)

        inputs = {}
        sizes = {} 
        for k,v in labels.items():
            possibilities = [self.label_idxs[k][v[i].item()] for i in range(len(x_idx))]
            sizes[k] = [len(X) for X in possibilities]
            input_idx = [np.random.choice(X, size=self.k_shot[k]) for X in possibilities]
            _inputs = [
                self.select_data(torch.LongTensor([I[j] for I in input_idx]))
                for j in range(self.k_shot[k])]
            if self.mode == "tensor":
                inputs[k] = torch.cat([x.unsqueeze(1) for x in _inputs], dim=1)
            elif self.mode == "list":
                inputs[k] = [[_inputs[j][i] for j in range(self.k_shot[k])]
                        for i in range(len(_inputs[0]))]

        batch = VHEBatch(target=x, inputs=inputs, sizes=sizes)
        for transform in self.transforms:
            batch = transform.apply(batch)
        return batch

class Transform():
    """
    A set of transformations to apply to a dataset
    Params:
        f is the transform function: f(data, args)
        args is a tensor containing possible transform arguments (so n_transforms = args.size(0))
        share_labels is the list of latent labels that should be shared between x and the transformed version of x
    """
    def __init__(self, f, args, share_labels=None):
        self.f = f
        self.args = args
        self.n_transforms = args.size(0)
        self.share_labels = [] if share_labels is None else share_labels

    def transform_tensor(self, t):
        # First dim of t is batch
        # Applies a random transform to every row of t
        transform_idxs = t.new_tensor(t.size(0)).random_(0, self.n_transforms).long()
        transform_args = t.index_select(0, transform_idxs)
        return self.f(t, transform_args)

    def apply(self, batch):
        target = self.transform_tensor(batch.target)
        inputs = {}
        for k, D in batch.inputs.items():
            if k in self.share_labels:
                D_unrolled = D.reshape(D.size(0) * D.size(1), *D.size()[2:])
                D_transformed = self.transform_tensor(D_unrolled)
                inputs[k] = D_transformed.reshape(D.size(0), D.size(1), *D_transformed.size()[1:])
            else:
                inputs[k] = D
        sizes = {k: [s*self.n_transforms for s in v] if k in self.share_labels else v
                for k,v in batch.sizes.items()}
        return VHEBatch(target=target, inputs=inputs, sizes=sizes)




class Factor(namedtuple('Factor', ['module', 'name', 'args'])):
    def __call__(self, *args, **kwargs):
        r = self.module.forward(*args, **kwargs)
        if Result.__instancecheck__(r):
            return r
        else:
            return Result(*r)


def createFactorFromModule(module):
    assert 'forward' in dir(module)
    spec = inspect.getargspec(module.forward)
    assert spec.defaults == (None,)
    name = spec.args[-1]
    args = set([k for k in spec.args[1:-1] if k != "inputs"]) #Not self, name or inputs 
    return Factor(module, name, args)
    
class Factors:
    def __init__(self, *args, **kwargs):
        unordered_factors = []
        for f in args:
            factor = createFactorFromModule(f)
            unordered_factors.append(factor)

        for name, f in kwargs.items():
            if isinstance(f, CustomDistribution):
                factor = f.make(name)
            else:
                factor = createFactorFromModule(f)
                assert factor.name == name 
            unordered_factors.append(factor)

        factors = OrderedDict()
        dependencies = {}
        try:
            while len(unordered_factors)>0:
                factor = next(f for f in unordered_factors if f.args <= set(factors.keys()))
                factors[factor.name] = factor 
                dependencies[factor.name] = factor.args | \
                        set([k for f2 in factor.args for k in dependencies[f2]])
                unordered_factors.remove(factor)
        except StopIteration:
            raise Exception("Can't make a tree out of factors: " + 
                    "".join("p(" + k + "|" + ",".join(v.args) + ")" for k,v in unordered_factors))

        self.factors = factors
        self.dependencies = dependencies
        self.variables = list(self.factors.keys())
        self.modules = list(f.module for f in self.factors.values())

def asFactors(f):
    if Factors.__instancecheck__(f):
        return f
    elif dict.__instancecheck__(f):
        return Factors(**f)
    elif "__iter__" in dir(f):
        return Factors(*f)
    else:
        return Factors(f)

class Vars():
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class KL(Vars):
    pass