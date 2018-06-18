from builtins import super

import inspect
from collections import namedtuple, OrderedDict

import torch
from torch import nn, distributions
from torch.autograd import Variable
import numpy as np

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
			self.norm = distributions.normal.Normal(0,1)

		def forward(self, batch_size=None, **kwargs):
			assert (batch_size != None) != (self.name in kwargs)

			if self.name in kwargs:  #Don't sample
				x = kwargs[self.name]
				if self.size is None:
					self.size = list(x.size()[1:])
					self.t = x.new_zeros(1,1)
				batch_size = x.size(0)
				return kwargs[self.name], self.norm.log_prob(x).view(batch_size,-1).sum(dim=1)
			else:
				size = [batch_size] + self.size
				dist = distributions.normal.Normal(self.t.new_zeros(size), self.t.new_ones(size))
				x = dist.rsample()
				return x, dist.log_prob(x)
	
	def __init__(self, sizes=None):
		self.size = None

	def make(self, name):
		module = self.F(name, self.size)
		args = set()
		return module, args

class DataLoader():
	VHEBatch = namedtuple("VHEBatch", ["inputs", "sizes", "target"])
	def __init__(self, data, batch_size, n_inputs=1, **kwargs):
		self.data = data
		self.labels = {}     # For each label type, a LongTensor assigning elements to labels
		self.label_idxs = {} # For each label type, for each label, a list of indices
		for k,v in kwargs.items():
			unique_oldlabels = list(set(v))
			map_label = {oldlabel:label for label, oldlabel in enumerate(unique_oldlabels)}
			self.labels[k] = torch.LongTensor([map_label[oldlabel] for oldlabel in v])
			self.label_idxs[k] = {}	
			for i in range(len(unique_oldlabels)):
				self.label_idxs[k][i] = (self.labels[k]==i).nonzero()[:,0]
		self.batch_size = batch_size
		self.n_inputs = n_inputs

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
		x = Variable(torch.index_select(self.data, 0, x_idx))
		inputs = {}
		sizes = {} 
		for k,v in labels.items():
			possibilities = [self.label_idxs[k][v[i].item()] for i in range(len(x_idx))]
			sizes[k] = torch.Tensor([len(X) for X in possibilities])
			input_idx = [np.random.choice(X, size=self.n_inputs) for X in possibilities]
			inputs[k] = [
				Variable(torch.index_select(self.data, 0, torch.LongTensor([I[j] for I in input_idx])))
				for j in range(self.n_inputs)]

		return self.VHEBatch(target=x, inputs=inputs, sizes=sizes)

class Factor(namedtuple('Factor', ['args', 'module'])):
	def __call__(self, *args, **kwargs):
		return self.module.forward(*args, **kwargs)

def Factors(**kwargs):
	unordered_factors = {}
	for name, f in kwargs.items():
		if isinstance(f, CustomDistribution):
			module, args = f.make(name)
		else:
			module, args = f, None

		assert 'forward' in dir(module)
		if args is None: args = set([k for k in inspect.getargspec(module.forward).args[1:] 
			if k != "inputs" and k != name])
		unordered_factors[name] = Factor(args, module)

	factors = OrderedDict()
	try:
		while len(unordered_factors)>0:
			k = next(k for k,v in unordered_factors.items() if v.args <= set(factors.keys()))
			factors[k] = unordered_factors[k]
			del unordered_factors[k]
	except StopIteration:
		raise Exception("Can't make a tree out of factors: " + 
				"".join("p(" + k + "|" + ",".join(v.args) + ")" for k,v in unordered_factors))

	return factors

class VHE(nn.Module):
	def __init__(self, prior, encoder):
		super().__init__()
		self.prior = prior
		self.encoder = encoder
		self.modules = nn.ModuleList([x.module for x in self.prior.values()] + 
									 [x.module for x in self.encoder.values()])
	
		assert len(prior) == len(encoder) + 1
		assert set(list(prior.keys())[:-1]) == set(encoder.keys())
		self.observation = list(prior.keys())[-1] 
		self.latents = list(prior.keys())[:-1] 
		self.Vars = namedtuple("Vars", prior.keys())
		self.KL = namedtuple("KL", self.latents)

	def score(self, inputs, sizes, return_kl=False, **kwargs):
		assert set(inputs.keys()) == set(self.encoder.keys())
		assert set(sizes.keys()) == set(self.latents)
		assert set(kwargs.keys()) == set([self.observation])

		# Sample from encoder
		sampled_vars = {}
		sampled_scores = {}
		for k, factor in self.encoder.items():
			sampled_vars[k], sampled_scores[k] = factor(inputs=inputs[k], **sampled_vars)
		sampled_vars[self.observation] = kwargs[self.observation]

		# Score under prior
		priors = {}
		for k, factor in self.prior.items():
			args = {k2:v for k2,v in sampled_vars.items() if k2==k or k2 in factor.args}
			_, priors[k] = factor(**args)

		# KL Divergence
		kl = {k:sampled_scores[k]-priors[k] for k in self.latents}

		# VHE Objective
		score = priors[self.observation] - sum(kl[k]/sizes[k] for k in self.latents) 

		if return_kl:
			return score.mean(), self.KL(**{k:v.mean() for k,v in kl.items()})
		else:
			return score.mean()

	def sample(self, inputs=None, batch_size=None):
		assert (inputs is None) != (batch_size is None)
		
		if inputs is None:
			samplers = self.prior
		else:
			batch_size = list(inputs.values())[0][0].size(0) #First latent, first example 
			samplers = {}
			for k in self.prior.keys():
				if k in inputs:
					samplers[k] = self.encoder[k]
				else:
					samplers[k] = self.prior[k]
			samplers = {k:self.encoder[k] if k in inputs else self.prior[k] for k in self.prior.keys()}

		sampled_vars = {}
		try:
			while len(samplers) > 0:
				k = next(k for k,v in samplers.items() if v.args <= set(sampled_vars.keys()))
				kwargs = {k2:sampled_vars[k2] for k2 in samplers[k].args}
				if k in inputs: kwargs['inputs'] = inputs[k]
				if len(kwargs)==0: kwargs['batch_size'] = batch_size

				sampled_vars[k], score = samplers[k](**kwargs)
				del samplers[k]
		except StopIteration:
			raise Exception("Can't find a valid sampling path")

		return self.Vars(**sampled_vars)
