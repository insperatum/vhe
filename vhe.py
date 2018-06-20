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
			self.norm = distributions.normal.Normal(torch.Tensor([0]).cuda(),torch.Tensor([1]).cuda())

		def forward(self, batch_size=None, **kwargs):
			assert (batch_size != None) != (self.name in kwargs)

			if self.name in kwargs:  #Don't sample
				x = kwargs[self.name]
				if self.size is None:
					self.size = list(x.size()[1:])
					self.t = x.new_zeros(1,1)
				batch_size = x.size(0)

				return Result(kwargs[self.name], self.norm.log_prob(x).view(batch_size,-1).sum(dim=1))
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
	def __init__(self, value, reparam_log_prob, reinforce_log_prob=None):
		self.value = value
		self.reparam_log_prob = reparam_log_prob
		self.reinforce_log_prob = reinforce_log_prob
		self.log_prob = self.reparam_log_prob
		if self.reinforce_log_prob is not None: self.log_prob += self.reinforce_log_prob
		
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

class Factor(namedtuple('Factor', ['module', 'name', 'args'])):
	def __call__(self, *args, **kwargs):
		return self.module.forward(*args, **kwargs)

def createFactorFromModule(module):
	assert 'forward' in dir(module)
	spec = inspect.getargspec(module.forward)
	assert spec.defaults == (None,)
	name = spec.args[-1]
	args = set([k for k in spec.args[1:-1] if k != "inputs"]) #Not self, name or inputs 
	return Factor(module, name, args)
	
class Factors:
	def __init__(self, **kwargs):
		unordered_factors = []
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

class VHE(nn.Module):
	def __init__(self, encoder, decoder, prior=None):
		super(VHE,self).__init__()
		self.encoder = encoder
		self.decoder = createFactorFromModule(decoder)
		if prior is not None:
			assert len(prior.factors) == len(encoder.factors) + 1
			assert set(prior.variables) == set(encoder.variables)
			self.prior = prior
		else:
			self.prior = Factors(**{k:NormalPrior() for k in encoder.variables})
		self.modules = nn.ModuleList(self.prior.modules + self.encoder.modules + [self.decoder.module])
	
		self.observation = self.decoder.name 
		self.latents = self.prior.variables 
		self.Vars = namedtuple("Vars", self.latents + [self.observation])
		self.KL = namedtuple("KL", self.latents)

	def score(self, inputs, sizes, return_kl=False, **kwargs):
		assert set(inputs.keys()) == set(self.latents)
		assert set(sizes.keys()) == set(self.latents)
		assert set(kwargs.keys()) == set([self.observation])

		# Sample from encoder
		sampled_vars = {}
		sampled_log_probs = {}
		sampled_reinforce_log_probs = {}
		for k, factor in self.encoder.factors.items():
			result = factor(inputs=inputs[k], **sampled_vars)
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
		lowerbound = ll - sum(kl[k]/sizes[k] for k in self.latents) 

		# Reinforce objective
		objective = lowerbound
		for k,v in sampled_reinforce_log_probs.items():
			objective += v * (ll - sum(kl[k]/sizes[k] for k2 in self.latents if k in self.dependencies[k2])).data

		if return_kl:
			return objective.mean(), self.KL(**{k:v.mean() for k,v in kl.items()})
		else:
			return objective.mean()

	def sample(self, inputs=None, batch_size=None):
		assert (inputs is None) != (batch_size is None)
		
		if inputs is None:
			samplers = self.prior
		else:
			batch_size = list(inputs.values())[0][0].size(0) #First latent, first example 
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

		return self.Vars(**sampled_vars)
