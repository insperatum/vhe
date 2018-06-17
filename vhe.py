from builtins import super

import inspect
from collections import namedtuple, OrderedDict

from torch import nn
from torch import distributions

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

		def score(self, **kwargs):
			x = kwargs[self.name]
			if self.size is None:
				self.size = list(x.size()[1:])
				self.t = x.new_zeros(1,1)
			batch_size = x.size(0)
			return self.norm.log_prob(x).view(batch_size,-1).sum(dim=1)

		def sample(self, batch_size):
			size = [batch_size] + self.size
			latents = distributions.normal.Normal(self.t.new_zeros(size), self.t.new_ones(size)).rsample()
			return latents
	
	def __init__(self, sizes=None):
		self.size = None

	def make(self, name):
		module = self.F(name, self.size)
		args = set()
		return module, args
		


def factors(**kwargs):
	Factor = namedtuple('Factor', ['args', 'module', 'sample', 'score'])

	unordered_factors = {}
	for name, f in kwargs.items():
		if isinstance(f, CustomDistribution):
			module, args = f.make(name)
		else:
			module, args = f, None

		sample, score = None, None
		if 'sample' in dir(module):
			sample = module.sample
			if args is None: args = set([k for k in inspect.getargspec(sample).args[1:] if k != "inputs"])
		if 'score' in dir(module):
			score = module.score
			if args is None: args = set([k for k in inspect.getargspec(score).args[1:] if k != name])

		unordered_factors[name] = Factor(args, module, sample, score)

	factors = OrderedDict()
	try:
		while len(unordered_factors)>0:
			k = next(k for k,v in unordered_factors.items() if v.args <= set(factors.keys()))
			factors[k] = unordered_factors[k]
			del unordered_factors[k]
	except StopIteration:
		print("Can't make a tree out of factors: " + 
				"".join("p(" + k + "|" + ",".join(v.args) + ")" for k,v in unordered_factors))
		raise Exception()
	return factors

class VHE(nn.Module):
	def __init__(self, encoder, decoder, prior=None, simple=False):
		super().__init__()
		self.simple = simple
		self.prior = NormalPrior(simple) if prior is None else prior
		self.encoder = encoder
		self.modules = nn.ModuleList([x.module for x in self.prior.values()] + 
									 [x.module for x in self.encoder.values()])
	
		assert len(prior) == len(encoder) + 1
		assert set(list(prior.keys())[:-1]) == set(encoder.keys())
		self.obs_name = list(prior.keys())[-1] 
		self.latent_names = list(prior.keys())[:-1] 
		self.Vars = namedtuple("Vars", prior.keys())

	def score(self, inputs, sizes, **kwargs):
		assert set(inputs.keys()) == set(self.encoder.keys())
		assert set(sizes.keys()) == set(self.latent_names)
		assert set(kwargs.keys()) == set([self.obs_name])

		# Sample from encoder
		sampled_vars = {}
		for k, factor in self.encoder.items():
			sampled_vars[k] = factor.sample(inputs=inputs[k], **sampled_vars)
		sampled_vars[self.obs_name] = kwargs[self.obs_name]

		# Score under prior
		priors = {}
		for k, factor in self.prior.items():
			args = {k2:v for k2,v in sampled_vars.items() if k2==k or k2 in factor.args}
			priors[k] = factor.score(**args)

		# VHE Objective
		score = sum(priors[k]/sizes[k] for k in sizes) + priors[self.obs_name]

		return score.mean()

	def sample(self, inputs=None, batch_size=None):
		assert (inputs is None) != (batch_size is None)
		
		if inputs is None:
			samplers = self.prior
		else:
			batch_size = list(inputs.values())[0][0].size(0) #First latent, first example 
			samplers = {k:self.encoder[k] if k in inputs else self.prior[k]
					for k in self.prior.keys()}

		sampled_vars = {}
		try:
			while len(samplers) > 0:
				k = next(k for k,v in samplers.items() if v.args <= set(sampled_vars.keys()))
				args = {k2:sampled_vars[k2] for k2 in samplers[k].args}
				if k in inputs: args['inputs'] = inputs[k]
				if len(args)==0: args['batch_size'] = batch_size

				sampled_vars[k] = samplers[k].sample(**args)
				del samplers[k]
		except StopIteration:
			print("Can't find a valid sampling path")
			raise Exception()

		return self.Vars(**sampled_vars)
