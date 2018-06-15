from builtins import super

from torch import nn
from torch.distributions.normal import Normal
#from collections import namedtuple

class NormalPrior(nn.Module):
	def __init__(self, simple=False):
		super().__init__()
		self.simple = simple
		self.norm = Normal(0,1)
		self.sizes = None
	def score(self, latents):
		if self.simple: latents=[latents]
		if self.sizes is None:
			self.sizes = [x.size()[1:] for x in latents]
			self.t = latents[0].new_zeros(1,1)
		batch_size = latents[0].size(0)
		return sum(self.norm.log_prob(x).view(batch_size,-1).sum(dim=1) for x in latents)
	def sample(self, batch_size):
		sizes = [[batch_size] + x for x in self.sizes]
		return [Normal(self.t.new_zeros(size), self.t.new_ones(size)).rsample() for size in sizes]

class VHE(nn.Module):
	def __init__(self, encoder, decoder, prior=None, simple=False):
		super().__init__()
		self.simple = simple
		self.prior = NormalPrior(simple) if prior is None else prior
		self.encoder = encoder
		self.decoder = decoder

	def score(self, inputs, cardinalities, output):
		latents = self.encoder.sample(inputs)
		priors = self.prior.score(latents)
		log_likelihood = self.decoder.score(latents, output)
		
		if self.simple:
			scores = priors/cardinalities + log_likelihood
		else:
			scores = sum(p/c for p,c in zip(priors, cardinalities)) + log_likelihood
		return scores.mean()

	def sample(self, inputs=None, batch_size=None):
		assert (inputs is None) != (batch_size is None)
		if inputs is None:
			latents = self.prior.sample(batch_size)
		else:
			latents = self.encoder.sample(inputs)
		return self.decoder.sample(latents)

class SimpleVHE(VHE):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs, simple=True)
