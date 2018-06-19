



model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix, prior_dims=prior_dims)
model = model.cuda()

#a pixelcnn px
class Px(nn.Module):
	def __init__(self):
		super().__init__()
		
		self.model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix, latent_dims=latent_dims)

	def sample(model, latents=None): 
		assert latents is not None
	    model.train(False)
	    data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
	    data = data.cuda()
	    for i in range(obs[1]):
	        for j in range(obs[2]):
	            data_v = Variable(data, volatile=True)
	            out   = model(data_v, sample=True, latents=latents)
	            out_sample = sample_op(out) #TODO: write this fn (I think Luke's job?)
	            data[:, :, i, j] = out_sample.data[:, :, i, j]
	    return data, out #the last output form the model should be the latents for the dist

	def forward(self, c, z, x=None):
		if x is None: 
			x, dist = self.sample(self.model, latents=(c,z))
			return x, loss_op(x, dist)
		else:
			#return x and distribution (or is it a loss?)
			return x, loss_op(x, self.model(x, latents=(c,z),  sample=False))




class Qc(nn.Module):
	def __init__(self):
		super().__init__()
		self.enc = nn.Linear(x_dim, h_dim)
		self.mu_c = nn.Linear(h_dim, c_dim)
		self.sigma_c = nn.Sequential(nn.Linear(h_dim, c_dim), nn.Softplus())

	def forward(self, inputs, c=None):	
		embeddings = [self.enc(x) for x in inputs]
		mean_embedding = sum(embeddings)/len(embeddings)
		mu_c = self.mu_c(mean_embedding)
		sigma_c = self.sigma_c(mean_embedding)
		dist = Normal(mu_c, sigma_c)
		if c is None: c = dist.rsample()
		return c, dist.log_prob(c).sum(dim=1)

class Qz(nn.Module):
	def __init__(self):
		super().__init__()
		self.mu_z = nn.Linear(x_dim, z_dim)
		self.sigma_z = nn.Sequential(nn.Linear(x_dim, z_dim), nn.Softplus())

	def forward(self, inputs, c, z=None):	
		mu_z = self.mu_z(inputs[0])
		sigma_z = self.sigma_z(inputs[0])
		dist = Normal(mu_z, sigma_z)
		if z is None: z = dist.rsample()
		return z, dist.log_prob(z).sum(dim=1)

encoder = Factors(c=Qc(), z=Qz())
decoder = Px()
vhe = VHE(encoder, decoder)




# Generate dataset
############
n = 0
classes = []
for i in range(1000):
	mu = torch.randn(1, x_dim)
	sigma = 0.1
	class_size = random.randint(10,20)
	classes.append(mu + sigma*torch.randn(class_size, x_dim))
data = torch.cat(classes)
class_labels = [i for i in range(len(classes)) for j in range(len(classes[i]))] 


# Training
batch_size = 100
n_inputs = 1
data_loader = DataLoader(data=data, c=class_labels, z=range(len(data)),
		batch_size=batch_size, n_inputs=n_inputs)
############


# Training
optimiser = optim.Adam(vhe.parameters(), lr=1e-3)
for epoch in range(1,11):
	for batch in data_loader:
		optimiser.zero_grad()
		score, kl = vhe.score(inputs=batch.inputs, sizes=batch.sizes, x=batch.target, return_kl=True)
		(-score).backward()
		optimiser.step()
	print("Epoch %d Score %3.3f KLc %3.3f KLz %3.3f" % (epoch, score.item(), kl.c.item(), kl.z.item()))

for mu in [-1, 0, 1]:
	test_D = [mu + 0.1*torch.randn(1,x_dim) for _ in range(n_inputs)]
	print("\nPosterior predictive for", test_D)
	print(vhe.sample(inputs={"c":test_D}).x)