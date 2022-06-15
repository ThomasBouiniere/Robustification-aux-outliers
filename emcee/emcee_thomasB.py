from matplotlib import pyplot as plt
from matplotlib.transforms import Transform
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import emcee
import numpy as np


def MCMC(img, func):

    pos = img + np.random.randn(28*28*2, 28*28)
    number_of_walkers, number_of_dimensions = pos.shape

    def func_prime(x):
        x.reshape(28*28)
        x = torch.Tensor(x)
        x = func(x)
        x = x.numpy().flatten()
        return x

    sampler = emcee.EnsembleSampler(
        nwalkers=number_of_walkers, ndim=number_of_dimensions, log_prob_fn=func_prime)

    sampler.run_mcmc(initial_state=pos, nsteps=150)
    samples = sampler.get_chain(flat=True)
    return samples


transform = transforms.ToTensor()


mnist_data = datasets.MNIST(root='./data', download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(
    dataset=mnist_data, batch_size=500, shuffle=True)


class Autoencodeur(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28*28, 612), nn.ReLU(), nn.Linear(612, 128), nn.ReLU(), nn.Linear(
            128, 64), nn.ReLU(), nn.Linear(64, 12), nn.ReLU(), nn.Linear(12, 3))
        self.decodeur = nn.Sequential(nn.Linear(3, 12), nn.ReLU(), nn.Linear(
            12, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 612), nn.ReLU(), nn.Linear(612, 28*28), nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decodeur(encoded)
        return decoded


model = Autoencodeur()
criteron = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

num_epochs = 10
outputs = []
for epoch in range(num_epochs):
    for (img, _) in data_loader:
        img = img.reshape(-1, 28*28)
        recon = model(img)
        loss = criteron(recon, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch+1}, Loss: {loss.item() : 4f}')
    outputs.append((epoch, img, recon))

with torch.no_grad():
    dataiter = iter(data_loader)
    target_label = 2  # CHOOSE IMAGE NUMBER TO TARGET VIA MCMC
    # "zero" for initial state @ zero array
    # "rd" for random initial states

    if target_label == "zero":
        image = [0]*28*28
    elif target_label == "rd":
        image = [0]*28*28 + np.random.rand(28*28)
    else:
        image, label = dataiter.next()
        print(label.shape)
        for j in range(len(image)):
            if label[j] == target_label:
                image = image[j]
                label = label[j].item()
                break
        print(label)

    def func(x): return criteron(model(x), x)
    res_MCMC = MCMC(image.reshape(28*28), func)


# VISU
# for k in range(0, num_epochs, 4):
#     plt.figure(figsize=(9, 2))
#     plt.gray()
#     imgs = outputs[k][1].detach().numpy()
#     recon = outputs[k][2].detach().numpy()
#     for i, item in enumerate(imgs):
#         if i >= 9:
#             break
#         plt.subplot(2, 9, i+1)
#         item = item.reshape(-1, 28, 28)
#         plt.imshow(item[0])

#     for i, item in enumerate(recon):
#         if i >= 9:
#             break
#         plt.subplot(2, 9, 9+i+1)
#         item = item.reshape(-1, 28, 28)
#         plt.imshow(item[0])
plt.title(label)
for i in range(3):
    for j in range(3):
        plt.subplot(3, 3, 1+i+3*j)
        plt.imshow(res_MCMC[i+3*j].reshape(28, 28))

plt.show()
