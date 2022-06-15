import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


'''img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])'''

# Tranformation images to pytorch tensor
tensor_transform = transforms.ToTensor()

dataset = MNIST('./data', train=True, download=True,
                transform=tensor_transform)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 36),
            nn.ReLU(True),
            nn.Linear(36, 18),
            nn.ReLU(True),
            nn.Linear(18, 9)
        )
        self.decoder = nn.Sequential(
            nn.Linear(9, 18),
            nn.ReLU(True),
            nn.Linear(18, 36),
            nn.ReLU(True),
            nn.Linear(36, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            # nn.Tanh())
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded  # , latent


def training(epochs):
    for epoch in range(epochs):
        for data in dataloader:
            ''' data est une liste de deux éléments, le premier (data[0])est un regroupement d'image (liste d'image) 
            donc c'est un mini batch et le deuxième (data[1]) est la liste des entiers correspondants aux images de la liste 1'''
            # print("hello")
            #img, labels = data
            images, labels = data
            images = images.reshape(-1, 28*28)
            #img = img.view(img.size(0), -1).cuda()
            assert len(images) == len(labels)

            output = model(images)
            loss = loss_function(output, images)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'epoch [{epoch + 1}/{epochs}], loss:{loss.data.item()}')


def visualisation():
    for data in dataloader:
        images = data[0]
        images_mod = images.reshape(-1, 28*28)
        reconstructeds = model(images_mod)
        for i, item in enumerate(images):

            # Reshape the array for plotting
            item = item.reshape(-1, 28, 28)
            item = item.detach().numpy()
            plt.imshow(item[0])

        for i, item in enumerate(reconstructeds):
            item = item.reshape(-1, 28, 28)
            item = item.detach().numpy()

            plt.imshow(item[0])


# visualisation()

def test():
    count = 0
    for (image, _) in dataloader:

        #        images_mod = images.reshape(-1, 28*28)
        #reconstructed_mod = model(image_mod)
        #reconstructed = reconstructed_mod.reshape(-1, 28, 28)

        for i in range(len(image)):
            image_mod = image[i][0].reshape(-1, 28*28)
            reconstructed_mod = model(image_mod)
            reconstructed_mod = reconstructed_mod.reshape(28, 28)
            reconstructed_mod = reconstructed_mod.detach().numpy()
            plt.imshow(image[i][0])
            plt.show()
            '''reconstructed_img = reconstructed_mod[i]
            reconstructed_img = reconstructed_img[1].reshape(-1, 28, 28)

            reconstructed_img = reconstructed_img.detach().numpy()'''
            plt.imshow(reconstructed_mod)
            plt.show()
        '''for i in range(len(images)):
            img = images[i]
            img_mod = img.reshape(-1, 28*28)
            img_reconst = model(img_mod)
            img = img.reshape(-1, 28, 28)
            img = img.detach().numpy()
            #img_reconst = reconstructeds[i]
            img_reconst = img_reconst.reshape(-1, 28, 28)
            img_reconst = img_reconst.detach().numpy()
            plt.imshow(img[0])
            plt.imshow(img_reconst[0])
            plt.show()'''
        '''for i, item in enumerate(images):
            print(images[i])
            # Reshape the array for plotting
            item = item.reshape(-1, 28, 28)
            item = item.detach().numpy()
            plt.imshow(item[0])
            plt.show()'''

        ''''for i, item in enumerate(reconstructeds):
            item = item.reshape(-1, 28, 28)
            item = item.detach().numpy()

            plt.imshow(item[0])
        plt.show()'''


if __name__ == "__main__":
    epochs = 5
    batch_size = 32
    learning_rate = 1e-2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder()
    # model.cuda()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate)

    training(epochs)

    # test()
