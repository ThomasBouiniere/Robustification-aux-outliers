# imports
import torch

import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.encoder = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.decoder = nn.Sequential(
            # Defining 2D upscaling layer
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            # Defining second 2D upscaling layer
            nn.ConvTranspose2d(4, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2)
        )

        # Defining the forward pass

    def forward(self, x):
        x = self.encoder(x)
        # x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


def train(epoch):

    model.train()
    tr_loss = 0
    for i, (images, labels) in enumerate(trainloader):
        # getting the training set
        x_train, y_train = Variable(images), Variable(labels)
        y_train = x_train  # images are ground truth

        # converting the data into GPU format
        if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()

        # prediction for training and validation set
        output_train = model(x_train)

        # computing the training loss
        loss_train = criterion(output_train, y_train)
        train_losses.append(loss_train.cpu())

        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()
    if epoch % 2 == 0:
        # printing the validation loss
        print('Epoch : ', epoch+1, '\t', 'loss :', loss_train)


if __name__ == "__main__":
    import torch
    from torchvision.transforms import ToTensor
    import torchvision.datasets as datasets
    import numpy as np
    import torch.nn as nn
    from torch.autograd import Variable
    import matplotlib.pyplot as plt

    # datasets
    mnist_trainset = datasets.MNIST(
        root='./data', train=True, download=True, transform=ToTensor())
    mnist_testset = datasets.MNIST(
        root='./data', train=False, download=True, transform=ToTensor())

    # holdout
    indexes = []
    held_digits = [1, 7]
    for i in range(len(mnist_trainset)):
        if mnist_trainset[i][1] not in held_digits:
            indexes.append(i)
    # print(len(indexes)/len(mnist_trainset))

    # torch.utils.data.Subset(trainset, idx)
    mnist_trainset_holdout = torch.utils.data.Subset(
        mnist_trainset, indexes)

    # dataloader
    trainloader = torch.utils.data.DataLoader(
        mnist_trainset_holdout, batch_size=50)
    testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=50)

    # defining the model
    model = Net()
    # defining the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.07)
    # defining the loss function
    # criterion = nn.CrossEntropyLoss() # for CNN
    criterion = nn.MSELoss()  # for autoencoder

    # checking if GPU is available
    if torch.cuda.is_available():
        print("On GPU !")
        model = model.cuda()
        criterion = criterion.cuda()

    # defining the number of epochs
    n_epochs = 5
    # empty list to store training losses
    train_losses = []
    # training the model
    for epoch in range(n_epochs):
        train(epoch)

    with torch.no_grad():
        plt.clf()
        # plotting the training and validation loss
        # plt.plot(train_losses, label='Training loss')
        # plt.legend()
        # plt.show()

        for i, (images, labels) in enumerate(testloader):
            if i == 4:
                break
            # getting the test images
            x_test, y_test = Variable(images), Variable(labels)
            if torch.cuda.is_available():
                x_test = x_test.cuda()
                y_test = y_test.cuda()

            y_pred = model(x_test)
            NB = 5
            images = np.random.randint(0, high=len(y_pred), size=NB)
            for i in range(NB):
                plt.subplot(NB, 2, 2*i+1)
                plt.imshow(np.squeeze(
                    x_test[images[i]].cpu().permute(1, 2, 0)))  # pfff
                # mdr c quoi
                plt.title(y_test[images[i]].cpu().detach().numpy())
                plt.subplot(NB, 2, 2*i+2)
                plt.imshow(np.squeeze(
                    y_pred[images[i]].cpu().permute(1, 2, 0)))
            plt.show()

            # print(type(y_pred))
