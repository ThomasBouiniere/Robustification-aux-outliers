# imports
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


# dataloaders
# train
trainloader = torch.utils.data.DataLoader(
    mnist_trainset, batch_size=1000)
# test 
testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=500) #batchsize = 1 for mcmc TO CHANGE FOR TESTING


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
            nn.ConvTranspose2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Defining the forward pass

    def forward(self, x):
        x = self.encoder(x)
        # x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


# defining the model
model = Net()
# defining the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
# defining the loss function
# criterion = nn.CrossEntropyLoss() # for CNN
criterion = nn.MSELoss()  # for autoencoder

# checking if GPU is available
if torch.cuda.is_available():
    print("On GPU !")
    model = model.cuda()
    criterion = criterion.cuda()


def train(epoch):

    model.train()
    tr_loss = 0
    for i, (images, labels) in enumerate(trainloader):
        # one batch
        # clearing the Gradients of the model parameters
        optimizer.zero_grad()
        # pos =================================================================
        # getting the pos training set
        x_train, y_train = Variable(images), Variable(labels)
        y_train = x_train  # images are ground truth
        n_true = len(x_train)
        # converting the data into GPU format
        if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()

        # prediction for training and validation set
        output_train = model(x_train)

        # computing the training loss for pos samples
        pos_loss_train = criterion(output_train, y_train)

        # neg =========================================================
        # print(iter(neg_trainloader))
        # next item of iterator made from torch dataloader
        if False: #condition on epoch
            def neg_samples_MCMC(modele, n=1000):
                samples = [y_train[0]]
                def f(x): return np.exp(-modele(x))
                for _ in range(10000):
                    candidate = np.random.normal(
                        samples[-1].cpu(), 4)  # a reecrire
                    p = min(1, f(candidate)/f(samples[-1]))
                    if np.random.random() < prob:
                        samples.append(candidate)
                    else:
                        samples.append(samples[-1])
                samples = samples[-1000:]
                return samples
            samples = neg_samples_MCMC(model, n=1000)
            x_train = Variable(samples)
            # converting the data into GPU format
            if torch.cuda.is_available():
                x_train = x_train.cuda()
            # prediction for training and validation set
            output_train = model(x_train)
            # computing the training loss
            neg_loss_train = criterion(output_train, x_train)
            n_false = len(x_train)
        else:
            neg_loss_train = 0
            n_false = 0

        # ================ computing final loss ===================

        loss_train = (pos_loss_train*n_true - n_false*neg_loss_train)/(n_true+n_false)
        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()

    print('Epoch : ', epoch+1, '\t', 'loss :',
          loss_train)


# defining the number of epochs
n_epochs = 5
# empty list to store training losses
train_losses = []
# training the model
for epoch in range(n_epochs):
    train(epoch)






# EMCEE from emcee_thomasB

def MCMC(img, f, use_emcee=False):
    
    if use_emcee:
        #emcee
        import emcee
        pos = img + np.random.randn(28*28*2, 28*28)
        number_of_walkers, number_of_dimensions = pos.shape
        sampler = emcee.EnsembleSampler(
            nwalkers=number_of_walkers, ndim=number_of_dimensions, log_prob_fn=f_likelihood)

        sampler.run_mcmc(initial_state=pos, nsteps=150)
        samples = sampler.get_chain(flat=True)
        return samples

    # mcmc maison 
    # TODO : optimiser etc
    # TODO : verif convergence
    res =[]
    number_of_walkers = 1
    pos = img + np.random.randn(number_of_walkers, 28*28)
    for i,sample in enumerate(pos):
        print(100*i/(number_of_walkers))
        samples = [sample.numpy()]
        for _ in range(1500):
                        candidate = np.random.normal(
                            samples[-1], 4)
                        a = f(candidate)
                        b = f(samples[-1])
                        p = min(1, a/b)
                        if np.random.random() < p:
                            samples.append(candidate)
                        else:
                            samples.append(samples[-1])
        samples = [samples[i] for i in [0,5,10,200,400,500,1000,1250,1449]]
        res += samples
    return res
    







def mcmc_out_of_model():
    with torch.no_grad():
        #get images
        dataiter = iter(testloader)

        # CHOOSE IMAGE NUMBER TO TARGET VIA MCMC
        target_label = 9
        # "zero" for initial state @ zero array
        # "rd" for random initial states
        if target_label == "zero":
            image = [0]*28*28
        elif target_label == "rd":
            image = [0]*28*28 + np.random.rand(28*28)*e-2  # TODO RESTER BORNES DATASET
        else:
            image, label = dataiter.next()
            for j in range(len(image)):
                if label[j] == target_label:
                    image = image[j]
                    label = label[j].item()
                    break
            print(label)

        # create f_likelihood log likelihood
        def adapt_to_cuda(f_likelihood):
            def f(x):
                x = torch.Tensor(x)
                x = x.reshape(1,1,28,28)
                if torch.cuda.is_available():
                    x = x.cuda()
                x = f_likelihood(x) 
                x = x.cpu().detach().numpy().flatten()
                return x
            return f
        @adapt_to_cuda
        def f_likelihood(x): return criterion(model(x), x)
        # print(image.shape)
        res_MCMC = MCMC(image.reshape(28*28), f_likelihood)
    # plot
    plt.title(label)
    N = 3
    for i in range(N):
        for j in range(N):
            plt.subplot(N, N, 1+i+N*j)
            plt.imshow(res_MCMC[-(i+N*j)].reshape(28, 28))
    plt.show()


## visu model test set
# with torch.no_grad():
#     plt.clf()
#     # plotting the training and validation loss
#     # plt.plot(train_losses, label='Training loss')
#     # plt.legend()
#     # plt.show()

#     for j, (images, labels) in enumerate(testloader):

#         # getting the test images
#         x_test, y_test = Variable(images), Variable(labels)
#         if torch.cuda.is_available():
#             x_test = x_test.cuda()
#             y_test = y_test.cuda()

#         y_pred = model(x_test)
#         NB = 5
#         images = np.random.randint(0, high=len(y_pred), size=NB)
#         for i in range(NB):
#             plt.subplot(NB, 2, 2*i+1)
#             plt.imshow(np.squeeze(
#                 x_test[images[i]].cpu().permute(1, 2, 0)))  # pfff
#             plt.title(y_test[images[i]].cpu().detach().numpy())  # mdr c quoi
#             plt.subplot(NB, 2, 2*i+2)
#             plt.imshow(np.squeeze(y_pred[images[i]].cpu().permute(1, 2, 0)))
#         plt.show()
#         if j == 4:
#             break
#         # print(type(y_pred))


if __name__ == '__main__':
    mcmc_out_of_model()