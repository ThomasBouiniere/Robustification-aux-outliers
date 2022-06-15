# imports
import torch
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt


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
        # to format following non-conv models(flattened models)
        if x.shape == torch.Size([1, 784]):
            x = x.reshape([1, 1, 28, 28])
            if torch.cuda.is_available():
                x = x.cuda()
            x = self.encoder(x)
            x = self.decoder(x)
            x = x.cpu().detach().numpy().flatten()
            x = x.reshape(1, 28*28)
            return x
        x = self.encoder(x)
        # x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


def MCMC(img, f, use_emcee=False):
    '''
    img: 1 28*28 flattened image to be used to initialise the mcmc
    f : function F the log energy function
    use_emcee : bool whether to use emcee or homemade mcmc algorithm
    // remi for details
    '''
    # TODO pass *emcee and *initial states parameters as args

    if use_emcee:
        # emcee
        import emcee
        pos = img + np.abs(np.random.randn(28*28*2, 28*28))

        number_of_walkers, number_of_dimensions = pos.shape
        sampler = emcee.EnsembleSampler(
            nwalkers=number_of_walkers, ndim=number_of_dimensions, log_prob_fn=f, live_dangerously=True)

        sampler.run_mcmc(initial_state=pos, nsteps=150)
        samples = sampler.get_chain(flat=True)
        return samples

    # mcmc maison
    # TODO : optimiser etc
    # TODO : verif convergence avec plot
    res = []
    number_of_walkers = 3
    pos = img + np.random.randn(number_of_walkers, 28*28)
    for i, sample in enumerate(pos):
        print(100*i/(number_of_walkers))
        try:
            samples = [sample.numpy()]  # if image taken from dataloader
        except:
            samples = [sample]  # if image gen with numpy
        for _ in range(1500):
            candidate = np.random.normal(
                samples[-1], 4)
            # PASSER EN EXP !!!
            a = f(candidate)
            b = f(samples[-1])
            p = min(1, a/b)
            if np.random.random() < p:
                samples.append(candidate)
            else:
                samples.append(samples[-1])
        res += samples
    return res


def train_model(held_digit):

    def train(epoch):

        model.train()  # sets the module in training mode
        tr_loss = 0
        # when to use mcmc
        use_neg_loss = epoch in [4, 5, 6, 7]
        if use_neg_loss:  # condition on epoch
            with torch.no_grad():
                # initial states
                # todo multiple inital states
                # une image aléatoire
                # initial_state = [0]*28*28 + np.random.rand(28*28)*0.1
                initial_state = np.array([0]*28*28)  # noise added in MCMC
                # create f_likelihood log likelihood

                def adapt_f(f_likelihood):
                    # todo adapt this  to model WHEN REFACTORISATION
                    '''
                    f_likelihood : function to adapt to present model (input shape different for each)
                    This is meant to be used as a decorator ( @ - see below).
                    // Remi for explanation if bugs or not clear
                    '''
                    def f(x):
                        x = torch.Tensor(x)
                        x = x.reshape(1, 1, 28, 28)
                        if torch.cuda.is_available():
                            x = x.cuda()
                        x = f_likelihood(x)
                        x = x.cpu().detach().numpy().flatten()
                        return x
                    return f

                @adapt_f
                def f_likelihood(x): return -criterion(model(x), x)
                res_mcmc = MCMC(initial_state.reshape(28*28),
                                f_likelihood, use_emcee=True)
                print("mcmc done")
                print("")
                # back to tensors
                n = len(res_mcmc)
                res_mcmc = np.array(res_mcmc)
                res_mcmc = np.reshape(res_mcmc, (n, 1, 28, 28))
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

            if use_neg_loss:
                # taking small bit of res_mcmc
                # pas toujours les mêmes # TODO mélanger res_mcmc
                x_mcmc = torch.Tensor(res_mcmc[100*i:100*(i+1)])
                if torch.cuda.is_available():
                    x_mcmc = x_mcmc.cuda()
                output_mcmc = model(x_mcmc)
                neg_loss_train = criterion(output_mcmc, x_mcmc)
                n_false = len(x_mcmc)
            else:
                neg_loss_train = 0
                n_false = 0

            # ================ computing final loss ===================

            loss_train = (n_true*pos_loss_train - n_false *
                          neg_loss_train) / (n_true+n_false)
            print(loss_train.__str__(), end='\r')
            loss_train.backward()
            optimizer.step()
            tr_loss = loss_train.item()
        print("\n")
        print('Epoch : ', epoch+1, '\t', 'loss :',
              loss_train)

    # todo ajouter params

    # # data
    # mnist_trainset = datasets.MNIST(
    #     root='./data', train=True, download=True, transform=ToTensor())
    # mnist_testset = datasets.MNIST(
    #     root='./data', train=False, download=True, transform=ToTensor())

    # # dataloaders
    # # train
    # trainloader = torch.utils.data.DataLoader(
    #     mnist_trainset, batch_size=5000)
    # # test
    # # batchsize = 1 for mcmc TO CHANGE FOR TESTING
    # testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=500)

    # avec holdouts pour tests
    # datasets
    mnist_trainset = datasets.MNIST(
        root='./data', train=True, download=True, transform=ToTensor())
    mnist_testset = datasets.MNIST(
        root='./data', train=False, download=True, transform=ToTensor())

    # holdout
    indexes = []
    neg_indexes = []
    held_digits = [held_digit]  # changes batch size !
    for i in range(len(mnist_trainset)):
        if mnist_trainset[i][1] not in held_digits:
            indexes.append(i)
        else:
            neg_indexes.append(i)
    # print(len(indexes)/len(mnist_trainset))

    # torch.utils.data.Subset(trainset, idx)
    mnist_trainset_pos = torch.utils.data.Subset(
        mnist_trainset, indexes)
    mnist_trainset_neg = torch.utils.data.Subset(
        mnist_trainset, neg_indexes)

    # dataloaders
    # train
    trainloader = torch.utils.data.DataLoader(
        mnist_trainset_pos, batch_size=5000)
    testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=500)
    # MODEL

    # defining the model
    model = Net()
    # defining the optimizer and lr
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    # defining the loss function
    # criterion = nn.CrossEntropyLoss() # for CNN
    criterion = nn.MSELoss()  # for autoencoder

    # checking if GPU is available
    if torch.cuda.is_available():
        print("On GPU !")
        model = model.cuda()
        criterion = criterion.cuda()

    # defining the number of epochs > 2 for mcmc loss
    n_epochs = 8
    # empty list to store training losses
    train_losses = []
    # training the model
    for epoch in range(n_epochs):
        train(epoch)
    return model


def mcmc_out_of_model(model):
    with torch.no_grad():
        # get images
        dataiter = iter(testloader)

        # CHOOSE IMAGE NUMBER TO TARGET VIA MCMC
        target_label = 9
        # "zero" for initial state @ zero array
        # "rd" for random initial states
        if target_label == "zero":
            image = [0]*28*28
        elif target_label == "rd":
            # TODO RESTER BORNES DATASET
            image = [0]*28*28 + np.random.rand(28*28)*e-2
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
                x = x.reshape(1, 1, 28, 28)
                if torch.cuda.is_available():
                    x = x.cuda()
                x = f_likelihood(x)
                x = x.cpu().detach().numpy().flatten()
                return x
            return f

        @adapt_to_cuda
        def f_likelihood(x): return -criterion(model(x), x)
        # print(image.shape)
        res_MCMC = MCMC(image.reshape(28*28), f_likelihood, use_emcee=True)
    # plot
    plt.title(label)
    N = 3
    for i in range(N):
        for j in range(N):
            plt.subplot(N, N, 1+i+N*j)
            plt.imshow(res_MCMC[-(i+N*j)].reshape(28, 28))
    plt.show()


def results(models):
    '''
    list of tuples (model, modelname)
    '''
    # decomposer le test set
    with torch.no_grad():
        mnist_testset = datasets.MNIST(
            root='./data', train=False, download=True, transform=ToTensor())
        models_res = []
        for model in models:
            model = model[0]
            res = [0]*10
            for i in range(10):
                indexes = []
                held_digits = [i]
                # select digit
                for idx in range(len(mnist_testset)):
                    if mnist_testset[idx][1] in held_digits:
                        indexes.append(idx)
                mnist_trainset_held = torch.utils.data.Subset(
                    mnist_testset, indexes)
                # dataloader
                test_loader = torch.utils.data.DataLoader(
                    mnist_trainset_held, batch_size=500)
                # results
                for j, (images, labels) in enumerate(test_loader):
                    # getting the test images
                    x_test, y_test = Variable(images), Variable(labels)
                    if torch.cuda.is_available():
                        x_test = x_test.cuda()
                        y_test = y_test.cuda()
                    y_pred = model(x_test)
                    criterion = nn.MSELoss()
                    res_i = criterion(x_test, y_pred)
                    res[i] += res_i.cpu().numpy()
                res[i] = res[i]/(j+1)  # keep mean value over j iterations
            mean_res = np.mean(res)
            res = [item/mean_res for item in res]
            models_res.append(res)
        n = len(models)
        for i in range(n):
            plt.bar([j+(i/(2*n)) for j in range(10)],
                    models_res[i], width=1/(2*n), label=models[i][1])
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        plt.legend()
        plt.show()


def visu_model(model):
    # # data
    # datasets
    mnist_testset = datasets.MNIST(
        root='./data', train=False, download=True, transform=ToTensor())
    # dataloaders
    testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=500)

    # visu model test set
    with torch.no_grad():
        plt.clf()
        # plotting the training and validation loss
        # plt.plot(train_losses, label='Training loss')
        # plt.legend()
        # plt.show()

        for j, (images, labels) in enumerate(testloader):

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
            if j == 4:
                break
            # print(type(y_pred))


if __name__ == '__main__':
    # # new NAE
    held_digit = 3
    model = train_model(held_digit)
    torch.save(model, "saved_models/model3.pth")  # gets stuck >< whyyyy
    # plot
    # models = []
    # models.append((torch.load("saved_models/model0.pth"), "model0"))
    # models.append((torch.load("saved_models/model1.pth"), "model1"))
    # models.append((torch.load("saved_models/model3.pth"), "model3"))
    # models.append((torch.load("saved_models/model5.pth"), "model5"))
    # models.append((torch.load("saved_models/model6.pth"), "model6"))
    # models.append((torch.load("saved_models/model8.pth"), "model8"))
    # models.append((torch.load("saved_models/model9.pth"), "model9"))

    # # mcmc_out_of_model(model) # samples model a posteriori
    # # plots images vs reconstruction
    # visu_model(torch.load("saved_models/model8.pth"))
    # results(models)

    print("all done !")
