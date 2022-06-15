# mcmc methods to be used in training (less code redundancy)
import numpy as np
import torch


def MCMC(img, f, use_emcee=False):
    '''
    img: 1 28*28 flattened image to be used to initialise the mcmc
    f : function F the log energy function
    use_emcee : bool whether to use emcee or homemade mcmc algorithm
    // remi for details
    '''
    # TODO pass *emcee and *initial states parameters as args
    # TODO different shape for nae remi

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


def get_negative_samples(model, criterion, n, use_emcee=False, shape=(28*28, 1), gpu=False):
    '''
    model : your pytorch model
    criterion: your loss function
    n: int  number of samples to return
    use_emcee
    shape: shape of images to use in model (//remi)
    cpu : use cuda (//remi)
    Returns n samples from the autoencoder distribution using mcmc 
    '''
    gpu = torch.cuda.is_available() and gpu
    # initial states
    # todo multiple inital states
    # une image aléatoire
    # initial_state = [0]*28*28 + np.random.rand(28*28)*0.1
    initial_state = np.array([0]*28*28)  # noise added in MCMC
    # create f_likelihood log likelihood

    def adapt_f(f_likelihood, shape=shape):
        # todo adapt this  to model WHEN REFACTORISATION
        '''
        f_likelihood : function to adapt to present model (input shape different for each)
        shape: size tuple of images to pass in autoencoder
        This is meant to be used as a decorator ( @ - see below).
        // Remi for explanation if bugs or not clear
        '''
        print(shape)
        if shape == (28*28, 1):
            print("hello")

            def f(x):
                x = torch.Tensor(x)
                x = x.reshape(1, 1, 784)
                if gpu:
                    x = x.cuda()
                x = f_likelihood(x)
                if gpu:
                    x = x.cpu()
                x = x.detach().numpy().flatten()
                return x

        def f(x):
            x = torch.Tensor(x)
            x = x.reshape(1, 1, 28, 28)
            if gpu:
                x = x.cuda()
            x = f_likelihood(x)
            if gpu:
                x = x.cpu()
            x = x.detach().numpy().flatten()
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
    return res_mcmc[:n]
