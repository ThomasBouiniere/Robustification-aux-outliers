import emcee
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.sum(np.exp((-x[0]**2-x[1]**2)/2))


print(f(np.array([1, 1])))

number_of_dimensions = 2
number_of_walkers = 10
initial_states = np.random.randn(number_of_walkers, number_of_dimensions)*10
sampler = emcee.EnsembleSampler(
    nwalkers=number_of_walkers, ndim=number_of_dimensions, log_prob_fn=f)
sampler.run_mcmc(initial_state=initial_states, nsteps=50)
samples = sampler.get_chain(flat=True)


samples1 = [samples[i][0] for i in range(len(samples))]
samples2 = [samples[i][1] for i in range(len(samples))]

# plt.hist(samples1, 35, color="k", histtype="step")
# plt.xlabel("$x$")
# plt.ylabel("$p(x)$")
# plt.gca().set_yticks([])
# plt.show()


plt.hist(samples1, range=(-1000, 1000), bins=45, color="k", histtype="step")
plt.xlabel("$x$")
plt.ylabel("$p(x)$")
plt.gca().set_yticks([])
plt.show()

plt.hist(samples2, range=(-1000, 1000), bins=45, color="k", histtype="step")
plt.xlabel("$x$")
plt.ylabel("$p(x)$")
plt.gca().set_yticks([])
plt.show()

# plt.hist(samples2, 35, color="k", histtype="step")
# plt.xlabel("$x$")
# plt.ylabel("$p(x)$")
# plt.gca().set_yticks([])
# plt.show()
