# End-to-end training of parameters of PNet, ANet, GNet
from train_utils import *
from losses import *
import torch.optim as optim

learning_rate = 0.002
beta_1 = 0.0
beta_2 = 0.99

params = list(PNet.parameters()) + list(ANet.parameters()) + list(GNet.parameters())  # TODO: intialise models
optimiser = optim.Adam(params, lr=learning_rate, betas=(beta_1, beta_2))


def training_loop(I_s, I_t, epochs):  # epochs not mentioned
    for epoch in range(epochs):
        I_s_prime, I_s_t_prime = get_generated_pairs(I_s, I_t)
        loss = compute_total_loss(I_s, I_t, I_s_prime, I_s_t_prime)
        optimizer.zero_grad()
        loss.backward()
        optimiser.step()

