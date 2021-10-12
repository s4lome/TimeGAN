import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules_and_training import Time_GAN_module

# set parameters for testing
gamma = 1
no, seq_len, dim = 12800, 24, 3

z_dim = dim
num_layers=3
device = 'cpu'
hidden_dim = 10

# create modules 
Embedder = Time_GAN_module(input_size=z_dim, output_size=hidden_dim
                             , hidden_dim=hidden_dim, num_layers=num_layers, device=device)
Recovery = Time_GAN_module(input_size=hidden_dim, output_size=dim
                             , hidden_dim=hidden_dim, num_layers=num_layers, device=device)
Generator = Time_GAN_module(input_size=dim, output_size=hidden_dim
                              , hidden_dim=hidden_dim, num_layers=num_layers, device=device)
Supervisor = Time_GAN_module(input_size=hidden_dim, output_size=hidden_dim
                               , hidden_dim=hidden_dim, num_layers=num_layers, device=device)
Discriminator = Time_GAN_module(input_size=hidden_dim, output_size=1, hidden_dim=hidden_dim
                                  , num_layers=num_layers, activation=nn.Identity,
                                  device=device)

#check forward pass of each module and output dimensions of each module

test_data = torch.rand(1, seq_len, dim)

embedder_output = Embedder(test_data)[0]    
recovery_output = Recovery(torch.unsqueeze(embedder_output, 0))[0]
generator_output = Generator(test_data)[0]
supervisor_output = Supervisor(torch.unsqueeze(embedder_output,0))[0]  
discriminator_output = Discriminator(torch.unsqueeze(generator_output,0))[0]

if embedder_output.shape == (seq_len,hidden_dim):
    print('Embedder Output is of correct Dimension.')
if recovery_output.shape == (seq_len, dim):
    print('Recovery Output is of correct Dimension.')
if generator_output.shape == (seq_len, hidden_dim):
    print('Geneator Output is of correct Dimension.')
if supervisor_output.shape == (seq_len, hidden_dim):
    print('Supervisor Output is of correct Dimension.')
if supervisor_output.shape == (seq_len, hidden_dim):
    print('Supervisor Output is of correct Dimension.')
    
