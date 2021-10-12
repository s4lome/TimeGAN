import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules_and_training import Time_GAN_module
import copy
from torch import optim

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
 
 # set up test optimizers   
embedder_optimizer = optim.Adam(Embedder.parameters(), lr=0.001)
recovery_optimizer = optim.Adam(Recovery.parameters(), lr=0.001)
supervisor_optimizer = optim.Adam(Supervisor.parameters(), lr=0.001)
discriminator_optimizer = optim.Adam(Discriminator.parameters(), lr=0.005)
generator_optimizer = optim.Adam(Generator.parameters(), lr=0.01)
   
##############################################################################
# Test Embedder Update 
# get params before update
params = []    
for name, param in Embedder.named_parameters():
    if param.requires_grad:
        params.append(param.data)

#deep copy neccessary as updating is done inplace
embedder_copy = copy.deepcopy(params)        

# do an update step
MSE_loss = nn.MSELoss()
test_loss = MSE_loss(embedder_output, torch.rand(24,hidden_dim))
test_loss.backward()
embedder_optimizer.step()

# get params after updating
params = []    
for name, param in Embedder.named_parameters():
    if param.requires_grad:
        params.append(param.data)  
    
if torch.all(embedder_copy[0] != params[0]):
    print('Embedder updating is working as intended.')

###############################################################################
# Test Recovery Update
# get params before update
params = []    
for name, param in Recovery.named_parameters():
    if param.requires_grad:
        params.append(param.data)

#deep copy neccessary as updating is done inplace
recovery_copy = copy.deepcopy(params)        

# do an update step
MSE_loss = nn.MSELoss()
test_loss = MSE_loss(Recovery(torch.unsqueeze(Embedder(torch.rand(1, seq_len, dim))[0], 0))[0]
                     , torch.rand(24,dim))
test_loss.backward()
recovery_optimizer.step()

# get params after updating
params = []    
for name, param in Recovery.named_parameters():
    if param.requires_grad:
        params.append(param.data)  
    
if torch.all(recovery_copy[0] != params[0]):
    print('Recovery updating is working as intended.')
    
###############################################################################
# Test Supervisor Update
# get params before update
params = []    
for name, param in Supervisor.named_parameters():
    if param.requires_grad:
        params.append(param.data)

#deep copy neccessary as updating is done inplace
supervisor_copy = copy.deepcopy(params)        

# do an update step
MSE_loss = nn.MSELoss()
test_loss = MSE_loss(Supervisor(torch.unsqueeze(Embedder(torch.rand(1, seq_len, dim))[0], 0))[0]
                     , torch.rand(24,hidden_dim))
test_loss.backward()
supervisor_optimizer.step()

# get params after updating
params = []    
for name, param in Recovery.named_parameters():
    if param.requires_grad:
        params.append(param.data)  
    
if torch.all(recovery_copy[0] != params[0]):
    print('Supervisor updating is working as intended.')
    
###############################################################################
# Test Generator Update
# get params before update
params = []    
for name, param in Generator.named_parameters():
    if param.requires_grad:
        params.append(param.data)

#deep copy neccessary as updating is done inplace
generator_copy = copy.deepcopy(params)        

# do an update step
MSE_loss = nn.MSELoss()
test_loss = MSE_loss(Generator(torch.rand(1, seq_len, dim))[0]
                     , torch.rand(24,hidden_dim))
test_loss.backward()
generator_optimizer.step()

# get params after updating
params = []    
for name, param in Recovery.named_parameters():
    if param.requires_grad:
        params.append(param.data)  
    
if torch.all(recovery_copy[0] != params[0]):
    print('Generator updating is working as intended.')
    
###############################################################################
# Test Discriminator Update
# get params before update
params = []    
for name, param in Discriminator.named_parameters():
    if param.requires_grad:
        params.append(param.data)

#deep copy neccessary as updating is done inplace
discriminator_copy = copy.deepcopy(params)        

# do an update step
MSE_loss = nn.MSELoss()
test_loss = MSE_loss(Discriminator(torch.unsqueeze(Generator(torch.rand(1, seq_len, dim))[0],0))[0]
                     , torch.rand(1))
test_loss.backward()
discriminator_optimizer.step()

# get params after updating
params = []    
for name, param in Recovery.named_parameters():
    if param.requires_grad:
        params.append(param.data)  
    
if torch.all(recovery_copy[0] != params[0]):
    print('Discriminator updating is working as intended.')
