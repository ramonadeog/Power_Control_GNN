# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 09:56:04 2022

@author: Daniel Abode

This code implements the functions for the power control GNN algorithm

References:
D. Abode, R. Adeogun, and G. Berardinelli, “Power control for 6g industrial wireless subnetworks: A graph neural network approach,”
2022. [Online]. Available: https://arxiv.org/abs/2212.14051  
"""

import numpy as np                         
import torch
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid

# --- EE hyperparameters (choose reasonable defaults) ---
BANDWIDTH_HZ = 5e6          # 5 MHz; use the same B used in noise calc
PA_EFFICIENCY = 1         # Typical OFDM PA efficiency (adjust as needed)
CIRCUIT_POWER_W = 0       # 100 mW per transmitter (adjust)
EPS = 1e-12                  # numerical floor



def create_features(dist_matrix, power_matrix):
    K = power_matrix.shape[1]
    mask = np.eye(K)
    mask = np.expand_dims(mask,axis=0)
    mask_1 = 1 - mask
    rcv_power = np.multiply(mask, power_matrix)
    int_dist_matrix = np.multiply(mask_1, dist_matrix)
    feature = rcv_power + int_dist_matrix
    return feature

def normalize_data(train_data, test_data):
    Nt = 1
    train_K = train_data.shape[1]
    test_K = test_data.shape[1]
    train_layouts = train_data.shape[0]
    tmp_mask = np.eye(train_K)
    mask = tmp_mask
    mask = np.expand_dims(mask,axis=0)
    
    train_copy = np.copy(train_data)
    diag_H = np.multiply(mask,train_copy)
    diag_mean = np.sum(diag_H/Nt)/train_layouts/train_K
    diag_var = np.sqrt(np.sum(np.square(diag_H))/train_layouts/train_K/Nt)
    tmp_diag = (diag_H - diag_mean)/diag_var

    off_diag = train_copy - diag_H 
    off_diag_mean = np.sum(off_diag/Nt)/train_layouts/train_K/(train_K-1)
    off_diag_var = np.sqrt(np.sum(np.square(off_diag))/Nt/train_layouts/train_K/(train_K-1))
    tmp_off = (off_diag - off_diag_mean)/off_diag_var 
    tmp_off_diag = tmp_off - np.multiply(tmp_off,mask) 
    
    norm_train = np.multiply(tmp_diag,mask) + tmp_off_diag 
    
    tmp_mask = np.eye(test_K)
    mask = tmp_mask
    mask = np.expand_dims(mask,axis=0)
    
    test_copy = np.copy(test_data)
    diag_H = np.multiply(mask,test_copy)
    tmp_diag = (diag_H - diag_mean)/diag_var
    
    off_diag = test_copy - diag_H
    tmp_off = (off_diag - off_diag_mean)/off_diag_var
    tmp_off_diag = tmp_off - np.multiply(tmp_off,mask)
    
    norm_test = np.multiply(tmp_diag,mask) + tmp_off_diag
    return norm_train, norm_test

def create_graph_list(features, powers):
    Graph_list = []
    for i in range(features.shape[0]):
        feature1 = features[i,:,:]
        mask = np.eye(feature1.shape[0])

        nodes_feature1 = np.sum(mask * feature1, axis=1)
        
        edges_features1 = (1-mask) * feature1

        nodes_features_ = np.concatenate((np.ones_like(np.expand_dims(nodes_feature1,-1)),np.expand_dims(nodes_feature1, -1)), axis=1)
        
        nodes_features = torch.tensor(nodes_features_,dtype=torch.float)
        
        edges_features1 = (1-mask) * feature1
        
        edges = torch.tensor(np.transpose(np.argwhere(edges_features1)), dtype=torch.long)
        
        edges_features1_ = np.expand_dims(edges_features1[np.nonzero(edges_features1)],-1) 
        
        edges_features = torch.tensor(edges_features1_,dtype=torch.float)
        
        graph = Data(nodes_features, edges, edges_features, y=torch.tensor(powers[i],dtype=torch.float))
        
        Graph_list.append(graph)
        
    return Graph_list

class NNConv(MessagePassing):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(NNConv, self).__init__(aggr='mean', **kwargs)

        self.mlp1 = mlp1
        self.mlp2 = mlp2
        
    def update(self, aggr_out, x): 
        tmp = torch.cat([x, aggr_out], dim=1) 
        comb = self.mlp2(tmp)
        
        return torch.cat([comb, x[:,1:3]],dim=1)
        
    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr) 

    def message(self, x_i, x_j, edge_attr): 
        tmp = torch.cat([x_j, edge_attr], dim=1)
        agg = self.mlp1(tmp)
        return agg

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1,self.mlp2)
    
class PCGNN(torch.nn.Module):
    def __init__(self):
        super(PCGNN, self).__init__()
        self.mlp1 = Seq(Lin(3,32), Lin(32,32),  Lin(32,32),  ReLU())
        self.mlp2 = Seq(Lin(34,32),  Lin(32,16), Lin(16,1), Sigmoid())
        self.conv = NNConv(self.mlp1,self.mlp2)

    def forward(self, data):
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x1 = self.conv(x = x0, edge_index = edge_index, edge_attr = edge_attr)
        x2 = self.conv(x = x1, edge_index = edge_index, edge_attr = edge_attr)
        out = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr)
        return out
    
def myloss2(out, data, batch_size, num_subnetworks,Noise_power, device):
    out = out.reshape([-1,num_subnetworks])

    out = out.reshape([-1,num_subnetworks,1,1])
    power_mat = data.y.reshape([-1,num_subnetworks,num_subnetworks,1])
    weighted_powers = torch.mul(out,power_mat)
    eye = torch.eye(num_subnetworks).to(device)
    desired_rcv_power = torch.sum(torch.mul(weighted_powers.squeeze(-1),eye), dim=1)
    
    Interference_power = torch.sum(torch.mul(weighted_powers.squeeze(-1),1-eye), dim=1)
    signal_interference_ratio = torch.divide(desired_rcv_power,Interference_power+Noise_power)
    capacity = torch.log2(1+signal_interference_ratio)

    
    Capacity_ = torch.mean(torch.sum(capacity, axis=1))
    
    return torch.neg(Capacity_/num_subnetworks)

# def myloss3(out, data, batch_size, num_subnetworks,Noise_power, device):
#     out = out.reshape([-1,num_subnetworks])

#     out = out.reshape([-1,num_subnetworks,1,1])
#     power_mat = data.y.reshape([-1,num_subnetworks,num_subnetworks,1])
#     weighted_powers = torch.mul(out,power_mat)
#     eye = torch.eye(num_subnetworks).to(device)
#     desired_rcv_power = torch.sum(torch.mul(weighted_powers.squeeze(-1),eye), dim=1)
    
#     Interference_power = torch.sum(torch.mul(weighted_powers.squeeze(-1),1-eye), dim=1)
#     signal_interference_ratio = torch.divide(desired_rcv_power,Interference_power+Noise_power)
#     capacity = torch.log2(1+signal_interference_ratio)
#     EE = capacity/(weighted_powers.squeeze(-1)/0.8+0.1)
#     EE_ = torch.mean(torch.sum(EE, axis=1))
#     #Capacity_ = torch.mean(torch.sum(capacity, axis=1))
    
#     return torch.neg(EE_/num_subnetworks)

def myloss3(
    out, 
    data, 
    batch_size, 
    num_subnetworks, 
    Noise_power, 
    device,
    Pmax_lin=1.0,        # max TX power in Watts used to scale the sigmoid outputs
    eta=0.8,             # PA efficiency (0<eta<=1). Your code used 0.8 implicitly.
    Pc_W=0.1,            # circuit power per TX in Watts
    bandwidth_Hz=None,   # set e.g. 5e6 to get bits/s; leave None for bits/s/Hz
    eps=1e-12
):
    # out: (batch*num_subnetworks, 1) or (batch, num_subnetworks, 1)
    out = out.reshape([-1, num_subnetworks])                     # [B, K]
    B = out.shape[0]

    # Transmit power per link (Watts); GNN outputs are fractions in (0,1)
    p_tx = out * Pmax_lin                                        # [B, K]

    # Channel gain matrix per sample
    H = data.y.reshape([B, num_subnetworks, num_subnetworks, 1]) # [B, K, K, 1]

    # Received powers H * p; build a [B,K,1] vector of per-link powers
    p_vec = p_tx.reshape([B, num_subnetworks, 1, 1])             # [B, K, 1, 1]
    weighted_powers = torch.mul(H, p_vec).squeeze(-1)            # [B, K, K]

    eye = torch.eye(num_subnetworks, device=device)              # [K, K]

    # Desired and interference received powers per link (Watts)
    desired_rcv_power = torch.sum(weighted_powers * eye, dim=1)          # [B, K]
    interference_power = torch.sum(weighted_powers * (1.0 - eye), dim=1) # [B, K]

    # SINR and per-link spectral efficiency (bits/s/Hz)
    sinr = desired_rcv_power / (interference_power + Noise_power + eps)  # [B, K]
    se_bits_per_s_per_Hz = torch.log2(1.0 + sinr)                        # [B, K]

    # If you want bits/s, scale by bandwidth; otherwise keep bits/s/Hz
    if bandwidth_Hz is not None:
        rate_bits_per_s = bandwidth_Hz * se_bits_per_s_per_Hz            # [B, K]
    else:
        rate_bits_per_s = se_bits_per_s_per_Hz

    # Power consumption per transmitter (Watts), per link
    # IMPORTANT: use TX power, not received power
    p_consumption = (p_tx / max(eta, eps)) + Pc_W                         # [B, K]

    # Per-link energy efficiency (bits/J) or (bits/J/Hz if bandwidth_Hz=None)
    ee = rate_bits_per_s / (p_consumption + eps)                          # [B, K]

    # Average over users and batch, then negate for loss
    ee_avg = torch.mean(torch.sum(ee, dim=1) / num_subnetworks)           # scalar
    return -ee_avg

def network_energy_efficiency_loss(
    out,
    data,
    batch_size, 
    num_subnetworks,
    Noise_power,
    device,
    Pmax_lin=1.0,       # max transmit power (Watts)
    eta=0.8,            # PA efficiency
    Pc_W=0.1,           # circuit power per transmitter (Watts)
    bandwidth_Hz=5e6,   # optional, set to None for bits/J/Hz
    eps=1e-12
):
    """
    Compute negative network-wide energy efficiency over the batch.
    
    Inputs:
        out            : [batch*K, 1] GNN output (Sigmoid power fractions)
        data.y         : [batch, K, K, 1] channel gain matrices
        num_subnetworks: K, number of links
        Noise_power    : receiver noise power (Watts)
        Pmax_lin       : maximum TX power (Watts)
        eta            : PA efficiency
        Pc_W           : circuit power per TX (Watts)
        bandwidth_Hz   : bandwidth to scale rate (bits/s). If None, leave bits/s/Hz
        eps            : small number to avoid division by zero
    Returns:
        loss : scalar (negative network EE averaged over batch)
    """
    batch_size = out.shape[0] // num_subnetworks
    B = batch_size
    K = num_subnetworks

    # reshape GNN output to [B, K]
    w = out.view(B, K)  # power fractions
    p_tx = w * Pmax_lin

    # reshape channel gains
    H = data.y.view(B, K, K, 1)
    weighted_powers = (H * p_tx.view(B, K, 1, 1)).squeeze(-1)  # [B, K, K]

    eye = torch.eye(K, device=weighted_powers.device)

    # Desired and interference received powers
    desired_rcv = torch.sum(weighted_powers * eye, dim=1)        # [B, K]
    interference = torch.sum(weighted_powers * (1 - eye), dim=1) # [B, K]

    # SINR and per-link rate
    sinr = desired_rcv / (interference + Noise_power + eps)
    rate = torch.log2(1.0 + sinr)                                # bits/s/Hz

    # Scale by bandwidth if desired
    if bandwidth_Hz is not None:
        rate *= bandwidth_Hz  # bits/s

    # Power consumption per link
    p_cons = (p_tx / max(eta, eps)) + Pc_W

    # Network EE = sum of rates / sum of powers
    numerator = torch.sum(rate, dim=1)        # [B]
    denominator = torch.sum(p_cons, dim=1)   # [B]
    network_EE = numerator / (denominator + eps)  # [B]

    # Negative EE for minimization
    loss = -torch.mean(network_EE)
    return loss

import torch

def multi_objective_loss(
    out,
    data,
    batch_size, 
    num_subnetworks,
    Noise_power,
    device,
    alpha=0.5,        # tradeoff factor: 1 = prioritize sum-rate, 0 = prioritize energy saving
    Pmax_lin=1.0,     # max transmit power (Watts)
    eta=0.8,          # PA efficiency
    Pc_W=0.1,         # circuit power per transmitter (Watts)
    bandwidth_Hz=5e6, # optional, for bits/s
    eps=1e-12
):
    """
    Multi-objective loss combining sum-rate and energy efficiency / power consumption.
    
    Returns:
        loss: scalar tensor to minimize
    """
    batch_size = out.shape[0] // num_subnetworks
    B = batch_size
    K = num_subnetworks

    # reshape GNN output to [B, K]
    w = out.view(B, K)  # power fractions
    p_tx = w * Pmax_lin

    # reshape channel gains
    H = data.y.view(B, K, K, 1)
    weighted_powers = (H * p_tx.view(B, K, 1, 1)).squeeze(-1)  # [B, K, K]

    eye = torch.eye(K, device=weighted_powers.device)

    # Desired and interference received powers
    desired_rcv = torch.sum(weighted_powers * eye, dim=1)        # [B, K]
    interference = torch.sum(weighted_powers * (1 - eye), dim=1) # [B, K]

    # SINR and per-link rate
    sinr = desired_rcv / (interference + Noise_power + eps)
    rate = torch.log2(1.0 + sinr)                                # bits/s/Hz

    # Optionally scale by bandwidth
    if bandwidth_Hz is not None:
        rate *= bandwidth_Hz  # bits/s

    # Power consumption per link
    p_cons = (p_tx / max(eta, eps)) + Pc_W

    # Network sum-rate and total consumed power
    sum_rate = torch.sum(rate, dim=1)         # [B]
    total_power = torch.sum(p_cons, dim=1)    # [B]

    # Multi-objective: weighted sum (maximize rate, minimize power)
    objective = alpha * sum_rate - (1 - alpha) * total_power

    # Convert to loss for minimization
    loss = -torch.mean(objective)
    return loss

def energy_efficiency_loss(out, data, Noise_power_lin, Pmax_lin=1.0):
    """
    Negative averaged per-link energy efficiency (bits/Joule) over the batch.
    Assumes `out` are Sigmoid outputs in (0,1) interpreted as power fractions.
    - data.y : channel gain matrix H per sample (K x K) or batched layout
    - Noise_power_lin : noise power in linear Watts at the receiver
    - Pmax_lin : max transmit power (Watts) that scales the fractions
    """
    # Shape handling matches the existing sum-rate code:
    # out: (num_nodes, 1) or (batch*K, 1); data.y provides H
    w = out.view(-1)                         # power fractions in (0,1)
    K = data.num_subnetworks                 # or however K is obtained in the file
    w = w.view(-1, K)                        # [batch, K]

    # H: [batch, K, K] channel gains (desired on diag, interference off-diag)
    H = data.y                               # reuse the repo’s convention
    # Received desired power
    desired = torch.diagonal(H, dim1=-2, dim2=-1) * (w * Pmax_lin)  # [batch, K]

    # Interference power = H @ p  minus desired
    p = (w * Pmax_lin).unsqueeze(-1)         # [batch, K, 1]
    total_rx = torch.matmul(H, p).squeeze(-1)  # [batch, K]
    interf = total_rx - desired

    # SINR and per-link rate (bits/s)
    sinr = desired / (interf + Noise_power_lin + EPS)
    rate_bits_per_s = BANDWIDTH_HZ * torch.log2(1.0 + sinr)

    # Power consumption per link (Watts)
    tx_consumption = (w * Pmax_lin) / PA_EFFICIENCY
    p_cons = tx_consumption + CIRCUIT_POWER_W

    # Per-link EE (bits/Joule)
    ee = rate_bits_per_s / (p_cons + EPS)    # [batch, K]

    # Negative average EE over users and batch
    loss = -ee.mean()
    return loss


def train(model2, train_loader, optimizer, num_of_subnetworks, Noise_power, device):
    model2.train()
    total_loss = 0
    count = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model2(data)
        loss = myloss3(out[:,0].to(device), data, data.num_graphs, num_of_subnetworks, Noise_power, device, Pmax_lin=1.0,eta=0.8, Pc_W=0.01, bandwidth_Hz=None, eps=1e-12)
        #loss = network_energy_efficiency_loss(out[:,0].to(device), data, data.num_graphs,num_of_subnetworks, Noise_power, device,  Pmax_lin=1.0, eta=0.8, Pc_W=0.1, bandwidth_Hz=None, eps=1e-12)
        #loss = multi_objective_loss(out[:,0].to(device), data, data.num_graphs, num_of_subnetworks, Noise_power, device, alpha=0.00,Pmax_lin=1.0, eta=0.8, Pc_W=0.1, bandwidth_Hz=None, eps=1e-12)
        total_loss += loss.item()
        count = count+1
        loss.backward()
        optimizer.step()
        
    total = total_loss / count   
    return total

def test(model2,validation_loader, num_of_subnetworks, Noise_power, device):
    model2.eval()
    total_loss = 0
    count = 0
    for data in validation_loader:
        data = data.to(device)
        with torch.no_grad():
            out = model2(data)
            loss = myloss3(out[:,0].to(device), data, data.num_graphs, num_of_subnetworks, Noise_power, device, Pmax_lin=1.0,eta=0.8, Pc_W=0.01, bandwidth_Hz=None, eps=1e-12)

            #loss = network_energy_efficiency_loss(out[:,0].to(device), data, data.num_graphs,num_of_subnetworks, Noise_power, device,  Pmax_lin=1.0, eta=0.8, Pc_W=0.1, bandwidth_Hz=None, eps=1e-12)
            #loss = multi_objective_loss(out[:,0].to(device), data, data.num_graphs, num_of_subnetworks, Noise_power, device, alpha=0.00,Pmax_lin=1.0, eta=0.8, Pc_W=0.1, bandwidth_Hz=None, eps=1e-12)
            total_loss += loss.item()
            count = count+1
    total = total_loss / count
    print('power weight for 1 snapshot \n', out[0:20,0])
    
    return total

def trainmodel(name, model2, scheduler, train_loader, validation_loader, optimizer, num_of_subnetworks, Noise_power, device):
    loss_ = []
    losst_ = []
    for epoch in range(1,500):
        losst = train(model2, train_loader, optimizer, num_of_subnetworks, Noise_power, device)
        loss1 = test(model2,validation_loader, num_of_subnetworks, Noise_power, device)
        loss_.append(loss1)
        losst_.append(losst)
        if (loss1 == min(loss_)):
            torch.save(model2, str(name))
        print('Epoch {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(
            epoch, losst, loss1))
        scheduler.step()
    return loss_, losst_

# def mycapacity(weights, data, batch_size, num_subnetworks, Noise_power):

#     weights = weights.reshape([-1,num_subnetworks,1,1])
    
#     power_mat = data.y.reshape([-1,num_subnetworks,num_subnetworks,1])

#     weighted_powers = torch.mul(weights,power_mat)
    
#     eye = torch.eye(num_subnetworks)
    
#     desired_rcv_power = torch.sum(torch.mul(weighted_powers.squeeze(-1),eye), dim=1)
   
#     Interference_power = torch.sum(torch.mul(weighted_powers.squeeze(-1),1-eye), dim=1)

#     signal_interference_ratio = torch.divide(desired_rcv_power,Interference_power+Noise_power)
    
#     capacity = torch.log2(1+signal_interference_ratio)
    
#     return capacity, weighted_powers 

def mycapacity(
    weights,
    data,
    batch_size,
    num_subnetworks,
    Noise_power,
    Pmax_lin=1.0,      # max TX power (Watts)
    eta=0.8,           # PA efficiency
    Pc_W=0.1,          # circuit power per transmitter (Watts)
    bandwidth_Hz=None, # if None, EE is in bits/J/Hz; else bits/J
    eps=1e-12
):
    """
    Compute per-link spectral efficiency and energy efficiency.

    Returns:
        capacity          : [batch, num_subnetworks] bits/s/Hz
        weighted_powers   : [batch, num_subnetworks, num_subnetworks]
                            (received powers)
        energy_efficiency : [batch, num_subnetworks] bits/J (or bits/J/Hz)
    """
    # [B, K, 1, 1]
    weights = weights.reshape([-1, num_subnetworks, 1, 1])
    power_mat = data.y.reshape([-1, num_subnetworks, num_subnetworks, 1])

    # Transmit power per link (Watts)
    p_tx = weights.squeeze(-1).squeeze(-1) * Pmax_lin  # [B, K]

    # Received powers matrix [B, K, K]
    weighted_powers = torch.mul(weights, power_mat).squeeze(-1)

    eye = torch.eye(num_subnetworks, device=weighted_powers.device)

    # Desired and interference powers per link
    desired_rcv_power = torch.sum(weighted_powers * eye, dim=1)            # [B, K]
    Interference_power = torch.sum(weighted_powers * (1 - eye), dim=1)     # [B, K]

    # SINR and per-link capacity
    sinr = desired_rcv_power / (Interference_power + Noise_power + eps)    # [B, K]
    capacity = torch.log2(1 + sinr)                                        # bits/s/Hz

    # Optional: convert to bits/s by multiplying with bandwidth
    if bandwidth_Hz is not None:
        rate_bits_per_s = capacity * bandwidth_Hz
    else:
        rate_bits_per_s = capacity

    # Energy efficiency (bits/J or bits/J/Hz)
    p_consumption = (p_tx / max(eta, eps)) + Pc_W                          # [B, K]
    energy_efficiency = rate_bits_per_s / (p_consumption + eps)            # [B, K]

    return capacity, weighted_powers, energy_efficiency


def GNN_test(GNNmodel, test_loader, num_of_subnetworks, Noise_power,device):    
    model2 = torch.load(GNNmodel, weights_only=False)
    model2.eval()
    capacities = torch.Tensor()
    energy_eff = torch.Tensor()
    GNN_powers = torch.Tensor() 
    GNN_weights = torch.Tensor() 
    GNN_sum_rate = torch.Tensor()
    Pmax = 1
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            out = model2(data)
            cap, GNN_pow, ee = mycapacity(Pmax*out[:,0].cpu(), data.cpu(), data.num_graphs,num_of_subnetworks, Noise_power, Pmax_lin=1.0, eta=0.8, Pc_W=0.01,bandwidth_Hz=None, eps=1e-12)        
        GNN_powers = torch.cat((GNN_powers, GNN_pow.cpu()),0)
        GNN_weights = torch.cat((GNN_weights, out[:,0].cpu()),0)
        capacities = torch.cat((capacities,cap.cpu()),0)
        energy_eff = torch.cat((energy_eff, ee.cpu()),0)
        GNN_sum_rate = torch.cat((GNN_sum_rate,torch.sum(cap,1)),0)
        
    return GNN_sum_rate, capacities, GNN_weights, GNN_powers, energy_eff

def generate_cdf(values, bins_):
    data = np.array(values)
    count, bins_count = np.histogram(data, bins=bins_)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    return bins_count[1:], cdf

def findcdfvalue(x,y,yval1,yval2):
    a = x[np.logical_and(y>yval1, y<yval2)]
    if a.size < 1:
        return 0
    else:
        m = np.mean(a)

        return m.item()




















