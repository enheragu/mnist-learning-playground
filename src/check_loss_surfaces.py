#!/usr/bin/env python3
# encoding: utf-8

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from utils.log_utils import log
from models import CNN_14L
from utils import output_path


model_output_path = os.path.join(output_path, 'CNN_14L')
analysis_path = './analysis_results/loss_surface'

plot_side_num_points = 50

# Seed of the random directions, so that every run gives the same surface
direction_seed = 0

# Evaluates the mean loss over a dataloader
def eval_loss(model, loss_fn, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / total_samples

# Filter normalization (Li et al., Visualizing the Loss Landscape of Neural Nets, NeurIPS 2018): each filter of the
# direction takes the norm of the same filter in the weights. A filter is one output channel of a Conv2d or one output
# neuron of a Linear (first dimension of the tensor). One-dimensional parameters (biases and BatchNorm) are not perturbed.
def normalize_direction(direction, weights):
    normalized = []
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            normalized.append(torch.zeros_like(d))
            continue
        d_norm = d.view(d.size(0), -1).norm(dim=1)
        w_norm = w.view(w.size(0), -1).norm(dim=1)
        scale = w_norm / (d_norm + 1e-10)
        normalized.append(d * scale.view(-1, *[1] * (d.dim() - 1)))
    return normalized

# Shifts the model parameters along two directions, scaled by alpha and beta
def add_direction_to_params(model, base_params, direction1, direction2, alpha, beta):
    for p, base_p, d1, d2 in zip(model.parameters(), base_params, direction1, direction2):
        p.data = base_p + alpha * d1 + beta * d2


def generateModelLossSurface(model_param_path, analysis_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # No output path: this script only reads a stored model, so it must not create a ./CNN_14L folder
    model = CNN_14L(input_size=28*28, output_path=None).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.load_state_dict(torch.load(model_param_path, map_location=device))

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    dataloader = DataLoader(dataset=test_dataset, batch_size=model.batch_size, shuffle=False)

    # Store the base parameters
    base_params = [p.data.clone() for p in model.parameters()]

    # Generate two random directions and normalize them
    torch.manual_seed(direction_seed)
    direction1 = [torch.randn_like(p) for p in model.parameters()]
    direction2 = [torch.randn_like(p) for p in model.parameters()]
    direction1 = normalize_direction(direction1, base_params)
    direction2 = normalize_direction(direction2, base_params)

    alpha_vals = np.linspace(-1, 1, plot_side_num_points)
    beta_vals = np.linspace(-1, 1, plot_side_num_points)
    loss_surface = np.zeros((plot_side_num_points, plot_side_num_points))

    # Loss func on each grid point
    for i, alpha in enumerate(alpha_vals):
        for j, beta in enumerate(beta_vals):
            add_direction_to_params(model, base_params, direction1, direction2, alpha, beta)
            loss_surface[i, j] = eval_loss(model, loss_fn, dataloader, device)

    add_direction_to_params(model, base_params, direction1, direction2, 0, 0)
    
    base_name = os.path.splitext(os.path.basename(model_param_path))[0]
    output_path = os.path.join(analysis_path, base_name)
    np.savez(f"{output_path}.npz",
         loss_surface=loss_surface,
         alpha=alpha_vals,
         beta=beta_vals)
    log(f"Stored surface data in {output_path}")


def plotModelLossSurface(surface_data_path, analysis_path):
    data = np.load(f"{surface_data_path}.npz")
    loss_surface = data['loss_surface']
    alpha_vals = data['alpha']
    beta_vals = data['beta']

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    A, B = np.meshgrid(alpha_vals, beta_vals)
    ax.plot_surface(A, B, loss_surface.T, cmap='viridis', edgecolor='none')

    ax.set_xlabel('Direction 1 (alpha)')
    ax.set_ylabel('Direction 2 (beta)')
    ax.set_zlabel('Loss')
    ax.set_title('Loss Landscape 3D')

    base_name = os.path.splitext(os.path.basename(surface_data_path))[0]
    output_path = os.path.join(analysis_path, base_name)
    plt.savefig(f"{output_path}.pdf", format="pdf")
    log(f"Stored surface data plot in {output_path}")


if __name__ == "__main__":
    os.makedirs(f"{analysis_path}", exist_ok=True)


    model_files = glob.glob(os.path.join(model_output_path, "*_best_model.pth"))

    for model_file in model_files:
        model_path = os.path.abspath(model_file)

        base_name = os.path.splitext(os.path.basename(model_file))[0]
        output_path = os.path.join(analysis_path, base_name)

        generateModelLossSurface(model_path, analysis_path)
        plotModelLossSurface(output_path, analysis_path)