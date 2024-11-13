#!/usr/bin/env python3
# encoding: utf-8

import os
import time
from datetime import datetime
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

from utils.yaml_utils import updateMetricsLogFile

# How many train loops are executed to study its variance
train_loops = 140
sameseed = False
output_path = "./output_data/SimplePerceptron"

# General configuration
input_size = 28 * 28  # Tamaño de cada imagen aplanada
num_classes = 10  # Números del 0 al 9
batch_size = 64
learning_rate = 0.001
patience = 10
num_epochs = 500

# Definir el dispositivo (GPU si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def set_seed(seed=42):
    """Establece la semilla para todas las librerías relevantes."""
    torch.manual_seed(seed)  # Para PyTorch
    np.random.seed(seed)  # Para NumPy
    torch.cuda.manual_seed(seed)  # Para PyTorch en GPU
    torch.cuda.manual_seed_all(seed)  # Para todos los dispositivos GPU
    torch.backends.cudnn.deterministic = True  # Garantiza la determinación del algoritmo de cuDNN
    torch.backends.cudnn.benchmark = False  # No optimiza los algoritmos si las dimensiones no cambian

    
class SimplePerceptron(nn.Module):
    def __init__(self, input_size, num_classes, seed=42):
        super(SimplePerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        self._initialize_weights(seed)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Aplanamos la imagen 28x28 a un vector de tamaño 784
        return self.fc1(x)

    def _initialize_weights(self,seed):
        """Inicializa los pesos de manera fija."""
        torch.manual_seed(seed)  # Asegura que la inicialización de pesos sea consistente
        nn.init.uniform_(self.fc1.weight, a=-0.1, b=0.1)  # Ejemplo de inicialización uniforme
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)  # Inicia el sesgo en 0


# Entrenamiento del modelo
def train(model, train_loader, test_loader, criterion, optimizer):
    global best_accuracy, num_epochs
    epochs_without_improvement = 0
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        for images, labels in train_loader:
            # Mover los datos a la GPU
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        accuracy = evaluate(model, test_loader)['accuracy']
        end_time = time.time()
        epoch_duration = end_time - start_time
        
        print(f"\tEpoch [{epoch + 1}/{num_epochs}], Accuracy: {accuracy * 100:.2f}%. Took {epoch_duration:.2f} seconds.")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_without_improvement = 0
            print("\tNew best accuracy, store model...")
            torch.save(model.state_dict(), os.path.join(output_path,"best_model.pth"))
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\tCould not get a better model for {patience} consecutive epochs. Stopping training process.")
            break
    best_epoch = epoch - patience
    return best_epoch

def evaluate(model, test_loader, print_log=False):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            # Mover los datos a la GPU
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    metrics = {
        "accuracy": accuracy.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1_score": f1.tolist(),
        "all_labels": [label.tolist() for label in all_labels],
        "all_preds": [pred.tolist() for pred in all_preds]
    }
    if print_log:
        print()
        print("\tMetrics:", metrics)
        print()
        print("\tClassification Report:\n", classification_report(all_labels, all_preds, digits=5))
        print("\tConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))

    return metrics


def get_model_summary(model):
    print("\nModel information:")

    # Mostrar el resumen del modelo
    print(model)

    # Número total de parámetros (incluye pesos y sesgos)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Número de parámetros entrenables (aquellos que requieren gradientes)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    # Obtener los tamaños de cada capa
    print("\nLayer details:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.size()}")

    print("")
def train_once(index, seed):

    # Cargar el dataset MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Init model, loss and optimizer
    model = SimplePerceptron(input_size=input_size, num_classes=num_classes, seed=seed).to(device)
    torch.save(model, os.path.join(output_path,"model_architecture.pth"))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"[{index}] Initialization of model complete, get into training process.")

    start_time = time.time()
    best_epoch = train(model, train_loader, test_loader, criterion, optimizer)
    end_time = time.time()
    train_duration = end_time - start_time
    print(f"Training took {train_duration:.2f} seconds.")
    print(f"[{index}] Final evaluation for best model for iteration {index}.")
    model.load_state_dict(torch.load(os.path.join(output_path,"best_model.pth"), weights_only=True))
    metrics = evaluate(model, test_loader, print_log=False)
    
    get_model_summary(model)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f".{datetime.now().microsecond // 1000:03d}"
    
    metrics.update({'best_epoch': best_epoch, 'train_duration': train_duration, 'total_epochs': best_epoch+patience})
    return {timestamp: metrics}



if __name__ == "__main__":
    seed = 0
    metrics = []
    
    print(f"Get {train_loops} iterations making use of {device}")
    for index in range(train_loops):
        if not sameseed:
            seed = random.randint(0, 2**32 - 1)  # Random 32 bits number

        set_seed(seed)
        metric_data = train_once(index, seed)
        updateMetricsLogFile(metric_data, os.path.join(output_path,f"{'sameseed_' if sameseed else 'randomseed_'}training_metrics.yaml"))

