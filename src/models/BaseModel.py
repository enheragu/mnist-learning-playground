#!/usr/bin/env python3
# encoding: utf-8

import os
import time
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

from utils.yaml_utils import updateMetricsLogFile
from utils.log_utils import log, logTable

class BaseModelTrainer(nn.Module):
    def __init__(self, input_size, num_classes, learning_rate=0.001, patience=10, device=None, output_path = ""):
        super(BaseModelTrainer, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.patience = patience
        self.to(self.device)

        self.model_name = type(self).__name__
        self.base_output_path = output_path

        self.batch_size = 64

        if self.base_output_path is not None:
            self.output_data_path = os.path.join(self.base_output_path, self.model_name)
            os.makedirs(self.output_data_path, exist_ok=True)

            self.best_trained_path = os.path.join(self.output_data_path,f"{os.getpid()}_best_model.pth")
            self.model_architecture_path = os.path.join(self.output_data_path,"model_architecture.pth")

        # List of accuracy on each epoch
        self.accuracy_each_epoch = []

    # Entrenamiento del modelo
    def train_model(self, train_loader, test_loader, num_epochs = 500):
        
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        epochs_without_improvement = 0
        best_accuracy = 0.0

        with tqdm(range(num_epochs), desc=f"{self.model_name}", unit="epoch") as pbar:
            for epoch in pbar:
                start_time = time.time()
                self.train()
                for images, labels in train_loader:
                    # Mover los datos a la GPU
                    images, labels = images.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                accuracy = self.evaluate_model(test_loader)['accuracy']
                end_time = time.time()
                epoch_duration = end_time - start_time
                self.accuracy_each_epoch.append(accuracy)
                
                log(f"\tEpoch [{epoch + 1}/{num_epochs}], Accuracy: {accuracy * 100:.2f}%. Took {epoch_duration:.2f} seconds.")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    epochs_without_improvement = 0
                    log("\tNew best accuracy, store model...")
                    
                    if self.base_output_path is not None:
                        torch.save(self.state_dict(), self.best_trained_path)
                    else:
                        log("[ERROR] Cannot store model if no path is provided")
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.patience:
                    log(f"\tCould not get a better model for {self.patience} consecutive epochs. Stopping training process.")
                    pbar.set_postfix_str(f"Stopping early at epoch {epoch + 1}/{num_epochs}.")
                    pbar.close()
                    break
                
                pbar.set_postfix(accuracy=accuracy)

        best_epoch = epoch - self.patience
        return best_epoch

    def evaluate_model(self, test_loader, print_log=False):
        self.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                # Mover los datos a la GPU
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds)

        # precision = precision_score(all_labels, all_preds, average='weighted')
        # recall = recall_score(all_labels, all_preds, average='weighted')
        # f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        metrics = {
            "accuracy": accuracy#.tolist()
            # "precision": precision.tolist(),
            # "recall": recall.tolist(),
            # "f1_score": f1.tolist(),
            # "all_labels": [label.tolist() for label in all_labels], # always the same...
            # "all_preds": [pred.tolist() for pred in all_preds]
        }
        if print_log:
            log()
            log("\tMetrics:", metrics)
            log()
            log("\tClassification Report:\n", classification_report(all_labels, all_preds, digits=5))
            log("\tConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))

        return metrics


    def get_model_summary(self):
        log("\nModel information:")

        # Mostrar el resumen del modelo
        log(self)

        # Número total de parámetros (incluye pesos y sesgos)
        total_params = sum(p.numel() for p in self.parameters())
        log(f"Total parameters: {total_params}")

        # Número de parámetros entrenables (aquellos que requieren gradientes)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log(f"Trainable parameters: {trainable_params}")

        # Obtener los tamaños de cada capa
        log("\nLayer details:")
        for name, param in self.named_parameters():
            log(f"{name}: {param.size()}")

        log("")

    def save_architecture(self):
        if self.base_output_path is not None:
            torch.save(self, self.model_architecture_path)
        else:
            log("[ERROR] Cannot store architecture if no path is provided.")

    def load_best_model(self):

        if self.base_output_path is not None:
            self.load_state_dict(torch.load(self.best_trained_path))
            log("Best model loaded.")
        else:
            log("Not known path from where to load best model.")

    def forward(self, x):
        raise NotImplementedError("This method should be implemented in subclasses.")
    

    def spinTrainEval(self, train_loader, test_loader, num_epochs = 500):
        
        log(f"[{self.model_name}] Init spin in {self.device} device.")

        start_time = time.time()
        best_epoch = self.train_model(train_loader, test_loader, num_epochs)
        end_time = time.time()
        train_duration = end_time - start_time
   
        log(f"[{self.model_name}] Training took {train_duration:.2f} seconds.")
        log(f"[{self.model_name}] Final evaluation for best model for iteration.")
        self.load_best_model()
        metrics = self.evaluate_model(test_loader, False)
    
        # self.get_model_summary()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f".{datetime.now().microsecond // 1000:03d}"
        
        metrics.update({'best_epoch': best_epoch, 
                        'train_duration': train_duration, 
                        'total_epochs': best_epoch+self.patience, 
                        'accuracy_plot': self.accuracy_each_epoch,
                        'seed': self.seed})
        return {timestamp: metrics}
    
    """
        Initialize each layer of a given model
        
        - Para capas lineales y convolucionales: inicialización uniforme.
        - Para capas recurrentes: inicialización xavier normal.
        - Los sesgos se inicializan en cero si existen.
    """
    def _initialize_weights(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)
        
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_normal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight,1)
                nn.init.constant_(module.bias,0)

