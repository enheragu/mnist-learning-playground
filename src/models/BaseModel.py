#!/usr/bin/env python3
# encoding: utf-8

import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

from utils.yaml_utils import updateMetricsLogFile

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

        if self.base_output_path is not None:
            self.output_data_path = os.path.join(self.base_output_path, self.model_name)
            os.makedirs(self.output_data_path, exist_ok=True)

            self.best_trained_path = os.path.join(self.output_data_path,f"{os.getpid()}_best_model.pth")
            self.model_architecture_path = os.path.join(self.output_data_path,"model_architecture.pth")

    # Entrenamiento del modelo
    def train_model(self, train_loader, test_loader, num_epochs = 500):
        
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        epochs_without_improvement = 0
        best_accuracy = 0.0

        for epoch in range(num_epochs):
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
            
            print(f"\tEpoch [{epoch + 1}/{num_epochs}], Accuracy: {accuracy * 100:.2f}%. Took {epoch_duration:.2f} seconds.")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                epochs_without_improvement = 0
                print("\tNew best accuracy, store model...")
                
                if self.base_output_path is not None:
                    torch.save(self.state_dict(), self.best_trained_path)
                else:
                    print("[ERROR] Cannot store model if no path is provided")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                print(f"\tCould not get a better model for {self.patience} consecutive epochs. Stopping training process.")
                break
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

        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        metrics = {
            "accuracy": accuracy.tolist()
            # "precision": precision.tolist(),
            # "recall": recall.tolist(),
            # "f1_score": f1.tolist(),
            # "all_labels": [label.tolist() for label in all_labels], # always the same...
            #"all_preds": [pred.tolist() for pred in all_preds]
        }
        if print_log:
            print()
            print("\tMetrics:", metrics)
            print()
            print("\tClassification Report:\n", classification_report(all_labels, all_preds, digits=5))
            print("\tConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))

        return metrics


    def get_model_summary(self):
        print("\nModel information:")

        # Mostrar el resumen del modelo
        print(self)

        # Número total de parámetros (incluye pesos y sesgos)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params}")

        # Número de parámetros entrenables (aquellos que requieren gradientes)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")

        # Obtener los tamaños de cada capa
        print("\nLayer details:")
        for name, param in self.named_parameters():
            print(f"{name}: {param.size()}")

        print("")

    def save_architecture(self):
        if self.base_output_path is not None:
            torch.save(self, self.model_architecture_path)
        else:
            print("[ERROR] Cannot store architecture if no path is provided.")

    def load_best_model(self):

        if self.base_output_path is not None:
            self.load_state_dict(torch.load(self.best_trained_path))
            print("Best model loaded.")
        else:
            print("Not known path from where to load best model.")

    def forward(self, x):
        raise NotImplementedError("This method should be implemented in subclasses.")
    

    def spinTrainEval(self, train_loader, test_loader, num_epochs = 500):
        
        print(f"[{self.model_name}] Init spin in {self.device} device.")

        start_time = time.time()
        best_epoch = self.train_model(train_loader, test_loader, num_epochs)
        end_time = time.time()
        train_duration = end_time - start_time
   
        print(f"[{self.model_name}] Training took {train_duration:.2f} seconds.")
        print(f"[{self.model_name}] Final evaluation for best model for iteration.")
        self.load_best_model()
        metrics = self.evaluate_model(test_loader, False)
    
        # self.get_model_summary()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f".{datetime.now().microsecond // 1000:03d}"
        
        metrics.update({'best_epoch': best_epoch, 'train_duration': train_duration, 'total_epochs': best_epoch+self.patience})
        return {timestamp: metrics}