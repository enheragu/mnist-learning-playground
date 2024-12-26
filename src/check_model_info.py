
import os
import torch
import torch.nn as nn
from torchsummary import summary

from models import SimplePerceptron, HiddenLayerPerceptron, DNN_6L, CNN_14L, CNN_3L, CNN_4L, CNN_5L

from utils.log_utils import log, logTable
from train_models import output_path

analysis_path = './analysis_results/model_info'
input_size = 28*28

## Custom color definitions
c_blue = "#0171ba"
c_green = "#78b01c"
c_yellow = "#f6ae2d"
c_red = "#f23535" 
c_purple = "#a66497"
c_grey = "#769393"
c_darkgrey = "#2a2b2e"

# Extended HEX color with 50% transparenci (last 80 number)
c_alpha_blue = "#0171ba4D"
c_alpha_green = "#78b01c4D"
c_alpha_yellow = "#f6ae2d4D"
c_alpha_red = "#f235354D"
c_alpha_purple = "#a664974D"
c_alpha_grey = "#7693934D"
c_alpha_darkgrey = "#2a2b2e4D"


color_palette_list = [c_blue,c_green,c_yellow,c_red,c_purple,c_grey,c_darkgrey]


def generate_mermaid_diagram(model, input_size):

    layers = [["input_image", 'Input Image (28x28, 1ch)', ':::noBox']]
    def process_layers(model):
        for name, layer in model.named_children():
            layer_type = layer.__class__.__name__
            if isinstance(layer, nn.Sequential):
                process_layers(layer)
            elif isinstance(layer, nn.Conv2d):
                layers.append([f"{layer_type}_{name}",f"<b>{layer_type}</b>({layer.kernel_size}, {layer.out_channels}ch)", ':::blockStyle'])
            elif isinstance(layer, nn.BatchNorm2d):
                layers[-1][1] += f"; <b>{layer_type}</b>({layer.num_features})"
            elif isinstance(layer, nn.BatchNorm1d):
                layers[-1][1] += f"; <b>{layer_type}</b>({layer.num_features})"
            elif isinstance(layer, nn.ReLU):
                layers[-1][1] += f"; <b>{layer_type}</b>"
            elif isinstance(layer, nn.MaxPool2d):
                layers.append([f"{layer_type}_{name}",f"<b>{layer_type}</b>({layer.kernel_size})", ':::blockStyle'])
            elif isinstance(layer, nn.Flatten):
                # layers.append([f"{layer_type}_{name}",f"{layer_type}"])
                pass
            elif isinstance(layer, nn.Linear):
                layers.append([f"{layer_type}_{name}",f"<b>{layer_type}</b>({layer.in_features}, {layer.out_features})", ':::blockStyle'])
            else:
                log(f"--- ERROR - {layer}")
    process_layers(model)
    layers.append(['output', 'Output (10)', ':::noBox'])
    # log(f"{model.model_name} layers:\n{layers}")

    # Group repeated layers!
    grouped_layers = []
    i = 0
    while i < len(layers):
        current_description = layers[i][1]
        modifier = layers[i][2]
        count = 1
        while i + count < len(layers) and layers[i + count][1] == current_description:
            count += 1
        if count > 1:
            # Crear un subgraph para capas repetidas
            subgraph_name = f"subgraph_{i}"
            grouped_layers.append((subgraph_name, current_description, modifier, count))
            i += count
        else:
            grouped_layers.append(layers[i])
            i += 1     

    diagram = "graph TD\n"
    
    for layer in grouped_layers:
        if isinstance(layer, tuple):
            # Subgraph para capas repetidas
            subgraph_name, description, modifier, count = layer
            diagram += f'    subgraph {subgraph_name} [" "]\n'
            diagram += f'        {subgraph_name}_count("<b>x{count}</b>"):::noBox\n'
            diagram += f'        {subgraph_name}_block("{description}"){modifier}\n'
            diagram += f'    end\n'
            diagram += f'    style {subgraph_name} fill:{c_alpha_purple},stroke:{c_purple},stroke-width:2px,rx:10px,ry:10px\n'
    
        else:
            # Nodo individual
            diagram += f'    {layer[0]}("{layer[1]}"){layer[2]}\n'
            
    diagram += "\n\n"
    diagram += f"    classDef blockStyle fill:{c_alpha_blue},stroke:{c_blue},stroke-width:2px\n"
    diagram += f"    classDef noBox fill:none,stroke:none;"
    diagram += "\n\n"
    
    connections = []
    prev_node = None
    for layer in grouped_layers:
        if isinstance(layer, tuple):
            subgraph_name = layer[0]
            current_node = f"{subgraph_name}"
        else:
            current_node = layer[0]
        if prev_node:
            connections.append(f"    {prev_node} --> {current_node}")
        prev_node = current_node

    diagram += "\n".join(connections)
    # diagram += "\n\n%% Min width for each node"
    # diagram += "\n".join([f'    style {layers[i][0]} width:500px;' for i in range(len(layers))])
    diagram += "\n"

    return diagram



if __name__ == "__main__":

    os.makedirs(analysis_path, exist_ok=True)
    for Model in [SimplePerceptron, HiddenLayerPerceptron, DNN_6L, CNN_14L, CNN_3L, CNN_4L, CNN_5L]:
        model = Model(input_size=input_size, num_classes=10, output_path=output_path)
        log(f"Summary of {model.model_name} model with input size {input_size} for 10 classes classification:")
        summary(model, input_size=(1, 28, 28))  # Tamaño de entrada: (canales, alto, ancho)


        diagram = generate_mermaid_diagram(model, (1, 28, 28))

        file_path = os.path.join(analysis_path,f"{model.model_name}_diagram.mmd")
        with open(file_path, "w") as file:
            file.write(diagram)
        
        log(f"Mermaid diagram for {model.model_name} saved to {file_path}.")