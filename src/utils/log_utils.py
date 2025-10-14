
#!/usr/bin/env python3
# encoding: utf-8

import os
from tqdm import tqdm
import tabulate

## Custom color definitions
c_blue = "#0171ba"
c_green = "#78b01c"
c_yellow = "#f6ae2d"
c_red = "#f23535" 
c_purple = "#a66497"
c_grey = "#769393"
c_darkgrey = "#2a2b2e"

# Extended HEX color 50% transparent (last 80 number)
c_alpha_blue = "#0171ba4D"
c_alpha_green = "#78b01c4D"
c_alpha_yellow = "#f6ae2d4D"
c_alpha_red = "#f235354D"
c_alpha_purple = "#a664974D"
c_alpha_grey = "#7693934D"
c_alpha_darkgrey = "#2a2b2e4D"

# Some corner cases need more colors... :(
extended_color_palette = [
    "#cd5500", "#01c201", "#1621c3", "#9a00a2", "#8c564b", "#da3c98",
    "#7f7f7f", "#bcbd22", "#17becf"
]

color_palette_list = [c_blue,c_green,c_yellow,c_red,c_purple,c_grey,c_darkgrey] + extended_color_palette


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log(*args, color=bcolors.ENDC, **kwargs):
    message = " ".join(str(arg) for arg in args)
    tqdm.write(f"{color}{message}{bcolors.ENDC}", **kwargs)


def logTable(row_data, output_path, filename, colalign = None, screen_log = True):

    table_str = tabulate.tabulate(row_data, headers="firstrow", tablefmt="fancy_grid", colalign = colalign)
    table_latex = tabulate.tabulate(row_data, headers="firstrow", tablefmt="latex", colalign = colalign)
    
    if screen_log:
        log(table_str)

    file_name = os.path.join(output_path, filename.lower().replace(' ','_'))
    log(f"Stored data table in {file_name}")
    with open(f"{file_name}.txt", 'w') as file:
        file.write(f'{filename}\n')
        file.write(table_str)

    headers = row_data[0]
    for i in range(len(headers)):
        table_latex = table_latex.replace(headers[i], f"\\textbf{{{headers[i]}}}")

    caption = f"{filename}"
    label = f"tab:{filename.lower().replace(' ','_')}"
    table_latex_with_caption = f"\\begin{{table}}[ht]\n\\centering\n{table_latex}\n\\captionsetup{{justification=centering}}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}"

    with open(f"{file_name}.tex", 'w') as file:
        file.write(table_latex_with_caption)

