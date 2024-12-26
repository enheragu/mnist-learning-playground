
#!/usr/bin/env python3
# encoding: utf-8

import os
from tqdm import tqdm
import tabulate

## log in terminal without affecting tqdm bar
def log(*args, **kwargs):
    tqdm.write(*args, **kwargs)   


def logTable(row_data, output_path, filename, colalign = None):

    table_str = tabulate.tabulate(row_data, headers="firstrow", tablefmt="fancy_grid", colalign = colalign)
    table_latex = tabulate.tabulate(row_data, headers="firstrow", tablefmt="latex", colalign = colalign)
    log(table_str)
    file_name = os.path.join(output_path, filename.lower().replace(' ','_'))
    log(f"Stored data in {file_name}")
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