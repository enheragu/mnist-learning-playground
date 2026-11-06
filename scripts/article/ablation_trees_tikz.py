"""TikZ sequential decision trees of the article from the output of ablation_trees.py.

Usage: python3 ablation_trees_tikz.py <trees_llvip.json> <trees_mnist.json> <out_dir>
Writes one tikzpicture per tree (fig_tree_*.tex); the node styles build on archblock (fig_tikz_styles.tex). Each node gives the fraction of simulated ablation
paths through it (Monte Carlo and Bootstrap); a leaf is the configuration the path ends in. Nodes on
the route to the global best are green, the fixed start value is yellow. Checks before writing: the
children of every node add up to the node, and the global-best leaf equals the probability of
reaching the global best.
"""
import json
import os
import sys

llvip_json, mnist_json, out_dir = sys.argv[1:4]

LABELS = {  # factor: (name in the start node, prefix in the other nodes, value names)
    "fusion": ("Fusion", "", {}),
    "rgb_eq": ("RGB eq.", "RGB: ", {"clahe": "CLAHE"}),
    "th_eq": ("Thermal eq.", "Th.: ", {"clahe": "CLAHE"}),
    "batch_size": ("Batch size", "Batch ", {}),
    "learning_rate": ("Learning rate", "LR ", {}),
}
TREES = [  # output name, json, start factor, start value
    ("fig_tree_llvip_fusion", llvip_json, "fusion", "VT"),
    ("fig_tree_llvip_rgb", llvip_json, "rgb_eq", "clahe"),
    ("fig_tree_llvip_th", llvip_json, "th_eq", "clahe"),
    ("fig_tree_mnist_batch", mnist_json, "batch_size", "10"),
    ("fig_tree_mnist_lr", mnist_json, "learning_rate", "0.001"),
]
COL_GAP = 2.0    # cm between the left edges of two columns
ROOT_GAP = 0.34  # cm between the start node (anchored east at x = 0) and the first column
TRUNK = 0.16     # cm from a column to the vertical segment that feeds it
ROW_GAP = 0.74   # cm between leaves


def label(factor, value, start=False):
    name, prefix, values = LABELS[factor]
    value = values.get(value, value)
    return f"{name}\n{value}" if start else f"{prefix}{value}"


def pct(p):
    return r"$<$0.1" if 0 < p < 0.0005 else f"{100 * p:.1f}"


def build(row_mc, row_bs, best):
    """Tree as nested dicts: level 1 and 2 are the remaining factors, leaves the re-chosen start factor."""
    remaining = row_mc["step1_factors"]
    start_factor = row_mc["start_factor"]

    def node(factor, value, mc, bs, path):
        on_best = all(best[f] == v for f, v in path.items())
        return {"label": label(factor, value), "mc": mc, "bs": bs, "best": on_best, "children": []}

    root = {"label": label(start_factor, row_mc["start_value"], start=True), "mc": None, "bs": None, "best": False,
            "start": True, "children": []}
    branches_bs = {b["step1_key"]: b for b in row_bs["branches"]}
    groups = {}
    for b_mc in row_mc["branches"]:
        b_bs = branches_bs[b_mc["step1_key"]]
        levels = dict(part.split("=", 1) for part in b_mc["step1_key"].split(" | "))
        leaves = []
        opts_bs = {o["step2_value"]: o for o in b_bs["step2_options"]}
        for o_mc in b_mc["step2_options"]:
            o_bs = opts_bs[o_mc["step2_value"]]
            leaf = node(start_factor, o_mc["step2_value"], b_mc["p_step1"] * o_mc["p_step2"],
                        b_bs["p_step1"] * o_bs["p_step2"], {**levels, start_factor: o_mc["step2_value"]})
            leaf["reach"] = (o_mc["p_reach_joint"], o_bs["p_reach_joint"])
            leaves.append(leaf)
        if len(remaining) == 1:
            n = node(remaining[0], levels[remaining[0]], b_mc["p_step1"], b_bs["p_step1"], levels)
            n["children"] = leaves
            root["children"].append(n)
        else:
            first = levels[remaining[0]]
            if first not in groups:
                groups[first] = node(remaining[0], first, 0.0, 0.0, {remaining[0]: first})
                root["children"].append(groups[first])
            n = node(remaining[1], levels[remaining[1]], b_mc["p_step1"], b_bs["p_step1"], levels)
            n["children"] = leaves
            groups[first]["children"].append(n)
            groups[first]["mc"] += b_mc["p_step1"]
            groups[first]["bs"] += b_bs["p_step1"]
    return root


def check(tree, row_mc, row_bs):
    def walk(n):
        if n["children"] and n["mc"] is not None:
            for k in ("mc", "bs"):
                if abs(sum(c[k] for c in n["children"]) - n[k]) > 1e-9:
                    raise SystemExit(f"children of {n['label']} do not add up")
        for c in n["children"]:
            walk(c)
    walk(tree)
    if abs(sum(c["mc"] for c in tree["children"]) - 1) > 1e-9:
        raise SystemExit("first level does not add up to 1")
    leaves = []
    stack = [tree]
    while stack:
        n = stack.pop()
        stack.extend(n["children"])
        if not n["children"]:
            leaves.append(n)
    best = [l for l in leaves if l["best"]]
    if len(best) != 1:
        raise SystemExit("expected exactly one global-best leaf")
    for k, row in (("mc", row_mc), ("bs", row_bs)):
        if abs(best[0][k] - row["p_reach_best"]) > 1e-9:
            raise SystemExit(f"global-best leaf ({k}) differs from p_reach_best")
    if any(abs(l["reach"][0]) > 0 for l in leaves if not l["best"]):
        raise SystemExit("a leaf that is not the global best reaches it")


def layout(tree):
    """Leaves on consecutive rows, parents centred on their children; returns depth."""
    rows = [0]

    def place(n, depth):
        n["x"] = 0.0 if depth == 0 else ROOT_GAP + (depth - 1) * COL_GAP
        if not n["children"]:
            n["y"] = -rows[0] * ROW_GAP
            rows[0] += 1
            return depth
        d = max(place(c, depth + 1) for c in n["children"])
        n["y"] = (n["children"][0]["y"] + n["children"][-1]["y"]) / 2
        return d
    return place(tree, 0)


def tikz(tree):
    lines = [r"\begin{tikzpicture}[",
             r"  treenode/.style={archblock, thin, minimum width=1.5cm, minimum height=0pt, inner sep=1.5pt,",
             r"                   fill=white},",
             r"  treebest/.style={treenode, thick, fill=gGreen!35},",
             r"  treestart/.style={treenode, thick, fill=gYellow!45},",
             r"  treeedge/.style={draw=gGrey, line width=0.5pt},",
             r"  treebestedge/.style={draw=gDarkGrey, line width=1pt}]"]
    counter = [0]

    def emit(n, parent):
        counter[0] += 1
        name = f"n{counter[0]}"
        style = "treestart" if n.get("start") else ("treebest" if n["best"] else "treenode")
        text = r"\\".join(rf"\textbf{{{part}}}" for part in n["label"].split("\n"))
        if n["mc"] is not None:
            text += rf"\\{pct(n['mc'])} $\mid$ {pct(n['bs'])}\,\%"
        anchor = "east" if n.get("start") else "west"
        lines.append(rf"  \node[{style}, anchor={anchor}] ({name}) at ({n['x']:.2f},{n['y']:.3f}) {{{text}}};")
        if parent is not None:
            edge = "treebestedge" if n["best"] else "treeedge"
            lines.append(rf"  \draw[{edge}] ({parent}.east) -| ({n['x'] - TRUNK:.2f},{n['y']:.3f}) -- ({name}.west);")
        for c in n["children"]:
            emit(c, name)
    emit(tree, None)
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines) + "\n"


cache = {}
for out_name, path, start_factor, start_value in TREES:
    data = cache.setdefault(path, json.load(open(path)))
    rows = {m: [r for r in data["trees"][m] if r["start_factor"] == start_factor and r["start_value"] == start_value]
            for m in ("mc", "bs")}
    if len(rows["mc"]) != 1 or len(rows["bs"]) != 1:
        raise SystemExit(f"{out_name}: start {start_factor}={start_value} not found")
    best = data["trees"]["mc"][0]["global_best_model"]
    model_best = data["model_factors"][best]
    tree = build(rows["mc"][0], rows["bs"][0], model_best)
    check(tree, rows["mc"][0], rows["bs"][0])
    layout(tree)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{out_name}.tex"), "w") as f:
        f.write(tikz(tree))
    print(f"{out_name}: start {start_factor}={start_value}, reach global best ({best}): "
          f"MC {100 * rows['mc'][0]['p_reach_best']:.2f} %, Bootstrap {100 * rows['bs'][0]['p_reach_best']:.2f} %")
