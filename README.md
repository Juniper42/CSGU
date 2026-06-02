# Certified Signed Graph Unlearning

## Abstract
Graph unlearning removes the influence of sensitive edges and nodes from trained Graph Neural Networks (GNNs) without full retraining, which is essential for privacy protection. However, existing graph unlearning methods do not account for the heterogeneity of positive and negative edges in signed graphs, thereby degrading both model utility and unlearning effectiveness when applied to widespread signed graph applications. To fill this research gap, we propose \underline{\textbf{C}}ertified \underline{\textbf{S}}igned \underline{\textbf{G}}raph \underline{\textbf{U}}nlearning (CSGU), which leverages the sociological principles underlying signed graphs, providing provable privacy guarantees while maintaining model utility. Specifically, CSGU efficiently identifies minimal influenced neighborhoods via triangular structures, and then applies sociological theories to quantify edge influence. Subsequently, it performs influence-weighted parameter updates with calibrated noise injection to achieve certified privacy guarantees with minimal utility degradation. Extensive experiments across five datasets show that CSGU outperforms four competing graph unlearning methods on four GNN architectures in most settings, achieving state-of-the-art results in both utility preservation and unlearning effectiveness.

<img src="figs/overview.png">

## Experients

```bash
conda create -n CSGU python=3.10
conda activate CSGU
pip install -r requirements.txt
```

## Signed Graphs

```bash
python main.py --model SGCN --dataset bitcoin_alpha --unlearning_method CSGU
```

## Unsiged Graphs
The extended directory contains extended experiments for unsigned graphs.

```bash
cd extended
python main.py --model SGCN --dataset Cora --unlearning_method CSGU
```
