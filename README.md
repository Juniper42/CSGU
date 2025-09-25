# Certified Signed Graph Unlearning

## Abstract
As data protection becomes increasingly important, the use of personal behavioral data in modern applications raises critical privacy concerns, making selective removal of information from models a key requirement. This issue is particularly challenging in signed graphs, which incorporate both positive and negative edges to model complex inverse relationships, with applications in social networks, recommendation systems, and financial analysis. While graph unlearning seeks to remove the influence of specific data from Graph Neural Networks (GNNs), existing approaches are designed for conventional GNNs and fail to capture the heterogeneous properties of signed graphs. Applied directly to Signed Graph Neural Networks (SGNNs), these methods overlook the dual-path aggregation mechanism that separately processes positive and negative edges, thereby breaking the semantic balance. To address this gap, we introduce $\textbf{\underline{C}ertified \underline{S}igned \underline{G}raph \underline{U}nlearning}$ (CSGU), a framework that provides provable privacy guarantees while preserving the sociological principles underlying SGNNs. CSGU consists of three stages: (1) efficiently identifying minimally affected neighborhoods through triangular structures, (2) leveraging sociological theories to quantify node importance for optimal privacy budget allocation, and (3) performing importance-weighted parameter updates to enable certified modifications with minimal utility loss. Extensive experiments show that CSGU consistently outperforms other baselines and achieves more reliable unlearning effects on SGNNs.

<img src="figs/overview.jpg">

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
