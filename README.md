# Relphormer

Source code for the paper: [Relphormer: Relational Graph Transformer for Knowledge Graph Representations](https://arxiv.org/abs/2205.10852)

> Transformers have achieved remarkable performance in widespread fields, including natural language processing, computer vision and graph mining. However, vanilla Transformer architectures have not yielded promising improvements in the Knowledge Graph (KG) representations, where the translational distance paradigm dominates this area. Note that vanilla Transformer architectures struggle to capture the intrinsically heterogeneous semantic and structural information of knowledge graphs. To this end, we propose a new variant of Transformer for knowledge graph representations dubbed Relphormer. Specifically, we introduce Triple2Seq which can dynamically sample contextualized sub-graph sequences as the input to alleviate the heterogeneity issue. We propose a novel structure-enhanced self-attention mechanism to encode the relational information and keep the globally semantic information among sub-graphs. Moreover, we propose masked knowledge modeling as a new paradigm for knowledge graph representation learning. We apply Relphormer to three tasks, namely, knowledge graph completion, KG-based question answering and KG-based recommendation for evaluation. Experimental results show that Relphormer can obtain better performance on benchmark datasets compared with baselines. Code is available in [this https URL](https://github.com/zjunlp/Relphormer).


# Model Architecture

<div align=center>
<img src="./resource/model.png" width="85%" height="75%" />
</div>
 

The model architecture of Relphormer. 
The contextualized sub-graph is sampled with Triple2Seq, and then it will be converted into sequences while maintaining its sub-graph structure.
Next, we conduct masked knowledge modeling, which randomly masks the nodes in the center triple in the contextualized sub-graph sequences.
For the transformer architecture, we design a novel structure-enhanced mechanism to preserve the structure feature.
Finally, we utilize our pre-trained KG transformer for KG-based downstream tasks. 

# Environments

- python (3.8.13)
- cuda(11.2)
- Ubuntu-18.04.6 (4.15.0-156-generic)

# Requirements

To run the codes, you need to install the requirements:
```
pip install -r requirements.txt
```

The expected structure of files is:

```
 ── Relphormer
    ├── data
    ├── dataset
    │   ├── FB15k-237
    │   ├── WN18RR
    │   ├── umls
    │   ├── create_neighbor.py
    ├── lit_models
    │   ├── _init_.py
    │   ├── base.py
    │   ├── transformer.py
    │   └── utils.py
    ├── models
    │   ├── _init_.py
    │   ├── huggingface_relformer.py
    │   ├── model.py
    │   └── utils.py    
    ├── resource
    │   └── model.png    
    ├── scripts
    │   ├── fb15k-237
    │   ├── wn18rr
    │   └── umls
    ├── QA
    ├── logs
    ├── main.py
    └── requirements.txt
```

# How to run

## KGC Task

### Generate Masked Neighbors

- Use the command below to generate the masked neighbors.
```shell
>> cd dataset
>> python create_neighbor.py --dataset xxx  # like python create_neighbor.py --dataset umls
```

### Entity Embedding Initialization

- Then use the command below to add entities to BERT and initialize the entity embedding layer to be used in the later training. For other datasets `FB15k-237`  and `WN18RR` ,  just replace the dataset name with  `fb15k-237` and  `wn18rr` will be fine.

```shell
>> cd pretrain
>> bash scripts/pretrain_umls.sh
>> tail -f -n 2000 logs/pretrain_umls.log
```

The pretrained models are saved in the `Relphormer/pretrain/output` directory.

### Entity Prediction

- Next use the command below to train the model to predict the correct entity in the masked position. Same as above for other datasets.

```shell
>> cd Relphormer
>> bash scripts/umls/umls.sh
>> tail -f -n 2000 logs/train_umls.log
```

The trained models are saved in the `Relphormer/output` directory.
