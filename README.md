# Relphormer

Code for the paper ["Relphormer: Relational Graph Transformer for Knowledge Graph Representations"]("https://arxiv.org/abs/2205.10852")


# Model Architecture

<div align=center>
<img src="./resource/model.png" width="85%" height="75%" />
</div>
 
 
The model architecture of Relphormer. 
The contextualized sub-graph is sampled with Triple2Seq, and then it will be converted into sequences while maintaining its sub-graph structure.
Next, we conduct masked knowledge modeling, which randomly masks the nodes in the center triple in the contextualized sub-graph sequences.
For the transformer architecture, we design a novel structure-enhanced mechanism to preserve the structure feature.
Finally, we utilize our pre-trained KG transformer for KG-based downstream tasks. 


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
    │   └── WN18RR
    ├── lit_models
    │   └── _init_.py
    │   └── base.py
    │   └── transformer.py
    │   └── utils.py
    ├── models
    │   └── _init_.py
    │   └── huggingface_relformer.py
    │   └── model.py
    │   └── utils.py    
    ├── resource
    │   └── model.png    
    └── scripts
        ├── fb15k-237
        └── wn18rr
```

# How to run


+ ## KGC Task
    - First run the ipython file to generate the masked neighbors.
    ```shell
        cd Relphormer/dataset/FB15k-237
        create_neighbor.ipynb
    ```

    - Then run pre-train script to initialize the bert embedding.

    ```shell
        cd Relphormer
        bash scripts/pretrain_fb15k.sh
    ```

    - Next do the Entity Prediction TASK.

    ```shell
        bash scripts/fb15k.sh
    ```
