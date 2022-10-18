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
# Papers for the Project & How to Cite
If you use or extend our work, please cite the paper as follows:

```bibtex
@article{DBLP:journals/corr/abs-2205-10852,
  author    = {Zhen Bi and
               Siyuan Cheng and
               Jing Chen and
               Xiaozhuan Liang and
               Ningyu Zhang and
               Feiyu Xiong and
               Huajun Chen},
  title     = {Relphormer: Relational Graph Transformer for Knowledge Graph Representation},
  journal   = {CoRR},
  volume    = {abs/2205.10852},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2205.10852},
  doi       = {10.48550/arXiv.2205.10852},
  eprinttype = {arXiv},
  eprint    = {2205.10852},
  timestamp = {Mon, 30 May 2022 15:47:29 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2205-10852.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```