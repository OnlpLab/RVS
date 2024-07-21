# Papaers:
## 1. Rendezvous (RVS): A Novel Dataset for Spatial Allocentric Reasoning 

Rendezvous (RVS) is a corpus for executing navigation instructions and resolving spatial descriptions based on maps. The task is to follow
geospatial instructions given in colloquial language based on a dense urban map. 

The details of the corpus and task are described in: [Where Do We Go from Here? Multi-scale Allocentric Relational Inference from Natural Spatial Descriptions](https://aclanthology.org/2024.eacl-long.62/).

## 2. Into the Unknown: Generating Geospatial Descriptions for New Environments
Using RVS data we propose a large-scale augmentation method for generating high-quality synthetic data for new environments using readily available geospatial data. Our method constructs a grounded knowledge-graph, capturing entity relationships. We compare data generated data with LLMs and CFG.

The details of the ugmentation experiments are described in: [Into the Unknown: Generating Geospatial Descriptions for New Environments]([https://arxiv.org/pdf/2402.16364.pdf](https://arxiv.org/abs/2406.19967).

## Data

The navigation instructions can be found in this repository -  [here](https://github.com/OnlpLab/RVS/tree/main/dataset) or can be downloaded from HuggingFace - [here](https://huggingface.co/datasets/tzufi/RVS/)

The map-graph can be found [here](https://drive.google.com/drive/folders/1bvxNeIlN1SKeup6aJgIUzWrQ8v-cL9Yq?usp=sharing)



## Run model
```
bazel-bin/rvs/model/text/model_trainer --processed_data_dir OUTDIR --train_region Manhattan --dev_region Manhattan_dev --test_region Manhattan_dev  --output_dir OUTDIR
```

## Installation:
### Install conda environment
```
conda create --name rvs -y
conda activate rvs
python3 -m pip install -r requirements.txt

```

### Install BAZEL
```
apt install bazel
```

### BAZEL: Build
```
source build_all.sh
```

### BAZEL: Test
```
source test_all.sh
```

### BAZEL: Run
```
source run_all.sh
```




