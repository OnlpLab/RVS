# Rendezvous (RVS): A Novel Dataset for Spatial Allocentric Reasoning 

## Data

The instructions can be found [here](https://github.com/OnlpLab/RVS/tree/main/dataset)

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




