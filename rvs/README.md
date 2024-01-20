# Rendezvous(RVS): Location Finding using Nearby Landmarks

### (1) Creating an OSM-based graph
```
bazel-bin/rvs/geo/map_processing/map_processor --region REGION --min_s2_level LEVEL --directory DIRECTORY_TO_MAP
```
### (2) Connecting the S2-Cells to the graph (created in 1) and calculating the environment embedding 
```
bazel-bin/rvs/data/metagraph/create_graph_embedding  --region REGION --s2_level LEVEL --s2_node_levels LEVEL --s2_node_levels LEVEL+1 --s2_node_levels LEVEL-1  --base_osm_map_filepath DIRECTORY_TO_MAP --save_embedding_path PATH_TO_SAVE_GRAPH --dimensions EMBED_DIM
```
### (3) Generating spatial samples based on the graph (created in 1)
```
bazel-bin/rvs/geo/sample_poi --region REGION --min_s2_level LEVEL --directory DIRECTORY_TO_MAP --path PATH_TO_SPATIAL_ITEMS.gpkg --n_samples NUMBER_OF_SAMPLES_TO_GENERATE
```

### (4) Running the model 
```
bazel-bin/rvs/model/text/model_trainer  --data_dir RAW_DATASET --dataset_dir PROCESSED_DATASET --train_region TRAIN_REGION --dev_region DEV_REGION --test_region TEST_REGION --s2_level LEVEL --output_dir PATH_TO_SAVE_RESULT --task human --model S2-Generation-T5-text-start-embedding-to-landmarks --graph_codebook 50 --train_graph_embed_path PATH_GRAPH_TRAIN --dev_graph_embed_path PATH_GRAPH_DEV --test_graph_embed_path PATH_GRAPH_TEST
```
