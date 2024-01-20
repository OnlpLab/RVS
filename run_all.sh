#!/bin/bash

REGION_NAME="UTAustin"

OUTPUT_DIR=$HOME/tmp/rvs_run/$REGION_NAME
MAP_DIR=$OUTPUT_DIR/map

OUTPUT_DIR_MODEL=$OUTPUT_DIR/rvs_run/manhattan
OUTPUT_DIR_MODEL_RVS=$OUTPUT_DIR_MODEL/rvs
OUTPUT_DIR_MODEL_RVS_FIXED_4=$OUTPUT_DIR_MODEL/rvs/fixed_4
OUTPUT_DIR_MODEL_RVS_FIXED_5=$OUTPUT_DIR_MODEL/rvs/fixed_5

OUTPUT_DIR_MODEL_HUMAN=$OUTPUT_DIR_MODEL/human



echo "****************************************"
echo "*                 Geo                  *"
echo "****************************************"

rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $MAP_DIR

bazel-bin/rvs/geo/map_processing/map_processor --region $REGION_NAME --min_s2_level 18 --directory $MAP_DIR
bazel-bin/rvs/geo/sample_poi --region $REGION_NAME --min_s2_level 18 --directory $MAP_DIR --path $MAP_DIR/utaustin_geo_paths.gpkg --n_samples 8

echo "****************************************"
echo "*                 graph embeddings     *"
echo "****************************************"

GRAPH_EMBEDDING_PATH=$MAP_DIR/graph_embedding.pth
bazel-bin/rvs/data/metagraph/create_graph_embedding  --region $REGION_NAME --dimensions 224 --s2_level 15 --s2_node_levels 15 --base_osm_map_filepath $MAP_DIR --save_embedding_path $GRAPH_EMBEDDING_PATH --num_walks 2 --walk_length 2

echo "****************************************"
echo "*                 models               *"
echo "****************************************"
mkdir -p $OUTPUT_DIR_MODEL
mkdir -p $OUTPUT_DIR_MODEL_RVS
mkdir -p $OUTPUT_DIR_MODEL_RVS_FIXED_4
mkdir -p $OUTPUT_DIR_MODEL_RVS_FIXED_5
mkdir -p $OUTPUT_DIR_MODEL_HUMAN

echo "*                 Dual-Encoder-Bert  - HUMAN DATA             *"
bazel-bin/rvs/model/text/model_trainer  --raw_data_dir ~/RVS/dataset --processed_data_dir $OUTPUT_DIR_MODEL_HUMAN --train_region Manhattan --dev_region Manhattan --test_region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_HUMAN --num_epochs 1 --task RVS --model Dual-Encoder-Bert 

echo "*                 Classification-Bert  - HUMAN DATA             *"
bazel-bin/rvs/model/text/model_trainer  --raw_data_dir ~/RVS/dataset --processed_data_dir $OUTPUT_DIR_MODEL_HUMAN --train_region Manhattan --dev_region Manhattan --test_region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_HUMAN --num_epochs 1 --task RVS --model Classification-Bert

echo "*                 S2-Generation-T5    - HUMAN DATA           *"
bazel-bin/rvs/model/text/model_trainer  --raw_data_dir ~/RVS/dataset --processed_data_dir $OUTPUT_DIR_MODEL_HUMAN --train_region Manhattan --dev_region Manhattan --test_region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_HUMAN --num_epochs 1 --task RVS --model S2-Generation-T5  --train_batch_size 20 --test_batch_size 40 

echo "*                 S2-Generation-T5-start-text-input    - HUMAN DATA           *"
bazel-bin/rvs/model/text/model_trainer  --raw_data_dir ~/RVS/dataset --processed_data_dir $OUTPUT_DIR_MODEL_HUMAN --train_region Manhattan --dev_region Manhattan --test_region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_HUMAN --num_epochs 1 --task RVS --model S2-Generation-T5-start-text-input  --train_batch_size 20 --test_batch_size 40

echo "*                 S2-Generation-T5-Landmarks   - HUMAN DATA            *"
bazel-bin/rvs/model/text/model_trainer  --raw_data_dir ~/RVS/dataset --processed_data_dir $OUTPUT_DIR_MODEL_HUMAN --train_region Manhattan --dev_region Manhattan --test_region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_HUMAN --num_epochs 1 --task RVS --model S2-Generation-T5-Landmarks  --train_batch_size 20 --test_batch_size 40

echo "*                Baseline           *"
bazel-bin/rvs/model/baselines --raw_data_dir ~/RVS/dataset --metrics_dir $OUTPUT_DIR_MODEL_HUMAN  --task RVS --region Philadelphia 


echo "****************************************"
echo "*              Wikidata                *"
echo "****************************************"
bazel-bin/rvs/data/wikidata/extract_geofenced_wikidata_items --region $REGION_NAME

echo "****************************************"
echo "*              Wikipedia               *"
echo "****************************************"
bazel-bin/rvs/data/wikipedia/extract_wikipedia_items --titles=New_York_Stock_Exchange,Empire_State_Building

echo "****************************************"
echo "*              Wikigeo                 *"
echo "****************************************"
bazel-bin/rvs/data/create_wikigeo_dataset --region $REGION_NAME --output_dir $OUTPUT_DIR/wikigeo
bazel-bin/rvs/data/create_wikigeo_dataset --region $REGION_NAME --output_dir $OUTPUT_DIR/wikigeo --osm_path $MAP_DIR/utaustin_poi.pkl


echo "Delete DATA"
rm -rf $OUTPUT_DIR

