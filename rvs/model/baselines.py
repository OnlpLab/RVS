# coding=utf-8
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Baseline models evaluation: 
(1) NO-MOVE - Uses the start point as the predicted target.
RUN - 
$ bazel-bin/rvs/model/baselines \
  --data_dir ~/data/  \
  --metrics_dir ~/eval/ \
"""

from absl import app
from absl import flags

from absl import logging
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
import osmnx as ox
from shapely.geometry.point import Point

import swifter


from rvs.model import datasets
from rvs.geo import util as gutil
from rvs.evals import utils as eu
from rvs.model import util
from rvs.geo import regions
from rvs.geo import osm

TASKS = ["RVS", "RUN", "human"]

FLAGS = flags.FLAGS

flags.DEFINE_string("raw_data_dir", None,
                    "The directory from which to load the dataset.")

flags.DEFINE_string("metrics_dir", None,
                    "The directory where the metrics evaluation witll be save to.")

flags.DEFINE_enum(
  "task", "RVS", TASKS,
  "Supported datasets to train\evaluate on: WikiGeo, RVS or RUN.")

flags.DEFINE_enum(
  "region", None, regions.SUPPORTED_REGION_NAMES,
  regions.REGION_SUPPORT_MESSAGE)

# Required flags.
flags.mark_flag_as_required("raw_data_dir")
flags.mark_flag_as_required("metrics_dir")


def main(argv):
  if not os.path.exists(FLAGS.raw_data_dir):
    sys.exit("Dataset path doesn't exist: {}.".format(FLAGS.raw_data_dir))

  metrics_path = os.path.join(FLAGS.metrics_dir, 'metrics.tsv')

  assert FLAGS.task in TASKS
  if FLAGS.task == "Synthetic":
    dataset_init = datasets.SyntheticDataset
  elif FLAGS.task == 'RUN':
    dataset_init = datasets.RUNDataset
  elif FLAGS.task == 'RVS':
    dataset_init = datasets.RVSDataset
  elif FLAGS.task == 'WikiGeo':
    dataset_init = datasets.WikiGeoDataset
  else:
    sys.exit("Dataset invalid")

  dataset = dataset_init(
    data_dir=FLAGS.raw_data_dir,
    train_region=FLAGS.region,
    dev_region=FLAGS.region,
    test_region=FLAGS.region,
    s2level=18,  
    size_of_train_split=7000,
  )


  # # STOP baseline
  end_points = dataset.test_raw.end_point.apply(gutil.list_yx_from_point).tolist()
  start_point = dataset.test_raw.start_point.apply(gutil.list_yx_from_point).tolist()

  logging.info(f"size of test: {dataset.test_raw.end_point.tolist()[0]}")
  util.save_metrics_last_only(
    metrics_path,
    end_points,
    start_point,
    start_point)

  logging.info(f"NO-MOVE evaluation for task {FLAGS.task}:")
  evaluator = eu.Evaluator()
  error_distances = evaluator.get_error_distances(metrics_path)
  evaluator.compute_metrics(error_distances)



  # # CENTER
  center_point = regions.get_region(FLAGS.region).polygon.centroid

  pred_points = dataset.test_raw.start_point.apply(lambda s: gutil.get_point_within_distance(s,center_point, 1000 ))
  
  pred_points_yx =pred_points.apply(gutil.list_yx_from_point).tolist()

  util.save_metrics_last_only(
    metrics_path,
    pred_points_yx,
    start_point,
    start_point)

  logging.info(f"CENTER-MOVE evaluation for task {FLAGS.task}:")
  evaluator = eu.Evaluator()

  error_distances = evaluator.get_error_distances(metrics_path)

  evaluator.compute_metrics(error_distances)


  # LANDMARK

  pred_points_yx = dataset.test_raw.start_point.swifter.apply(get_prominent_osm)

  util.save_metrics_last_only(
    metrics_path,
    pred_points_yx,
    start_point,
    start_point)

  logging.info(f"LANDMARK evaluation for task {FLAGS.task}:")
  evaluator = eu.Evaluator()

  error_distances = evaluator.get_error_distances(metrics_path)

  evaluator.compute_metrics(error_distances)

  
dict_prominent_tags = {x: True for x in osm.PROMINENT_TAGS_ORDERED}

def get_prominent_osm(start_point):
  
  poi_near = ox.geometries.geometries_from_point(
    center_point = gutil.list_yx_from_point(start_point),tags=dict_prominent_tags,dist=1000)

  for type_poi in osm.PROMINENT_TAGS_ORDERED:
    pois_prominent = poi_near[poi_near[type_poi].isnull()==False]
    if pois_prominent.shape[0]:

      geom = pois_prominent.geometry.iloc[0]
      if not isinstance(geom, Point):
        geom = geom.centroid
      return gutil.list_yx_from_point(geom)

  return start_point


if __name__ == '__main__':
  app.run(main)
