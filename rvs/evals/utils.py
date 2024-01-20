# coding=utf-8
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Library for evaluation functions for RVS models."""

import collections

import numpy as np
from geopy.distance import great_circle

from absl import logging

# Object for storing each evaluation tuple parsed from the input file.
EvalDataTuple = collections.namedtuple(
  "EvalDataTuple",
  ["example_id", "true_lat", "true_lng", "predicted_lat", "predicted_lng"])
# Object for evaluation metrics.
EvalMetrics = collections.namedtuple(
  "EvalMetrics",
  [
    "accuracy", 'accuracy_10m', 'accuracy_100m', 'accuracy_1000m', "mean_distance",
    "median_distance", "max_error", "norm_auc"])
# 20039 kms is half of earth's circumference (max. great circle distance)
_MAX_LOG_HAVERSINE_DIST = np.log(20039 * 1000)  # in meters.
_EPSILON = 1e-5


class Evaluator:
  """Class for evaluating geo models."""

  def __init__(self):
    logging.info("Starting evaluation.")

  def get_error_distances(self, input_file):
    """Compute error distance in meters between true and predicted coordinates.

        Args:
        input_file: TSV file containing example_id and true and
          predicted co-ordinates. One example per line.
        eval_logger: Logger object.

        Returns:
        Array of distance error - one per example.
    """
    error_distances = []
    total_examples = 0
    logging.info(f"Opening file <= {input_file}")
    for line in open(input_file):
      toks = line.strip().split("\t")
      if len(toks) != 7:
        logging.warning(
          "Unexpected line format: [%s]. Skipping", line)
        continue
      eval_tuple = EvalDataTuple(toks[0], float(toks[1]), float(toks[2]),
                     float(toks[3]), float(toks[4]))
      err = great_circle((eval_tuple.true_lat, eval_tuple.true_lng),
                 (eval_tuple.predicted_lat, eval_tuple.predicted_lng)).m
      error_distances.append(err)
      total_examples += 1
    return np.array(error_distances)

  def compute_metrics(self, error_distances):
    """Compute distance error metrics given an array of error distances.

        Args:
        error_distances: Array of distance errors.
        eval_logger: Logger object.
    """
    num_examples = len(error_distances)
    logging.info(f"Started evaluation with {num_examples} samples") 
    if num_examples == 0:
      logging.error("No examples to be evaluated.")
    accuracy = float(
      len(np.where(np.array(error_distances) == 0.)[0])) / num_examples

    accuracy_10m = float(
      len(np.where(np.array(error_distances) <= 10.)[0])) / num_examples

    accuracy_100m = float(
      len(np.where(np.array(error_distances) <= 100.)[0])) / num_examples

    accuracy_250m = float(
      len(np.where(np.array(error_distances) <= 250.)[0])) / num_examples

    mean_distance, median_distance, max_error = np.mean(error_distances), np.median(
      error_distances), np.max(error_distances)
    log_distance = np.sort(
      np.log(error_distances + np.ones_like(error_distances) * _EPSILON))
    # AUC for the distance errors curve. Smaller the better.
    auc = np.trapz(log_distance)

    # Normalized AUC by maximum error possible.
    norm_auc = auc / (_MAX_LOG_HAVERSINE_DIST * (num_examples - 1))

    log_message = f"Metrics: \
                    Exact accuracy : [{accuracy:.4f}]\n\
                    10 m accuracy : [{accuracy_10m:.4f}]\n\
                    100 m accuracy : [{accuracy_100m:.4f}]\n\
                    250 m accuracy : [{accuracy_250m:.4f}]\n\
                    mean error [{mean_distance:.2f}],\n\
                    median error [{median_distance:.2f}],\n\
                    max error [{max_error:.2f}]\n\
                    AUC of error curve [{norm_auc:.2f}]"
   
    logging.info(log_message)
    print(log_message)

    return EvalMetrics(accuracy, accuracy_10m, accuracy_100m, accuracy_250m,
               mean_distance, median_distance, max_error, norm_auc)


  def compute_cdf(self, error_distances, max_error_distance = -1):

    # Round the area to the nearest 10
    error_distances = np.around(error_distances, -1)

    # Calculate the CDF.
    max_error_distance = max(error_distances) if max_error_distance==-1 else max_error_distance
    cdf = np.zeros(int(max_error_distance/10)+1)
    
    lables = list(range(0, len(cdf)*10, 10))
    for i in range(len(cdf)):
      cdf[i] = np.sum(error_distances <= i*10) / len(error_distances)
    
    return cdf, lables, max_error_distance
