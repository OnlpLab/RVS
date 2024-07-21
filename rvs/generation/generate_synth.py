# coding=utf-8
# Copyright 2020 Google LLC
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

'''
Output synthetic instructions by templates.

Example command line call:
$ bazel-bin/rvs/generation/generate_synth \
  --geo_data_path /path/to/REGION_geo_paths.gpkg \
  --save_instruction_path /tmp/pittsburgh_instructions.json

Example output: 
  "Meet at Swirl Crepe. Walk past Wellington. Swirl Crepe will be near Gyros."

See rvs/geo/map_processing/README.md for instructions to generate the gpkg 
data file.
'''

import sys
from sklearn.utils import shuffle

from absl import logging
from absl import app
from absl import flags

from rvs.geo import walk
from rvs.generation import templates

FLAGS = flags.FLAGS

flags.DEFINE_string("geo_data_path", None,
          "The path of the synthetic data file to use for generating the synthetic instructions.")

flags.DEFINE_string("save_instruction_dir", None,
          "The path of the file where the generated instructions will be saved. ")

flags.DEFINE_float("train_proportion", 0.8,
          "The train proportion of the dataset (0,1)")

flags.DEFINE_float("dev_proportion", 0.1,
          "The dev proportion of the dataset (0,1)")

# Required flags.
flags.mark_flag_as_required('geo_data_path')
flags.mark_flag_as_required('save_instruction_dir')


def main(argv):
  del argv  # Unused.

  if not FLAGS.train_proportion + FLAGS.dev_proportion < 1:
    sys.exit("Proportion of train and dev combined should be less then 1.")

  logging.info(f"Starting to generate synthetic samples")

  entities = walk.load_entities(FLAGS.geo_data_path)

  if entities is None:
    sys.exit("No entities found.")

  logging.info(f"Number of synthetic samples to create: {len(entities)}")

  # Get templates.
  gen_templates = templates.create_templates()

  # Save templates.
  gen_templates['sentence'].to_csv('templates.csv')

  # Shuffle templates.
  gen_templates = shuffle(gen_templates)

  # Split into Train, Dev and Test sets.
  size_templates = gen_templates.shape[0]
  train_size = round(size_templates*FLAGS.train_proportion)
  dev_size = round(size_templates*FLAGS.dev_proportion)

  train_gen_templates = gen_templates[:train_size]
  dev_gen_templates = gen_templates[train_size:train_size+dev_size]
  test_gen_templates = gen_templates[train_size+dev_size:]

  size_entities = len(entities)
  entities_train_size = round(size_entities*FLAGS.train_proportion)
  entities_dev_size = round(size_entities*FLAGS.dev_proportion)

  train_entities = entities[:entities_train_size]
  dev_entities = entities[entities_train_size:entities_train_size+entities_dev_size]
  test_entities = entities[entities_train_size+entities_dev_size:]

  templates.generate_instruction_by_split(train_entities, train_gen_templates, "train", FLAGS.save_instruction_dir)
  templates.generate_instruction_by_split(dev_entities, dev_gen_templates, "dev", FLAGS.save_instruction_dir)
  templates.generate_instruction_by_split(test_entities, test_gen_templates, "test", FLAGS.save_instruction_dir)


if __name__ == '__main__':
  app.run(main)