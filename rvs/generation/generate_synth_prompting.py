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


import sys

from absl import logging
from absl import app
from absl import flags

from rvs.geo import walk
from rvs.generation import prompt_LLM_gen_functions

FLAGS = flags.FLAGS


flags.DEFINE_string("geo_data_path", None,
          "The path of the synthetic data file to use for generating the synthetic instructions.")

flags.DEFINE_string("save_instruction_dir", None,
          "The path of the file where the generated instructions will be saved. ")

flags.DEFINE_enum(
  "type_model_prompting", 'Bard', prompt_LLM_gen_functions.LLMs, prompt_LLM_gen_functions.LLMs_SUPPORT_MESSAGE)


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

  size_entities = len(entities)
  entities_train_size = round(size_entities*FLAGS.train_proportion)
  entities_dev_size = round(size_entities*FLAGS.dev_proportion)

  train_entities = entities[:entities_train_size]
  dev_entities = entities[entities_train_size:entities_train_size+entities_dev_size]
  test_entities = entities[entities_train_size+entities_dev_size:]

  prompt_LLM_gen_functions.gen_and_save_entities_prompting_LLM_instructions_with_meta_data(
    list_entities=train_entities, 
    split='train', 
    dir=FLAGS.save_instruction_dir,
    model=FLAGS.type_model_prompting,
  )

  prompt_LLM_gen_functions.gen_and_save_entities_prompting_LLM_instructions_with_meta_data(
    list_entities=dev_entities, 
    split='dev', 
    dir=FLAGS.save_instruction_dir,
    model=FLAGS.type_model_prompting,
  )

  prompt_LLM_gen_functions.gen_and_save_entities_prompting_LLM_instructions_with_meta_data(
    list_entities=test_entities, 
    split='test', 
    dir=FLAGS.save_instruction_dir,
    model=FLAGS.type_model_prompting,
  )
 

if __name__ == '__main__':
  app.run(main)