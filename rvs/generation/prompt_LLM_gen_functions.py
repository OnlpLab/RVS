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

import os
import openai
import sys

import time

from absl import logging
import json

import google.generativeai as palm

from rvs.geo import geo_item


palm.configure(api_key='API_KEY_GCP')

models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
model = models[0]


openai.api_type = "azure"
openai.api_base = 'API_BASE'
openai.api_version = "API_VERSION"

openai.api_key = 'API_KEY_OPENAI' 

INIT_MESSAGE = 'Rephrase the subsequent navigation instruction, ensuring it explains how to travel from the starting position to the destination:\n'

LLMs = ['ChatGPT', 'Bard']

LLMs_SUPPORT_MESSAGE = (
  'Supported LLMs: ' + ', '.join(LLMs))

def get_basic_instruction(entity):
   start_point = entity.geo_landmarks['start_point'].main_tag
   end_point = entity.geo_landmarks['end_point'].main_tag
   end_point_side = entity.geo_features['spatial_rel_goal']
   main_near_pivot = entity.geo_landmarks['main_near_pivot'].main_tag
   near_pivot = entity.geo_landmarks['near_pivot'].main_tag


   main_near_pivot_side = entity.geo_features['spatial_rel_main_near']

   main_pivot = entity.geo_landmarks['main_pivot'].main_tag
   main_pivot_side = entity.geo_features['spatial_rel_pivot']
   intersections = entity.geo_features['intersections']
   position_on_block = entity.geo_features['goal_position']

   cardinal_direction = entity.geo_features['cardinal_direction']
   instruction = f"The starting point is {start_point}. " + \
   f"You need to get to {end_point} which is {cardinal_direction} of the {start_point}. "+\
   f"On your way you will pass {main_pivot} on your {main_pivot_side}. "+\
   f"The goal will be on your {end_point_side}. "

   if intersections:
      instruction += f'The goal is {intersections} intersections or blocks away from {main_pivot}. '

   if main_near_pivot!='None':
      if main_near_pivot_side == end_point_side:
         instruction += f'Before reaching the destination, you will see a {main_near_pivot} on the same side of the street as your destination. ' 

      else:
         instruction += f'Before reaching the destination, you will see a {main_near_pivot} on the other side of the street from your destination. ' 

   if near_pivot:
      instruction += f'It will be near a {near_pivot}. '

   if position_on_block:
      if position_on_block[0]=='middle_block':
         instruction += f'The goal is in the middle of the block.'
      elif position_on_block[0]=='first_intersection':
         instruction += f'The goal will be on the closest corner of the block'
      else:
         instruction += f'The goal is on the far corner of the block'
      
      if len(position_on_block)==2:
         instruction += f', on the {position_on_block[-1]} corner.'
      else:
         instruction += '.'

   return instruction

def repharse_instruciton_with_chatGPT(instruction):
  message = INIT_MESSAGE+instruction

  response = openai.ChatCompletion.create(
      engine="chatgpt", # engine = "deployment_name".
      messages=[
          {"role": "user", "content": message},
      ]
  )

  entities_per_type = response['choices'][0]['message']['content']
    
  return entities_per_type



def repharse_instruciton_with_Bard(instruction):
   prompt = INIT_MESSAGE+instruction

   try:
      completion = palm.generate_text(
         model=model,
         prompt=prompt,
         temperature=0,
         max_output_tokens=800,
      )

      time.sleep(3)
   except: 
      logging.info("Wait before another attempt at request")
      time.sleep(800)
      return repharse_instruciton_with_Bard(instruction)
   return completion.result



def gen_from_entities_prompt_instructions(list_entities, model = 'ChatGPT'):
   instructions_list = []
   for e_idx, e in enumerate(list_entities):
      basic_instruction = get_basic_instruction(e)
      print (f"basic example: {basic_instruction}")
      assert model in LLMs, sys.exit(f"No such model ({model}) for prompting")
      logging.info(f"running {model}")
      if model == 'ChatGPT':
         rephrased_instruction = repharse_instruciton_with_chatGPT(basic_instruction)
      else: # model == 'Bard':
         rephrased_instruction = repharse_instruciton_with_Bard(basic_instruction)
      print (f"bard instructuin: {rephrased_instruction}\n")
         
      instructions_list.append(rephrased_instruction)
   return instructions_list


def gen_and_save_entities_prompting_LLM_instructions_with_meta_data(
      list_entities, split, dir, model = 'Bard'):
   gen_samples = []

   save_instruction_path = os.path.join(dir, f"ds_{split}.json")
   n_writen = 0
   for entity_idx, entity in enumerate(list_entities): 
         gen_instructions = gen_from_entities_prompt_instructions([entity], model=model)[0]

         rvs_entity = geo_item.RVSSample.to_rvs_sample(
         instructions=gen_instructions,
         id=entity_idx,
         geo_entity=entity,
         entity_span=''
         )
         gen_samples.append(rvs_entity)

         with open(save_instruction_path, 'a') as outfile:
            try:
               synth_entity_dict = rvs_entity.__dict__
               if is_jsonable(synth_entity_dict, entity_idx):
                  json.dump(synth_entity_dict, outfile)
                  outfile.write('\n')
                  outfile.flush()
                  n_writen += 1
                  logging.info(f"Written {entity_idx}: {n_writen/len(list_entities)}")
            except Exception as e:
               logging.info(f"Failed writing sample {entity_idx}. Error: {e}")

   
   logging.info(f"Synthetic Bard {split}-set generated: {n_writen} out of {len(list_entities)}")
   


def is_jsonable(sample_dict, idx):
  try:
    json.dumps(sample_dict)
    return True
  except Exception as e:
    logging.info(f"Failed writing sample {idx}. Error: {e}")
    return False