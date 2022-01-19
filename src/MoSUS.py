# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 16:30:06 2022

@author: nisri
"""

import os
import io
import numpy as np
from json import load, dump


# Load the given problem, which can be a json file
def load_instance(json_file):
    """
    Inputs: path to json file
    Outputs: json file object if it exists, or else returns NoneType
    """
    if os.path.exists(path=json_file):
        with io.open(json_file, 'rt', newline='') as file_object:
            return load(file_object)
    return None

  ########################   DATA   ########################################
  
json_instance = load_instance('../data/insects_data.json') 
insects = json_instance['Number_of_insects']
feed = json_instance['Number_of_feed']
countries = json_instance['Number_of_countries']
equipments = json_instance['Number_of_equipments']
scales = json_instance['Number_of_scales']
min_employees = 5
max_employees = 40
class Chromosome:
  #  amount_feed = 2 * np.random.random_sample((insects,feed ))
    Amount_feed = 2 * np.random.random_sample((insects,feed ))
    randnums = np.random.randint(0,countries)   
    country = np.zeros(countries)
    country[randnums] = 1
    randnums = np.random.randint(0,scales)  
    scale = np.zeros(scales)
    scale[randnums] = 1
    equipments = np.random.randint(0,1,equipments)
    Number_of_employees = np.random.randint(min_employees,max_employees) 
    insect_biomass = 0
    labor = 0
    labor_safety = 0
    insect_frass = 0

    def fitness(self,json_instance):
        insect_biomass = 0
        insect_frass = 0
        for insect_id in range(insects):
            for feed_id in range(feed):
                insect_biomass += json_instance["feedSF"][insect_id][feed_id]*json_instance["NRF"][feed_id]*self.Amount_feed[insect_id][feed_id]
                for scale_id in range(scales):
                    insect_frass += self.Amount_feed[insect_id][feed_id]/json_instance["NRF"][feed_id]*(1-json_instance["feedSF"][insect_id][feed_id])*json_instance["NRF"][feed_id]*json_instance["FeSF"][insect_id][scale_id]*self.scale[scale_id]
        
        labor = 0       
        for country_id in range(countries):            
            labor += json_instance["RW"][country_id]*json_instance["RWT"]*json_instance["CFfw"][country_id]*self.country[country_id]*self.Number_of_employee
        labor_safety = 0
        for equipment_id in range(equipments):
            for scale_id in range(scales):
                labor_safety += json_instance["PPE"][equipment_id]*self.equipments[equipment_id]*json_instance["SFls"][scale_id]*self.scale[scale_id]
        
     ########################   Initial population   ########################################     
          
            

              
  

Pop=Chromosome()
print(Pop.Amount_feed)
print("==============================")
Pop.Amount_feed=7 * np.random.random_sample((insects,feed ))
print(Pop.Amount_feed)


        
