#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This notebook will serve as a baseline with the regular
#Single Agent logic implemented in gymnasium as was done in 
#MATLAB
#This will allow us to try certain concepts in a controlled
#environment rather than the more complex MARL environment


# In[1]:


import gymnasium as gym
import numpy as np
import matplotlib
from typing import Optional
from gymnasium import spaces
import random


# In[2]:


#Setup environment
class SingleSatelliteEnv(gym.Env):

    def __init__(self):
        #We define the key environment variables here
        
        
        
        #Observation space for the satellite
        #Contact plan
        #Current data to downlink
        self.timestep=None
        self.contact_plan=np.full((10, 1), None)
        self.satellite_remaining_data=None
        self.delivered_data=None
        self.Energy_Expended=None
        self.initial_data_volume=None
        
        self.observation_space = gym.spaces.Dict(
            {
            "contact plan": spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32),
            "Remaining data": spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            "current timestep": spaces.Box(low=0.0, high=11, shape=(1,), dtype=np.float32),
            }
        )
        
        self.action_space = gym.spaces.Discrete(2)
    
    
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.
        
        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)
        
        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        self.timestep=0
        self.contact_plan=self.np_random.uniform(low=0.0, high=1.0, size=(10,)).astype(np.float32)
        self.contact_plan= np.round(self.contact_plan*10)/10
        self.satellite_remaining_data=np.array(
        [self.np_random.integers(5, 100) / 100.0], dtype=np.float32
    )
        self.delivered_data=0
        self.initial_data_volume=self.satellite_remaining_data
        self.Energy_Expended=0
        observation = self._get_obs()
        info = self._get_info()
        #for k, v in observation.items():
            #print(f"{k}: shape={v.shape}, dtype={v.dtype}")
            #print(f"Contained in space? {self.observation_space[k].contains(v)}")
        return observation, info
    def updateDeliveryandEnergy(self,weather,length,remaining_data):
        delivered_packets=0
        excess_energy_expended=0
        random_sample=self.np_random.uniform(low=0.0, high=1.0, size=(10,)).astype(np.float32)
        for i in range(length-1):
            if remaining_data>0:
                if random_sample[i]> weather:
                    delivered_packets=delivered_packets+1
                    remaining_data=remaining_data-1
                else:
                    excess_energy_expended=excess_energy_expended+1
            else:
                excess_energy_expended=excess_energy_expended+1
        
        return delivered_packets,excess_energy_expended
    def step(self, action):
        """Execute one timestep within the environment.
        
        Args:
            action: The action to take (0-3 for directions)
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        if action==0:
            reward=0
        else:
            #print(self.timestep)
            #print(self.contact_plan[self.timestep])
            current_delivered_data,current_energy_expenditure=self.updateDeliveryandEnergy(self.contact_plan[self.timestep],10,self.satellite_remaining_data*100)
            #Use link availability model to update the environment state

            if current_delivered_data>0:
                reward=(10/(self.initial_data_volume*100))*current_delivered_data-5/(100*self.initial_data_volume)*current_delivered_data*(current_energy_expenditure/(current_energy_expenditure+10))
                #print("Step Reward 1")
                #print(reward)
            else:
                reward=-(5*current_energy_expenditure)/(self.initial_data_volume*100)
                #print("Step Reward 2")
                #print(reward)
            #print("Current Delivered Data")
            #print(current_delivered_data)
            #print(current_energy_expenditure)
            #print("Initial Remaining Data")
            #print((self.satellite_remaining_data)*100)
            self.delivered_data=self.delivered_data+current_delivered_data
            self.Energy_Expended=self.Energy_Expended+current_energy_expenditure
            self.satellite_remaining_data=(self.satellite_remaining_data)*100-current_delivered_data
            self.satellite_remaining_data=self.satellite_remaining_data/100
            #print("remaining data")
            #print((self.satellite_remaining_data)*100)
        self.updateTimestep()
        terminated=False
        truncated=False
        if self.timestep>=10:
            terminated=True
            #Episode Reward
            #print("Delivered Data")
            #print(self.delivered_data)
            #print("Initial Data volume")
            #print(self.initial_data_volume)
            reward=reward+10*(self.delivered_data/(self.initial_data_volume*100))-5*(self.Energy_Expended/100)
            #print("Episode Reward 1")
            #print(reward)
        else:
            terminated=False
            
        if self.satellite_remaining_data<=0:
            truncated=True
            #Episode Reward
            reward=reward+10-(self.Energy_Expended/(10*(self.timestep)))*5
            #print("Episode Reward 2")
            #print(reward)
        else:
            truncated=False
        observation = self._get_obs()
        info = self._get_info()
        #print("Remaining_data")
        #print(self.satellite_remaining_data)
        #print("Step Reward")
        #print(reward)
        return observation, reward, terminated, truncated, info
    def _get_info(self):
        """"
        Returns:
        dict: Info with energy expenditure and delivered packets
        """
        return {"contact plan":self.contact_plan, "Remaining data":self.satellite_remaining_data,"current timestep":self.timestep,"Energy Expended":self.Energy_Expended,"Delivered Data":self.delivered_data}
    def _get_obs(self):
        """
        Returns:
            dict: With satellite observation information
        """
        return {"contact plan": self.contact_plan, "Remaining data": self.satellite_remaining_data,"current timestep":np.array([float(self.timestep)], dtype=np.float32)}
    def updateTimestep(self):
        self.timestep=self.timestep+1
        return self.timestep


# In[ ]:


#Next need to produce training graphs and compare between pre-existing agents

