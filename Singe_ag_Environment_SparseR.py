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
from gymnasium.utils import seeding


# In[4]:


#This environment gives us insight into the basic stochastic knapsack problem that we need to solve in the multi-agent environment
#Basically it is the same problem as the multi-agent environment without the complexities of fairness and collisions
class SingleSatelliteEnvSR(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],  # or ["human", "rgb_array"]
        "render_fps": 30
    }
    def __init__(self,render_mode="rgb_array"):
        print("DEBUG: SingleSatelliteEnvSR __init__ called")
        print("DEBUG metadata:", self.metadata)
        self.render_mode = render_mode
        self.np_random = np.random.default_rng()
        self.timestep=None#Current contact that we are using (indicates wher we are in the contact plan
        #Contact plan in this case only contains weather condition information (as we assume all equal contact lengths)
        self.contact_plan=np.full((10,), None)#Contact Plan
        self.satellite_remaining_data=None#Data that we still need to transmit to the ground before downlink period ends
        self.delivered_data=None#Data that has been delivered to the ground 
        self.Energy_Expended=None#Excess Energy Expended
        self.initial_data_volume=None#Initial Amount of data we need to transmit
        self.num_of_contacts=None#Number of contacts that we have used
        self.max_episode_steps = 10
        #Observation space is a (12,) array containing the timestep, remaining satellite data and the weather information
        self.observation_space = gym.spaces.Box(
    low=np.array([0.0]*10 + [0.0] + [0.0], dtype=np.float32),
    high=np.array([1.0]*10 + [1.0] + [10.0], dtype=np.float32),
    dtype=np.float32
)
        
        self.action_space = gym.spaces.Discrete(2)#2 possible values either satellite chooses to use contact or not
    
    
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)#Set seed
        self.timestep=0#Initialize timstep
        #
        self.contact_plan=self.np_random.uniform(low=0.0, high=1.0, size=(10,)).astype(np.float32)#Randomly initialize the weather conditions
        self.contact_plan= np.round(self.contact_plan*10)/10#Round the weather conditions to the nearest decade
        self.satellite_remaining_data=np.array(self.np_random.integers(5, 100) / 100.0, dtype=np.float32)#randomly initialize the data that needs to be delivered
        self.delivered_data=0#Initialize the amount of data delivered to the ground
        self.initial_data_volume=self.satellite_remaining_data#Store the amount of data that we will try to deliver during the downlink period
        self.Energy_Expended=0#Initialize the amount of excess energy wasted
        self.num_of_contacts=0#Intialize the number of contacts used during the downlink period by the satellite
        observation = self._get_obs()#Get the observation vector
        info = self._get_info()#Get info 
        return observation, info#Return Observation and Info
    def updateDeliveryandEnergy(self,weather,length,remaining_data):
        delivered_packets=0#Local variable, number of packets delivered during the contact
        excess_energy_expended=0#Local variable, excess energy expended during the contact
        random_sample=self.np_random.uniform(low=0.0, high=1.0, size=(length,)).astype(np.float32)#10 random samples that we will compare to the cloud cover value
        remaining_data=np.round(remaining_data)#just rounding to be safe (probably should be handling the types better)
        for i in range(0,length):
            if remaining_data>=1:
                if random_sample[i]> weather:
                    delivered_packets=delivered_packets+1#successfully transmit packet
                    remaining_data=remaining_data-1
                else:
                    excess_energy_expended=excess_energy_expended+1#not successful in transmission
            else:
                excess_energy_expended=excess_energy_expended+1#not successful in transmitting
        return delivered_packets,excess_energy_expended#Return the number of packets delivered and excess energy expended
    def step(self, action):
        if action==0:
            reward=0#Satellite does not receive any reward if we don't try to use a contact, this is done to incentivize 
        else:
            self.num_of_contacts+=1#In add to the number of contacts that we used
            data_remaining=self.satellite_remaining_data*100#Just to make the math easier we convert data remaining to an integer
            current_weather=self.contact_plan[self.timestep]#We get the current weather conditions from the contact plan
            current_delivered_data,current_energy_expenditure=self.updateDeliveryandEnergy(current_weather,10,data_remaining)#We use the link availability model to get the number of packets delivered and the energy expended
            init_data_volume=self.initial_data_volume#Local variable of the initial data volume (might not be necessary)
            #if we do deliver some data during the contact, reward will always be positive
            self.delivered_data=self.delivered_data+current_delivered_data#We update the number of delivered packets
            self.Energy_Expended=self.Energy_Expended+current_energy_expenditure#We update the amount of enegy expended
            self.satellite_remaining_data=np.round((self.satellite_remaining_data)*100)-np.round(current_delivered_data)#Update the remaining data
            self.satellite_remaining_data=self.satellite_remaining_data/100
            if self.satellite_remaining_data<0:
                self.satellite_remaining_data=0
        self.updateTimestep()#Update timestep
        terminated=False
        truncated=False
        #We check if we are at the end of the episode
        if self.timestep>=self.max_episode_steps:
            terminated=True
            delivered_data=self.delivered_data
            init_data_volume=self.initial_data_volume*100
            energy_expended=self.Energy_Expended
            if np.round(self.delivered_data)==np.round(self.initial_data_volume*100):
                reward=self.delivered_data/(self.initial_data_volume*100)+0.5*(self.Energy_Expended/(self.num_of_contacts*10))
            else:
                reward=self.delivered_data/(self.initial_data_volume*100)
            
        elif self.satellite_remaining_data<=0:
            truncated=True
            #Episode Reward
            energy_expended=self.Energy_Expended
            number_of_contacts_used=self.num_of_contacts
            if np.round(self.delivered_data)==np.round(self.initial_data_volume*100):
                reward=self.delivered_data/(self.initial_data_volume*100)+0.5*(self.Energy_Expended/(self.num_of_contacts*10))
            else:
                reward=self.delivered_data/(self.initial_data_volume*100)
        else:
            reward=0
        observation = self._get_obs()
        #print(observation)
        info = self._get_info()
        reward=float(reward)
        #print(reward)
        return observation, reward, terminated, truncated, info
    def _get_info(self):
        
        return {"contact plan":self.contact_plan, "Remaining data":self.satellite_remaining_data,"current timestep":self.timestep,"Energy Expended":self.Energy_Expended,"Delivered Data":self.delivered_data,"Initial Data":self.initial_data_volume,"Number of Contacts":self.num_of_contacts}
        #Info is structured as a dictionary to ease usability during graphing and analysis
    def _get_obs(self):
       return np.concatenate([
        np.array(self.contact_plan, dtype=np.float32),
        np.array([self.satellite_remaining_data], dtype=np.float32),
        np.array([self.timestep], dtype=np.float32)
    ])
    def render(self):
        if self.render_mode == "rgb_array":
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            return frame
        #Observation get function, also structures observations properly as a 12 element vector
    def updateTimestep(self):
        self.timestep=self.timestep+1
        return self.timestep
    def updateDeliveredPackets(self,new_deliveries):
        self.delivered_data=self.delivered_data+new_deliveries
    def getDeliveredPackets(self):
        return self.delivered_data
    def getEnergyExpenditure(self):
        return self.Energy_Expended
    def updateEnergyExpenditure(self,new_expend):
        self.Energy_Expended=self.Energy_Expended+new_expend
    def updateRemainingData(self,new_deliveries):
        self.satellite_remaining_data=(self.satellite_remaining_data)*100-current_delivered_data
        self.satellite_remaining_data=self.satellite_remaining_data/100
        


# In[3]:


#Next need to produce training graphs and compare between pre-existing agents
from gymnasium.envs.registration import register
register(
    id="SingleSatelliteEnvSR-v1",
    entry_point="Singe_ag_Environment_SparseR:SingleSatelliteEnvSR",  # Adjust the module path
)

# Test if the environment works
#env = gym.make("SingleSatelliteEnv")
#obs, info = env.reset()
#print("Custom Env Loaded Successfully!")


# In[ ]:




