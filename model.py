#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
from gurobipy import *
import math


df = pd.read_csv('data_w_distance.csv', encoding='ISO-8859-1')
df_selected = df[['Order Item Id','Days for shipment (scheduled)','Order Item Product Price','Distance_km']].reset_index(drop=True)
df_selected = df_selected.sample(n=40000, random_state=42).reset_index(drop=True)

shipping_speed = {
    'First_Class': 7000, #distance travel per day 
    'Same_Day': 5000,
    'Second_Class': 3000,
    'Standard_Class': 1000,
}

cost_factors = {
    'First_Class': 0.045,  
    'Same_Day': 0.025,
    'Second_Class': 0.015,
    'Standard_Class': 0.009,
}

penalty_per_day_late = 0.02

# Total number of orders
total_orders = len(df_selected)

#Initialize the model
m = Model("ShippingMode")

#Setting up the parameters
orders = [i for i in range (total_orders)]
shipping_mode = [i for i in shipping_speed.keys()]

#Decision Variables. X[i,j] binary variable. 
x = m.addVars(orders, shipping_mode, vtype=GRB.BINARY)

#immediate shipping cost matrix. C[i,j] means the cost of shipping the ith order using jth shipping mode . 
#This should be calculated as the product price matching the ith order* cost_factors values for the chosen shipping mode.
cost = [
    [df_selected.loc[i, 'Order Item Product Price'] * cost_factors[shipping_mode[j]] for j in range(len(shipping_mode))]
    for i in range(len(orders))
]

immediate_cost = pd.DataFrame(cost, columns=shipping_mode)

#create a real delivery dataframe for the ith order having chosen jth shipping mode.
#The real delivery time is calculated as distance for delivering ith order/ shipping speed for jth shipping mode.
days = [
    [math.ceil(df_selected.loc[i, 'Distance_km'] / shipping_speed[shipping_mode[j]]) for j in range(len(shipping_mode))]
    for i in range(len(orders))
]
delivery_day = pd.DataFrame(days, columns=shipping_mode)

#create a late day dataframe
#late days are calculated as actual delivery days- scheduled delivery days
late_days = [
    [max(0,delivery_day.loc[i,shipping_mode[j]]- int(df_selected.loc[i,"Days for shipment (scheduled)"]))for j in range(len(shipping_mode))]
    for i in orders
]
late_day= pd.DataFrame(late_days, columns=shipping_mode)

#create a late delivery cost for ith order having chosen jth shipping mode.
#the value will be late_days* penalty_per_day_late*product price.
late_delivery_cost = [[late_day.loc[i,shipping_mode[j]]*penalty_per_day_late*df_selected.loc[i,"Order Item Product Price"]
                    for j in range(len(shipping_mode))]
                    for i in range(len(orders))
                    ]
late_cost = pd.DataFrame(late_delivery_cost, columns=shipping_mode)

#The objective is to minimize the total cost associated with choosing the specific shipping mode for each order.
#the total cost is the late delivery cost+ immediate_cost
m.setObjective(
    quicksum(immediate_cost.loc[i, shipping_mode[j]]*x[i,shipping_mode[j]] + late_cost.loc[i, shipping_mode[j]]*x[i,shipping_mode[j]] for j in range(len(shipping_mode)) for i in orders),
    GRB.MINIMIZE
)

# set constraints 

#first constraint is that for each order (for all i), there should only be one shipping mode chosen.
m.addConstrs(quicksum(x[i, shipping_mode[j]] for j in range(len(shipping_mode))) == 1 for i in orders)

#second constraint is that for each shipping mode, it has a maximum delivery capacity of total_orders / 4
m.addConstrs(quicksum(x[i, shipping_mode[j]] for i in orders) <= total_orders/4 for j in range(len(shipping_mode)))

# #third constraint is that the maximum late delivery day should be less than 5 days to ensure that the scheduled days are not overly underestimated.
m.addConstrs(late_day.loc[i,shipping_mode[j]]*x[i,shipping_mode[j]] <= 5 for i in orders for j in range(len(shipping_mode)))

m.update()

# Solve
m.optimize()

print("Retrieve Optimal Solution.")

# Print objective value
print("Objective value =", m.objVal)

print("***")

#print out the optimal choice dataframe
selected_modes = [(i, j) for i in orders for j in shipping_mode if x[i, j].x == 1]
optimal_choice = pd.DataFrame(selected_modes, columns=['Order','Selected_Shipping_Mode'])
optimal_choice.set_index('Order', inplace=True)
optimal_choice = pd.merge(df_selected["Order Item Id"], optimal_choice, left_index=True, right_index=True)
optimal_choice

