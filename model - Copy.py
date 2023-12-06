#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
from gurobipy import *
import math

# OPTIGUIDE DATA CODE GOES HERE
df = pd.read_csv('data_w_distance.csv', encoding='ISO-8859-1')

df_selected = df[['Type', 'Days for shipping (real)', 'Days for shipment (scheduled)',
       'Benefit per order', 'Sales per customer', 'Delivery Status',
       'Late_delivery_risk', 'Category Id', 'Category Name', 'Customer City',
       'Customer Country', 'Customer Email', 'Customer Fname', 'Customer Id',
       'Customer Lname', 'Customer Password', 'Customer Segment',
       'Customer State', 'Customer Street', 'Customer Zipcode',
       'Department Id', 'Department Name', 'Latitude', 'Longitude', 'Market',
       'Order City', 'Order Country', 'Order Customer Id',
       'order date (DateOrders)', 'Order Id', 'Order Item Cardprod Id',
       'Order Item Discount', 'Order Item Discount Rate', 'Order Item Id',
       'Order Item Product Price', 'Order Item Profit Ratio',
       'Order Item Quantity', 'Sales', 'Order Item Total',
       'Order Profit Per Order', 'Order Region', 'Order State', 'Order Status',
       'Order Zipcode', 'Product Card Id', 'Product Category Id',
       'Product Description', 'Product Image', 'Product Name', 'Product Price',
       'Product Status', 'shipping date (DateOrders)', 'Shipping Mode',
       'Order City Coordinates', 'Customer City Coordinates', 'Distance_km']].reset_index(drop=True)

df_selected = df_selected.sample(n=40000, random_state=42).reset_index(drop=True)

shipping_speed = {
    'First_Class': 1500, # distance travel per day in km
    'Same_Day': 1000, # distance travel per day in km
    'Second_Class': 800, # distance travel per day in km
    'Standard_Class': 500, # distance travel per day in km
}

cost_factors = {
    'First_Class': 0.2,  # cost per km
    'Same_Day': 0.15,  # cost per km
    'Second_Class': 0.1,  # cost per km
    'Standard_Class': 0.05,  # cost per km
}
total_orders = 40000

#benefit_factors = {
#    'First_Class': 0.05,  
#    'Same_Day': 0.03,
#    'Second_Class': 0.02,
#    'Standard_Class': 0.01,
#}

penalty_per_day_late = 0.02


m = Model("ShippingMode")

orders = [i for i in range (total_orders)]
shipping_mode = [i for i in shipping_speed.keys()]

x = m.addVars(orders, shipping_mode, vtype=GRB.BINARY)

cost = [
    [df_selected.loc[i, 'Order Item Product Price'] * cost_factors[shipping_mode[j]] for j in range(len(shipping_mode))]
    for i in range(len(orders))
]

immediate_cost = pd.DataFrame(cost, columns=shipping_mode)

days = [
    [math.ceil(df_selected.loc[i, 'Distance_km'] / shipping_speed[shipping_mode[j]]) for j in range(len(shipping_mode))]
    for i in range(len(orders))
]

delivery_day = pd.DataFrame(days, columns=shipping_mode)

late_days = [
    [max(0,delivery_day.loc[i,shipping_mode[j]]- int(df_selected.loc[i,"Days for shipment (scheduled)"]))for j in range(len(shipping_mode))]
    for i in orders
]

late_day= pd.DataFrame(late_days, columns=shipping_mode)

late_delivery_cost = [
    [late_day.loc[i,shipping_mode[j]]*penalty_per_day_late*df_selected.loc[i,"Order Item Product Price"]
     for j in range(len(shipping_mode))]
    for i in range(len(orders))
]

late_cost = pd.DataFrame(late_delivery_cost, columns=shipping_mode)


m.setObjective(
    quicksum(immediate_cost.loc[i, shipping_mode[j]]*x[i,shipping_mode[j]] + late_cost.loc[i, shipping_mode[j]]*x[i,shipping_mode[j]] for j in range(len(shipping_mode)) for i in orders),
    GRB.MINIMIZE
)


# OPTIGUIDE CONSTRAINT CODE GOES HERE
m.addConstrs(quicksum(x[i, shipping_mode[j]] for j in range(len(shipping_mode))) == 1 for i in orders)
m.addConstrs(quicksum(x[i, shipping_mode[j]] for i in orders) <= total_orders/4 for j in range(len(shipping_mode)))
m.addConstrs(late_day.loc[i,shipping_mode[j]]*x[i,shipping_mode[j]] <= 13 for i in orders for j in range(len(shipping_mode)))

m.update()
m.optimize()

print("Retrieve Optimal Solution.")
print("Objective value =", m.objVal)

selected_modes = [(i, j) for i in orders for j in shipping_mode if x[i, j].x == 1]
optimal_choice = pd.DataFrame(selected_modes, columns=['Order','Selected_Shipping_Mode'])
optimal_choice.set_index('Order', inplace=True)
optimal_choice = pd.merge(df_selected["Order Item Id"], optimal_choice, left_index=True, right_index=True)
optimal_choice
