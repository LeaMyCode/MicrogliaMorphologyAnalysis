# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:44:07 2023

@author: leaga
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 17:11:17 2022

@author: leaga
"""
import numpy as np
from tkinter import filedialog
import os
import pandas as pd


def processDat_dict (data):
    data_list = []
    try:
        for dat in data:
            for i in range(len(dat)):
                data_list.append(dat[i])
    except: 
        print("You probably do have less than 4 channel. Check your data in case you want to analyze 4 channel. Error in: processDat_dict")
        data_list = [0] * cell_counter
    
    return data_list
            

def prepDat_dict (Data):
    datas = []
    for dat in Data:
        datas.append(dat[0])
  
    return datas

def microglia_percentage (DAPI, Microglia):
    percentage_list = []
    for i in range(len(DAPI)):
        percentage_list.append(float(Microglia[i]/DAPI[i][0]) *100)
    return percentage_list


def CountCell(cells_in_list):
    total_cells = 0
    for cells in cells_in_list:
        total_cells += int(cells)
    
    return total_cells
        
#where are your data located?
root_path = filedialog.askdirectory(title="Please select your folder with images to analyze")

xls_list = []
diction_data  = {}


CD68_area_list = []
CD68_IntDen_list = []
CD68_Mean_list = []

IBA1_area_list = []
IBA1_IntDen_list = []
IBA1_Mean_list = []


filename_list = []
microglia_number_list = []
microglia_number_total = []


#extracts all neccessary data from each .xls file
for file in os.listdir(root_path):
    results_path = os.path.join(root_path,"Analysis")
    if file == "Analysis":
        continue
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    #print(data.columns)
    
    if file.endswith(".png"):
        continue
    
    if file.endswith(".zip"):
        continue
        
    if file.endswith("-CD68.csv"):
        filename = file.split("-CD68.csv")[0]
        filename_list.append(filename)
        xls_path = os.path.join(root_path,file) 
        data = pd.read_csv(xls_path)
        df_CD68_area = prepDat_dict(np.asarray(pd.DataFrame(data, columns=['Area'])[0:]).astype(float))
        df_CD68_Int = prepDat_dict(np.asarray(pd.DataFrame(data, columns=['IntDen'])[0:]).astype(float))
        df_CD68_Mean = prepDat_dict(np.asarray(pd.DataFrame(data, columns=['Mean'])[0:]).astype(float))
        CD68_area_list.append(df_CD68_area)
        CD68_IntDen_list.append(df_CD68_Int)
        CD68_Mean_list.append(df_CD68_Mean)
         
    
    if file.endswith("-IBA1.csv"):
        file = filename + "-IBA1.csv"
        xls_path = os.path.join(root_path,file) 
        data = pd.read_csv(xls_path)
        df_IBA1_area = prepDat_dict(np.asarray(pd.DataFrame(data, columns=['Area'])[0:]).astype(float))
        df_IBA1_Int = prepDat_dict(np.asarray(pd.DataFrame(data, columns=['IntDen'])[0:]).astype(float))
        df_IBA1_Mean = prepDat_dict(np.asarray(pd.DataFrame(data, columns=['Mean'])[0:]).astype(float))
        IBA1_area_list.append(df_IBA1_area)
        IBA1_IntDen_list.append(df_IBA1_Int)
        IBA1_Mean_list.append(df_IBA1_Mean)
        microglia_number_list.append(list(range(0, len(df_CD68_area))))
        microglia_number_total.append(len(df_CD68_area))  
        
    else:
        continue


global cell_counter
cell_counter = CountCell(microglia_number_total)


filename_list_perCell = []

for nmbr in range(len(microglia_number_total)):
    filename_list_perCell.append([filename_list[nmbr]]*microglia_number_total[nmbr])


'''
if len(IBA1_area_list) != cell_counter:
    IBA1_area_list = [0] * cell_counter
    IBA1_IntDen_list = [0] * cell_counter
    IBA1_Mean_list = [0] * cell_counter
'''

diction_data["Filename"] = processDat_dict(filename_list_perCell)
diction_data["#"] = processDat_dict(microglia_number_list)
diction_data["microglia_area"] = processDat_dict(CD68_area_list)
diction_data["CD68_IntDen"] = processDat_dict(CD68_IntDen_list)
diction_data["CD68_Mean"] = processDat_dict(CD68_Mean_list)
diction_data["IBA1_IntDen"] = processDat_dict(IBA1_IntDen_list)
diction_data["IBA1_Mean"] = processDat_dict(IBA1_Mean_list)


"""
for key in diction_data:
    print(key, len(diction_data[key]))
"""

# Find the maximum length of all values (lists) in the dictionary
max_length = max(len(lst) for lst in diction_data.values())
#print("max length: ", max_length)

# Pad each list in the dictionary with NaN to ensure the same length
for key in diction_data:
    current_length = len(diction_data[key])
    if current_length < max_length:
        diction_data[key] += [pd.NA] * (max_length - current_length)


# create a dataframe from the dictionary
df_diction1 = pd.DataFrame(diction_data)

# write the dataframe to a CSV file with headers
df_diction1.to_csv(results_path + '\Analysis_Intensity.csv', index=False, header=True)







