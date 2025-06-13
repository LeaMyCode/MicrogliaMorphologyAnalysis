# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:15:48 2023

@author: leaga
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import filedialog
import os
import csv
import statistics
from statistics import mean



def RadarPlotting(data, save_path, characteristics, conditions, colors, title, limit):
    
    # trim the data for the characteristics you want + condition name
    scaling_factor = 1000
    df = data.loc[:, characteristics] * scaling_factor
    #print(df.index)
    
    # Number of variables we're plotting.
    num_vars = len(characteristics)
    
    # Split the circle into even parts and save the angles
    # so we know where to put each axis.
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # The plot is a circle, so we need to "complete the loop" and append the start value to the end.
    angles += angles[:1]
    #print("len angles :", len(angles))    
    # ax = plt.subplot(polar=True)
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    
    # Helper function to plot each condition on the radar chart.
    def add_to_radar(cond, color):
      values = df.loc[cond].tolist()
      values += values[:1]
      ax.plot(angles, values, color=color, linewidth=3, label=cond)
      #ax.fill(angles, values, color=color, alpha=0.25)
    
    # Add each condition
    for i in range(len(conditions)):
        add_to_radar(conditions[i], colors[i]) 

    # Fix axis to go in the right order and start at 12 o'clock.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label.
    ax.set_thetagrids(np.degrees(angles[:-1]), characteristics)
    
    # Go through labels and adjust alignment based on where
    # it is in the circle.
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
        # Make the font a little bigger
        label.set_fontsize(13)  
        
    # Ensure radar goes from 0 to x.
    #max_value = df.max().max()
    #ax.set_ylim(0, max_value + 1)
    ax.set_ylim(0, limit)
    
    # You can also set gridlines manually like this:
    # ax.set_rgrids([20, 40, 60, 80, 100])

    # Set position of y-labels (0-100) to be in the middle
    # of the first two axes.
    ax.set_rlabel_position(180 / num_vars)
    
    # Add some custom styling.
    # Change the color of the tick labels.
    ax.tick_params(colors='#222222')
    # Make the y-axis (0-100) labels smaller.
    ax.tick_params(axis='y', labelsize=0)
    # Change the color of the circular gridlines.
    ax.grid(color='#AAAAAA')
    # Change the color of the outermost gridline (the spine).
    ax.spines['polar'].set_color('#222222')
    
    

    # Change the background color inside the circle itself.
    
    # Convert RGB values to the format used in ax.patch.set_facecolor
    
    red = 210/ 255
    green = 210/ 255
    blue = 210/ 255
    
    # Set the background color inside the circle itself.
    ax.patch.set_facecolor((red, green, blue, 1))
    
    # Set the background color of the entire figure.
    fig.patch.set_facecolor((1, 1, 1, 1))
   

    # Add title.
    #ax.set_title('Microglia Morphology 24 hpi', y=1.08, fontweight='bold', fontsize=15)
    
    # Add a legend as well.
    #ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15))
    
    for r_label in ax.get_yticklabels():
       r_label.set_text('')
    
    # save the plot
    fig.savefig(rf"{save_path}\radar_plot_morphology_{title}.svg", format="svg")
    fig.savefig(rf"{save_path}\radar_plot_morphology_transparent_{title}.png", facecolor=fig.get_facecolor(), dpi=600, transparent=True)
    
    
def drop_rows_with_any_na(dictionary):
    # Find indices with any NaN value across all keys
    indices_to_drop = set()
    for values in dictionary.values():
        indices_to_drop.update([i for i, value in enumerate(values) if value is np.nan or pd.isna(value)])

    # Drop indices from all keys
    for key, values in dictionary.items():
        dictionary[key] = [value for i, value in enumerate(values) if i not in indices_to_drop]
        
# normalize morphological data per condition (normalize to female ctrl)
def NormalizeData(df, condition_1, conditions, plot_path, dates, first, last):
    characteristics = list(df.columns.values)[first:last]
    print("Characteristics: ", characteristics)
    diction_data = {}
    for cond in conditions: 
        diction_data[cond] = {}
        for dat in dates:
            # Filter both `cond` and `dat`
            df_cond = df[df["Filename"].str.contains(cond, na=False, regex=False) &
                df["Filename"].str.contains(dat, na=False, regex=False)]
            
            # normalize to cond_1
            df_cond_1 = df[df["Filename"].str.contains(condition_1, na=False, regex=False) &
                df["Filename"].str.contains(dat, na=False, regex=False)]
    
            for char in characteristics:
                if char == 'condition':
                    continue
                else:
                    # calculate mean of cond_1 (for normalization)
                    df_cond[char] = pd.to_numeric(df_cond[char], errors='coerce')
                    df_cond_1[char] = pd.to_numeric(df_cond_1[char], errors='coerce')
                    cond_1_mean = df_cond_1[char].mean()
                    
                    
                    # normalize
                    if char not in diction_data[cond]:
                        diction_data[cond][char] = (df_cond[char]/cond_1_mean).tolist()
            
                    else:
                        diction_data[cond][char].extend((df_cond[char]/cond_1_mean).tolist())
        
        #get rid of zeros and add NaNs instead
        for key, values in diction_data[cond].items():
            diction_data[cond][key] = [np.nan if value == 0 else value for value in values]
        
        # save your data 
        #for key, values in diction_data[cond].items():
            #print(f"Key '{key}' hat {len(values)} Werte.")
        pd.DataFrame(diction_data[cond]).to_csv(rf"{plot_path}\normalized_MORPH_{cond}.csv", header=True, index = False)

    dataframes = []
    for cond in conditions:
        #copy dicts
        drop_rows_with_any_na(diction_data[cond])
        # prepare data for radarplotting
        # build the mean of all characteristics
        mean_cond = pd.DataFrame(diction_data[cond]).mean()
        #transpose the data to obtain the correct orientation of the data
        mean_cond = pd.DataFrame(mean_cond).T
        mean_cond["condition"] = cond
        
        dataframes.append(mean_cond)
        
    #combine the dataframes
    radarplotting_df = pd.concat(dataframes, ignore_index=True)
    # drop old index and set condition as index
    radarplotting_df.reset_index(inplace=True, drop=True)
    radarplotting_df.set_index("condition", inplace = True)
    radarplotting_df.to_csv(rf"{plot_path}\RadarPlotting.csv", header=True)


    return radarplotting_df
        
def __main__():  
    # where are your data located? The folder with your summary data
    root_path = filedialog.askdirectory(title="Please select your folder with data")
    direction = os.path.join(root_path,"summary_data_with_intensity.csv")
    
    # load your data to normalize
    data = pd.read_csv(direction, index_col=False)
    data.dropna(inplace=True)
    data.to_csv(rf"{root_path}\cleaned_summary.csv", index=False)
    
    # create the results folder if not there jet
    plot_path = os.path.join(root_path,"Radarplot")
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    

    # specify your conditions
    condition_1 = "HC_CTRL"
    condition_2 = "HC_POLY"
    condition_3 = "CTX_CTRL"
    condition_4 = "CTX_POLY"
    
    conditions = [condition_1, condition_2, condition_3, condition_4]
    
    
    #define colors
    color_1 = "violet"
    color_2 = "darkmagenta"
    color_3 = "cornflowerblue"
    color_4 = "midnightblue"
    
    
    colors = [color_1,color_2,color_3,color_4]
    
    
    #specify your dates/rounds
    dates_list = ["_A_", "_B_", "_C_"]
    
    
    #characteristics = data.columns.tolist()[4:34]
    
    
    
#Morphology data
    characteristics = data.columns.tolist()[10:29]
    print("Characteristics: ", characteristics)
    
    #radar_df = NormalizeData(data, condition_1, conditions, plot_path, dates_list, 4, 34  )

    characteristics = ['# Branches','convex_hull_area', 'cell_area', 'roughness','cell_circularity','density','Average Branch Length']

    title = "Morph"
    radar_df = NormalizeData(data, condition_1, conditions, plot_path, dates_list, 10, 29)
    RadarPlotting(radar_df, plot_path, characteristics, conditions, colors, title, 1700)


    # normalize your morphologcial data, save and get the radarplotting data
    title = "HC_Morph"
    radar_df_1 = NormalizeData(data, condition_1, [condition_1, condition_2],  plot_path, dates_list, 10, 29)
    RadarPlotting(radar_df_1, plot_path, characteristics, [condition_1, condition_2], [color_1,color_2], title, 1700)
    
    # normalize your morphologcial data, save and get the radarplotting data
    title = "CTX_Morph"
    radar_df_2 = NormalizeData(data, condition_3, [condition_3, condition_4], plot_path, dates_list, 10, 29)
    RadarPlotting(radar_df_2, plot_path, characteristics, [condition_3, condition_4], [color_3, color_4], title, 1700)


#Immunostaining data
    characteristics = data.columns.tolist()[4:8]
    print("CHARACTERISTICS: ", characteristics)
    
    characteristics = ['IBA1_Mean', 'IBA1_IntDen', 'CD68_Mean', 'CD68_IntDen']

    title = "Intensities"
    radar_df = NormalizeData(data, condition_1, conditions, plot_path, dates_list, 4, 8 )
    RadarPlotting(radar_df, plot_path, characteristics, conditions, colors, title, 2200)
    

    # normalize your morphologcial data, save and get the radarplotting data
    title = "HC_Intensites"
    radar_df_1 = NormalizeData(data, condition_1, [condition_1, condition_2],  plot_path, dates_list, 4, 8)
    RadarPlotting(radar_df_1, plot_path, characteristics, [condition_1, condition_2], [color_1,color_2], title, 1200)
    
    # normalize your morphologcial data, save and get the radarplotting data
    title = "CTX_Intensities"
    radar_df_2 = NormalizeData(data, condition_3, [condition_3, condition_4], plot_path, dates_list, 4, 8)
    RadarPlotting(radar_df_2, plot_path, characteristics, [condition_3, condition_4], [color_3, color_4], title, 1200)

__main__()
        
        
    

        
    