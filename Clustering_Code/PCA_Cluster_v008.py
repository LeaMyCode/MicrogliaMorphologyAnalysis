0# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:55:21 2022

@author: leaga

PCA analysis microglia morphology data
"""
from tkinter import filedialog
import pandas as pd
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt # NOTE: This was tested with matplotlib v. 2.1.0
import matplotlib.patches as mpatches



#########################
#
# Please specify your conditions
#
#########################

condition_1 = "HC_CTRL"
condition_2 = "HC_POLY"
condition_3 = "CTX_CTRL"
condition_4 = "CTX_POLY"


conditions = [condition_1, condition_2, condition_3, condition_4]


# Define colors for the conditions manually

color_1 = "violet"
color_2 = "darkmagenta"
color_3 = "cornflowerblue"
color_4 = "midnightblue"


custom_colors = [color_1, color_2, color_3, color_4]

#########################
#
# Please specify your dates!
#
#########################

#get the number of Rounds and the name of the dates 
global dates_list
dates_list = ["_A_", "_B_", "_C_"]

#########################
#
# Please decode your conditions (if you have it coded, otherwise leave this empty)
#
#########################
global replacements
replacements = {
    'A1' : '_A_A1_CTX_CTRL',
    'A2' : '_A_A2_CTX_POLY',
    'A3' : '_A_A3_HC_CTRL',
    'A4' : '_A_A4_HC_POLY',
    'B1' : '_B_B1_HC_CTRL',
    'B2' : '_B_B2_CTX_CTRL',
    'B3' : '_B_B3_CTX_POLY', 
    'B4' : '_B_B4_HC_POLY', 
    'C1' : '_C_C1_HC_POLY', 
    'C2' : '_C_C2_CTX_POLY', 
    'C3' : '_C_C3_HC_CTRL', 
    'C4' : '_C_C4_CTX_CTRL', 
}

#########################
#
# Load your data
# Please open the whole Dataset (all conditions included)
#
#########################

def load_data_from_directory(directory, file_prefix, columns, header):
    dataframes = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(file_prefix) and file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, usecols=columns, names=header, encoding='latin1', header=None)[1:]
                dataframes.append(df)
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    else:
        return None

#########################
#
# Modify microglia names (remove the date)
# Unfortunately, Fraclac and MorphData save dots differently (fraclac data deletes dots)
#
#########################

def modify_filename_fraclac(s):
    try:
        name = s.split('_')
        name = ('_'.join((s.split('_'))[6:]))
        name = name.replace(" ", "")
        name = name.split('.')
        name = ''.join(name[0:])
        name.rstrip()
    except:
        name = int(s)
    
    return name

def modify_filename_skeleton(s):
    try:
        name = s.split('_')
        name = ('_'.join((s.split('_'))[2:]))
        name = name.replace(" ", "")
        name = name.split('.')
        name = ''.join(name[0:])
        name.rstrip()
    except:
        name = int(s)
    
    return name


def modify_filename_intensity(s):
    try:
        name = s.split('.')
        name = ''.join(name[0:])
        #print(name)
    except:
        name = s
    return name

########################
#
# Decode your data
#
#########################

def Decoder (value):
    for old, new in replacements.items():
        if old in value:
            value = value.replace(old, new)
            
    return value

######################### 
# 
# Data Load
#
######################### 


root_path = filedialog.askdirectory(title="Please select your folder with data")
morph_data = ['fractal_dimension', 'lacunarity', 'density', 'span_ratio_major_minor', 'convex_hull_area', 'convex_hull_perimeter', 'convex_hull_circularity', 'diameter_bounding_circle', 'mean_radius', 'max_span_across_convex_hull', 'max_min_radii', 'cell_area', 'cell_perimeter', 'roughness', 'cell_circularity', '# Branches', '# Junctions', 'Average Branch Length', 'Maximum Branch Length']

data_fraclac = load_data_from_directory(root_path, 'fraclac', [2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21], ['microglia','fractal_dimension', 'lacunarity', 'density', 'span_ratio_major_minor', 'convex_hull_area', 'convex_hull_perimeter', 'convex_hull_circularity', 'diameter_bounding_circle', 'mean_radius', 'max_span_across_convex_hull', 'max_min_radii', 'cell_area', 'cell_perimeter', 'roughness', 'cell_circularity'])
data_skeleton = load_data_from_directory(root_path, 'skeleton', [2, 3, 7, 10, 12], ['# Branches','# Junctions','Average Branch Length', 'Maximum Branch Length', 'microglia'])

results_path = os.path.join(root_path,"Analysis_all_Characteristics")
if not os.path.exists(results_path):
    os.makedirs(results_path)


if data_fraclac is not None and data_skeleton is not None:
    # Modify microglia names
    data_fraclac['microglia'] = data_fraclac['microglia'].apply(lambda x: modify_filename_fraclac(x) if isinstance(x, str) else x)
    data_skeleton['microglia'] = data_skeleton['microglia'].apply(lambda x: modify_filename_skeleton(x) if isinstance(x, str) else x)


    # Further processing using the loaded dataframes
else:
    print("No data found in the specified directory.")
    
    
#drop all values with zeros inside#data_skeleton = data_skeleton.replace(0, pd.np.nan).dropna(axis=0, how='any', subset=['microglia']).fillna(0).astype(str)
data_skeleton =data_skeleton[data_skeleton['microglia'] !=0]


data_fraclac = data_fraclac.reset_index(drop=True)
data_skeleton = data_skeleton.reset_index(drop=True)
#data_skeleton.to_csv (results_path + "\export_dataframe_skeleton.csv", index = 'microglia', header=True) 
#data_fraclac.to_csv (results_path + "\export_dataframe_fraclac.csv", index = 'microglia', header=True) 

#merge skeleton and fraclac data
data = pd.merge(data_fraclac, data_skeleton, left_on = 'microglia', right_on = 'microglia', how = 'inner')
data_summary = data.copy()

#load the round names to the data_summary
df_roundnames = load_data_from_directory(root_path, 'fraclac', [1], ['animal']).astype(str)
roundnames = df_roundnames['animal'].tolist()
#data_summary.loc[:,"dates"] = roundnames

#data_summary.to_csv (results_path + "\export_data_summary.csv", index = 'microglia', header=True) 
#data=data_fraclac.set_index('microglia').join(data_skeleton.set_index('microglia'), how='inner')
#data.to_csv (results_path + "\summary_data.csv", header=True) 

data['microglia'] = data['microglia'].astype(str)

#Decode your data
if len(replacements) != 0:
    data['microglia'] = data['microglia'].apply(Decoder)
    data_summary['microglia'] = data_summary['microglia'].apply(Decoder)
    #data_summary.to_csv (results_path + "\export_data_summary_decoded.csv", index = 'microglia', header=True) 


data.set_index('microglia', inplace=True, drop=True)

data = data.fillna(0)

cells = list(data.index.values)

#print("Columns data: ", data.columns.tolist())

#########################
#
# Perform PCA on the data
#
#########################
# First center and scale the data
scaled_data = preprocessing.scale(data)

pca = PCA() # create a PCA object
pca.fit(scaled_data) # do the math
pca_data = pca.transform(scaled_data) # get PCA coordinates for scaled_data


#########################
#
# How much Variance is explained by each PC?
#
#########################
fig = plt.figure(figsize = (10,8))
plt.plot((pca.explained_variance_ratio_.cumsum())[:9], marker = 'o', linestyle = '--')
plt.title("Explained Variance by Compontents")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.show

fig.savefig(os.path.join(results_path, "Explained_Variance.png"))   

#########################
#
# Plotting PCA
#
#########################

per_var = np.round(pca.explained_variance_ratio_* 100, decimals=3)

labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
pca_df = pd.DataFrame(pca_data, index=cells, columns=labels)


# Calculate the number of additional conditions beyond the first four
additional_conditions = max(0, len(conditions) - len(custom_colors))

# Get additional colors dynamically using tab10 colormap
additional_colors = [plt.cm.tab10(i) for i in range(additional_conditions)]

# Combine custom colors and additional colors
colors = custom_colors + additional_colors

# Plotting
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')
plt.title('3 Component PCA', fontsize=20)
ax.set_xlabel('PC1 - {0}%'.format(per_var[0]), fontsize = 15)
ax.set_ylabel('PC2 - {0}%'.format(per_var[1]), fontsize = 15)
ax.set_zlabel('PC3 - {0}%'.format(per_var[2]), fontsize = 15)

scatter_handles = []  # List to store scatter plot handles for legend

for condition, color in zip(conditions, colors):
    scatter = None
    for i in range(len(pca_df)):
        if condition in pca_df.index[i]:
            scatter = ax.scatter(pca_df.PC1[i], pca_df.PC2[i], pca_df.PC3[i], c=color, s=50)
    if scatter:
        scatter_handles.append(scatter)


# Customize legend with scatter handles and conditions
ax.legend(scatter_handles, conditions, bbox_to_anchor=(0.75, 1.02), loc=2)

plt.show()

fig.savefig(os.path.join(results_path, "PCA_Analysis.png"))  

#########################
#
# Determine which data had the biggest influence on PC1
#
#########################
 
## get the name of the top 10 measurements (genes) that contribute
## most to pc1.
## first, get the loading scores
loading_scores = pd.Series(pca.components_[0], index=morph_data)
## now sort the loading scores based on their magnitude
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
 
# get the names of the top 10 genes
top_10_morphData = sorted_loading_scores[0:10].index.values

 
## print the gene names and their scores (and +/- sign)
loading_scores[top_10_morphData].to_csv (results_path + "\PC_influence.csv", header=True) 

explained_Variance = pca.explained_variance_ratio_
pcs = []
j=1
for j in range (len(explained_Variance)+1):
    pcs.append("PC" + str(j))
pd.DataFrame(explained_Variance).to_csv(results_path+"\explained_variance.csv", header = None, index = None)

#########################
#
# Cluster Analysis
#
#########################

#how many clusters do you have? --> ellbow method
wcss = []
for k in range(1,21):
    kmeans_pca = KMeans(n_clusters = k, init = 'k-means++', random_state=42)
    kmeans_pca.fit(pca_data)
    wcss.append(kmeans_pca.inertia_)
fig = plt.figure(figsize = (10,8))
plt.plot(wcss[:21], marker = 'o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('PCA with K-means Clustering')
plt.show()
fig.savefig(os.path.join(results_path, "K-Means.png"))    


#create k clusters
kmeans_pca = KMeans(n_clusters=3, init= 'k-means++', random_state=42)
kmeans = kmeans_pca.fit(pca_df)
centroids = kmeans.cluster_centers_


# Calculate the number of additional conditions beyond the first four
additional_conditions = max(0, len(conditions) - len(custom_colors))

# Get additional colors dynamically using tab10 colormap
additional_colors = [plt.cm.tab10(i) for i in range(additional_conditions)]

# Combine custom colors and additional colors
colors = custom_colors + additional_colors

# Plotting
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')
plt.title('PCA with K-Means Clustering', fontsize=20)
ax.set_xlabel('PC1 - {0}%'.format(per_var[0]), fontsize = 15)
ax.set_ylabel('PC2 - {0}%'.format(per_var[1]), fontsize = 15)
ax.set_zlabel('PC3 - {0}%'.format(per_var[2]), fontsize = 15)

scatter_handles = []  # List to store scatter plot handles for legend

for condition, color in zip(conditions, colors):
    scatter = None
    for i in range(len(pca_df)):
        if condition in pca_df.index[i]:
            scatter = ax.scatter(pca_df.PC1[i], pca_df.PC2[i], pca_df.PC3[i], c=color, s=50)
    if scatter:
        scatter_handles.append(scatter)


# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='k', marker='x', label='Centroids')

# Customize legend with scatter handles and conditions
ax.legend(scatter_handles, conditions, bbox_to_anchor=(0.75, 1.02), loc=2)

plt.show()

fig.savefig(os.path.join(results_path, "PCA_Analysis_KMeans.png"))  

#########################
#
# Save labels
#
#########################
label_data = kmeans.labels_

pca_df.loc[:,"cluster"] = label_data
pca_df.to_csv(results_path + '\cluster_pca.csv')

#create a list with the conditions of the files
condition_data_list = []
for i in range(len(pca_df)):
    #print(pca_df.index[i])
    for condition in conditions:  # Iterate over your list of conditions
        if condition in pca_df.index[i]:  
            condition_data_list.append(condition)
        
#add condition column
data_summary.loc[:,'condition'] = condition_data_list
#add the cluster column to data

data_summary.loc[:,"cluster"] = label_data

#determine which cluster is which morphological type (ramified, intermediate, amoeboid)
from morphType import ClusterDefiner
#print("data summary columns: ", data_summary.columns)
data_summary, label_name = ClusterDefiner(data_summary)

#data_summary.to_csv(results_path + "\summary_test.csv")
#sort the data via the condition column (primary) and the cluster column (secondary)
data_sort = data_summary.sort_values(by=['condition', 'cluster'], ascending=[True, False])

#print the final csv
data_sort.to_csv (results_path + "\summary_data.csv", header=True) 


#########################
#
# Cluster Analysis color clusters, not conditions
#
#########################

# Use the first three components of PCA data for clustering
pca_data_for_clustering = pca_data

# Specify the number of clusters you want to find
num_clusters = 3

# Fit k-means to the data
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(pca_data_for_clustering)

# Get cluster labels for each sample
cluster_labels = kmeans.labels_

# Plot the clusters
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('PC1 - {0}%'.format(per_var[0]), fontsize = 15)
ax.set_ylabel('PC2 - {0}%'.format(per_var[1]), fontsize = 15)
ax.set_zlabel('PC3 - {0}%'.format(per_var[2]), fontsize = 15)



# Define specific colors for each cluster
#cluster_colors = [purple, pink, grey]  # Add more colors as needed
red = (139/255, 144/255, 206/255)
blue = (0, 0, 128/255)
grey = (0, 0, 4/255)

cluster_colors = [red, blue, grey]

legend_patches = []

for cluster, color in zip(set(cluster_labels), cluster_colors):
    cluster_mask = cluster_labels == cluster
    cluster_name = label_name[cluster]
    ax.scatter(
        pca_data_for_clustering[cluster_mask, 0],
        pca_data_for_clustering[cluster_mask, 1],
        pca_data_for_clustering[cluster_mask, 2],
        label=cluster_name,
        c=color,
    )
    legend_patches.append(mpatches.Patch(color=color, label=cluster_name))
# Plot the legend
ax.legend(handles=legend_patches, loc='upper right')


# Convert RGB values to the format used in ax.patch.set_facecolor
"""
red = 218 / 255
green = 218 / 255
blue = 218 / 255
"""

red = 1
green = 1
blue = 1

# Set the background color inside the circle itself.
ax.patch.set_facecolor((red, green, blue, 1))

# Set the background color of the entire figure.
fig.patch.set_facecolor((red, green, blue, 1))

plt.scatter(centroids[:,0] , centroids[:,1] , s = 100, color = 'k', marker = 'x')

fig.savefig(os.path.join(results_path, "3_PCA_Analysis_KMeans.png"), facecolor=fig.get_facecolor(), dpi=600, transparent=True)  


plt.show()


#########################
#
# Print labels on intensity data
#
#########################

cluster_data = data_summary.copy()
cluster_data.reset_index(inplace=True)


#convert the condition column to purely string
cluster_data['condition'] = cluster_data['condition'].astype(str)

#cluster_data.to_csv(results_path + "\summary_data_with_dates.csv", header=True) 

def modify_string_with_dot(s):
    name = s.split('_')
    new_name = '_'.join(name[0:])
    name_without_dot = new_name.split('.')
    new_name_without_dot = ''.join(name_without_dot[0:])
    new_name_without_dot.rstrip()
    return new_name_without_dot

    
#load intensity datas 
intensity_data = load_data_from_directory(root_path, "Analysis_Intensity", [0,1,2,3,4,5,6], ['Filename','#','microglia_area','CD68_IntDen', 'CD68_Mean','IBA1_IntDen', 'IBA1_Mean'])
#decode your data if they are coded
if len(replacements) != 0:
    intensity_data['Filename'] = intensity_data['Filename'].apply(Decoder)

#data_sort.to_csv (results_path + "\summary_data_prior_intensity.csv", header=True) 

#change filename in order to alling intensity and cluster_data
intensity_data['Filename'] = intensity_data['Filename'].apply(modify_filename_intensity)
#print("INTENSITÄTSDATEN LEN: ", len(intensity_data))

intensity_data['Filename'] = intensity_data['Filename'] + '-' + intensity_data['#']
#print("INTENSITÄTSDATEN LEN AFTER: ", len(intensity_data))

#intensity_data = intensity_data.drop_duplicates(subset=['Filename'])
#cluster_data = cluster_data.drop_duplicates(subset=['microglia'])

#merge intensity_data and cluster_data
data = pd.merge(intensity_data, cluster_data, left_on = 'Filename', right_on = 'microglia', how = 'outer')

#sort data first by condition, then by cluster, then dates
data_sort = data.sort_values(by=['condition', 'cluster'], ascending=[True, False])
data_sort.to_csv (results_path + "\summary_data_with_intensity.csv", header=True) 


#########################
#
# Cluster counter & Data Summer
#
#########################

# counts the amount and calculates percentages of ramified, intermediate and amoeboid microglia

def percentage_calculator(dat_0, dat_1, dat_2):
    percentage_ramified_list = []
    percentage_intermediate_list = []
    percentage_amoeboid_list = []
    
    for i in range(len(dat_0)):
        #print(dat_0)
        try:
            percentage_ramified_list.append((dat_0[i]/(dat_1[i]+dat_2[i]+dat_0[i])) * 100)
        except:
            percentage_ramified_list.append(0)
            
        try:
            percentage_intermediate_list.append((dat_1[i]/(dat_1[i]+dat_2[i]+dat_0[i])) * 100)
        except:
            percentage_intermediate_list.append(0)
            
        try:
            percentage_amoeboid_list.append((dat_2[i]/(dat_1[i]+dat_2[i]+dat_0[i])) * 100)
        except:
            percentage_amoeboid_list.append(0)
        
        
    return percentage_ramified_list, percentage_intermediate_list, percentage_amoeboid_list


#get the names of each image (header from all single cell data names)
img_names_list = []
#analyze characteristics per image
microglia_names = cluster_data['microglia'].values.tolist()
for nam in microglia_names:
    nam = nam.split("czi")
    img_name = ''.join(nam[0])
    img_names_list.append(img_name)
    
#get the img names out of the list via set method
unique_img_names = set(img_names_list)
# Convert the set back to a list
img_names = list(unique_img_names)

#define cluster conditions
cluster_names = ["ramified", "intermediate", "amoeboid", "sum"]
cluster_ramified = (cluster_data["cluster"] == "ramified")
cluster_intermediate = (cluster_data["cluster"] == "intermediate")
cluster_amoeboid = (cluster_data["cluster"] == "amoeboid")


# Initialize dictionaries to store percentages
percentage_data = {}

# Initialize dictionaries to store condition data
condition_data_mean = {condition: {"ramified": {}, "intermediate": {}, "amoeboid": {}} for condition in conditions}



characteristics = data.columns.values[3:42] 


# Loop through conditions dynamically

    
for condition in conditions:
    
        
    condition_data = {}
    
    condition_data_mean_ramified = {}
    condition_data_mean_intermediate = {}
    condition_data_mean_amoeboid = {}
    condition_data_mean_sum = {}
    
    
    cond_ramified = []
    cond_intermediate = []
    cond_amoeboid = []
    cond_sum = []
    
    diction_data = {cluster: {} for cluster in cluster_names}
    for img in img_names:

        name_cond = data["microglia"].str.contains(str(img))
        if condition in img:
            if "filename" not in condition_data_mean_ramified:
                condition_data_mean_ramified["filename"] = []
                condition_data_mean_ramified["filename"].append(img)
            else:
                condition_data_mean_ramified["filename"].append(img)
                
            if "filename" not in condition_data_mean_intermediate:
                condition_data_mean_intermediate["filename"] = []
                condition_data_mean_intermediate["filename"].append(img)
            else:
                condition_data_mean_intermediate["filename"].append(img)
                
            if "filename" not in condition_data_mean_amoeboid:
                condition_data_mean_amoeboid["filename"] = []
                condition_data_mean_amoeboid["filename"].append(img)
            else:
                condition_data_mean_amoeboid["filename"].append(img)
                
            if "filename" not in condition_data_mean_sum:
                condition_data_mean_sum["filename"] = []
                condition_data_mean_sum["filename"].append(img)
            else:
                condition_data_mean_sum["filename"].append(img)
    
            

            #get the amout of true values for your condition to count the cluster percentages
            cond_ramified.append(len(data[(data["condition"] == condition) & (cluster_ramified) & (name_cond)]))
            cond_intermediate.append(len(data[(data["condition"] == condition) & (cluster_intermediate) & (name_cond)]))
            cond_amoeboid.append(len(data[(data["condition"] == condition) & (cluster_amoeboid) & (name_cond)]))
            
            #define the conditions for the reorganizhation and mean calculation for the characteristics
            df_cond_ramified= data[(data["condition"] == condition) & (cluster_ramified) & (name_cond)]
            df_cond_intermediate= data[(data["condition"] == condition) & (cluster_intermediate) & (name_cond)]
            df_cond_amoeboid= data[(data["condition"] == condition) & (cluster_amoeboid) & (name_cond)]
            
            df_cond_sum= data[(data["condition"] == condition) & (name_cond)]
            
            for char in characteristics: 
                if char == 'condition' or char == 'cluster' or char == 'dates':
                    continue
                if char == 'microglia':
                    if char not in condition_data_mean_ramified:
                        condition_data_mean_ramified[char]= []
                        condition_data_mean_ramified[char].append(img)
                    else:
                        condition_data_mean_ramified[char].append(img)
                        
                    if char not in condition_data_mean_intermediate:
                        condition_data_mean_intermediate[char]= []
                        condition_data_mean_intermediate[char].append(img)
                    else:
                        condition_data_mean_intermediate[char].append(img)
                        
                    if char not in condition_data_mean_amoeboid:
                        condition_data_mean_amoeboid[char]=[]
                        condition_data_mean_amoeboid[char].append(img)
                    else:
                        condition_data_mean_amoeboid[char].append(img)
                        
                    if char not in condition_data_mean_sum:
                        condition_data_mean_sum[char]=[]
                        condition_data_mean_sum[char].append(img)
                    else:
                        condition_data_mean_sum[char].append(img)
                  
                else:
                    # calculate mean 
                    if char not in condition_data_mean_ramified:
                        condition_data_mean_ramified[char]= []
                        condition_data_mean_ramified[char].append(np.nan if df_cond_ramified[char].empty else df_cond_ramified[char].astype(float).mean())

                    else:
                        condition_data_mean_ramified[char].append(np.nan if df_cond_ramified[char].empty else df_cond_ramified[char].astype(float).mean())
                        
                    if char not in condition_data_mean_intermediate:
                        condition_data_mean_intermediate[char]= []
                        condition_data_mean_intermediate[char].append(np.nan if df_cond_intermediate[char].empty else df_cond_intermediate[char].astype(float).mean())
                    else:
                        condition_data_mean_intermediate[char].append(np.nan if df_cond_intermediate[char].empty else df_cond_intermediate[char].astype(float).mean())
                        
                    if char not in condition_data_mean_amoeboid:
                        condition_data_mean_amoeboid[char]=[]
                        condition_data_mean_amoeboid[char].append(np.nan if df_cond_amoeboid[char].empty else df_cond_amoeboid[char].astype(float).mean())
                    else:
                        condition_data_mean_amoeboid[char].append(np.nan if df_cond_amoeboid[char].empty else df_cond_amoeboid[char].astype(float).mean())
                    
                    if char not in condition_data_mean_sum:
                        condition_data_mean_sum[char]=[]
                        condition_data_mean_sum[char].append(np.nan if df_cond_sum[char].empty else df_cond_sum[char].astype(float).mean())
                    else:
                        condition_data_mean_sum[char].append(np.nan if df_cond_sum[char].empty else df_cond_sum[char].astype(float).mean())
            
                        
    # Store mean data for each cluster in the main dictionary
    condition_data_mean[condition]["ramified"] = condition_data_mean_ramified
    print("len cond data ramified: ", len(condition_data_mean_ramified))
    condition_data_mean[condition]["intermediate"] = condition_data_mean_intermediate
    condition_data_mean[condition]["amoeboid"] = condition_data_mean_amoeboid
    condition_data_mean[condition]["sum"] = condition_data_mean_sum
                
    # Create a DataFrame from the dictionary
    df_ramified = pd.DataFrame(condition_data_mean_ramified)
    df_intermediate = pd.DataFrame(condition_data_mean_intermediate)
    df_amoeboid = pd.DataFrame(condition_data_mean_amoeboid)
    df_sum = pd.DataFrame(condition_data_mean_sum)

    
    # Write the sorted DataFrames to CSV files with headers
    df_ramified.to_csv(results_path + rf'\ramified_{condition}.csv', index=False, header=True)
    df_intermediate.to_csv(results_path + rf'\intermediate_{condition}.csv', index=False, header=True)
    df_amoeboid.to_csv(results_path + rf'\amoeboid_{condition}.csv', index=False, header=True)
    df_sum.to_csv(results_path + rf'\sum_{condition}.csv', index=False, header=True)

    
    
    #calculate cluster percentages
    percentages = percentage_calculator(cond_ramified, cond_intermediate, cond_amoeboid)
    
    # Store percentages in dictionary
    percentage_data[f"{condition}_ramified_percentage"] = percentages[0]
    percentage_data[f"{condition}_intermediate_percentage"] = percentages[1]
    percentage_data[f"{condition}_amoeboid_percentage"] = percentages[2]
    

# Find the maximum length of all values (lists) in the dictionary
max_length = max(len(lst) for lst in percentage_data.values())
#print("max length: ", max_length)

# Pad each list in the dictionary with NaN to ensure the same length
for key in percentage_data:
    current_length = len(percentage_data[key])
    if current_length < max_length:
        percentage_data[key] += [pd.NA] * (max_length - current_length)
        
# Create a DataFrame from the dictionary
df_percentage = pd.DataFrame(percentage_data)

# Write the DataFrame to a CSV file with headers
df_percentage.to_csv(results_path + '\Analysis_cluster_count_percentage.csv', index=False, header=True)

#########################
#
# Normalize your data to condition 1
#
#########################


#initialize norm dictionaries
norm_ramified = {}
norm_intermediate = {}
norm_amoeboid = {}
norm_sum = {}

# get the mean values of all charateritsics and intenisties of condition 1
for dat in dates_list :
    for char in characteristics:
        if char == 'microglia' or char == 'cluster' or char == 'condition' or char == 'dates':
            continue
        else:
            # set the date variable as an filter criteria
            # Convert the list to a Series and then apply .str.contains() method
            # Filter out NaN values and calculate mean
            #print(f"Before filtering, {dat}: {len(condition_data_mean[condition_1]['ramified'][char])} values")
            ramified_values = [x for i, x in enumerate(condition_data_mean[condition_1]["ramified"][char]) if pd.Series(condition_data_mean[condition_1]["ramified"]['filename']).str.contains(str(dat))[i] and not np.isnan(x)]
            intermediate_values = [x for i, x in enumerate(condition_data_mean[condition_1]["intermediate"][char]) if pd.Series(condition_data_mean[condition_1]["intermediate"]['filename']).str.contains(str(dat))[i] and not np.isnan(x)]
            amoeboid_values = [x for i, x in enumerate(condition_data_mean[condition_1]["amoeboid"][char]) if pd.Series(condition_data_mean[condition_1]["amoeboid"]['filename']).str.contains(str(dat))[i] and not np.isnan(x)]
            sum_values = [x for i, x in enumerate(condition_data_mean[condition_1]["sum"][char]) if pd.Series(condition_data_mean[condition_1]["sum"]['filename']).str.contains(str(dat))[i] and not np.isnan(x)]
            #print(f"After filtering, {dat}: {len(ramified_values)} values")
            
            #initilaize dictionary
            if char not in norm_ramified:
                norm_ramified[char] = {}
            if char not in norm_intermediate:
                norm_intermediate[char] = {}
            if char not in norm_amoeboid:
                norm_amoeboid[char] = {}
            
            if char not in norm_sum:
                norm_sum[char] = {}
            
            norm_ramified[char][dat] = []
            norm_intermediate[char][dat] = []
            norm_amoeboid[char][dat] = []
            norm_sum[char][dat] = []
            # Calculate mean for non-NaN values
            norm_ramified[char][dat].append(np.mean(ramified_values) if ramified_values else np.nan)
            norm_intermediate[char][dat].append(np.mean(intermediate_values) if intermediate_values else np.nan)
            norm_amoeboid[char][dat].append(np.mean(amoeboid_values) if amoeboid_values else np.nan)
            norm_sum[char][dat].append(np.mean(sum_values) if sum_values else np.nan)
        



for con in conditions:
    # initialize temporary dictionaries
    condition_norm_ramified = {}
    condition_norm_intermediate = {}
    condition_norm_amoeboid = {}
    condition_norm_sum = {}
    for dat in dates_list:
        for char in characteristics:
            if char == 'cluster' or char == 'condition' or char == 'dates':
                continue
            
            if char == 'microglia':
                continue
    
            else:
                # normalize to condition_1
                if char not in condition_norm_ramified:
                    condition_norm_ramified[char] = []
                for i, x in enumerate(condition_data_mean[con]["ramified"][char]):
                    if pd.Series(condition_data_mean[con]["ramified"]['filename']).str.contains(str(dat))[i] and not np.isnan(x):
                        condition_norm_ramified[char].append(float(x / norm_amoeboid[char][dat]))
                
                if char not in condition_norm_intermediate:
                    condition_norm_intermediate[char] = []
                for i, x in enumerate(condition_data_mean[con]["intermediate"][char]):
                    if pd.Series(condition_data_mean[con]["intermediate"]['filename']).str.contains(str(dat))[i] and not np.isnan(x):
                        condition_norm_intermediate[char].append(float(x / norm_amoeboid[char][dat]))
                        
                if char not in condition_norm_amoeboid:
                    condition_norm_amoeboid[char] = []
                for i, x in enumerate(condition_data_mean[con]["amoeboid"][char]):
                    if pd.Series(condition_data_mean[con]["amoeboid"]['filename']).str.contains(str(dat))[i] and not np.isnan(x):
                        condition_norm_amoeboid[char].append(float(x / norm_amoeboid[char][dat]))
                        
                if char not in condition_norm_sum:
                    condition_norm_sum[char] = []
                for i, x in enumerate(condition_data_mean[con]["sum"][char]):
                    if pd.Series(condition_data_mean[con]["sum"]['filename']).str.contains(str(dat))[i] and not np.isnan(x):
                        condition_norm_sum[char].append(float(x / norm_sum[char][dat]))

        
    
    for key, values in condition_norm_ramified.items():
        print(f"Key '{key}' hat {len(values)} Werte.")
    
    # Create a DataFrame from the dictionary
    df_norm_ramified = pd.DataFrame(condition_norm_ramified)
    df_norm_intermediate = pd.DataFrame(condition_norm_intermediate)
    df_norm_amoeboid = pd.DataFrame(condition_norm_amoeboid)
    df_norm_sum = pd.DataFrame(condition_norm_sum)

    # Write the DataFrame to a CSV file with headers
    df_norm_ramified.to_csv(results_path + rf'\norm_ramified_{con}.csv', index=False, header=True)
    df_norm_intermediate.to_csv(results_path + rf'\norm_intermediate_{con}.csv', index=False, header=True)
    df_norm_amoeboid.to_csv(results_path + rf'\norm_amoeboid_{con}.csv', index=False, header=True)
    df_norm_sum.to_csv(results_path + rf'\norm_sum_{con}.csv', index=False, header=True)


        
    
    
    