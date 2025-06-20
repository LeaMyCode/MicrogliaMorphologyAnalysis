# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 09:45:20 2023

@author: leaga
"""

import pandas as pd

def ClusterDefiner (df):
    origin_df = df.copy()
    df = df.dropna()

    df["roughness"] = df["roughness"].astype(float)
    roughness_0 = df.loc[df["cluster"] == 0, "roughness"].mean()
    roughness_1 = df.loc[df["cluster"] == 1, "roughness"].mean()
    roughness_2 = df.loc[df["cluster"] == 2, "roughness"].mean()
    
    df["density"] = df["density"].astype(float)
    density_0 = df.loc[df["cluster"] == 0, "density"].mean()
    density_1 = df.loc[df["cluster"] == 1, "density"].mean()
    density_2 = df.loc[df["cluster"] == 2, "density"].mean()
    
    df["cell_circularity"] = df["cell_circularity"].astype(float)
    cell_circularity_0 = df.loc[df["cluster"] == 0, "cell_circularity"].mean()
    cell_circularity_1 = df.loc[df["cluster"] == 1, "cell_circularity"].mean()
    cell_circularity_2 = df.loc[df["cluster"] == 2, "cell_circularity"].mean()
    
    df['# Branches'] = df['# Branches'].astype(float)
    branches_0 = df.loc[df["cluster"] == 0, '# Branches'].mean()
    branches_1 = df.loc[df["cluster"] == 1, '# Branches'].mean()
    branches_2 = df.loc[df["cluster"] == 2, '# Branches'].mean()
    
    
    
    diction = {"cluster": [0,1,2], "roughness": [roughness_0, roughness_1, roughness_2], "density": [density_0, density_1, density_2], 
               "cell_circularity": [cell_circularity_0, cell_circularity_1, cell_circularity_2], 
               '# Branches': [branches_0, branches_1, branches_2]}
    
    df_typ = pd.DataFrame(diction)
    df_typ_copie = df_typ.copy() # use the copy to later find the last morphology
    df_typ = df_typ.set_index("cluster")
    
    #Ramified microglia find
    max_roughness_cluster = df_typ['roughness'].idxmax()
    min_circularity_cluster = df_typ['cell_circularity'].idxmin()
    max_branch_cluster = df_typ['# Branches'].idxmax()
    min_density_cluster = df_typ['density'].idxmin()

    
    #amöboid microglia find
    max_circularity_cluster = df_typ['cell_circularity'].idxmax()
    max_density_cluster = df_typ['density'].idxmax()
    min_branch_cluster = df_typ['# Branches'].idxmin()
    min_roughness_cluster = df_typ['roughness'].idxmin()


    print("max density : ", max_density_cluster)
    print("min density : ", min_density_cluster)
    print("max roughness: ", max_roughness_cluster)
    print("min roughness: ", min_roughness_cluster)
    print("max circularity: ", max_circularity_cluster)
    print("min circularity: ", min_circularity_cluster)
    print("max branches: ", max_branch_cluster)
    print("min branches: ", min_branch_cluster)
 
    
    # write the true morphological name in the dataframe for ramified and amoeboid
    if min_density_cluster:
        origin_df.loc[origin_df['cluster'] == min_density_cluster, 'cluster'] = 'ramified'
    else:
        print("somethings wrong with your cluster data")
    if max_circularity_cluster == max_density_cluster:
        origin_df.loc[origin_df['cluster'] == max_circularity_cluster, 'cluster'] = 'amoeboid'
    else:
        print("somethings wrong with your cluster data")
    
    #create a label_list for later plotting
    label_list = []
    
    # the cluster thats neither amoeboid nor ramified is intermediate
    for index, row in df_typ_copie.iterrows():
        if row["cluster"] == min_density_cluster:
            label_list.append("ramified")
            continue
        elif row["cluster"] == max_circularity_cluster:
            label_list.append("amoeboid")
            continue
        else:
            origin_df.loc[origin_df["cluster"] == row["cluster"], "cluster"] = "intermediate"
            label_list.append("intermediate")
    print(label_list)
    
    return origin_df, label_list
        

    

    
    
    
    
    
    
    

    
