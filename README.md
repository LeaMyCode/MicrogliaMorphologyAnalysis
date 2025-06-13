# MicrogliaMorphology
A Pipeline to analyze microglia morphology and expressions (via immunohistochemistry stainings).

Changes in microglial morphology can be analyzed as previously described (Fernández-Arjona et al., 2017). The branching data are obtained using the MorphData macro (Campos et al., 2021), which utilizes the AnalyzeSkeleton plugin (Arganda-Carreras et al., 2010) and processes the data for further analysis. The different morphological parameters (cell perimeter, convex hull perimeter, cell circularity, convex hull circularity, cell area, convex hull area, fractal dimension, lacunarity, density, roughness, convex hull span ratio, diameter of the bounding circle, maximum span across the convex hull, ratio maximum/minimum convex hull radii, mean radius) are analyzed using the FracLac plugin (Karperien, 2007-2012). Finally, a three-component principal component analysis (PCA) can be performed by first scaling the data using the preprocessing scale function of the Sklearn library and then using the functions of the same library to fit the scaled data to obtain the coordinate data for the PCA graphs and further analysis (Sklearn version 1.4.0). The optimal number of clusters was selected using the elbow method, followed by K-Means clustering (Sklearn version 1.0.2) to categorize microglia cells according to the degree of dissimilarity of morphological parameters (fractal dimension, lacunarity, density, span ratio major_minor, area of convex hull, convex hull perimeter, convex hull circularity, diameter of bounding circle, average radius, maximum span of convex hull, max_min radii, area of cell, perimeter of cell, roughness, circularity of cell, number of branches, average length of branches, maximum length of branches). The matplotlib library was used to display the obtained data (matplotlib version 3.8.2). In addition, the NumPy (version 1.26.3) and pandas (version 2.2.0) libraries were used for data processing. 

For a _how to guide_, please see the manual. Examples are added above. 

Please download the morphdata macro and the code for the first postprocessing step [here](https://github.com/anabelacampos/MorphData).

Please download the FracLac plugin [here](https://imagej.net/ij/plugins/fraclac/FLHelp/t4.htm).

## Example of data obtainable with the presented pipeline

### **Microglial morphology changes following stimulation**
![Fig 3_morph corrected](https://github.com/user-attachments/assets/3c9ec50d-7c4e-44d4-b9a3-09b4c38abbda)

### **Morphology-divided expression profiles**

![Fig 4_IBA1_CD68](https://github.com/user-attachments/assets/f71cc464-82c0-4266-b9f9-bd22c995a598)


## Please cite the following if you use the code
### Clustering, morphology type categorization and intensity analysis
>Gabele, L., Bochow, I., Rieke, N., Sieben, C., Michaelsen-Preusse, K., Hosseini, S., et al. (2024). H7N7 viral infection elicits pronounced, sex-specific neuroinflammatory responses in vitro. Front Cell Neurosci 18. doi: 10.3389/fncel.2024.1444876
### morphdata
>Campos A. B., Duarte-Silva S., Ambrósio A. F., Maciel P., Fernandes B. (2021). MorphData: Automating the data extraction process of morphological features of microglial cells in ImageJ. bioRxiv [Preprint]. 10.1101/2021.08.05.455282 
### FracLac
>Karperien A. L. (2007-2012). Fraclac for ImageJ. Albury-Wodonga: Charles Sturt University, 1–36. Available online at: https://imagej.net/ij/plugins/fraclac/FLHelp/Introduction.htm
