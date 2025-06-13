input_path = getDirectory("Choose your data to be analyzed");
output_path = getDirectory("Select destination directory for your intensity data");
output_path_binary = getDirectory("Selectt destination directory for your binary data (all microglia included)");
dir3 = getDirectory("Selecttt Destination Directory for single binary data");
fileList = getFileList(input_path) 

for (f = 0; f<fileList.length; f++){
		
		open(input_path + fileList[f]); 
		run("Z Project...", "projection=[Max Intensity]");
		Stack.getDimensions(width, height, channels, slices, frames); 
		title = getTitle();
		run("Set Measurements...", "area mean integrated redirect=None decimal=3");
		if (channels == 2) {
			run("Split Channels");
			selectWindow ("C1-" + title);
			rename("DAPI");
			selectWindow ("C2-" + title);
			rename("Microglia");
		        
	        //Microglia analysis
	        //Duplicate microglia
			selectWindow("Microglia");
			run("Duplicate...", "title=IBA1");
			selectWindow("Microglia");
			run("Duplicate...", "title=Microglia_threshold");
			
			//Microglia with Nuclei
			run("Merge Channels...", "c1=Microglia c2=DAPI create");
			
			//Thresholding Microglia
			selectWindow("Microglia_threshold");
			run("Enhance Contrast", "saturated=0.35");
			run("Apply LUT"); 
			run("Despeckle");
			saveAs("Tiff", output_path_binary + title + " -Microglia");
			waitForUser("Please threshold the image. Intermodes or Huang are good options");
			//setAutoThreshold("Li dark");
			setOption("BlackBackground", true);
			run("Convert to Mask");
			run("Despeckle");
			run("Close-");
			waitForUser("Compare the original image with the binary image. Adjust the binary image by using the paintbrush tool to enhance the accuracy, then press ok.");
			saveAs("Tiff", output_path_binary + title + "-Microglia_binary");
			
			//cut out each individual microglia
			run("Analyze Particles...", "size=100-Infinity show=Outlines summarize add");
			n = roiManager('count');
			for (j = 0; j < n; j++) {
	    		selectWindow(title + "-Microglia_binary.tif");
	    		run("Duplicate...", " ");
	    		selectWindow(title + "-Microglia_binary-1.tif");
	    		roiManager('select', j);
				run("Make Inverse");
				setBackgroundColor(0, 0, 0);
				run("Clear", "slice");
				saveAs("Tiff", dir3 + title + "-" +j);
			}
			
			//anayze IBA1
			n = roiManager('count');
			selectWindow("IBA1");
			for (j = 0; j < n; j++) {
				roiManager('select', j);
	            run("Measure");
			}
			//save results
			selectWindow("Results");
			saveAs("Measurements", output_path + title + "-IBA1.csv");
			run("Close");
			
			//Save all microglia rois
			roiManager('select', "*");
			roiManager("Save", output_path_binary + title + "microglia_ROI.zip");
			
			roiManager("reset");
			
		}
		
		if (channels == 3) {
			run("Split Channels");
			selectWindow ("C1-" + title);
			rename("DAPI");
			selectWindow ("C2-" + title);
			rename("Microglia");
			selectWindow ("C3-" + title);
			rename("CD68");
		        
	        //Microglia analysis
	        //Duplicate microglia
			selectWindow("Microglia");
			run("Duplicate...", "title=IBA1");
			selectWindow("Microglia");
			run("Duplicate...", "title=Microglia_threshold");
			
			//Microglia with Nuclei
			run("Merge Channels...", "c1=Microglia c3=DAPI create");
			
			//Thresholding Microglia
			selectWindow("Microglia_threshold");
			run("Enhance Contrast", "saturated=0.35");
			run("Apply LUT"); 
			run("Despeckle");
			saveAs("Tiff", output_path_binary + title + " -Microglia");
			waitForUser("Please threshold the image. Intermodes or Huang are good options");
			//setAutoThreshold("Li dark");
			setOption("BlackBackground", true);
			run("Convert to Mask");
			run("Despeckle");
			run("Close-");
			waitForUser("Compare the original image with the binary image. Adjust the binary image by using the paintbrush tool to enhance the accuracy, then press ok.");
			saveAs("Tiff", output_path_binary + title + "-Microglia_binary");
			
			//cut out each individual microglia
			run("Analyze Particles...", "size=100-Infinity show=Outlines summarize add");
			n = roiManager('count');
			for (j = 0; j < n; j++) {
	    		selectWindow(title + "-Microglia_binary.tif");
	    		run("Duplicate...", " ");
	    		selectWindow(title + "-Microglia_binary-1.tif");
	    		roiManager('select', j);
				run("Make Inverse");
				setBackgroundColor(0, 0, 0);
				run("Clear", "slice");
				saveAs("Tiff", dir3 + title + "-" +j);
			}
			
			//anayze IBA1
			n = roiManager('count');
			selectWindow("IBA1");
			for (j = 0; j < n; j++) {
				roiManager('select', j);
	            run("Measure");
			}
			//save results
			selectWindow("Results");
			saveAs("Measurements", output_path + title + "-IBA1.csv");
			run("Close");
	
			//anayze CD68
			n = roiManager('count');
			selectWindow("CD68");
			for (j = 0; j < n; j++) {
				roiManager('select', j);
	            run("Measure");
			}
			//save results
			selectWindow("Results");
			saveAs("Measurements", output_path + title + "-CD68.csv");
			run("Close");
			
			//Save all microglia rois
			roiManager('select', "*");
			roiManager("Save", output_path_binary + title + "microglia_ROI.zip");
			
			roiManager("reset");
			
		}
			
		
		
		if (channels == 4) {
			run("Split Channels");
			selectWindow ("C1-" + title);
			rename("DAPI");
			selectWindow ("C2-" + title);
			run("Close");
			selectWindow ("C3-" + title);
			rename("Microglia");
			selectWindow ("C4-" + title);
			rename("CD68");
		        
	        //Microglia analysis
	        //Duplicate microglia
			selectWindow("Microglia");
			run("Duplicate...", "title=IBA1");
			selectWindow("Microglia");
			run("Duplicate...", "title=Microglia_threshold");
			
			//Microglia with Nuclei
			run("Merge Channels...", "c1=Microglia c3=DAPI create");
			
			//Thresholding Microglia
			selectWindow("Microglia_threshold");
			run("Enhance Contrast", "saturated=0.35");
			run("Apply LUT"); 
			run("Despeckle");
			saveAs("Tiff", output_path_binary + title + " -Microglia");
			waitForUser("Please threshold the image. Intermodes or Huang are good options");
			//setAutoThreshold("Li dark");
			setOption("BlackBackground", true);
			run("Convert to Mask");
			run("Despeckle");
			run("Close-");
			waitForUser("Compare the original image with the binary image. Adjust the binary image by using the paintbrush tool to enhance the accuracy, then press ok.");
			saveAs("Tiff", output_path_binary + title + "-Microglia_binary");
			
			//cut out each individual microglia
			run("Analyze Particles...", "size=100-Infinity show=Outlines summarize add");
			n = roiManager('count');
			for (j = 0; j < n; j++) {
	    		selectWindow(title + "-Microglia_binary.tif");
	    		run("Duplicate...", " ");
	    		selectWindow(title + "-Microglia_binary-1.tif");
	    		roiManager('select', j);
				run("Make Inverse");
				setBackgroundColor(0, 0, 0);
				run("Clear", "slice");
				saveAs("Tiff", dir3 + title + "-" +j);
			}
			
			//anayze IBA1
			n = roiManager('count');
			selectWindow("IBA1");
			for (j = 0; j < n; j++) {
				roiManager('select', j);
	            run("Measure");
			}
			//save results
			selectWindow("Results");
			saveAs("Measurements", output_path + title + "-IBA1.csv");
			run("Close");
	
			//anayze CD68
			n = roiManager('count');
			selectWindow("CD68");
			for (j = 0; j < n; j++) {
				roiManager('select', j);
	            run("Measure");
			}
			//save results
			selectWindow("Results");
			saveAs("Measurements", output_path + title + "-CD68.csv");
			run("Close");
			
			//Save all microglia rois
			roiManager('select', "*");
			roiManager("Save", output_path_binary + title + "microglia_ROI.zip");
			
			roiManager("reset");
			
		}
		
		
//Clean-up to prepare for next image
	roiManager("reset");
	run("Close All");
	run("Clear Results");
	close("*");
	
	if (isOpen("Log")) {
         selectWindow("Log");
         run("Close");
	}
	if (isOpen("Summary")) {
         selectWindow("Summary");
         run("Close");
	}
	
		
}
print("Jeah, finished!");