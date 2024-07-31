# Data Analysis Scripts

Updated: March 2024

This repository contains the scripts for processing and analyzing the data collected as part of the study descibed in "Mind the data gap: Using a multi-measurement synthesis for identifying the challenges and opportunities in studying plant drought response and recovery".

This code is run in Python 3.10, and the necessary accompanying packages and versions are listed in the requirements.txt file.

Stage 1 data are available at: 10.5281/zenodo.10685214 

Author Info: Jean Wilkening (jvwilkening@berkeley.edu)

### Code

The data is processed in four stages which are represented by individual functions. In the first stage (raw_data_to_stage_1), the raw data files (in Raw_Data directory) are imported and formatted. In the second stage (stage_1_to_stage_2), the datasets are subset to the experimental period and corrections are made based on notes from during the experiment (e.g. mislabelled data or noted quality issues with measurements). In the third stage (stage_2_to_stage_3), derived values from the data are calculted. In the final stage (stage_3_to_stage_4), additional data cleaning and corrections are made. All of these processing steps can be run consecutively using the process_all_data_from_raw script. All of the processing functions for individual stages and data types are within the utility_functions file. The scripts also all save every stage of data into indivdual directories, both as csv files and pkl files. Plots of the final processed data as presented in the manuscript can be created using the plot_results script.

The mutual information analyses are completed using the info_analysis script, which uses the Stage 4 data files.
