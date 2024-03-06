from raw_data_to_stage_1 import raw_to_S1
from stage_1_to_stage_2 import S1_to_S2
from stage_2_to_stage_3 import S2_to_S3
from stage_3_to_stage_4 import S3_to_S4

'''
This script runs all the data processing steps to go from raw data to fully processed data for all data types. 
Processing steps can also be run as individual scripts. Intermediate data stages are saved as both csv and pkl files.
'''
#runs all processing scripts starting from raw data
raw_to_S1()
S1_to_S2()
S2_to_S3()
S3_to_S4()