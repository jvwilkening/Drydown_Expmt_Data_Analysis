import pandas as pd
from utility_functions import plot_results

#stand alone script for plotting all the results data. Makes a lot more plots than what is done between data stages.

input_folder = 'Stage_4_Data'

soil_moisture_df = pd.read_pickle('%s/soil_moisture.pkl' % input_folder)
sap_flow_df = pd.read_pickle('%s/sap_flow.pkl' % input_folder)
gasex_df = pd.read_pickle('%s/gasex.pkl' % input_folder)
isotopes_df = pd.read_pickle('%s/isotopes.pkl' % input_folder)
lwp_df = pd.read_pickle('%s/lwp.pkl' % input_folder)
psych_df = pd.read_pickle('%s/psych.pkl' % input_folder)
met_df = pd.read_pickle('%s/met.pkl' % input_folder)
leaf_area_df = pd.read_pickle('%s/leaf_area.pkl' % input_folder)
root_df = pd.read_pickle('%s/root_dist.pkl' % input_folder)
gasex_curves_df = pd.read_pickle('%s/gasex_curves.pkl' % input_folder)

#enter inputs for which data you want to plot
#plot_results(sap_flow=sap_flow_df, psych=psych_df, soil_moisture=soil_moisture_df, lwp=lwp_df, gasex=gasex_df, met=met_df, isotopes=isotopes_df, leaf_area=leaf_area_df, root=root_df, stage=4)

plot_results(sap_flow=sap_flow_df, stage=4)