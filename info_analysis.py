import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from utility_functions import classify_sampling_day, custom_mi_reg, isotope_mixing

'''
This script takes the processed data and calculates summaries (as needed) and formats data to calculate mutual 
information and make plots
'''

neighbors = 2 #number of neighbors to use for mi_regression, if change also update in custom_mi_reg in utility_functions
random_seed = 13

#folder with final processed data
input_folder = 'Stage_4_Data'

#load all dataframes
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

###Sap flow over sampling days only
#Add sampling day category column and filter only for measurements during sampling days
sap_flow_df['Sampling_Day'] = sap_flow_df.apply(classify_sampling_day, axis=1)
sap_flow_df_short = sap_flow_df[sap_flow_df['Sampling_Day']!='NaN']
#filter to only daylight hours (6:00-20:00)
sap_flow_daytime = sap_flow_df_short.set_index('TIMESTAMP').between_time('06:00', '20:00').reset_index()
#Daytime hour mean sap flow rate
mean_day_sap_flow = sap_flow_daytime.groupby(['Sampling_Day','Tree_Number'], as_index=False)['Total_Sapflow'].mean()
mean_day_sap_flow = mean_day_sap_flow.rename(columns={"Total_Sapflow": "Day_Mean_Sap_Rate"})
#Find 95th percentile of sap flow rate (rather than max which is going to be more affected by oscillations in sensor)
quant95_sap_flow = sap_flow_daytime.groupby(['Sampling_Day','Tree_Number'], as_index=False)['Total_Sapflow'].quantile(.95)
quant95_sap_flow = quant95_sap_flow.rename(columns={"Total_Sapflow": "95th_Perc_Sap_Rate"})
#Find 5th and 95th percentile of stem water content (including overnight measurements)
quant95_swc = sap_flow_df_short.groupby(['Sampling_Day','Tree_Number'], as_index=False)['SWC_Outer'].quantile(.95)
quant95_swc = quant95_swc.rename(columns={"SWC_Outer": "95th_SWC"})
quant05_swc = sap_flow_df_short.groupby(['Sampling_Day','Tree_Number'], as_index=False)['SWC_Outer'].quantile(.05)
quant05_swc = quant05_swc.rename(columns={"SWC_Outer": "5th_SWC"})
sap_flow_summary = pd.merge(mean_day_sap_flow, quant95_sap_flow, on=['Sampling_Day', 'Tree_Number'])
sap_flow_summary = pd.merge(sap_flow_summary, quant95_swc, on=['Sampling_Day', 'Tree_Number'])
sap_flow_summary = pd.merge(sap_flow_summary, quant05_swc, on=['Sampling_Day', 'Tree_Number'])

###leaf water potentials
lwp_df = lwp_df[lwp_df['Sampling_Day']!='NaN']
predawn_lwps = lwp_df[lwp_df['LWP_Type']=='Predawn']
midday_lwps = lwp_df[lwp_df['LWP_Type']=='Midday']
mean_predawn_lwp = predawn_lwps.groupby(['Sampling_Day', 'Tree_Number'], as_index=False)['LWP_(MPa)'].mean()
mean_predawn_lwp = mean_predawn_lwp.rename(columns={"LWP_(MPa)": "Mean_Predawn_LWP"})
mean_midday_lwp = midday_lwps.groupby(['Sampling_Day', 'Tree_Number'], as_index=False)['LWP_(MPa)'].mean()
mean_midday_lwp = mean_midday_lwp.rename(columns={"LWP_(MPa)": "Mean_Midday_LWP"})
rolling_min_predawn = mean_predawn_lwp.groupby(['Tree_Number'], as_index=False)['Mean_Predawn_LWP'].rolling(5, min_periods=1).min()
rolling_min_predawn = rolling_min_predawn.drop(['Tree_Number'], axis=1)
mean_predawn_lwp = pd.merge(mean_predawn_lwp, rolling_min_predawn, left_index=True, right_index=True, suffixes=('', '_rolling_min'))
rolling_min_midday = mean_midday_lwp.groupby(['Tree_Number'], as_index=False)['Mean_Midday_LWP'].rolling(5, min_periods=1).min()
rolling_min_midday = rolling_min_midday.drop(['Tree_Number'], axis=1)
mean_midday_lwp = pd.merge(mean_midday_lwp, rolling_min_midday, left_index=True, right_index=True, suffixes=('', '_rolling_min'))
lwp_summary = pd.merge(mean_predawn_lwp, mean_midday_lwp, on=['Sampling_Day', 'Tree_Number'])

###soil moisture
mean_soil_moisture = soil_moisture_df.groupby(['Sampling_Day', 'Tree_Number'], as_index=False)['VWC_Percent'].mean()
mean_soil_moisture = mean_soil_moisture.rename(columns={"VWC_Percent": "Mean_VWC"})
min_soil_moisture = soil_moisture_df.groupby(['Sampling_Day', 'Tree_Number'], as_index=False)['VWC_Percent'].min()
min_soil_moisture = min_soil_moisture.rename(columns={"VWC_Percent": "Min_VWC"})
max_soil_moisture = soil_moisture_df.groupby(['Sampling_Day', 'Tree_Number'], as_index=False)['VWC_Percent'].max()
max_soil_moisture = max_soil_moisture.rename(columns={"VWC_Percent": "Max_VWC"})

mean_psi_moisture = soil_moisture_df.groupby(['Sampling_Day', 'Tree_Number'], as_index=False)['psi_soil'].mean()
mean_psi_moisture = mean_psi_moisture.rename(columns={"psi_soil": "Mean_psi_soil"})
min_psi_moisture = soil_moisture_df.groupby(['Sampling_Day', 'Tree_Number'], as_index=False)['psi_soil'].min()
min_psi_moisture = min_psi_moisture.rename(columns={"psi_soil": "Min_psi_soil"})
max_psi_moisture = soil_moisture_df.groupby(['Sampling_Day', 'Tree_Number'], as_index=False)['psi_soil'].max()
max_psi_moisture = max_psi_moisture.rename(columns={"psi_soil": "Max_psi_soil"})
soil_moisture_summary = pd.merge(mean_soil_moisture, min_soil_moisture, on=['Sampling_Day', 'Tree_Number'])
soil_moisture_summary = pd.merge(soil_moisture_summary, max_soil_moisture, on=['Sampling_Day', 'Tree_Number'])
soil_moisture_summary = pd.merge(soil_moisture_summary, mean_psi_moisture, on=['Sampling_Day', 'Tree_Number'])
soil_moisture_summary = pd.merge(soil_moisture_summary, min_psi_moisture, on=['Sampling_Day', 'Tree_Number'])
soil_moisture_summary = pd.merge(soil_moisture_summary, max_psi_moisture, on=['Sampling_Day', 'Tree_Number'])

###met data
met_df['Sampling_Day'] = met_df.apply(classify_sampling_day, axis=1)
met_df_short = met_df[met_df['Sampling_Day']!='NaN']
mean_VPD = met_df_short.groupby(['Sampling_Day'], as_index=False)['VPD_kPa'].mean()
mean_Irr = met_df_short.groupby(['Sampling_Day'], as_index=False)['SlrW_Avg'].mean()
met_summary = pd.merge(mean_Irr, mean_VPD, on='Sampling_Day')

##gas exchange
mean_gsw = gasex_df.groupby(['Sampling_Day', 'Tree_Number'], as_index=False)['gsw'].mean()
mean_A = gasex_df.groupby(['Sampling_Day', 'Tree_Number'], as_index=False)['A'].mean()
gasex_summary = pd.merge(mean_gsw, mean_A, on=['Sampling_Day', 'Tree_Number'])

##isotopes
water_fractions = isotope_mixing(isotopes_df)
water_fractions = water_fractions[['Tree_Number', 'Sampling_Day', 'Mean_Uptake_Depth']]

##leaf area
unknown_tree = leaf_area_df[leaf_area_df['Tree'].isna()] #filter for naturally dropped leaves (vs cut off for sampling)
daily_sums = unknown_tree.groupby(['TIMESTAMP'])['Area_cm2'].sum()
sums_df = daily_sums.to_frame()
sums_df['cumulative'] = sums_df['Area_cm2'].cumsum(axis=0)
sums_df = sums_df.reset_index()
sums_df['Sampling_Day'] = sums_df.apply(classify_sampling_day, axis=1)
leaf_loss = sums_df[sums_df['Sampling_Day']!='NaN']
leaf_loss = leaf_loss[['Sampling_Day', 'cumulative']]
leaf_loss = leaf_loss.rename(columns={"cumulative": "Cumulative_Leaf_Loss"})

###create composite dataframe
full_summary_df = pd.merge(sap_flow_summary, lwp_summary, on=['Sampling_Day', 'Tree_Number'])
full_summary_df = pd.merge(full_summary_df, soil_moisture_summary, on=['Sampling_Day', 'Tree_Number'])
full_summary_df = pd.merge(full_summary_df, water_fractions, on=['Sampling_Day', 'Tree_Number'])
full_summary_df = pd.merge(full_summary_df, met_summary, on=['Sampling_Day'])
full_summary_df = pd.merge(full_summary_df, leaf_loss, how='left', on=['Sampling_Day']).fillna(0)
full_summary_df = pd.merge(full_summary_df, gasex_summary, how='left',  on=['Sampling_Day', 'Tree_Number'])

#drop variables that aren't considered
full_summary_df = full_summary_df.drop(labels=['Mean_Predawn_LWP_rolling_min', 'Mean_Midday_LWP_rolling_min', '95th_Perc_Sap_Rate', 'Min_VWC', 'Max_VWC', 'Mean_psi_soil', 'Min_psi_soil', 'Max_psi_soil'], axis=1)

summary_drydown_df = full_summary_df[full_summary_df['Sampling_Day']<=3]
summary_recovery_df = full_summary_df[full_summary_df['Sampling_Day']>3]

#Subsample drydown to match same sample size as recovery period
summary_drydown_df = summary_drydown_df.sample(n=12, random_state=random_seed)

only_gasex_dates = full_summary_df.dropna()
gasex_dates_drydown = only_gasex_dates[only_gasex_dates['Sampling_Day']<=3]
gasex_dates_recovery = only_gasex_dates[only_gasex_dates['Sampling_Day']>3]

###Matrices of mutual information
features = full_summary_df.drop(labels=['Tree_Number', 'Sampling_Day', 'gsw', 'A'], axis=1)
features = features[['Mean_VWC', 'VPD_kPa', 'SlrW_Avg', 'Mean_Uptake_Depth', 'Mean_Predawn_LWP', 'Mean_Midday_LWP', 'Cumulative_Leaf_Loss', 'Day_Mean_Sap_Rate', '95th_SWC', '5th_SWC']]
df_full_mi = features.corr(method=custom_mi_reg)

gasex_sub = full_summary_df.dropna()
features_gasex = gasex_sub.drop(labels=['Tree_Number', 'Sampling_Day'], axis=1)
features_gasex = features_gasex[['Mean_VWC', 'VPD_kPa', 'SlrW_Avg', 'Mean_Uptake_Depth', 'Mean_Predawn_LWP', 'Mean_Midday_LWP', 'Cumulative_Leaf_Loss', 'Day_Mean_Sap_Rate', '95th_SWC', '5th_SWC', 'A', 'gsw']]
df_gasex_mi = features_gasex.corr(method=custom_mi_reg)

df_gasex_mi.update(df_full_mi) ###join all
dummy_ones_gasex = np.ones_like(df_gasex_mi)
matrix_gasex = np.triu(dummy_ones_gasex)
plt.figure(figsize=(10,8))
sns.heatmap(df_gasex_mi, annot=True, fmt=".2f", linewidth=.5, mask=matrix_gasex, vmin=0, vmax=1.5)
plt.tight_layout()
plt.savefig('Output_Plots/MI_matrix_full_experiment.pdf')
plt.show()

features_drydown = summary_drydown_df.drop(labels=['Tree_Number', 'Sampling_Day', 'gsw', 'A'], axis=1)
features_drydown = features_drydown[['Mean_VWC', 'VPD_kPa', 'SlrW_Avg', 'Mean_Uptake_Depth', 'Mean_Predawn_LWP', 'Mean_Midday_LWP', 'Cumulative_Leaf_Loss', 'Day_Mean_Sap_Rate', '95th_SWC', '5th_SWC']]
df_drydown_mi = features_drydown.corr(method=custom_mi_reg)

gasex_sub_drydown_matrix = gasex_dates_drydown
features_gasex_drydown = gasex_sub_drydown_matrix.drop(labels=['Tree_Number', 'Sampling_Day'], axis=1)
features_gasex_drydown = features_gasex_drydown[['Mean_VWC', 'VPD_kPa', 'SlrW_Avg', 'Mean_Uptake_Depth', 'Mean_Predawn_LWP', 'Mean_Midday_LWP', 'Cumulative_Leaf_Loss', 'Day_Mean_Sap_Rate', '95th_SWC', '5th_SWC', 'A', 'gsw']]
df_gasex_drydown_mi = features_gasex_drydown.corr(method=custom_mi_reg)

df_gasex_drydown_mi.update(df_drydown_mi) ###join all
dummy_ones_gasex_drydown = np.ones_like(df_gasex_drydown_mi)
matrix_gasex_drydown = np.triu(dummy_ones_gasex_drydown)
plt.figure(figsize=(10,8))
sns.heatmap(df_gasex_drydown_mi, annot=True, fmt=".2f", linewidth=.5, mask=matrix_gasex_drydown, vmin=0, vmax=1.5)
plt.tight_layout()
plt.savefig('Output_Plots/MI_matrix_drydown.pdf')
plt.show()

features_recovery = summary_recovery_df.drop(labels=['Tree_Number', 'Sampling_Day', 'gsw', 'A'], axis=1)
features_recovery = features_recovery[['Mean_VWC', 'VPD_kPa', 'SlrW_Avg', 'Mean_Uptake_Depth', 'Mean_Predawn_LWP', 'Mean_Midday_LWP', 'Cumulative_Leaf_Loss', 'Day_Mean_Sap_Rate', '95th_SWC', '5th_SWC']]
df_recovery_mi = features_recovery.corr(method=custom_mi_reg)

gasex_sub_recovery_matrix = gasex_dates_recovery
features_gasex_recovery = gasex_sub_recovery_matrix.drop(labels=['Tree_Number', 'Sampling_Day'], axis=1)
features_gasex_recovery = features_gasex_recovery[['Mean_VWC', 'VPD_kPa', 'SlrW_Avg', 'Mean_Uptake_Depth', 'Mean_Predawn_LWP', 'Mean_Midday_LWP', 'Cumulative_Leaf_Loss', 'Day_Mean_Sap_Rate', '95th_SWC', '5th_SWC', 'A', 'gsw']]
df_gasex_recovery_mi = features_gasex_recovery.corr(method=custom_mi_reg)

df_gasex_recovery_mi.update(df_recovery_mi) ###join all
dummy_ones_gasex_recovery = np.ones_like(df_gasex_recovery_mi)
matrix_gasex_recovery = np.triu(dummy_ones_gasex_recovery)
plt.figure(figsize=(10,8))
sns.heatmap(df_gasex_recovery_mi, annot=True, fmt=".2f", linewidth=.5, mask=matrix_gasex_recovery, vmin=0, vmax=1.5)
plt.tight_layout()
plt.savefig('Output_Plots/MI_matrix_recovery.pdf')
plt.show()

#Bar Chart for Sap Flow
sapflow_full = df_gasex_mi[['Day_Mean_Sap_Rate']]
sapflow_full = sapflow_full.rename(columns={'Day_Mean_Sap_Rate': 'Full_Period'})
sapflow_drydown = df_gasex_drydown_mi[['Day_Mean_Sap_Rate']]
sapflow_drydown = sapflow_drydown.rename(columns={'Day_Mean_Sap_Rate': 'Drydown'})
sapflow_full = pd.merge(sapflow_full, sapflow_drydown, left_index=True, right_index=True)
sapflow_recovery = df_gasex_recovery_mi[['Day_Mean_Sap_Rate']]
sapflow_recovery = sapflow_recovery.rename(columns={'Day_Mean_Sap_Rate': 'Recovery'})
sapflow_full = pd.merge(sapflow_full, sapflow_recovery, left_index=True, right_index=True)
sapflow_full = sapflow_full.drop(index='Day_Mean_Sap_Rate')
sapflow_full = sapflow_full.reset_index()
sapflow_full = sapflow_full.rename(columns={"index": "Measurement"})
sapflow_full = sapflow_full.melt(id_vars=["Measurement"])
sapflow_full = sapflow_full.rename(columns={"variable": "Time_Period"})
sapflow_full = sapflow_full.rename(columns={"value": "Mutual_Information"})

palette = {"Full_Period":'#8E85A3',
           "Drydown":'#C14655',
           "Recovery":'#4D84BC'}
order = ['Mean_VWC', 'VPD_kPa', 'SlrW_Avg','Mean_Uptake_Depth', 'Mean_Predawn_LWP', 'Mean_Midday_LWP', 'Cumulative_Leaf_Loss', '95th_SWC', '5th_SWC', 'A', 'gsw']
plt.figure(figsize=(11,7))
sns.barplot(data=sapflow_full, x="Measurement", y="Mutual_Information", hue="Time_Period", palette=palette, order=order)
plt.xticks(rotation=70)
plt.ylim(0.0, 1.2)
plt.tight_layout()
plt.savefig('Output_Plots/MI_with_sapflow.pdf')
plt.show()

#Bar Chart for Predawn
predawn_full = df_gasex_mi[['Mean_Predawn_LWP']]
predawn_full = predawn_full.rename(columns={'Mean_Predawn_LWP': 'Full_Period'})
predawn_drydown = df_gasex_drydown_mi[['Mean_Predawn_LWP']]
predawn_drydown = predawn_drydown.rename(columns={'Mean_Predawn_LWP': 'Drydown'})
predawn_full = pd.merge(predawn_full, predawn_drydown, left_index=True, right_index=True)
predawn_recovery = df_gasex_recovery_mi[['Mean_Predawn_LWP']]
predawn_recovery = predawn_recovery.rename(columns={'Mean_Predawn_LWP': 'Recovery'})
predawn_full = pd.merge(predawn_full, predawn_recovery, left_index=True, right_index=True)
predawn_full = predawn_full.drop(index='Mean_Predawn_LWP')
predawn_full = predawn_full.reset_index()
predawn_full = predawn_full.rename(columns={"index": "Measurement"})
predawn_full = predawn_full.melt(id_vars=["Measurement"])
predawn_full = predawn_full.rename(columns={"variable": "Time_Period"})
predawn_full = predawn_full.rename(columns={"value": "Mutual_Information"})

palette = {"Full_Period":'#8E85A3',
           "Drydown":'#C14655',
           "Recovery":'#4D84BC'}
order = ['Mean_VWC', 'VPD_kPa', 'SlrW_Avg','Mean_Uptake_Depth', 'Mean_Midday_LWP', 'Cumulative_Leaf_Loss', 'Day_Mean_Sap_Rate', '95th_SWC', '5th_SWC', 'A', 'gsw']
plt.figure(figsize=(11,7))
sns.barplot(data=predawn_full, x="Measurement", y="Mutual_Information", hue="Time_Period", palette=palette, order=order)
plt.xticks(rotation=70)
plt.ylim(0.0, 1.2)
plt.tight_layout()
plt.savefig('Output_Plots/MI_with_predawn.pdf')
plt.show()

#Bar Chart for Midday
midday_full = df_gasex_mi[['Mean_Midday_LWP']]
midday_full = midday_full.rename(columns={'Mean_Midday_LWP': 'Full_Period'})
midday_drydown = df_gasex_drydown_mi[['Mean_Midday_LWP']]
midday_drydown = midday_drydown.rename(columns={'Mean_Midday_LWP': 'Drydown'})
midday_full = pd.merge(midday_full, midday_drydown, left_index=True, right_index=True)
midday_recovery = df_gasex_recovery_mi[['Mean_Midday_LWP']]
midday_recovery = midday_recovery.rename(columns={'Mean_Midday_LWP': 'Recovery'})
midday_full = pd.merge(midday_full, midday_recovery, left_index=True, right_index=True)
midday_full = midday_full.drop(index='Mean_Midday_LWP')
midday_full = midday_full.reset_index()
midday_full = midday_full.rename(columns={"index": "Measurement"})
midday_full = midday_full.melt(id_vars=["Measurement"])
midday_full = midday_full.rename(columns={"variable": "Time_Period"})
midday_full = midday_full.rename(columns={"value": "Mutual_Information"})

palette = {"Full_Period":'#8E85A3',
           "Drydown":'#C14655',
           "Recovery":'#4D84BC'}
order = ['Mean_VWC', 'VPD_kPa', 'SlrW_Avg','Mean_Uptake_Depth', 'Mean_Predawn_LWP', 'Cumulative_Leaf_Loss', 'Day_Mean_Sap_Rate', '95th_SWC', '5th_SWC', 'A', 'gsw']
plt.figure(figsize=(11,7))
sns.barplot(data=midday_full, x="Measurement", y="Mutual_Information", hue="Time_Period", palette=palette, order=order)
plt.xticks(rotation=70)
plt.ylim(0.0, 1.2)
plt.tight_layout()
plt.savefig('Output_Plots/MI_with_midday.pdf')
plt.show()