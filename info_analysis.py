import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from utility_functions import classify_sampling_day, custom_mi_reg, isotope_mixing
from sklearn.feature_selection import mutual_info_regression

'''
This script takes the processed data and calculates summaries (as needed) and formats data to calculate mutual 
information and make plots
'''

neighbors = 2 #number of neighbors to use for mi_regression, if change also update in custom_mi_reg in utility_functions

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

###Sap flow over full experimental period
#filter to only daylight hours (6:00-20:00)
sap_flow_daytime_long = sap_flow_df.set_index('TIMESTAMP').between_time('06:00', '20:00').reset_index()
sap_flow_daytime_long['Date'] = pd.DatetimeIndex(sap_flow_daytime_long['TIMESTAMP']).normalize()
#Daytime hour mean sap flow rate
mean_day_sap_flow_long = sap_flow_daytime_long.groupby(['Date','Tree_Number'], as_index=False)['Total_Sapflow'].mean()
mean_day_sap_flow_long = mean_day_sap_flow_long.rename(columns={"Total_Sapflow": "Day_Mean_Sap_Rate"})
#Find 95th percentile of sap flow rate (rather than max which is going to be more affected by oscillations in sensor)
quant95_sap_flow_long = sap_flow_daytime_long.groupby(['Date','Tree_Number'], as_index=False)['Total_Sapflow'].quantile(.95)
quant95_sap_flow_long = quant95_sap_flow_long.rename(columns={"Total_Sapflow": "95th_Perc_Sap_Rate"})
#Find 5th and 95th percentile of stem water content (including overnight measurements)
sap_flow_df['Date'] = pd.DatetimeIndex(sap_flow_df['TIMESTAMP']).normalize()
quant95_swc_long = sap_flow_df.groupby(['Date','Tree_Number'], as_index=False)['SWC_Outer'].quantile(.95)
quant95_swc_long = quant95_swc_long.rename(columns={"SWC_Outer": "95th_SWC"})
quant05_swc_long = sap_flow_df.groupby(['Date','Tree_Number'], as_index=False)['SWC_Outer'].quantile(.05)
quant05_swc_long = quant05_swc_long.rename(columns={"SWC_Outer": "5th_SWC"})
sap_flow_summary_long = pd.merge(mean_day_sap_flow_long, quant95_sap_flow_long, on=['Date', 'Tree_Number'])
sap_flow_summary_long = pd.merge(sap_flow_summary_long, quant95_swc_long, on=['Date', 'Tree_Number'])
sap_flow_summary_long = pd.merge(sap_flow_summary_long, quant05_swc_long, on=['Date', 'Tree_Number'])

###leaf water potentials
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

###met data full series
met_df['Date'] = pd.DatetimeIndex(met_df['TIMESTAMP']).normalize()
mean_VPD_long = met_df.groupby(['Date'], as_index=False)['VPD_kPa'].mean()
mean_Irr_long = met_df.groupby(['Date'], as_index=False)['SlrW_Avg'].mean()
met_summary_long = pd.merge(mean_Irr_long, mean_VPD_long, on='Date')

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

long_series_df = pd.merge(sap_flow_summary_long, met_summary_long, on=['Date'])

long_target = long_series_df['Day_Mean_Sap_Rate']
long_features = long_series_df[['VPD_kPa', 'SlrW_Avg', '95th_SWC', '5th_SWC']]
long_drydown = long_series_df[long_series_df['Date'] < pd.Timestamp('2021-07-16 00:00:00')]
long_target_drydown = long_drydown['Day_Mean_Sap_Rate']
long_features_drydown = long_drydown[['VPD_kPa', 'SlrW_Avg', '95th_SWC', '5th_SWC']]
long_recovery = long_series_df[long_series_df['Date'] >= pd.Timestamp('2021-07-16 00:00:00')]
long_target_recovery = long_recovery['Day_Mean_Sap_Rate']
long_features_recovery = long_recovery[['VPD_kPa', 'SlrW_Avg', '95th_SWC', '5th_SWC']]

mutual_info_long = mutual_info_regression(long_features, long_target, n_neighbors=neighbors)
mutual_info_long = pd.Series(mutual_info_long)
mutual_info_long.index = long_features.columns
mutual_info_long.sort_values(ascending=False)
mutual_info_long
MI_df_long = mutual_info_long.to_frame()
MI_df_long = MI_df_long.rename(columns={0: "Full_Period"})

mutual_info_drydown_long = mutual_info_regression(long_features_drydown, long_target_drydown, n_neighbors=neighbors)
mutual_info_drydown_long = pd.Series(mutual_info_drydown_long)
mutual_info_drydown_long.index = long_features_drydown.columns
mutual_info_drydown_long.sort_values(ascending=False)
MI_drydown_df_long = mutual_info_drydown_long.to_frame()
MI_drydown_df_long = MI_drydown_df_long.rename(columns={0: "Drydown"})
MI_df_long = pd.merge(MI_df_long, MI_drydown_df_long, left_index=True, right_index=True)

mutual_info_recovery_long = mutual_info_regression(long_features_recovery, long_target_recovery, n_neighbors=neighbors)
mutual_info_recovery_long = pd.Series(mutual_info_recovery_long)
mutual_info_recovery_long.index = long_features_recovery.columns
mutual_info_recovery_long.sort_values(ascending=False)
MI_recovery_df_long = mutual_info_recovery_long.to_frame()
MI_recovery_df_long = MI_recovery_df_long.rename(columns={0: "Recovery"})
MI_df_long = pd.merge(MI_df_long, MI_recovery_df_long, left_index=True, right_index=True)

full_summary_df = full_summary_df.drop(labels=['Mean_Predawn_LWP_rolling_min', 'Mean_Midday_LWP_rolling_min', 'Min_VWC', 'Max_VWC', 'Mean_psi_soil', 'Min_psi_soil', 'Max_psi_soil'], axis=1)

##Mutual information with sap flow as target

target = full_summary_df['Day_Mean_Sap_Rate']
features = full_summary_df.drop(labels=['Tree_Number', 'Sampling_Day', 'Day_Mean_Sap_Rate', '95th_Perc_Sap_Rate', 'gsw', 'A', 'VPD_kPa', 'SlrW_Avg', '95th_SWC', '5th_SWC'], axis=1)

mutual_info = mutual_info_regression(features, target, n_neighbors=neighbors)
mutual_info = pd.Series(mutual_info)
mutual_info.index = features.columns
mutual_info.sort_values(ascending=False)
mutual_info
MI_df = mutual_info.to_frame()
MI_df = MI_df.rename(columns={0: "Full_Period"})

summary_drydown_df = full_summary_df[full_summary_df['Sampling_Day']<=3]
summary_recovery_df = full_summary_df[full_summary_df['Sampling_Day']>3]

target_drydown = summary_drydown_df['Day_Mean_Sap_Rate']
features_drydown = summary_drydown_df.drop(labels=['Tree_Number', 'Sampling_Day', 'Day_Mean_Sap_Rate', '95th_Perc_Sap_Rate', 'gsw', 'A', 'VPD_kPa', 'SlrW_Avg', '95th_SWC', '5th_SWC'], axis=1)

mutual_info_drydown = mutual_info_regression(features_drydown, target_drydown, n_neighbors=neighbors)
mutual_info_drydown = pd.Series(mutual_info_drydown)
mutual_info_drydown.index = features_drydown.columns
mutual_info_drydown.sort_values(ascending=False)
MI_drydown_df = mutual_info_drydown.to_frame()
MI_drydown_df = MI_drydown_df.rename(columns={0: "Drydown"})
MI_df = pd.merge(MI_df, MI_drydown_df, left_index=True, right_index=True)

target_recovery = summary_recovery_df['Day_Mean_Sap_Rate']
features_recovery = summary_recovery_df.drop(labels=['Tree_Number', 'Sampling_Day', 'Day_Mean_Sap_Rate', '95th_Perc_Sap_Rate', 'gsw', 'A', 'VPD_kPa', 'SlrW_Avg', '95th_SWC', '5th_SWC'], axis=1)

mutual_info_recovery = mutual_info_regression(features_recovery, target_recovery, n_neighbors=neighbors)
mutual_info_recovery = pd.Series(mutual_info_recovery)
mutual_info_recovery.index = features_recovery.columns
mutual_info_recovery.sort_values(ascending=False)
MI_recovery_df = mutual_info_recovery.to_frame()
MI_recovery_df = MI_recovery_df.rename(columns={0: "Recovery"})
MI_df = pd.merge(MI_df, MI_recovery_df, left_index=True, right_index=True)

only_gasex_dates = full_summary_df.dropna()
gasex_dates_drydown = only_gasex_dates[only_gasex_dates['Sampling_Day']<=3]
gasex_dates_recovery = only_gasex_dates[only_gasex_dates['Sampling_Day']>3]

target_gasex_sub = only_gasex_dates['Day_Mean_Sap_Rate']
features_gasex_sub = only_gasex_dates[['gsw', 'A']]
mutual_info_gasex = mutual_info_regression(features_gasex_sub, target_gasex_sub, n_neighbors=neighbors)
mutual_info_gasex = pd.Series(mutual_info_gasex)
mutual_info_gasex.index = features_gasex_sub.columns
mutual_info_gasex.sort_values(ascending=False)
MI_gasex_df = mutual_info_gasex.to_frame()
MI_gasex_df = MI_gasex_df.rename(columns={0: "Full_Period"})


target_gasex_drydown = gasex_dates_drydown['Day_Mean_Sap_Rate']
features_gasex_drydown= gasex_dates_drydown[['gsw', 'A']]
mutual_info_gasex_drydown = mutual_info_regression(features_gasex_drydown, target_gasex_drydown, n_neighbors=neighbors)
mutual_info_gasex_drydown = pd.Series(mutual_info_gasex_drydown)
mutual_info_gasex_drydown.index = features_gasex_drydown.columns
mutual_info_gasex_drydown.sort_values(ascending=False)
MI_gasex_drydown_df = mutual_info_gasex_drydown.to_frame()
MI_gasex_drydown_df = MI_gasex_drydown_df.rename(columns={0: "Drydown"})
MI_gasex_df = pd.merge(MI_gasex_df, MI_gasex_drydown_df, left_index=True, right_index=True)

target_gasex_recovery = gasex_dates_recovery['Day_Mean_Sap_Rate']
features_gasex_recovery = gasex_dates_recovery[['gsw', 'A']]
mutual_info_gasex_recovery = mutual_info_regression(features_gasex_recovery, target_gasex_recovery, n_neighbors=neighbors)
mutual_info_gasex_recovery = pd.Series(mutual_info_gasex_recovery)
mutual_info_gasex_recovery.index = features_gasex_recovery.columns
mutual_info_gasex_recovery.sort_values(ascending=False)
MI_gasex_recovery_df = mutual_info_gasex_recovery.to_frame()
MI_gasex_recovery_df = MI_gasex_recovery_df.rename(columns={0: "Recovery"})
MI_gasex_df = pd.merge(MI_gasex_df, MI_gasex_recovery_df, left_index=True, right_index=True)

MI_df = MI_df.append(MI_gasex_df)
MI_df = MI_df.append(MI_df_long)
MI_df = MI_df.reset_index()
MI_df = MI_df.rename(columns={"index": "Measurement"})
MI_df_melt = MI_df.melt(id_vars=["Measurement"])
MI_df_melt = MI_df_melt.rename(columns={"variable": "Time_Period"})
MI_df_melt = MI_df_melt.rename(columns={"value": "Mutual_Information"})

palette = {"Full_Period":'#8E85A3',
           "Drydown":'#C14655',
           "Recovery":'#4D84BC'}
order = ['Mean_VWC', 'VPD_kPa', 'SlrW_Avg','Mean_Uptake_Depth', 'Mean_Predawn_LWP', 'Mean_Midday_LWP', 'Cumulative_Leaf_Loss', '95th_SWC', '5th_SWC', 'A', 'gsw']
plt.figure(figsize=(11,7))
sns.barplot(data=MI_df_melt, x="Measurement", y="Mutual_Information", hue="Time_Period", palette=palette, order=order)
plt.xticks(rotation=70)
plt.ylim(0.0, 1.2)
plt.tight_layout()
plt.savefig('Output_Plots/MI_with_sapflow.pdf')
plt.show()

###Mutual information with predawn LWP as target variable

target = full_summary_df['Mean_Predawn_LWP']
features = full_summary_df.drop(labels=['Tree_Number', 'Sampling_Day', 'Mean_Predawn_LWP', '95th_Perc_Sap_Rate', 'gsw', 'A'], axis=1)

mutual_info = mutual_info_regression(features, target, n_neighbors=neighbors)
mutual_info = pd.Series(mutual_info)
mutual_info.index = features.columns
mutual_info.sort_values(ascending=False)
mutual_info
MI_df = mutual_info.to_frame()
MI_df = MI_df.rename(columns={0: "Full_Period"})

summary_drydown_df = full_summary_df[full_summary_df['Sampling_Day']<=3]
summary_recovery_df = full_summary_df[full_summary_df['Sampling_Day']>3]

target_drydown = summary_drydown_df['Mean_Predawn_LWP']
features_drydown = summary_drydown_df.drop(labels=['Tree_Number', 'Sampling_Day', 'Mean_Predawn_LWP', '95th_Perc_Sap_Rate', 'gsw', 'A'], axis=1)

mutual_info_drydown = mutual_info_regression(features_drydown, target_drydown, n_neighbors=neighbors)
mutual_info_drydown = pd.Series(mutual_info_drydown)
mutual_info_drydown.index = features_drydown.columns
mutual_info_drydown.sort_values(ascending=False)
MI_drydown_df = mutual_info_drydown.to_frame()
MI_drydown_df = MI_drydown_df.rename(columns={0: "Drydown"})
MI_df = pd.merge(MI_df, MI_drydown_df, left_index=True, right_index=True)

target_recovery = summary_recovery_df['Mean_Predawn_LWP']
features_recovery = summary_recovery_df.drop(labels=['Tree_Number', 'Sampling_Day', 'Mean_Predawn_LWP', '95th_Perc_Sap_Rate', 'gsw', 'A'], axis=1)

mutual_info_recovery = mutual_info_regression(features_recovery, target_recovery, n_neighbors=neighbors)
mutual_info_recovery = pd.Series(mutual_info_recovery)
mutual_info_recovery.index = features_recovery.columns
mutual_info_recovery.sort_values(ascending=False)
MI_recovery_df = mutual_info_recovery.to_frame()
MI_recovery_df = MI_recovery_df.rename(columns={0: "Recovery"})
MI_df = pd.merge(MI_df, MI_recovery_df, left_index=True, right_index=True)

only_gasex_dates = full_summary_df.dropna()
gasex_dates_drydown = only_gasex_dates[only_gasex_dates['Sampling_Day']<=3]
gasex_dates_recovery = only_gasex_dates[only_gasex_dates['Sampling_Day']>3]

target_gasex_sub = only_gasex_dates['Mean_Predawn_LWP']
features_gasex_sub = only_gasex_dates[['gsw', 'A']]
mutual_info_gasex = mutual_info_regression(features_gasex_sub, target_gasex_sub, n_neighbors=neighbors)
mutual_info_gasex = pd.Series(mutual_info_gasex)
mutual_info_gasex.index = features_gasex_sub.columns
mutual_info_gasex.sort_values(ascending=False)
MI_gasex_df = mutual_info_gasex.to_frame()
MI_gasex_df = MI_gasex_df.rename(columns={0: "Full_Period"})


target_gasex_drydown = gasex_dates_drydown['Mean_Predawn_LWP']
features_gasex_drydown= gasex_dates_drydown[['gsw', 'A']]
mutual_info_gasex_drydown = mutual_info_regression(features_gasex_drydown, target_gasex_drydown, n_neighbors=neighbors)
mutual_info_gasex_drydown = pd.Series(mutual_info_gasex_drydown)
mutual_info_gasex_drydown.index = features_gasex_drydown.columns
mutual_info_gasex_drydown.sort_values(ascending=False)
MI_gasex_drydown_df = mutual_info_gasex_drydown.to_frame()
MI_gasex_drydown_df = MI_gasex_drydown_df.rename(columns={0: "Drydown"})
MI_gasex_df = pd.merge(MI_gasex_df, MI_gasex_drydown_df, left_index=True, right_index=True)

target_gasex_recovery = gasex_dates_recovery['Mean_Predawn_LWP']
features_gasex_recovery = gasex_dates_recovery[['gsw', 'A']]
mutual_info_gasex_recovery = mutual_info_regression(features_gasex_recovery, target_gasex_recovery, n_neighbors=neighbors)
mutual_info_gasex_recovery = pd.Series(mutual_info_gasex_recovery)
mutual_info_gasex_recovery.index = features_gasex_recovery.columns
mutual_info_gasex_recovery.sort_values(ascending=False)
MI_gasex_recovery_df = mutual_info_gasex_recovery.to_frame()
MI_gasex_recovery_df = MI_gasex_recovery_df.rename(columns={0: "Recovery"})
MI_gasex_df = pd.merge(MI_gasex_df, MI_gasex_recovery_df, left_index=True, right_index=True)

MI_df = MI_df.append(MI_gasex_df)
MI_df = MI_df.reset_index()
MI_df = MI_df.rename(columns={"index": "Measurement"})
MI_df_melt = MI_df.melt(id_vars=["Measurement"])
MI_df_melt = MI_df_melt.rename(columns={"variable": "Time_Period"})
MI_df_melt = MI_df_melt.rename(columns={"value": "Mutual_Information"})

palette = {"Full_Period":'#8E85A3',
           "Drydown":'#C14655',
           "Recovery":'#4D84BC'}
order = ['Mean_VWC', 'VPD_kPa', 'SlrW_Avg','Mean_Uptake_Depth', 'Mean_Midday_LWP', 'Cumulative_Leaf_Loss', 'Day_Mean_Sap_Rate', '95th_SWC', '5th_SWC', 'A', 'gsw']
plt.figure(figsize=(11,7))
sns.barplot(data=MI_df_melt, x="Measurement", y="Mutual_Information", hue="Time_Period", palette=palette, order=order)
plt.xticks(rotation=70)
plt.ylim(0.0, 1.2)
plt.tight_layout()
plt.savefig('Output_Plots/MI_with_predawn.pdf')
plt.show()


###MI with midday leaf water potential as target variable

target = full_summary_df['Mean_Midday_LWP']
features = full_summary_df.drop(labels=['Tree_Number', 'Sampling_Day', 'Mean_Midday_LWP', '95th_Perc_Sap_Rate', 'gsw', 'A'], axis=1)

mutual_info = mutual_info_regression(features, target, n_neighbors=neighbors)
mutual_info = pd.Series(mutual_info)
mutual_info.index = features.columns
mutual_info.sort_values(ascending=False)
mutual_info
MI_df = mutual_info.to_frame()
MI_df = MI_df.rename(columns={0: "Full_Period"})

summary_drydown_df = full_summary_df[full_summary_df['Sampling_Day']<=3]
summary_recovery_df = full_summary_df[full_summary_df['Sampling_Day']>3]

target_drydown = summary_drydown_df['Mean_Midday_LWP']
features_drydown = summary_drydown_df.drop(labels=['Tree_Number', 'Sampling_Day', 'Mean_Midday_LWP', '95th_Perc_Sap_Rate', 'gsw', 'A'], axis=1)

mutual_info_drydown = mutual_info_regression(features_drydown, target_drydown, n_neighbors=neighbors)
mutual_info_drydown = pd.Series(mutual_info_drydown)
mutual_info_drydown.index = features_drydown.columns
mutual_info_drydown.sort_values(ascending=False)
MI_drydown_df = mutual_info_drydown.to_frame()
MI_drydown_df = MI_drydown_df.rename(columns={0: "Drydown"})
MI_df = pd.merge(MI_df, MI_drydown_df, left_index=True, right_index=True)

target_recovery = summary_recovery_df['Mean_Midday_LWP']
features_recovery = summary_recovery_df.drop(labels=['Tree_Number', 'Sampling_Day', 'Mean_Midday_LWP', '95th_Perc_Sap_Rate', 'gsw', 'A'], axis=1)

mutual_info_recovery = mutual_info_regression(features_recovery, target_recovery, n_neighbors=neighbors)
mutual_info_recovery = pd.Series(mutual_info_recovery)
mutual_info_recovery.index = features_recovery.columns
mutual_info_recovery.sort_values(ascending=False)
MI_recovery_df = mutual_info_recovery.to_frame()
MI_recovery_df = MI_recovery_df.rename(columns={0: "Recovery"})
MI_df = pd.merge(MI_df, MI_recovery_df, left_index=True, right_index=True)

only_gasex_dates = full_summary_df.dropna()
gasex_dates_drydown = only_gasex_dates[only_gasex_dates['Sampling_Day']<=3]
gasex_dates_recovery = only_gasex_dates[only_gasex_dates['Sampling_Day']>3]

target_gasex_sub = only_gasex_dates['Mean_Midday_LWP']
features_gasex_sub = only_gasex_dates[['gsw', 'A']]
mutual_info_gasex = mutual_info_regression(features_gasex_sub, target_gasex_sub, n_neighbors=neighbors)
mutual_info_gasex = pd.Series(mutual_info_gasex)
mutual_info_gasex.index = features_gasex_sub.columns
mutual_info_gasex.sort_values(ascending=False)
MI_gasex_df = mutual_info_gasex.to_frame()
MI_gasex_df = MI_gasex_df.rename(columns={0: "Full_Period"})


target_gasex_drydown = gasex_dates_drydown['Mean_Midday_LWP']
features_gasex_drydown= gasex_dates_drydown[['gsw', 'A']]
mutual_info_gasex_drydown = mutual_info_regression(features_gasex_drydown, target_gasex_drydown, n_neighbors=neighbors)
mutual_info_gasex_drydown = pd.Series(mutual_info_gasex_drydown)
mutual_info_gasex_drydown.index = features_gasex_drydown.columns
mutual_info_gasex_drydown.sort_values(ascending=False)
MI_gasex_drydown_df = mutual_info_gasex_drydown.to_frame()
MI_gasex_drydown_df = MI_gasex_drydown_df.rename(columns={0: "Drydown"})
MI_gasex_df = pd.merge(MI_gasex_df, MI_gasex_drydown_df, left_index=True, right_index=True)

target_gasex_recovery = gasex_dates_recovery['Mean_Midday_LWP']
features_gasex_recovery = gasex_dates_recovery[['gsw', 'A']]
mutual_info_gasex_recovery = mutual_info_regression(features_gasex_recovery, target_gasex_recovery, n_neighbors=neighbors)
mutual_info_gasex_recovery = pd.Series(mutual_info_gasex_recovery)
mutual_info_gasex_recovery.index = features_gasex_recovery.columns
mutual_info_gasex_recovery.sort_values(ascending=False)
MI_gasex_recovery_df = mutual_info_gasex_recovery.to_frame()
MI_gasex_recovery_df = MI_gasex_recovery_df.rename(columns={0: "Recovery"})
MI_gasex_df = pd.merge(MI_gasex_df, MI_gasex_recovery_df, left_index=True, right_index=True)

MI_df = MI_df.append(MI_gasex_df)
MI_df = MI_df.reset_index()
MI_df = MI_df.rename(columns={"index": "Measurement"})
MI_df_melt = MI_df.melt(id_vars=["Measurement"])
MI_df_melt = MI_df_melt.rename(columns={"variable": "Time_Period"})
MI_df_melt = MI_df_melt.rename(columns={"value": "Mutual_Information"})

palette = {"Full_Period":'#8E85A3',
           "Drydown":'#C14655',
           "Recovery":'#4D84BC'}
order = ['Mean_VWC', 'VPD_kPa', 'SlrW_Avg','Mean_Uptake_Depth', 'Mean_Predawn_LWP', 'Cumulative_Leaf_Loss', 'Day_Mean_Sap_Rate', '95th_SWC', '5th_SWC', 'A', 'gsw']
plt.figure(figsize=(11,7))
sns.barplot(data=MI_df_melt, x="Measurement", y="Mutual_Information", hue="Time_Period", palette=palette, order=order)
plt.xticks(rotation=70)
plt.ylim(0.0, 1.2)
plt.tight_layout()
plt.savefig('Output_Plots/MI_with_midday.pdf')
plt.show()


###Matrices of mutual information
long_features = long_series_df[['VPD_kPa', 'SlrW_Avg', '95th_SWC', '5th_SWC', 'Day_Mean_Sap_Rate']]
df_long_mi = long_features.corr(method=custom_mi_reg)

features = full_summary_df.drop(labels=['Tree_Number', 'Sampling_Day', '95th_Perc_Sap_Rate', 'gsw', 'A'], axis=1)
features = features[['Mean_VWC', 'VPD_kPa', 'SlrW_Avg', 'Mean_Uptake_Depth', 'Mean_Predawn_LWP', 'Mean_Midday_LWP', 'Cumulative_Leaf_Loss', 'Day_Mean_Sap_Rate', '95th_SWC', '5th_SWC']]
df_full_mi = features.corr(method=custom_mi_reg)

# dummy_ones = np.ones_like(df_full_mi)
# matrix = np.triu(dummy_ones)
# sns.heatmap(df_full_mi, annot=True, fmt=".2f", linewidth=.5, mask=matrix)

gasex_sub = full_summary_df.dropna()
features_gasex = gasex_sub.drop(labels=['Tree_Number', 'Sampling_Day', '95th_Perc_Sap_Rate'], axis=1)
features_gasex = features_gasex[['Mean_VWC', 'VPD_kPa', 'SlrW_Avg', 'Mean_Uptake_Depth', 'Mean_Predawn_LWP', 'Mean_Midday_LWP', 'Cumulative_Leaf_Loss', 'Day_Mean_Sap_Rate', '95th_SWC', '5th_SWC', 'A', 'gsw']]
df_gasex_mi = features_gasex.corr(method=custom_mi_reg)
#dummy_ones_gasex = np.ones_like(df_gasex_mi)
#matrix_gasex = np.triu(dummy_ones_gasex)
#sns.heatmap(df_gasex_mi, annot=True, fmt=".2f", linewidth=.5, mask=matrix_gasex)

df_gasex_mi.update(df_full_mi) ###join all
df_gasex_mi.update(df_long_mi)
dummy_ones_gasex = np.ones_like(df_gasex_mi)
matrix_gasex = np.triu(dummy_ones_gasex)
plt.figure(figsize=(10,8))
sns.heatmap(df_gasex_mi, annot=True, fmt=".2f", linewidth=.5, mask=matrix_gasex, vmin=0, vmax=1.5)
plt.tight_layout()
plt.savefig('Output_Plots/MI_matrix_full_experiment.pdf')
plt.show()


###matrix for drydown
long_features_drydown = long_drydown[['VPD_kPa', 'SlrW_Avg', '95th_SWC', '5th_SWC', 'Day_Mean_Sap_Rate']]
df_long_mi_drydown = long_features_drydown.corr(method=custom_mi_reg)

summary_drydown_df = full_summary_df[full_summary_df['Sampling_Day']<=3]
features = summary_drydown_df.drop(labels=['Tree_Number', 'Sampling_Day', '95th_Perc_Sap_Rate', 'gsw', 'A'], axis=1)
features = features[['Mean_VWC', 'VPD_kPa', 'SlrW_Avg', 'Mean_Uptake_Depth', 'Mean_Predawn_LWP', 'Mean_Midday_LWP', 'Cumulative_Leaf_Loss', 'Day_Mean_Sap_Rate', '95th_SWC', '5th_SWC']]
df_full_mi = features.corr(method=custom_mi_reg)

# dummy_ones = np.ones_like(df_full_mi)
# matrix = np.triu(dummy_ones)
# sns.heatmap(df_full_mi, annot=True, fmt=".2f", linewidth=.5, mask=matrix)

gasex_sub = summary_drydown_df.dropna()
features_gasex = gasex_sub.drop(labels=['Tree_Number', 'Sampling_Day', '95th_Perc_Sap_Rate'], axis=1)
features_gasex = features_gasex[['Mean_VWC', 'VPD_kPa', 'SlrW_Avg', 'Mean_Uptake_Depth', 'Mean_Predawn_LWP', 'Mean_Midday_LWP', 'Cumulative_Leaf_Loss', 'Day_Mean_Sap_Rate', '95th_SWC', '5th_SWC', 'A', 'gsw']]
df_gasex_mi = features_gasex.corr(method=custom_mi_reg)
#dummy_ones_gasex = np.ones_like(df_gasex_mi)
#matrix_gasex = np.triu(dummy_ones_gasex)
#sns.heatmap(df_gasex_mi, annot=True, fmt=".2f", linewidth=.5, mask=matrix_gasex)

df_gasex_mi.update(df_full_mi) ###join all
df_gasex_mi.update(df_long_mi_drydown)
dummy_ones_gasex = np.ones_like(df_gasex_mi)
matrix_gasex = np.triu(dummy_ones_gasex)
plt.figure(figsize=(10,8))
sns.heatmap(df_gasex_mi, annot=True, fmt=".2f", linewidth=.5, mask=matrix_gasex, vmin=0, vmax=1.5)
plt.tight_layout()
plt.savefig('Output_Plots/MI_matrix_drydown.pdf')
plt.show()

###matrix for recovery
long_features_recovery = long_recovery[['VPD_kPa', 'SlrW_Avg', '95th_SWC', '5th_SWC', 'Day_Mean_Sap_Rate']]
df_long_mi_recovery = long_features_recovery.corr(method=custom_mi_reg)

summary_recovery_df = full_summary_df[full_summary_df['Sampling_Day']>3]
features = summary_recovery_df.drop(labels=['Tree_Number', 'Sampling_Day', '95th_Perc_Sap_Rate', 'gsw', 'A'], axis=1)
features = features[['Mean_VWC', 'VPD_kPa', 'SlrW_Avg', 'Mean_Uptake_Depth', 'Mean_Predawn_LWP', 'Mean_Midday_LWP', 'Cumulative_Leaf_Loss', 'Day_Mean_Sap_Rate', '95th_SWC', '5th_SWC']]
df_full_mi = features.corr(method=custom_mi_reg)

# dummy_ones = np.ones_like(df_full_mi)
# matrix = np.triu(dummy_ones)
# sns.heatmap(df_full_mi, annot=True, fmt=".2f", linewidth=.5, mask=matrix)

gasex_sub = summary_recovery_df.dropna()
features_gasex = gasex_sub.drop(labels=['Tree_Number', 'Sampling_Day', '95th_Perc_Sap_Rate'], axis=1)
features_gasex = features_gasex[['Mean_VWC', 'VPD_kPa', 'SlrW_Avg', 'Mean_Uptake_Depth', 'Mean_Predawn_LWP', 'Mean_Midday_LWP', 'Cumulative_Leaf_Loss', 'Day_Mean_Sap_Rate', '95th_SWC', '5th_SWC', 'A', 'gsw']]
df_gasex_mi = features_gasex.corr(method=custom_mi_reg)
#dummy_ones_gasex = np.ones_like(df_gasex_mi)
#matrix_gasex = np.triu(dummy_ones_gasex)
#sns.heatmap(df_gasex_mi, annot=True, fmt=".2f", linewidth=.5, mask=matrix_gasex)

df_gasex_mi.update(df_full_mi) ###join all
df_gasex_mi.update(df_long_mi_recovery)
dummy_ones_gasex = np.ones_like(df_gasex_mi)
matrix_gasex = np.triu(dummy_ones_gasex)
plt.figure(figsize=(10,8))
sns.heatmap(df_gasex_mi, annot=True, fmt=".2f", linewidth=.5, mask=matrix_gasex, vmin=0, vmax=1.5)
plt.tight_layout()
plt.savefig('Output_Plots/MI_matrix_recovery.pdf')
plt.show()