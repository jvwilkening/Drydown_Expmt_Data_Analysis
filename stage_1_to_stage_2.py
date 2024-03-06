from utility_functions import sap_flow_S1_to_S2, gasex_S1_to_S2, met_S1_to_S2, psych_S1_to_S2, soil_moisture_S1_to_S2, \
    lwp_S1_to_S2, isotopes_S1_to_S2, sap_install_S1_to_S2, leaf_area_S1_to_S2, root_S1_to_S2, gasex_curves_S1_to_S2, \
    plot_data_stage
'''
This script cleans the stage 1 datasets (subsets to date range of experiment, corrects mislabeled data, removes
obviously incorrect datapoints, etc.) and saves results as Stage 2 Data
'''
def S1_to_S2(input_folder = 'Stage_1_Data', output_folder='Stage_2_Data', plot_results=False):

    sap_flow_df = sap_flow_S1_to_S2(input_folder)
    gasex_df = gasex_S1_to_S2(input_folder)
    met_df = met_S1_to_S2(input_folder)
    psych_df = psych_S1_to_S2(input_folder)
    soil_moisture_df = soil_moisture_S1_to_S2(input_folder) #currently doesn't do any cleaning
    lwp_df = lwp_S1_to_S2(input_folder) #currently doesn't do any cleaning
    isotopes_df = isotopes_S1_to_S2(input_folder) #currently doesn't do any cleaning
    sap_install_df = sap_install_S1_to_S2(input_folder)
    leaf_area_df = leaf_area_S1_to_S2(input_folder) #currently doesn't do any cleaning
    root_df = root_S1_to_S2(input_folder)  # currently doesn't do any cleaning
    gasex_curves_df = gasex_curves_S1_to_S2(input_folder)


    #Saves as pickles (to use most easily in next step) and csv files (for recordkeeping)
    sap_flow_df.to_pickle('%s/sap_flow.pkl' % output_folder)
    sap_flow_df.to_csv('%s/csv_Files/sap_flow.csv' % output_folder)
    gasex_df.to_pickle('%s/gasex.pkl' % output_folder)
    gasex_df.to_csv('%s/csv_Files/gasex.csv' % output_folder)
    met_df.to_pickle('%s/met.pkl' % output_folder)
    met_df.to_csv('%s/csv_Files/met.csv' % output_folder)
    psych_df.to_pickle('%s/psych.pkl' % output_folder)
    psych_df.to_csv('%s/csv_Files/psych.csv' % output_folder)
    lwp_df.to_pickle('%s/lwp.pkl' % output_folder)
    lwp_df.to_csv('%s/csv_Files/lwp.csv' % output_folder)
    soil_moisture_df.to_pickle('%s/soil_moisture.pkl' % output_folder)
    soil_moisture_df.to_csv('%s/csv_Files/soil_moisture.csv' % output_folder)
    isotopes_df.to_pickle('%s/isotopes.pkl' % output_folder)
    isotopes_df.to_csv('%s/csv_Files/isotopes.csv' % output_folder)
    sap_install_df.to_pickle('%s/sap_install.pkl' % output_folder)
    sap_install_df.to_csv('%s/csv_Files/sap_install.csv' % output_folder)
    leaf_area_df.to_pickle('%s/leaf_area.pkl' % output_folder)
    leaf_area_df.to_csv('%s/csv_Files/leaf_area.csv' % output_folder)
    root_df.to_pickle('%s/root_dist.pkl' % output_folder)
    root_df.to_csv('%s/csv_Files/root_dist.csv' % output_folder)
    gasex_curves_df.to_pickle('%s/gasex_curves.pkl' % output_folder)
    gasex_curves_df.to_csv('%s/gasex_curves.csv' % output_folder)

    if plot_results == True:
        plot_data_stage(sap_flow=sap_flow_df, psych=psych_df, soil_moisture=soil_moisture_df, lwp=lwp_df, gasex=gasex_df,
                        met=met_df,isotopes=isotopes_df)

if __name__ == '__main__':
    S1_to_S2(input_folder='Stage_1_Data', output_folder='Stage_2_Data', plot_results=False)