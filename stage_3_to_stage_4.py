from utility_functions import sap_flow_S3_to_S4, met_S3_to_S4, gasex_S3_to_S4, isotopes_S3_to_S4, \
    sap_install_S3_to_S4, soil_moisture_S3_to_S4, lwp_S3_to_S4, leaf_area_S3_to_S4, root_S3_to_S4, gasex_curves_S3_to_S4, \
    plot_data_stage
'''
This script takes the Stage 3 data products and does further cleaning and corrections to create the Stage 4 (final) data
'''
def S3_to_S4(input_folder = 'Stage_3_Data', output_folder='Stage_4_Data', plot_results=False):

    sap_flow_df = sap_flow_S3_to_S4(input_folder)
    gasex_df = gasex_S3_to_S4(input_folder) #doesn't do anything
    met_df = met_S3_to_S4(input_folder)
    soil_moisture_df = soil_moisture_S3_to_S4(input_folder) #currently doesn't do any cleaning
    lwp_df = lwp_S3_to_S4(input_folder) #currently doesn't do any cleaning
    isotopes_df = isotopes_S3_to_S4(input_folder) #currently doesn't do any cleaning
    sap_install_df = sap_install_S3_to_S4(input_folder)
    leaf_area_df = leaf_area_S3_to_S4(input_folder)
    root_df = root_S3_to_S4(input_folder)
    gasex_curves_df = gasex_curves_S3_to_S4(input_folder)


    #Saves as pickles (to use most easily in next step) and csv files (for recordkeeping)
    sap_flow_df.to_pickle('%s/sap_flow.pkl' % output_folder)
    sap_flow_df.to_csv('%s/csv_Files/sap_flow.csv' % output_folder)
    gasex_df.to_pickle('%s/gasex.pkl' % output_folder)
    gasex_df.to_csv('%s/csv_Files/gasex.csv' % output_folder)
    met_df.to_pickle('%s/met.pkl' % output_folder)
    met_df.to_csv('%s/csv_Files/met.csv' % output_folder)
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
        plot_data_stage(sap_flow=sap_flow_df, soil_moisture=soil_moisture_df, lwp=lwp_df, gasex=gasex_df,
                        met=met_df,isotopes=isotopes_df, stage=4)

if __name__ == '__main__':
    S3_to_S4(input_folder='Stage_3_Data', output_folder='Stage_4_Data', plot_results=False)