from utility_functions import import_raw_sapflow, import_raw_met, import_raw_gas_exchange, \
    import_raw_lwp, import_raw_soil_moisture, import_raw_isotopes, import_raw_install, import_raw_leaf_area, \
    import_raw_root_data, import_raw_gasex_curve
'''
This script imports the raw data from the raw data folder, formats the data into machine-friendly formats
and then saves the "Stage 1" data as both csv and python formatted pickle files
NOTE: some data sheets had symbols in them that do not get translated to python files
'''
def raw_to_S1(input_folder = 'Raw_Data', output_folder='Stage_1_Data'):

    #imports data and formats to dataframes
    sap_flow_df= import_raw_sapflow(input_folder)
    met_df = import_raw_met(input_folder)
    gasex_df = import_raw_gas_exchange(input_folder)
    lwp_df = import_raw_lwp(input_folder)
    soil_moisture_df = import_raw_soil_moisture(input_folder)
    isotopes_df = import_raw_isotopes(input_folder)
    sap_install_df = import_raw_install(input_folder)
    leaf_area_df = import_raw_leaf_area(input_folder)
    root_df = import_raw_root_data(input_folder)
    gasex_curves_df = import_raw_gasex_curve(input_folder)

    #Saves as pickles (to use most easily in next step) and csv files (for recordkeeping)
    sap_flow_df.to_pickle('%s/sap_flow.pkl' % output_folder)
    sap_flow_df.to_csv('%s/csv_Files/sap_flow.csv' % output_folder)
    met_df.to_pickle('%s/met.pkl' % output_folder)
    met_df.to_csv('%s/csv_Files/met.csv' % output_folder)
    gasex_df.to_pickle('%s/gasex.pkl' % output_folder)
    gasex_df.to_csv('%s/csv_Files/gasex.csv' % output_folder)
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
    gasex_curves_df.to_csv('%s/csv_Files/gasex_curves.csv' % output_folder)

if __name__ == '__main__':
    raw_to_S1(input_folder='Raw_Data', output_folder='Stage_1_Data')