import pandas as pd
import matplotlib.pyplot  as plt
import datetime as dt
import math
import scipy.optimize
from scipy.optimize import minimize
import numpy as np
from sklearn.feature_selection import mutual_info_regression


def custom_mi_reg(a, b, neighbors=2, random_seed=13):
    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)
    return  mutual_info_regression(a, b, n_neighbors=neighbors, random_state=random_seed)[0] # should return a float value

def classify_sampling_day(row):
    if row['TIMESTAMP'] >= pd.Timestamp('2021-07-08 00:00:00') and row['TIMESTAMP'] < pd.Timestamp('2021-07-09 00:00:00'):
        val = 1
    elif row['TIMESTAMP'] >= pd.Timestamp('2021-07-11 00:00:00') and row['TIMESTAMP'] < pd.Timestamp('2021-07-12 00:00:00'):
        val = 2
    elif row['TIMESTAMP'] >= pd.Timestamp('2021-07-14 00:00:00') and row['TIMESTAMP'] < pd.Timestamp('2021-07-15 00:00:00'):
        val = 3
    elif row['TIMESTAMP'] >= pd.Timestamp('2021-07-18 00:00:00') and row['TIMESTAMP'] < pd.Timestamp('2021-07-19 00:00:00'):
        val = 4
    elif row['TIMESTAMP'] >= pd.Timestamp('2021-07-21 00:00:00') and row['TIMESTAMP'] < pd.Timestamp('2021-07-22 00:00:00'):
        val = 5
    else:
        val = 'NaN'
    return val

def classify_lwps(row):
    row_time = dt.datetime.strptime(row['Time'], '%H:%M')
    pd_start = dt.datetime.strptime('5:00', '%H:%M')
    pd_end = dt.datetime.strptime('6:00', '%H:%M')
    md_start = dt.datetime.strptime('13:30', '%H:%M')
    md_end = dt.datetime.strptime('14:30', '%H:%M')

    if row_time.time() >= pd_start.time() and row_time.time() <= pd_end.time():
        val = 'Predawn'
    elif row_time.time() >= md_start.time() and row_time.time() <= md_end.time():
        val = 'Midday'
    else:
        val = 'Extra'
    return val

def remove_big_sap_diffs(df, allowable_Vh_diff_percent_max, allowable_Vh_diff_percent):
    df.reset_index(inplace=True)

    x = 0
    while x < (np.max(df.index) - 1):
        x_msmt = df.loc[x].Vh_Outer
        y = x + 1
        z = y + 1
        check_flag = False
        if x < (np.max(df.index) - 3) and x > 3:
            max_diff = allowable_Vh_diff_percent_max/100.0 * np.max(df.Vh_Outer[(x-3):(x+3)])
        elif x <3:
            max_diff = allowable_Vh_diff_percent_max / 100.0 * np.max(df.Vh_Outer[0:(x + 3)])
        else:
            remaining=np.max(df.index) - x
            max_diff = allowable_Vh_diff_percent_max / 100.0 * np.max(df.Vh_Outer[x:(x + remaining)])
        if max_diff < 10.0: max_diff = 10.0
        while check_flag == False:
            y_msmt = df.loc[y].Vh_Outer
            z_msmt = df.loc[z].Vh_Outer
            xy_diff = abs(x_msmt - y_msmt)
            if x_msmt >= 30.0: #only use percent diff when values aren't small
                xy_percent = xy_diff/abs(x_msmt)*100.0
            else:
                xy_percent = 0.0
            if z_msmt >= 30.0:
                yz_percent = abs(z_msmt - y_msmt)/abs(z_msmt)*100.0
            else:
                yz_percent = 0.0
            if xy_diff <= max_diff: #checks magnitude diff from prior step allowable & percent diff from either side
                x = y
                check_flag = True
            else:
                df=df.drop(index=y)
                y = y + 1
                z= y + 1
                if y > (np.max(df.index)-1):
                    x=np.max(df.index)
                    break
    return df

def hampel_filter_pandas(input_df, window_size, n_sigmas=3):
    k = 1.4826  # scale factor for Gaussian distribution
    #new_series = input_series.copy()

    # helper lambda function
    MAD = lambda x: np.median(np.abs(x - np.median(x)))

    rolling_median = input_df.Total_Sapflow.rolling(window=2 * window_size, center=True).median()
    rolling_mad = k * input_df.Total_Sapflow.rolling(window=2 * window_size, center=True).apply(MAD)
    diff = np.abs(input_df.Total_Sapflow - rolling_median)

    beg_index=np.min(input_df.index)
    end_index = np.max(input_df.index)

    index_list = input_df.index
    outlier_counter=0

    for i in index_list:
        if diff[i] > (n_sigmas * rolling_mad[i]):
            input_df.Total_Sapflow[i] = rolling_median[i]
            outlier_counter=outlier_counter+1

    print(outlier_counter)
    return input_df

def calc_VPD(met_df):
    met_df["SVP"] = 610.7*10.0**(7.5*met_df.AirTC/(237.3+met_df.AirTC)) #Saturation vapor pressure [Pa]
    met_df["VPD_kPa"] = (1.0-(met_df.RH/100.0))*met_df.SVP/1000.0 #Vapor pressure deficit [kPa]
    return met_df

def calculate_sap_water_flux(df, tree_number, install_df, only_outer=True):
    '''
    This calculation is based on the Implexx sap flow calculation spreadsheet. It calculates sap flux using the dual method
    approach
    :param df: sap flow data frame
    :param tree_number: tree number, used to reference morphology measurements
    :param install_df: dataframe with morphology data of sensor install site
    :param only_outer: if true, will calculate sap flux based only on outer sensor, if false also uses inner sensor
    :return: returns dataframe with new columns of calculated corrected heat velocity, sap flux density, and total sapflow [L/hr]

    '''

    #Get install site parameters for tree number
    TTD = install_df.loc[install_df['Tree_Number'] == tree_number, 'Stem_Diameter_cm'].iloc[0] # Diameter over bark where sensor installed [cm]
    BDD = install_df.loc[install_df['Tree_Number'] == tree_number, 'Bark_Depth_cm'].iloc[0] # Depth of bark where sensor installed [cm]
    SD = install_df.loc[install_df['Tree_Number'] == tree_number, 'Sapwood_Depth_cm'].iloc[0] # Depth of sapwood where sensor installed [cm]

    #Constants used in calculation
    x_d_o = 0.6                 # Distance between heater and downstream, outer thermister [cm]
    x_u_o = 0.6                 # Distance between heater and upstream, outer thermister [cm]
    x_d_i = 0.6                 # Distance between heater and downstream, inner thermister [cm]
    x_u_i = 0.6                 # Distance between heater and upstream, inner thermister [cm]
    outer_Peclet_Tmax = 33.0    # Tmax value to transition between slow and fast flow for outer position [s]
    inner_Peclet_Tmax = 33.0    # Tmax value to transition between slow and fast flow for inner position [s]
    w_f = 3.577/1000.0          # Sapwood fresh weight [kg]
    w_d = 1.8896/1000.0         # Sapwood dry weight [kg]
    v_f = 0.000003321           # Sapwood fresh volume [m3]
    rho = w_f / v_f             # Basic density of fresh wood [kg/m3]
    rho_d = w_d / v_f           # Basic density of dry wood [kg/m3]
    c_d = 1200.0                # Wood matrix specific heat capacity at 20C [J/kg/C]
    mc = (w_f - w_d)/w_d        # Gravimetric water content of sapwood [kg/kg]
    c_w = 4182.0                # Sap specific heat capacity at 20C [J/kg/C]
    rho_w = 1000.0              # Density of water [kg/m3]
    mc_FSP = 0.2*((rho_d*(1.0/rho_w))**(-0.5)) # Water content of sapwood at fiber saturation point [-]
    rho_cw = 1530.0             # Density of cell wall [kg/m3]
    Fv_FSP = 1.0 - (rho_d/rho_w)*((rho_w/rho_cw)+mc_FSP) # Void fraciton of wood at fiber saturation point from Vandergehucte & Steppe (2012) [-]
    K_w = 0.5984                # Thermal conductivity of water at 20C (Lide 1992) [J/kg/C]
    K_Van = K_w*(mc-mc_FSP)*(rho_d/rho_w) + 0.04186*(21.0-20.0*Fv_FSP) #Stem thermal conductivity at 20C from Vandegehuchte & Steppe (2012) [J/m/s/C]
    c = ((w_d*c_d)+c_w*(w_f-w_d))/w_f # Specific heat capacity of fresh wood at 20 C (Edwards and Warwick 1984) [J/kg/C]
    rho_c = rho * c             # Volumetric specific heat capacity of fresh wood at 20C [J/m3/C]
    k_van = (K_Van/rho_c) * 10000.0 # Thermal diffusivity in wood-sap matrix calculated from Vandegehuchte & Steppe (2012) [cm2/s]
    beta = 2.2817               # Correction parameter for wounding -value based on Burgess (2001) Table 1. Mean wounding of.26 cm, probe spacing 0.6

    #Calculate sapwood morphology parameters
    if SD < 1.51:
        heart_A = math.pi*((TTD/2.0)-BDD-SD)**2.0 #Heartwood area [cm2]
        inner_sap_A = 0.0 #Inner sapwood area [cm2]
    else:
        heart_A = math.pi * ((TTD / 2.0) - BDD - 2.0) ** 2.0
        inner_sap_A = math.pi*(((TTD/2.0) - BDD -1.0) ** 2.0) - heart_A
    outer_sap_A = math.pi * ((TTD/2.0) - BDD) ** 2.0 - heart_A - inner_sap_A #Outer sapwood area [cm2]
    total_sap_A = math.pi * ((TTD/2.0) - BDD) ** 2.0 #Total sapwood area assuming no heartwood [cm2]

    #Add new columns with NaN fillers for calculated results to dataframe
    df["DMA_inner"] = np.nan
    df["DMA_outer"] = np.nan
    df["Sap_Flux_Density_inner"] = np.nan
    df["Sap_Flux_Density_outer"] = np.nan
    df["Total_Sapflow"] = np.nan
    df["Flux_Method_inner"] = ""
    df["Flux_Method_outer"] = ""
    if only_outer == True:
        df["Sensors_used"] = "Outer"
    else:
        df["Sensors_used"] = "Outer+Inner"

    #Loop through all entries and calculate sap flow variables
    index_list = df.index

    for i in index_list:
        #calculate heat velocity at inner and outer sensors [cm/hr]
        if df.Tmax_Outer[i] < outer_Peclet_Tmax: #Uses Tmax method
            DMA_outer = beta * 3600.0 * np.sqrt((4.0*k_van/3.0)*np.log(1.0-(3.0/df.Tmax_Outer[i]))+((x_d_o**2.0)/(df.Tmax_Outer[i]*(df.Tmax_Outer[i]-3.0))))
            df.Flux_Method_outer[i] = "Tmax"
        else: #Uses HRM method
            DMA_outer =beta * 3600.0 *(2.0*k_van/(x_d_o+x_u_o)*df.Alpha_Outer[i] + (x_d_o - x_u_o)/(2.0*58.5))
            df.Flux_Method_outer[i] = "HRM"

        if df.Tmax_Inner[i] < inner_Peclet_Tmax: #Uses Tmax method
            DMA_inner = beta * 3600.0 * np.sqrt((4.0*k_van/3.0)*np.log(1.0-(3.0/df.Tmax_Inner[i]))+((x_d_i**2.0)/(df.Tmax_Inner[i]*(df.Tmax_Inner[i]-3.0))))
            df.Flux_Method_inner[i] = "Tmax"
        else: #Uses HRM method
            DMA_inner =beta * 3600.0 *(2.0*k_van/(x_d_i+x_u_i)*df.Alpha_Inner[i] + (x_d_i - x_u_i)/(2.0*58.5))
            df.Flux_Method_inner[i] = "HRM"
        df.DMA_inner[i] = DMA_inner
        df.DMA_outer[i] = DMA_outer

        #calculate sap flux density at inner and outer sensors [cm3/cm2/hr]
        J_outer = (DMA_outer * rho_d * (c_d + (mc + c_w))) / (rho_w * c_w)
        J_inner = (DMA_inner * rho_d * (c_d + (mc + c_w))) / (rho_w * c_w)
        df.Sap_Flux_Density_inner[i] = J_inner
        df.Sap_Flux_Density_outer[i] = J_outer

        #Calculate total sapflow [L/hr]
        if only_outer == True:
            Total_Sapflow = (J_outer * total_sap_A) / 1000.0
        else:
            Total_Sapflow = ((J_outer * outer_sap_A) + (J_inner * inner_sap_A)) / 1000.0
        df.Total_Sapflow[i] = Total_Sapflow

        #rename column that sensor initially calculated sap flow in - doesn't include correct tree parameters
        df = df.rename(columns={"TotalSapFlow": "Sensor_calc_flow"})

    return df

def correct_sap_flow_baseline(df):
    min_sapflow = np.min(df.Total_Sapflow)
    df.Total_Sapflow = df.Total_Sapflow - min_sapflow

    return df

def soil_wrc(vwc):
    #Takes soil volumetric water content value (%) and returns soil water potential [MPa] from water retention curve

    #Bimodel van genuchten parameter fit using soilHyP package in R
    vg_df = pd.read_csv('Raw_Data/Soil_WRC/vg_parameters_2022.csv', header=0)

    cols=[i for i in vg_df.columns if i not in ["Row"]]
    for col in cols:
        vg_df[col]=pd.to_numeric(vg_df[col])

    #Two replicate curves were measured. Function calculates psi_soil using both and returns mean

    theta_h1 = vwc / 100.0
    theta_r1 = vg_df.iloc[0]['thr']
    theta_s1 = vg_df.iloc[0]['ths']
    w_21 = vg_df.iloc[0]['w2']
    alpha_11 = vg_df.iloc[0]['alfa']
    n_11 = vg_df.iloc[0]['n']
    alpha_21 = vg_df.iloc[0]['alfa2']
    n_21 = vg_df.iloc[0]['n2']

    theta_h2 = vwc / 100.0
    theta_r2 = vg_df.iloc[1]['thr']
    theta_s2 = vg_df.iloc[1]['ths']
    w_22 = vg_df.iloc[1]['w2']
    alpha_12 = vg_df.iloc[1]['alfa']
    n_12 = vg_df.iloc[1]['n']
    alpha_22 = vg_df.iloc[1]['alfa2']
    n_22 = vg_df.iloc[1]['n2']

    def wrc1(h):
        y1 = -theta_h1 + theta_r1 + (theta_s1 - theta_r1) * ((1.0-w_21)*(1.0/(1.0+(alpha_11*h)**(n_11)))**(1.0-(1.0/n_11)) + w_21*(1.0/(1.0+(alpha_21*h)**(n_21)))**(1.0-(1.0/n_21)))
        return y1

    def wrc2(h):
        y2 = -theta_h2 + theta_r2 + (theta_s2 - theta_r2) * ((1.0-w_22)*(1.0/(1.0+(alpha_12*h)**(n_12)))**(1.0-(1.0/n_12)) + w_22*(1.0/(1.0+(alpha_22*h)**(n_22)))**(1.0-(1.0/n_22)))
        return y2

    h1 = scipy.optimize.fsolve(wrc1, [100.0])
    psi1 = -h1 / 100.0 /101.99773339984 #convert from cm head to MPa

    h2 = scipy.optimize.fsolve(wrc2, [100.0])
    psi2 = -h2 / 100.0 / 101.99773339984

    psi_mean = (psi1 + psi2)/2.0
    psi_mean_float = psi_mean[0]

    return psi_mean_float

def import_raw_sapflow(raw_data_folder):
    #import sapflow data
    sap_flow_df  = pd.read_csv('%s/Sapflow_Sensors/CR1000_5_Sap_Flow_Data.csv' % raw_data_folder, skiprows=[0,2,3], header = 0)
    #format timestamp
    sap_flow_df['TIMESTAMP'] = pd.to_datetime(sap_flow_df['TIMESTAMP'], format="%Y-%m-%d %H:%M:%S")
    #format values as floats
    cols = sap_flow_df.columns
    sap_flow_df[cols[1:]] = sap_flow_df[cols[1:]].apply(pd.to_numeric, errors='coerce')

    # repeat process for stem water content data
    stem_water_df = pd.read_csv('%s/Sapflow_Sensors/CR1000_5_Stem_Water_Data.csv' % raw_data_folder, skiprows=[0, 2, 3],
                                header=0)
    stem_water_df['TIMESTAMP'] = pd.to_datetime(stem_water_df['TIMESTAMP'], format="%Y-%m-%d %H:%M:%S")
    cols = stem_water_df.columns
    stem_water_df[cols[1:]] = stem_water_df[cols[1:]].apply(pd.to_numeric, errors='coerce')

    #merge sapflow and stem water content measurements
    sap_flow_df = sap_flow_df.merge(stem_water_df, on=['TIMESTAMP', 'RECORD'], how='outer')

    #for some reason merging creates duplicate rows??
    sap_flow_df = sap_flow_df.drop_duplicates()

    #subset by tree number
    sap_flow_df_1 = sap_flow_df[["TIMESTAMP", "RECORD", "TotalSapFlow_a", "Vh_Outer_a", "Vh_Inner_a", "Alpha_Outer_a",\
                                 "Alpha_Inner_a", "Beta_Outer_a", "Beta_Inner_a", "Tmax_Outer_a", "Tmax_Inner_a",\
                                 "SWC_Outer_a", "SWC_Inner_a", "Heat_Capacity_Outer_a", "Heat_Capacity_Inner_a",\
                                 "Pulse_Energy_a", "Pulse_Duration_a"]]
    sap_flow_df_1['Tree_Number'] = 1
    sap_flow_df_1.columns = sap_flow_df_1.columns.str.replace("_a", "")

    sap_flow_df_2 = sap_flow_df[["TIMESTAMP", "RECORD", "TotalSapFlow_b", "Vh_Outer_b", "Vh_Inner_b", "Alpha_Outer_b", \
                                 "Alpha_Inner_b", "Beta_Outer_b", "Beta_Inner_b", "Tmax_Outer_b", "Tmax_Inner_b",\
                                 "SWC_Outer_b", "SWC_Inner_b", "Heat_Capacity_Outer_b", "Heat_Capacity_Inner_b",\
                                 "Pulse_Energy_b", "Pulse_Duration_b"]]
    sap_flow_df_2['Tree_Number'] = 2
    sap_flow_df_2.columns = sap_flow_df_2.columns.str.replace("_b", "")

    sap_flow_df_3 = sap_flow_df[["TIMESTAMP", "RECORD", "TotalSapFlow_c", "Vh_Outer_c", "Vh_Inner_c", "Alpha_Outer_c", \
                                 "Alpha_Inner_c", "Beta_Outer_c", "Beta_Inner_c", "Tmax_Outer_c", "Tmax_Inner_c",\
                                 "SWC_Outer_c", "SWC_Inner_c", "Heat_Capacity_Outer_c", "Heat_Capacity_Inner_c",\
                                 "Pulse_Energy_c", "Pulse_Duration_c"]]
    sap_flow_df_3['Tree_Number'] = 3
    sap_flow_df_3.columns = sap_flow_df_3.columns.str.replace("_c", "")

    sap_flow_df_4 = sap_flow_df[["TIMESTAMP", "RECORD", "TotalSapFlow_d", "Vh_Outer_d", "Vh_Inner_d", "Alpha_Outer_d", \
                                 "Alpha_Inner_d", "Beta_Outer_d", "Beta_Inner_d", "Tmax_Outer_d", "Tmax_Inner_d",\
                                 "SWC_Outer_d", "SWC_Inner_d", "Heat_Capacity_Outer_d", "Heat_Capacity_Inner_d",\
                                 "Pulse_Energy_d", "Pulse_Duration_d"]]
    sap_flow_df_4['Tree_Number'] = 4
    sap_flow_df_4.columns = sap_flow_df_4.columns.str.replace("_d", "")

    sap_flow_df_5 = sap_flow_df[["TIMESTAMP", "RECORD", "TotalSapFlow_e", "Vh_Outer_e", "Vh_Inner_e", "Alpha_Outer_e", \
                                 "Alpha_Inner_e", "Beta_Outer_e", "Beta_Inner_e", "Tmax_Outer_e", "Tmax_Inner_e",\
                                 "SWC_Outer_e", "SWC_Inner_e", "Heat_Capacity_Outer_e", "Heat_Capacity_Inner_e",\
                                 "Pulse_Energy_e", "Pulse_Duration_e"]]
    sap_flow_df_5['Tree_Number'] = 5
    sap_flow_df_5.columns = sap_flow_df_5.columns.str.replace("_e", "")

    sap_flow_df_6 = sap_flow_df[["TIMESTAMP", "RECORD", "TotalSapFlow_f", "Vh_Outer_f", "Vh_Inner_f", "Alpha_Outer_f", \
                                 "Alpha_Inner_f", "Beta_Outer_f", "Beta_Inner_f", "Tmax_Outer_f", "Tmax_Inner_f",\
                                 "SWC_Outer_f", "SWC_Inner_f", "Heat_Capacity_Outer_f", "Heat_Capacity_Inner_f",\
                                 "Pulse_Energy_f", "Pulse_Duration_f"]]
    sap_flow_df_6['Tree_Number'] = 6
    sap_flow_df_6.columns = sap_flow_df_6.columns.str.replace("_f", "")

    sap_flow_combined_df=pd.concat([sap_flow_df_1, sap_flow_df_2, sap_flow_df_3, sap_flow_df_4, sap_flow_df_5, sap_flow_df_6], ignore_index=True,
              sort=False)

    return sap_flow_combined_df

def import_raw_install(raw_data_folder):
    install_df = pd.read_csv('%s/Sapflow_Sensors/sapflow_install.csv' % raw_data_folder, header=0)
    cols = [i for i in install_df.columns]
    for col in cols:
        install_df[col] = pd.to_numeric(install_df[col])
    return install_df

def import_raw_met(raw_data_folder):
    met_df = pd.read_csv('%s/Met_Station/CR1000_3_Table1.csv' % raw_data_folder, skiprows=[0, 2, 3], header=0)
    met_df['TIMESTAMP'] = pd.to_datetime(met_df['TIMESTAMP'], format="%Y-%m-%d %H:%M:%S")
    cols = met_df.columns
    met_df[cols[1:]] = met_df[cols[1:]].apply(pd.to_numeric, errors='coerce')
    return met_df


def import_raw_gas_exchange(raw_data_folder):
    #import csv version of licor data sheet
    Day_1_AM_df = pd.read_csv('%s/Licor/2021-07-08-0954_logdata_AM_msmts.csv' % raw_data_folder,
                              skiprows=[0,1,2,3,4,5,6,7,8,9,10,11,12,14], header=0, encoding_errors='ignore')
    #create timestamp that python recognizes
    Day_1_AM_df['TIMESTAMP'] = pd.to_datetime(Day_1_AM_df['date'], format="%Y%m%d %H:%M:%S")
    Day_1_AM_df['Type'] = 'AM'
    Day_1_AM_df['Sampling_Day'] = 1
    #Format other numerical data as floats
    cols = [i for i in Day_1_AM_df.columns if i not in ["TIMESTAMP", "date", "Geometry", "hhmmss", "State", "Type"]]
    for col in cols:
        Day_1_AM_df[col] = pd.to_numeric(Day_1_AM_df[col])

    # repeat for next data sheet
    Day_1_PM_df = pd.read_csv('%s/Licor/2021-07-08-1416_logdata_PM_msmts.csv' % raw_data_folder,
                              skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14], header=0, encoding_errors='ignore')
    Day_1_PM_df['TIMESTAMP'] = pd.to_datetime(Day_1_PM_df['date'], format="%Y%m%d %H:%M:%S")
    Day_1_PM_df['Type'] = 'PM'
    Day_1_PM_df['Sampling_Day'] = 1
    cols = [i for i in Day_1_PM_df.columns if i not in ["TIMESTAMP", "date", "Geometry", "hhmmss", "State", "Type"]]
    for col in cols:
        Day_1_PM_df[col] = pd.to_numeric(Day_1_PM_df[col])

    # repeat for next data sheet
    Day_2_AM_df = pd.read_csv('%s/Licor/2021-07-11-1027_logdata_AM_msmts.csv' % raw_data_folder,
                              skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14], header=0, encoding_errors='ignore')
    Day_2_AM_df['TIMESTAMP'] = pd.to_datetime(Day_2_AM_df['date'], format="%Y%m%d %H:%M:%S")
    Day_2_AM_df['Type'] = 'AM'
    Day_2_AM_df['Sampling_Day'] = 2
    cols = [i for i in Day_2_AM_df.columns if i not in ["TIMESTAMP", "date", "Geometry", "hhmmss", "State", "Type"]]
    for col in cols:
        Day_2_AM_df[col] = pd.to_numeric(Day_2_AM_df[col])

    # repeat for next data sheet
    Day_2_PM_df = pd.read_csv('%s/Licor/2021-07-11-1437_logdata_PM_msmts.csv' % raw_data_folder,
                              skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14], header=0, encoding_errors='ignore')
    Day_2_PM_df['TIMESTAMP'] = pd.to_datetime(Day_2_PM_df['date'], format="%Y%m%d %H:%M:%S")
    Day_2_PM_df['Type'] = 'PM'
    Day_2_PM_df['Sampling_Day'] = 2
    cols = [i for i in Day_2_PM_df.columns if i not in ["TIMESTAMP", "date", "Geometry", "hhmmss", "State", "Type"]]
    for col in cols:
        Day_2_PM_df[col] = pd.to_numeric(Day_2_PM_df[col])

    # repeat for next data sheet
    Day_4_AM_df = pd.read_csv('%s/Licor/2021-07-18-1003_logdata_AM_msmts.csv' % raw_data_folder,
                              skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14], header=0, encoding_errors='ignore')
    Day_4_AM_df['TIMESTAMP'] = pd.to_datetime(Day_4_AM_df['date'], format="%Y%m%d %H:%M:%S")
    Day_4_AM_df['Type'] = 'AM'
    Day_4_AM_df['Sampling_Day'] = 4
    cols = [i for i in Day_4_AM_df.columns if i not in ["TIMESTAMP", "date", "Geometry", "hhmmss", "State", "Type"]]
    for col in cols:
        Day_4_AM_df[col] = pd.to_numeric(Day_4_AM_df[col])

    # repeat for next data sheet
    Day_4_PM_df = pd.read_csv('%s/Licor/2021-07-18-1435_logdata_PM_msmts.csv' % raw_data_folder,
                              skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14], header=0, encoding_errors='ignore')
    Day_4_PM_df['TIMESTAMP'] = pd.to_datetime(Day_4_PM_df['date'], format="%Y%m%d %H:%M:%S")
    Day_4_PM_df['Type'] = 'PM'
    Day_4_PM_df['Sampling_Day'] = 4
    cols = [i for i in Day_4_PM_df.columns if i not in ["TIMESTAMP", "date", "Geometry", "hhmmss", "State", "Type"]]
    for col in cols:
        Day_4_PM_df[col] = pd.to_numeric(Day_4_PM_df[col])

    gasex_df = pd.concat([Day_1_AM_df, Day_1_PM_df, Day_2_AM_df, Day_2_PM_df, Day_4_AM_df, Day_4_PM_df],
                         ignore_index=True, sort=False)

    gasex_df = gasex_df.rename({'tree_number': 'Tree_Number'}, axis='columns')


    return gasex_df

def import_raw_gasex_curve(raw_data_folder):
    #import csv version of licor data sheet
    ACi_Curves_df = pd.read_csv('%s/GasEx_Curves/2021-06-22-1445_logdata_pop_ACi.csv' % raw_data_folder,
                              skiprows=[0,1,2,3,4,5,6,7,8,9,10,11,12,14], header=0, encoding_errors='ignore')
    #create timestamp that python recognizes
    ACi_Curves_df['TIMESTAMP'] = pd.to_datetime(ACi_Curves_df['date'], format="%Y%m%d %H:%M:%S")
    ACi_Curves_df['Curve_Type'] = 'ACi'
    #Format other numerical data as floats
    cols = [i for i in ACi_Curves_df.columns if i not in ["TIMESTAMP", "date", "Geometry", "hhmmss", "State", "Curve_Type"]]
    for col in cols:
        ACi_Curves_df[col] = pd.to_numeric(ACi_Curves_df[col])

    Light_Curves_df = pd.read_csv('%s/GasEx_Curves/2021-06-29-1113_logdata_pop_light_curves.csv' % raw_data_folder,
                                skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14], header=0,
                                encoding_errors='ignore')
    # create timestamp that python recognizes
    Light_Curves_df['TIMESTAMP'] = pd.to_datetime(Light_Curves_df['date'], format="%Y%m%d %H:%M:%S")
    Light_Curves_df['Curve_Type'] = 'Light'
    # Format other numerical data as floats
    cols = [i for i in Light_Curves_df.columns if
            i not in ["TIMESTAMP", "date", "Geometry", "hhmmss", "State", "Curve_Type"]]
    for col in cols:
        Light_Curves_df[col] = pd.to_numeric(Light_Curves_df[col])

    gasex_curves_df = pd.concat([ACi_Curves_df, Light_Curves_df],
                         ignore_index=True, sort=False)

    gasex_curves_df = gasex_curves_df.rename({'tree_number': 'Tree_Number'}, axis='columns')

    return gasex_curves_df


def import_raw_lwp(raw_data_folder):
    lwp_df = pd.read_csv('%s/Leaf_Water_Potentials/LWP_measurements.csv' % raw_data_folder, header=0)
    lwp_df['TIMESTAMP'] = pd.to_datetime(lwp_df['Date'].apply(str) + ' ' + lwp_df['Time'])
    lwp_df['TIMESTAMP'] = pd.to_datetime(lwp_df['TIMESTAMP'], format="%Y-%m-%d %H:%M:%S")
    cols = [i for i in lwp_df.columns if i not in ["TIMESTAMP", "Date", "Time", "Quality_Notes"]]
    for col in cols:
        lwp_df[col] = pd.to_numeric(lwp_df[col])
    lwp_df['Sampling_Day'] = lwp_df.apply(classify_sampling_day, axis=1)
    lwp_df['LWP_Type'] = lwp_df.apply(classify_lwps, axis=1)
    return lwp_df

def import_raw_leaf_area(raw_data_folder):
    leaf_area_df = pd.read_csv('%s/Leaf_area_collected/collected_leaf_area.csv' % raw_data_folder, header=0)
    leaf_area_df['TIMESTAMP'] = pd.to_datetime(leaf_area_df['Date_Collected'], format="%m/%d/%y")
    cols = [i for i in leaf_area_df.columns if i not in ["TIMESTAMP", "Date_Collected", "Notes"]]
    for col in cols:
        leaf_area_df[col] = pd.to_numeric(leaf_area_df[col])
    return leaf_area_df

def import_raw_root_data(raw_data_folder):
    root_df = pd.read_csv('%s/Root_Data/Root_distribution.csv' % raw_data_folder, header=0)
    cols = [i for i in root_df.columns]
    for col in cols:
        root_df[col] = pd.to_numeric(root_df[col])
    return root_df

def import_raw_soil_moisture(raw_data_folder):
    soil_moisture_df = pd.read_csv('%s/Soil_Moisture/soil_moisture_measurements.csv' % raw_data_folder, header=0)
    soil_moisture_df['TIMESTAMP'] = pd.to_datetime(soil_moisture_df['Date'].apply(str) + ' ' + soil_moisture_df['Time'])
    cols = [i for i in soil_moisture_df.columns if i not in ["TIMESTAMP", "Date", "Time"]]
    for col in cols:
        soil_moisture_df[col] = pd.to_numeric(soil_moisture_df[col])
    soil_moisture_df['Sampling_Day'] = soil_moisture_df.apply(classify_sampling_day, axis=1)
    return soil_moisture_df

def import_raw_isotopes(raw_data_folder):
    isotopes_df = pd.read_csv('%s/Isotopes/compiled_isotope_data.csv' % raw_data_folder, header=0)
    isotopes_df['TIMESTAMP'] = pd.to_datetime(isotopes_df['TIMESTAMP'], format="%m/%d/%y %H:%M")
    cols = [i for i in isotopes_df.columns if i not in ["TIMESTAMP", "Sample_ID", "Analysis_Run", "Sample_Type"]]
    for col in cols:
        isotopes_df[col] = pd.to_numeric(isotopes_df[col])
    isotopes_df['Sampling_Day'] = isotopes_df.apply(classify_sampling_day, axis=1)
    return isotopes_df

def sap_flow_S1_to_S2(input_folder):
    #load pickled dataframe from prior stage
    sap_flow_df = pd.read_pickle('%s/sap_flow.pkl' % input_folder)

    #remove data from outside date range of experiment
    sap_flow_df = sap_flow_df[sap_flow_df['TIMESTAMP'] >= "2021-07-06 00:00:00"]
    sap_flow_df = sap_flow_df[sap_flow_df['TIMESTAMP'] <= "2021-07-27 23:59:59"]

    #remove data points with blatantly incorrect readings
    sap_flow_df = sap_flow_df[sap_flow_df['Vh_Outer'] > -2.0]
    sap_flow_df = sap_flow_df[sap_flow_df['SWC_Outer'] > -2.0]

    #Tree 4 sensor uninstalled earlier
    sap_flow_4 = sap_flow_df[sap_flow_df['Tree_Number'] == 4]
    sap_flow_all_others = sap_flow_df[sap_flow_df['Tree_Number'] != 4]
    sap_flow_4 = sap_flow_4[sap_flow_4['TIMESTAMP'] <= "2021-07-27 06:00:00"]

    sap_flow_df = pd.concat([sap_flow_4, sap_flow_all_others],
                            ignore_index=True, sort=False)

    return sap_flow_df

def gasex_S1_to_S2(input_folder):
    #load pickled dataframe from prior stage
    gasex_df = pd.read_pickle('%s/gasex.pkl' % input_folder)

    #fix datapoints where tree number was mislabeled
    gasex_df.loc[(gasex_df['TIMESTAMP'] == "2021-07-17 14:49:19"), 'Tree_Number'] = 1
    gasex_df.loc[(gasex_df['TIMESTAMP'] == "2021-07-17 14:51:40"), 'Tree_Number'] = 1
    gasex_df.loc[(gasex_df['TIMESTAMP'] == "2021-07-07 14:31:57"), 'Tree_Number'] = 1

    #Delete accidental duplicate reading
    gasex_df.drop(gasex_df[gasex_df.TIMESTAMP == "2021-07-07 14:32:54"].index, inplace=True)

    #Remove points with negative Ci value that indicates bad measurement
    gasex_df = gasex_df[gasex_df['Ci'] >= 0.0]

    return gasex_df

def gasex_curves_S1_to_S2(input_folder):
    # load pickled dataframe from prior stage
    gasex_curves_df = pd.read_pickle('%s/gasex_curves.pkl' % input_folder)

    #Remove NaN labelled points that aren't part of response curves
    gasex_curves_df = gasex_curves_df[gasex_curves_df['Curve_Rep'].notna()]

    return gasex_curves_df

def met_S1_to_S2(input_folder):
    #load pickled dataframe from prior stage
    met_df = pd.read_pickle('%s/met.pkl' % input_folder)

    #remove data from outside date range of experiment
    met_df = met_df[met_df['TIMESTAMP'] >= "2021-07-06 00:00:00"]
    met_df = met_df[met_df['TIMESTAMP'] <= "2021-07-27 23:59:59"]

    return met_df

def lwp_S1_to_S2(input_folder):
    #load pickled dataframe from prior stage
    lwp_df = pd.read_pickle('%s/lwp.pkl' % input_folder)

    #no data cleaning required

    return lwp_df

def root_S1_to_S2(input_folder):
    #load pickled dataframe from prior stage
    root_df = pd.read_pickle('%s/root_dist.pkl' % input_folder)

    #no data cleaning required

    return root_df

def leaf_area_S1_to_S2(input_folder):
    #load pickled dataframe from prior stage
    leaf_area_df = pd.read_pickle('%s/leaf_area.pkl' % input_folder)

    #no data cleaning required

    return leaf_area_df

def soil_moisture_S1_to_S2(input_folder):
    #load pickled dataframe from prior stage
    soil_moisture_df = pd.read_pickle('%s/soil_moisture.pkl' % input_folder)

    #no data cleaning required

    return soil_moisture_df

def isotopes_S1_to_S2(input_folder):
    #load pickled dataframe from prior stage
    isotopes_df = pd.read_pickle('%s/isotopes.pkl' % input_folder)

    #no data cleaning required

    return isotopes_df

def sap_install_S1_to_S2(input_folder):
    # load pickled dataframe from prior stage
    sap_install_df = pd.read_pickle('%s/sap_install.pkl' % input_folder)

    # no data cleaning required

    return sap_install_df

def root_S2_to_S3(input_folder):
    # load pickled dataframe from prior stage
    root_df = pd.read_pickle('%s/root_dist.pkl' % input_folder)

    #calculate % of root in each layer
    Tree_totals = root_df.groupby(['Tree'])['Mass_g'].sum()
    total_df = Tree_totals.to_frame()
    root_df = root_df.merge(total_df, on=['Tree'])
    root_df.rename(columns={'Mass_g_x': 'Mass_g', 'Mass_g_y': 'Total_tree_mass_g'}, inplace=True)
    root_df['Root_percent'] = root_df['Mass_g'] / root_df['Total_tree_mass_g'] * 100.0
    return root_df

def gasex_curves_S2_to_S3(input_folder):
    # load pickled dataframe from prior stage
    gasex_curves_df = pd.read_pickle('%s/gasex_curves.pkl' % input_folder)

    #no derived variables
    return gasex_curves_df

def sap_flow_S2_to_S3(input_folder):
    # load pickled dataframe from prior stage
    sap_flow_df = pd.read_pickle('%s/sap_flow.pkl' % input_folder)

    # load install data frame
    install_df = pd.read_pickle('%s/sap_install.pkl' % input_folder)

    #divide into separate trees
    sap_flow_1 = sap_flow_df[sap_flow_df['Tree_Number'] == 1]
    sap_flow_2 = sap_flow_df[sap_flow_df['Tree_Number'] == 2]
    sap_flow_3 = sap_flow_df[sap_flow_df['Tree_Number'] == 3]
    sap_flow_4 = sap_flow_df[sap_flow_df['Tree_Number'] == 4]
    sap_flow_5 = sap_flow_df[sap_flow_df['Tree_Number'] == 5]
    sap_flow_6 = sap_flow_df[sap_flow_df['Tree_Number'] == 6]

    #calculate water fluxes from sensor data
    sap_flow_1 = calculate_sap_water_flux(sap_flow_1, 1, install_df, only_outer=True)
    sap_flow_2 = calculate_sap_water_flux(sap_flow_2, 2, install_df, only_outer=True)
    sap_flow_3 = calculate_sap_water_flux(sap_flow_3, 3, install_df, only_outer=True)
    sap_flow_4 = calculate_sap_water_flux(sap_flow_4, 4, install_df, only_outer=True)
    sap_flow_5 = calculate_sap_water_flux(sap_flow_5, 5, install_df, only_outer=True)
    sap_flow_6 = calculate_sap_water_flux(sap_flow_6, 6, install_df, only_outer=True)

    sap_flow_df = pd.concat([sap_flow_1, sap_flow_2, sap_flow_3, sap_flow_4, sap_flow_5, sap_flow_6],
                            ignore_index=True, sort=False)

    return sap_flow_df

def met_S2_to_S3(input_folder):
    # load pickled dataframe from prior stage
    met_df = pd.read_pickle('%s/met.pkl' % input_folder)

    #calculates VPD at each time step based on T and RH
    met_df = calc_VPD(met_df)

    return met_df

def gasex_S2_to_S3(input_folder):
    #load pickled dataframe from prior stage
    gasex_df = pd.read_pickle('%s/gasex.pkl' % input_folder)

    #calculate intrinsic and instantaneous water use efficiency
    gasex_df["WUE_intr"] = gasex_df.A/gasex_df.gsw
    gasex_df["WUE_inst"] = gasex_df.A/gasex_df.E

    return gasex_df

def leaf_area_S2_to_S3(input_folder):
    #load pickled dataframe from prior stage
    leaf_area_df = pd.read_pickle('%s/leaf_area.pkl' % input_folder)

    return leaf_area_df


def isotopes_S2_to_S3(input_folder):
    #load pickled dataframe from prior stage
    isotopes_df = pd.read_pickle('%s/isotopes.pkl' % input_folder)

    #no derived data products

    return isotopes_df

def sap_install_S2_to_S3(input_folder):
    # load pickled dataframe from prior stage
    sap_install_df = pd.read_pickle('%s/sap_install.pkl' % input_folder)

    # no derived data products

    return sap_install_df

def lwp_S2_to_S3(input_folder):
    #load pickled dataframe from prior stage
    lwp_df = pd.read_pickle('%s/lwp.pkl' % input_folder)

    #no derived data products

    return lwp_df

def soil_moisture_S2_to_S3(input_folder):
    #load pickled dataframe from prior stage
    soil_moisture_df = pd.read_pickle('%s/soil_moisture.pkl' % input_folder)

    #Calculates soil water potentials based on measured water retention curve
    soil_moisture_df['psi_soil'] = soil_moisture_df['VWC_Percent'].apply(soil_wrc)

    return soil_moisture_df


def gasex_curves_S3_to_S4(input_folder):
    # load pickled dataframe from prior stage
    gasex_curves_df = pd.read_pickle('%s/gasex_curves.pkl' % input_folder)

    # no further cleaning or corrections
    return gasex_curves_df

def sap_flow_S3_to_S4(input_folder):
    # load pickled dataframe from prior stage
    sap_flow_df = pd.read_pickle('%s/sap_flow.pkl' % input_folder)

    sap_flow_1 = sap_flow_df[sap_flow_df['Tree_Number'] == 1]
    sap_flow_2 = sap_flow_df[sap_flow_df['Tree_Number'] == 2]
    sap_flow_3 = sap_flow_df[sap_flow_df['Tree_Number'] == 3]
    sap_flow_4 = sap_flow_df[sap_flow_df['Tree_Number'] == 4]
    sap_flow_5 = sap_flow_df[sap_flow_df['Tree_Number'] == 5]
    sap_flow_6 = sap_flow_df[sap_flow_df['Tree_Number'] == 6]

    # #Uses a Hampel Filter to remove outliers (see Poyatos et al 2021 Sapfluxnet paper)
    window_size=8
    n_sigmas = 1.0
    sap_flow_1 = hampel_filter_pandas(sap_flow_1, window_size, n_sigmas)
    sap_flow_2 = hampel_filter_pandas(sap_flow_2, window_size, n_sigmas)
    sap_flow_3 = hampel_filter_pandas(sap_flow_3, window_size, n_sigmas)
    sap_flow_4 = hampel_filter_pandas(sap_flow_4, window_size, n_sigmas)
    sap_flow_5 = hampel_filter_pandas(sap_flow_5, window_size, n_sigmas)
    sap_flow_6 = hampel_filter_pandas(sap_flow_6, window_size, n_sigmas)

    # Corrects baseline to zero based on min value over this time period (ie overnight assumed zero sap flow)
    sap_flow_1 = correct_sap_flow_baseline(sap_flow_1)
    sap_flow_2 = correct_sap_flow_baseline(sap_flow_2)
    sap_flow_3 = correct_sap_flow_baseline(sap_flow_3)
    sap_flow_4 = correct_sap_flow_baseline(sap_flow_4)
    sap_flow_5 = correct_sap_flow_baseline(sap_flow_5)
    sap_flow_6 = correct_sap_flow_baseline(sap_flow_6)

    sap_flow_df = pd.concat([sap_flow_1, sap_flow_2, sap_flow_3, sap_flow_4, sap_flow_5, sap_flow_6],
                            ignore_index=True, sort=False)

    return sap_flow_df

def met_S3_to_S4(input_folder):
    # load pickled dataframe from prior stage
    met_df = pd.read_pickle('%s/met.pkl' % input_folder)

    # no other corrections currently

    return met_df

def leaf_area_S3_to_S4(input_folder):
    #load pickled dataframe from prior stage
    leaf_area_df = pd.read_pickle('%s/leaf_area.pkl' % input_folder)

    return leaf_area_df

def root_S3_to_S4(input_folder):
    #load pickled dataframe from prior stage
    root_df = pd.read_pickle('%s/root_dist.pkl' % input_folder)

    return root_df

def gasex_S3_to_S4(input_folder):
    #load pickled dataframe from prior stage
    gasex_df = pd.read_pickle('%s/gasex.pkl' % input_folder)

    #no other corrections
    return gasex_df


def isotopes_S3_to_S4(input_folder):
    #load pickled dataframe from prior stage
    isotopes_df = pd.read_pickle('%s/isotopes.pkl' % input_folder)

    #no other corrections currently

    return isotopes_df

def sap_install_S3_to_S4(input_folder):
    # load pickled dataframe from prior stage
    sap_install_df = pd.read_pickle('%s/sap_install.pkl' % input_folder)

    # no other corrections currently

    return sap_install_df

def lwp_S3_to_S4(input_folder):
    #load pickled dataframe from prior stage
    lwp_df = pd.read_pickle('%s/lwp.pkl' % input_folder)

    #no other corrections currently

    return lwp_df

def soil_moisture_S3_to_S4(input_folder):
    #load pickled dataframe from prior stage
    soil_moisture_df = pd.read_pickle('%s/soil_moisture.pkl' % input_folder)

    #no other corrections currently

    return soil_moisture_df

def plot_data_stage(sap_flow=None, soil_moisture=None, lwp=None, gasex=None, met=None, isotopes=None, stage=1):
    if sap_flow is not None:
        sap_flow['Clock'] = pd.to_datetime(sap_flow['TIMESTAMP'], format="%Y-%m-%d %H:%M:%S").dt.time
        #sap_flow['Clock'] = pd.to_datetime(sap_flow['TIMESTAMP']) - pd.to_datetime(sap_flow['TIMESTAMP']).dt.normalize()
        sap_flow['Sampling_Day'] = sap_flow.apply(classify_sampling_day, axis=1)
        sap_flow_1 = sap_flow[sap_flow['Tree_Number'] == 1]
        sap_flow_2 = sap_flow[sap_flow['Tree_Number'] == 2]
        sap_flow_3 = sap_flow[sap_flow['Tree_Number'] == 3]
        sap_flow_4 = sap_flow[sap_flow['Tree_Number'] == 4]
        sap_flow_5 = sap_flow[sap_flow['Tree_Number'] == 5]
        sap_flow_6 = sap_flow[sap_flow['Tree_Number'] == 6]

        if stage == 1 or stage == 2:
            y_data = 'Vh_Outer'
        else:
            y_data = 'Total_Sapflow'

        ax = plt.gca()
        sap_flow_1.plot(kind='line', x='TIMESTAMP', y=y_data, label='1', ax=ax)
        sap_flow_2.plot(kind='line', x='TIMESTAMP', y=y_data, label='2', ax=ax)
        sap_flow_3.plot(kind='line', x='TIMESTAMP', y=y_data, label='3', ax=ax)
        sap_flow_4.plot(kind='line', x='TIMESTAMP', y=y_data, label='4', ax=ax)
        sap_flow_5.plot(kind='line', x='TIMESTAMP', y=y_data, label='5', ax=ax)
        sap_flow_6.plot(kind='line', x='TIMESTAMP', y=y_data, label='6', ax=ax)
        plt.legend()
        plt.ylabel('Vh outer (cm/hr)')
        plt.show()

        ax = plt.gca()
        sap_flow_1.plot(kind='line', x='TIMESTAMP', y='SWC_Outer', label='1', ax=ax)
        sap_flow_2.plot(kind='line', x='TIMESTAMP', y='SWC_Outer', label='2', ax=ax)
        sap_flow_3.plot(kind='line', x='TIMESTAMP', y='SWC_Outer', label='3', ax=ax)
        sap_flow_4.plot(kind='line', x='TIMESTAMP', y='SWC_Outer', label='4', ax=ax)
        sap_flow_5.plot(kind='line', x='TIMESTAMP', y='SWC_Outer', label='5', ax=ax)
        sap_flow_6.plot(kind='line', x='TIMESTAMP', y='SWC_Outer', label='6', ax=ax)
        plt.legend()
        plt.ylabel('Stem Water Content (%)')
        plt.show()

        sap_flow_1_1 = sap_flow_1[sap_flow_1['Sampling_Day'] == 1]
        sap_flow_1_2 = sap_flow_1[sap_flow_1['Sampling_Day'] == 2]
        sap_flow_1_3 = sap_flow_1[sap_flow_1['Sampling_Day'] == 3]
        sap_flow_1_4 = sap_flow_1[sap_flow_1['Sampling_Day'] == 4]
        sap_flow_1_5 = sap_flow_1[sap_flow_1['Sampling_Day'] == 5]

        #col = sap_flow_1.Sampling_Day.map({1: '#41b6c4', 2: '#1d91c0', 3: '#225ea8', 4: '#253494', 5: '#081d58'})
        colors = ['#f4a582', '#d6604d', '#b2182b', '#92c5de', '#2166ac']
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1,6, sharey=True, figsize=(12,6))
        sap_flow_1_1.plot(kind='line', x='Clock', y=y_data, c=colors[0], ax=ax1, label='Day 1', legend=None)
        sap_flow_1_2.plot(kind='line', x='Clock', y=y_data, c=colors[1], ax=ax1, label='Day 2', legend=None)
        sap_flow_1_3.plot(kind='line', x='Clock', y=y_data, c=colors[2], ax=ax1, label='Day 3', legend=None)
        sap_flow_1_4.plot(kind='line', x='Clock', y=y_data, c=colors[3], ax=ax1, label='Day 4', legend=None)
        sap_flow_1_5.plot(kind='line', x='Clock', y=y_data, c=colors[4], ax=ax1, label='Day 5', legend=None)
        ax1.set_xticks(['0:00', '12:00'])



        sap_flow_2_1 = sap_flow_2[sap_flow_2['Sampling_Day'] == 1]
        sap_flow_2_2 = sap_flow_2[sap_flow_2['Sampling_Day'] == 2]
        sap_flow_2_3 = sap_flow_2[sap_flow_2['Sampling_Day'] == 3]
        sap_flow_2_4 = sap_flow_2[sap_flow_2['Sampling_Day'] == 4]
        sap_flow_2_5 = sap_flow_2[sap_flow_2['Sampling_Day'] == 5]
        sap_flow_2_1.plot(kind='line', x='Clock', y=y_data, c=colors[0], ax=ax2, label='Day 1', legend=None)
        sap_flow_2_2.plot(kind='line', x='Clock', y=y_data, c=colors[1], ax=ax2, label='Day 2', legend=None)
        sap_flow_2_3.plot(kind='line', x='Clock', y=y_data, c=colors[2], ax=ax2, label='Day 3', legend=None)
        sap_flow_2_4.plot(kind='line', x='Clock', y=y_data, c=colors[3], ax=ax2, label='Day 4', legend=None)
        sap_flow_2_5.plot(kind='line', x='Clock', y=y_data, c=colors[4], ax=ax2, label='Day 5', legend=None)
        ax2.set_xticks(['0:00', '12:00'])



        sap_flow_3_1 = sap_flow_3[sap_flow_3['Sampling_Day'] == 1]
        sap_flow_3_2 = sap_flow_3[sap_flow_3['Sampling_Day'] == 2]
        sap_flow_3_3 = sap_flow_3[sap_flow_3['Sampling_Day'] == 3]
        sap_flow_3_4 = sap_flow_3[sap_flow_3['Sampling_Day'] == 4]
        sap_flow_3_5 = sap_flow_3[sap_flow_3['Sampling_Day'] == 5]
        sap_flow_3_1.plot(kind='line', x='Clock', y=y_data, c=colors[0], ax=ax3, label='Day 1', legend=None)
        sap_flow_3_2.plot(kind='line', x='Clock', y=y_data, c=colors[1], ax=ax3, label='Day 2', legend=None)
        sap_flow_3_3.plot(kind='line', x='Clock', y=y_data, c=colors[2], ax=ax3, label='Day 3', legend=None)
        sap_flow_3_4.plot(kind='line', x='Clock', y=y_data, c=colors[3], ax=ax3, label='Day 4', legend=None)
        sap_flow_3_5.plot(kind='line', x='Clock', y=y_data, c=colors[4], ax=ax3, label='Day 5', legend=None)
        ax3.set_xticks(['0:00', '12:00'])



        sap_flow_4_1 = sap_flow_4[sap_flow_4['Sampling_Day'] == 1]
        sap_flow_4_2 = sap_flow_4[sap_flow_4['Sampling_Day'] == 2]
        sap_flow_4_3 = sap_flow_4[sap_flow_4['Sampling_Day'] == 3]
        sap_flow_4_4 = sap_flow_4[sap_flow_4['Sampling_Day'] == 4]
        sap_flow_4_5 = sap_flow_4[sap_flow_4['Sampling_Day'] == 5]
        sap_flow_4_1.plot(kind='line', x='Clock', y=y_data, c=colors[0], ax=ax4, label='Day 1', legend=None)
        sap_flow_4_2.plot(kind='line', x='Clock', y=y_data, c=colors[1], ax=ax4, label='Day 2', legend=None)
        sap_flow_4_3.plot(kind='line', x='Clock', y=y_data, c=colors[2], ax=ax4, label='Day 3', legend=None)
        sap_flow_4_4.plot(kind='line', x='Clock', y=y_data, c=colors[3], ax=ax4, label='Day 4', legend=None)
        sap_flow_4_5.plot(kind='line', x='Clock', y=y_data, c=colors[4], ax=ax4, label='Day 5', legend=None)
        ax4.set_xticks(['0:00', '12:00'])


        sap_flow_5_1 = sap_flow_5[sap_flow_5['Sampling_Day'] == 1]
        sap_flow_5_2 = sap_flow_5[sap_flow_5['Sampling_Day'] == 2]
        sap_flow_5_3 = sap_flow_5[sap_flow_5['Sampling_Day'] == 3]
        sap_flow_5_4 = sap_flow_5[sap_flow_5['Sampling_Day'] == 4]
        sap_flow_5_5 = sap_flow_5[sap_flow_5['Sampling_Day'] == 5]
        sap_flow_5_1.plot(kind='line', x='Clock', y=y_data, c=colors[0], ax=ax5, label='Day 1', legend=None)
        sap_flow_5_2.plot(kind='line', x='Clock', y=y_data, c=colors[1], ax=ax5, label='Day 2', legend=None)
        sap_flow_5_3.plot(kind='line', x='Clock', y=y_data, c=colors[2], ax=ax5, label='Day 3', legend=None)
        sap_flow_5_4.plot(kind='line', x='Clock', y=y_data, c=colors[3], ax=ax5, label='Day 4', legend=None)
        sap_flow_5_5.plot(kind='line', x='Clock', y=y_data, c=colors[4], ax=ax5, label='Day 5', legend=None)
        ax5.set_xticks(['0:00', '12:00'])


        sap_flow_6_1 = sap_flow_6[sap_flow_6['Sampling_Day'] == 1]
        sap_flow_6_2 = sap_flow_6[sap_flow_6['Sampling_Day'] == 2]
        sap_flow_6_3 = sap_flow_6[sap_flow_6['Sampling_Day'] == 3]
        sap_flow_6_4 = sap_flow_6[sap_flow_6['Sampling_Day'] == 4]
        sap_flow_6_5 = sap_flow_6[sap_flow_6['Sampling_Day'] == 5]
        sap_flow_6_1.plot(kind='line', x='Clock', y=y_data, c=colors[0], ax=ax6, label='Day 1', legend=None)
        sap_flow_6_2.plot(kind='line', x='Clock', y=y_data, c=colors[1], ax=ax6, label='Day 2', legend=None)
        sap_flow_6_3.plot(kind='line', x='Clock', y=y_data, c=colors[2], ax=ax6, label='Day 3', legend=None)
        sap_flow_6_4.plot(kind='line', x='Clock', y=y_data, c=colors[3], ax=ax6, label='Day 4', legend=None)
        sap_flow_6_5.plot(kind='line', x='Clock', y=y_data, c=colors[4], ax=ax6, label='Day 5', legend=None)
        ax6.set_xticks(['0:00', '12:00'])

        plt.savefig('Output_Plots/sapflow_tree_split.pdf')
        plt.show()

    if lwp is not None:
        full_lwp=lwp
        lwp=lwp[lwp['Sampling_Day'] != 'NaN']

        lwp_1 = lwp[lwp['Tree_Number'] == 1]
        lwp_2 = lwp[lwp['Tree_Number'] == 2]
        lwp_3 = lwp[lwp['Tree_Number'] == 3]
        lwp_4 = lwp[lwp['Tree_Number'] == 4]
        lwp_5 = lwp[lwp['Tree_Number'] == 5]
        lwp_6 = lwp[lwp['Tree_Number'] == 6]

        # ax = plt.gca()
        # lwp_1.plot(kind='scatter', x='TIMESTAMP', y='LWP_(MPa)', label='1', ax=ax)
        # lwp_2.plot(kind='scatter', x='TIMESTAMP', y='LWP_(MPa)', label='2', ax=ax)
        # lwp_3.plot(kind='scatter', x='TIMESTAMP', y='LWP_(MPa)', label='3', ax=ax)
        # lwp_4.plot(kind='scatter', x='TIMESTAMP', y='LWP_(MPa)', label='4', ax=ax)
        # lwp_5.plot(kind='scatter', x='TIMESTAMP', y='LWP_(MPa)', label='5', ax=ax)
        # lwp_6.plot(kind='scatter', x='TIMESTAMP', y='LWP_(MPa)', label='6', ax=ax)
        # plt.legend()
        # plt.ylabel('Leaf Water Potential (MPa)')
        # plt.show()

        col = lwp.Tree_Number.map({1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        lwp.plot(kind='scatter', x='TIMESTAMP', y='LWP_(MPa)', c=col)
        plt.show()

        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1,6, sharey=True)
        col = lwp_1.LWP_Type.map({'Predawn': 'r', 'Midday':'b', 'Extra':'k'})
        lwp_1.plot(kind='scatter', x='TIMESTAMP', y='LWP_(MPa)', c=col, ax=ax1)
        col = lwp_2.LWP_Type.map({'Predawn': 'r', 'Midday': 'b', 'Extra': 'k'})
        lwp_2.plot(kind='scatter', x='TIMESTAMP', y='LWP_(MPa)', c=col, ax=ax2)
        col = lwp_3.LWP_Type.map({'Predawn': 'r', 'Midday': 'b', 'Extra': 'k'})
        lwp_3.plot(kind='scatter', x='TIMESTAMP', y='LWP_(MPa)', c=col, ax=ax3)
        col = lwp_4.LWP_Type.map({'Predawn': 'r', 'Midday': 'b', 'Extra': 'k'})
        lwp_4.plot(kind='scatter', x='TIMESTAMP', y='LWP_(MPa)', c=col, ax=ax4)
        col = lwp_5.LWP_Type.map({'Predawn': 'r', 'Midday': 'b', 'Extra': 'k'})
        lwp_5.plot(kind='scatter', x='TIMESTAMP', y='LWP_(MPa)', c=col, ax=ax5)
        col = lwp_6.LWP_Type.map({'Predawn': 'r', 'Midday': 'b', 'Extra': 'k'})
        lwp_6.plot(kind='scatter', x='TIMESTAMP', y='LWP_(MPa)', c=col, ax=ax6)
        plt.show()


    if gasex is not None:
        gasex_1 = gasex[gasex['Tree_Number'] == 1]
        gasex_2 = gasex[gasex['Tree_Number'] == 2]
        gasex_3 = gasex[gasex['Tree_Number'] == 3]
        gasex_4 = gasex[gasex['Tree_Number'] == 4]
        gasex_5 = gasex[gasex['Tree_Number'] == 5]
        gasex_6 = gasex[gasex['Tree_Number'] == 6]

        col = gasex.Tree_Number.map({1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        gasex.plot(kind='scatter', x='TIMESTAMP', y='A', c=col)
        plt.show()

        gasex.plot(kind='scatter', x='TIMESTAMP', y='gsw', c=col)
        plt.show()


        # ax = plt.gca()
        # gasex_1.plot(kind='scatter', x='TIMESTAMP', y='A', label='1', ax=ax)
        # gasex_2.plot(kind='scatter', x='TIMESTAMP', y='A', label='2', ax=ax)
        # gasex_3.plot(kind='scatter', x='TIMESTAMP', y='A', label='3', ax=ax)
        # gasex_4.plot(kind='scatter', x='TIMESTAMP', y='A', label='4', ax=ax)
        # gasex_5.plot(kind='scatter', x='TIMESTAMP', y='A', label='5', ax=ax)
        # gasex_6.plot(kind='scatter', x='TIMESTAMP', y='A', label='6', ax=ax)
        # plt.legend()
        # plt.ylabel('Assimilation')
        # plt.show()

        # ax = plt.gca()
        # gasex_1.plot(kind='scatter', x='TIMESTAMP', y='gsw', label='1', ax=ax)
        # gasex_2.plot(kind='scatter', x='TIMESTAMP', y='gsw', label='2', ax=ax)
        # gasex_3.plot(kind='scatter', x='TIMESTAMP', y='gsw', label='3', ax=ax)
        # gasex_4.plot(kind='scatter', x='TIMESTAMP', y='gsw', label='4', ax=ax)
        # gasex_5.plot(kind='scatter', x='TIMESTAMP', y='gsw', label='5', ax=ax)
        # gasex_6.plot(kind='scatter', x='TIMESTAMP', y='gsw', label='6', ax=ax)
        # plt.legend()
        # plt.ylabel('gsw')
        # plt.show()

    if met is not None:
        met.plot(kind='line', x='TIMESTAMP', y='AirTC')
        plt.title('Air Temp')
        plt.show()

        met.plot(kind='line', x='TIMESTAMP', y='RH')
        plt.title('RH')
        plt.show()

        met.plot(kind='line', x='TIMESTAMP', y='SlrW_Avg')
        plt.title('Solar Irradiation')
        plt.show()

        if stage >= 3:
            met.plot(kind='line', x='TIMESTAMP', y='VPD_kPa')
            plt.title('Solar Irradiation')
            plt.show()

    if isotopes is not None:
        soil_isotopes = isotopes[isotopes['Sample_Type'] == 'Soil']
        plant_isotopes = isotopes[isotopes['Sample_Type'] == 'Plant']

        soil_day1 = soil_isotopes[soil_isotopes['TIMESTAMP'] >= "2021-07-08 00:00:00"]
        soil_day1 = soil_day1[soil_day1['TIMESTAMP'] <= "2021-07-09 00:00:00"]

        plant_day1 = plant_isotopes[plant_isotopes['TIMESTAMP'] >= "2021-07-08 00:00:00"]
        plant_day1 = plant_day1[plant_day1['TIMESTAMP'] <= "2021-07-09 00:00:00"]

        soil_day2 = soil_isotopes[soil_isotopes['TIMESTAMP'] >= "2021-07-11 00:00:00"]
        soil_day2 = soil_day2[soil_day2['TIMESTAMP'] <= "2021-07-12 00:00:00"]

        plant_day2 = plant_isotopes[plant_isotopes['TIMESTAMP'] >= "2021-07-11 00:00:00"]
        plant_day2 = plant_day2[plant_day2['TIMESTAMP'] <= "2021-07-12 00:00:00"]

        soil_day3 = soil_isotopes[soil_isotopes['TIMESTAMP'] >= "2021-07-14 00:00:00"]
        soil_day3 = soil_day3[soil_day3['TIMESTAMP'] <= "2021-07-15 00:00:00"]

        plant_day3 = plant_isotopes[plant_isotopes['TIMESTAMP'] >= "2021-07-14 00:00:00"]
        plant_day3 = plant_day3[plant_day3['TIMESTAMP'] <= "2021-07-15 00:00:00"]

        soil_day4 = soil_isotopes[soil_isotopes['TIMESTAMP'] >= "2021-07-18 00:00:00"]
        soil_day4 = soil_day4[soil_day4['TIMESTAMP'] <= "2021-07-19 00:00:00"]

        plant_day4 = plant_isotopes[plant_isotopes['TIMESTAMP'] >= "2021-07-18 00:00:00"]
        plant_day4 = plant_day4[plant_day4['TIMESTAMP'] <= "2021-07-19 00:00:00"]

        soil_day5 = soil_isotopes[soil_isotopes['TIMESTAMP'] >= "2021-07-21 00:00:00"]
        soil_day5 = soil_day5[soil_day5['TIMESTAMP'] <= "2021-07-22 00:00:00"]

        plant_day5 = plant_isotopes[plant_isotopes['TIMESTAMP'] >= "2021-07-21 00:00:00"]
        plant_day5 = plant_day5[plant_day5['TIMESTAMP'] <= "2021-07-22 00:00:00"]

        first_round = soil_isotopes[soil_isotopes['TIMESTAMP'] <= "2021-07-15 00:00:00"]
        second_round = soil_isotopes[soil_isotopes['TIMESTAMP'] > "2021-07-15 00:00:00"]


        col = soil_day1.Tree_Number.map({1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        soil_day1.plot(kind='scatter', x='del_O', y='Depth', c=col)
        plt.show()

        col = soil_day2.Tree_Number.map({1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        soil_day2.plot(kind='scatter', x='del_O', y='Depth', c=col)
        plt.show()

        col = soil_day3.Tree_Number.map({1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        soil_day3.plot(kind='scatter', x='del_O', y='Depth', c=col)
        plt.show()

        col = soil_day4.Tree_Number.map({1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        soil_day4.plot(kind='scatter', x='del_O', y='Depth', c=col)
        plt.show()

        col = soil_day5.Tree_Number.map({1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        soil_day5.plot(kind='scatter', x='del_O', y='Depth', c=col)
        plt.show()

        ax = plt.gca()
        col = first_round.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
                                     50: '#253494', 60: '#081d58'})
        first_round.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax)
        plant_day1.plot(kind='scatter', x='del_O', y='del_D', marker="s", c="grey", ax=ax)
        plant_day2.plot(kind='scatter', x='del_O', y='del_D', marker="^", c="grey", ax=ax)
        plant_day3.plot(kind='scatter', x='del_O', y='del_D', marker="x", c="grey", ax=ax)
        plt.show()

        ax = plt.gca()
        col = second_round.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
                                     50: '#253494', 60: '#081d58'})
        second_round.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax)
        plant_day3.plot(kind='scatter', x='del_O', y='del_D', marker="s", c="black", ax=ax)
        plant_day4.plot(kind='scatter', x='del_O', y='del_D', marker="s", c="grey", ax=ax)
        plant_day5.plot(kind='scatter', x='del_O', y='del_D', marker="^", c="grey", ax=ax)
        plt.show()

        col = plant_isotopes.Tree_Number.map({1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        plant_isotopes.plot(kind='scatter', x='del_O', y='del_D', c=col)
        plt.show()

    if soil_moisture is not None:
        soil_day1 = soil_moisture[soil_moisture['TIMESTAMP'] >= "2021-07-08 00:00:00"]
        soil_day1 = soil_day1[soil_day1['TIMESTAMP'] <= "2021-07-09 00:00:00"]

        soil_day2 = soil_moisture[soil_moisture['TIMESTAMP'] >= "2021-07-11 00:00:00"]
        soil_day2 = soil_day2[soil_day2['TIMESTAMP'] <= "2021-07-12 00:00:00"]

        soil_day3 = soil_moisture[soil_moisture['TIMESTAMP'] >= "2021-07-14 00:00:00"]
        soil_day3 = soil_day3[soil_day3['TIMESTAMP'] <= "2021-07-15 00:00:00"]

        soil_day4 = soil_moisture[soil_moisture['TIMESTAMP'] >= "2021-07-18 00:00:00"]
        soil_day4 = soil_day4[soil_day4['TIMESTAMP'] <= "2021-07-19 00:00:00"]

        soil_day5 = soil_moisture[soil_moisture['TIMESTAMP'] >= "2021-07-21 00:00:00"]
        soil_day5 = soil_day5[soil_day5['TIMESTAMP'] <= "2021-07-22 00:00:00"]

        ax = plt.gca()
        col = soil_day1.Tree_Number.map(
            {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        soil_day1.plot(kind='scatter', x='VWC_Percent', y='Depth_cm', c=col, ax=ax)
        plt.show()

        col = soil_day2.Tree_Number.map(
            {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        soil_day2.plot(kind='scatter', x='VWC_Percent', y='Depth_cm', c=col)
        plt.show()

        col = soil_day3.Tree_Number.map(
            {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        soil_day3.plot(kind='scatter', x='VWC_Percent', y='Depth_cm', c=col)
        plt.show()

        col = soil_day4.Tree_Number.map(
            {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        soil_day4.plot(kind='scatter', x='VWC_Percent', y='Depth_cm', c=col)
        plt.show()

        col = soil_day5.Tree_Number.map(
            {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        soil_day5.plot(kind='scatter', x='VWC_Percent', y='Depth_cm', c=col)
        plt.show()

def plot_results(sap_flow=None, soil_moisture=None, lwp=None, gasex=None, met=None, isotopes=None, leaf_area = None, root=None, gasex_curves=None,  stage=1):
    #makes assortment of plots for each data input provided. Makes a lot more plots than the plot_data_stage_function
    tree_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    day_colors = ['#f4a582', '#d6604d', '#b2182b', '#92c5de', '#2166ac']
    if sap_flow is not None:
        sap_flow['Clock'] = pd.to_datetime(sap_flow['TIMESTAMP'], format="%Y-%m-%d %H:%M:%S").dt.time
        #sap_flow['Clock'] = pd.to_datetime(sap_flow['TIMESTAMP']) - pd.to_datetime(sap_flow['TIMESTAMP']).dt.normalize()
        sap_flow['Sampling_Day'] = sap_flow.apply(classify_sampling_day, axis=1)
        sap_flow_1 = sap_flow[sap_flow['Tree_Number'] == 1]
        sap_flow_2 = sap_flow[sap_flow['Tree_Number'] == 2]
        sap_flow_3 = sap_flow[sap_flow['Tree_Number'] == 3]
        sap_flow_4 = sap_flow[sap_flow['Tree_Number'] == 4]
        sap_flow_5 = sap_flow[sap_flow['Tree_Number'] == 5]
        sap_flow_6 = sap_flow[sap_flow['Tree_Number'] == 6]

        rewatering_sap_flow = sap_flow[(sap_flow['TIMESTAMP'] >= "2021-07-16 00:00:00") & (sap_flow['TIMESTAMP'] < "2021-07-17 00:00:00")]
        rw_sap_flow_1 = rewatering_sap_flow[rewatering_sap_flow['Tree_Number'] == 1]
        rw_sap_flow_2 = rewatering_sap_flow[rewatering_sap_flow['Tree_Number'] == 2]
        rw_sap_flow_3 = rewatering_sap_flow[rewatering_sap_flow['Tree_Number'] == 3]
        rw_sap_flow_4 = rewatering_sap_flow[rewatering_sap_flow['Tree_Number'] == 4]
        rw_sap_flow_5 = rewatering_sap_flow[rewatering_sap_flow['Tree_Number'] == 5]
        rw_sap_flow_6 = rewatering_sap_flow[rewatering_sap_flow['Tree_Number'] == 6]

        if stage == 1 or stage == 2:
            y_data = 'Vh_Outer'
        else:
            y_data = 'Total_Sapflow'

        plt.figure(figsize=(12,6))
        ax = plt.gca()
        sap_flow_1.plot(kind='line', x='TIMESTAMP', y=y_data, label='Tree 1', ax=ax)
        sap_flow_2.plot(kind='line', x='TIMESTAMP', y=y_data, label='Tree 2', ax=ax)
        sap_flow_3.plot(kind='line', x='TIMESTAMP', y=y_data, label='Tree 3', ax=ax)
        sap_flow_4.plot(kind='line', x='TIMESTAMP', y=y_data, label='Tree 4', ax=ax)
        sap_flow_5.plot(kind='line', x='TIMESTAMP', y=y_data, label='Tree 5', ax=ax)
        sap_flow_6.plot(kind='line', x='TIMESTAMP', y=y_data, label='Tree 6', ax=ax)
        plt.legend()
        plt.ylabel('Sapflow (L/hr)')
        plt.savefig('Output_Plots/sapflow_full_length.pdf')
        plt.show()


        ax = plt.gca()
        rw_sap_flow_1.plot(kind='line', x='TIMESTAMP', y=y_data, label='Tree 1', ax=ax)
        rw_sap_flow_2.plot(kind='line', x='TIMESTAMP', y=y_data, label='Tree 2', ax=ax)
        rw_sap_flow_3.plot(kind='line', x='TIMESTAMP', y=y_data, label='Tree 3', ax=ax)
        rw_sap_flow_4.plot(kind='line', x='TIMESTAMP', y=y_data, label='Tree 4', ax=ax)
        rw_sap_flow_5.plot(kind='line', x='TIMESTAMP', y=y_data, label='Tree 5', ax=ax)
        rw_sap_flow_6.plot(kind='line', x='TIMESTAMP', y=y_data, label='Tree 6', ax=ax)
        plt.legend()
        plt.ylabel('Sapflow (L/hr)')
        plt.savefig('Output_Plots/sapflow_rewatering.pdf')
        plt.show()

        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        sap_flow_1.plot(kind='line', x='TIMESTAMP', y='SWC_Outer', label='1', ax=ax)
        sap_flow_2.plot(kind='line', x='TIMESTAMP', y='SWC_Outer', label='2', ax=ax)
        sap_flow_3.plot(kind='line', x='TIMESTAMP', y='SWC_Outer', label='3', ax=ax)
        sap_flow_4.plot(kind='line', x='TIMESTAMP', y='SWC_Outer', label='4', ax=ax)
        sap_flow_5.plot(kind='line', x='TIMESTAMP', y='SWC_Outer', label='5', ax=ax)
        sap_flow_6.plot(kind='line', x='TIMESTAMP', y='SWC_Outer', label='6', ax=ax)
        plt.legend()
        plt.ylabel('Stem Water Content (%)')
        plt.savefig('Output_Plots/stem_water_content_full_length.pdf')
        plt.show()


        sap_flow_1_1 = sap_flow_1[sap_flow_1['Sampling_Day'] == 1]
        sap_flow_1_2 = sap_flow_1[sap_flow_1['Sampling_Day'] == 2]
        sap_flow_1_3 = sap_flow_1[sap_flow_1['Sampling_Day'] == 3]
        sap_flow_1_4 = sap_flow_1[sap_flow_1['Sampling_Day'] == 4]
        sap_flow_1_5 = sap_flow_1[sap_flow_1['Sampling_Day'] == 5]

        #col = sap_flow_1.Sampling_Day.map({1: '#41b6c4', 2: '#1d91c0', 3: '#225ea8', 4: '#253494', 5: '#081d58'})
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1,6, sharey=True, figsize=(12,6))
        sap_flow_1_1.plot(kind='line', x='Clock', y=y_data, c=day_colors[0], ax=ax1, label='Day 1', legend=None)
        sap_flow_1_2.plot(kind='line', x='Clock', y=y_data, c=day_colors[1], ax=ax1, label='Day 2', legend=None)
        sap_flow_1_3.plot(kind='line', x='Clock', y=y_data, c=day_colors[2], ax=ax1, label='Day 3', legend=None)
        sap_flow_1_4.plot(kind='line', x='Clock', y=y_data, c=day_colors[3], ax=ax1, label='Day 4', legend=None)
        sap_flow_1_5.plot(kind='line', x='Clock', y=y_data, c=day_colors[4], ax=ax1, label='Day 5', legend=None)
        ax1.set_xticks(['0:00', '12:00'])



        sap_flow_2_1 = sap_flow_2[sap_flow_2['Sampling_Day'] == 1]
        sap_flow_2_2 = sap_flow_2[sap_flow_2['Sampling_Day'] == 2]
        sap_flow_2_3 = sap_flow_2[sap_flow_2['Sampling_Day'] == 3]
        sap_flow_2_4 = sap_flow_2[sap_flow_2['Sampling_Day'] == 4]
        sap_flow_2_5 = sap_flow_2[sap_flow_2['Sampling_Day'] == 5]
        sap_flow_2_1.plot(kind='line', x='Clock', y=y_data, c=day_colors[0], ax=ax2, label='Day 1', legend=None)
        sap_flow_2_2.plot(kind='line', x='Clock', y=y_data, c=day_colors[1], ax=ax2, label='Day 2', legend=None)
        sap_flow_2_3.plot(kind='line', x='Clock', y=y_data, c=day_colors[2], ax=ax2, label='Day 3', legend=None)
        sap_flow_2_4.plot(kind='line', x='Clock', y=y_data, c=day_colors[3], ax=ax2, label='Day 4', legend=None)
        sap_flow_2_5.plot(kind='line', x='Clock', y=y_data, c=day_colors[4], ax=ax2, label='Day 5', legend=None)
        ax2.set_xticks(['0:00', '12:00'])



        sap_flow_3_1 = sap_flow_3[sap_flow_3['Sampling_Day'] == 1]
        sap_flow_3_2 = sap_flow_3[sap_flow_3['Sampling_Day'] == 2]
        sap_flow_3_3 = sap_flow_3[sap_flow_3['Sampling_Day'] == 3]
        sap_flow_3_4 = sap_flow_3[sap_flow_3['Sampling_Day'] == 4]
        sap_flow_3_5 = sap_flow_3[sap_flow_3['Sampling_Day'] == 5]
        sap_flow_3_1.plot(kind='line', x='Clock', y=y_data, c=day_colors[0], ax=ax3, label='Day 1', legend=None)
        sap_flow_3_2.plot(kind='line', x='Clock', y=y_data, c=day_colors[1], ax=ax3, label='Day 2', legend=None)
        sap_flow_3_3.plot(kind='line', x='Clock', y=y_data, c=day_colors[2], ax=ax3, label='Day 3', legend=None)
        sap_flow_3_4.plot(kind='line', x='Clock', y=y_data, c=day_colors[3], ax=ax3, label='Day 4', legend=None)
        sap_flow_3_5.plot(kind='line', x='Clock', y=y_data, c=day_colors[4], ax=ax3, label='Day 5', legend=None)
        ax3.set_xticks(['0:00', '12:00'])



        sap_flow_4_1 = sap_flow_4[sap_flow_4['Sampling_Day'] == 1]
        sap_flow_4_2 = sap_flow_4[sap_flow_4['Sampling_Day'] == 2]
        sap_flow_4_3 = sap_flow_4[sap_flow_4['Sampling_Day'] == 3]
        sap_flow_4_4 = sap_flow_4[sap_flow_4['Sampling_Day'] == 4]
        sap_flow_4_5 = sap_flow_4[sap_flow_4['Sampling_Day'] == 5]
        sap_flow_4_1.plot(kind='line', x='Clock', y=y_data, c=day_colors[0], ax=ax4, label='Day 1', legend=None)
        sap_flow_4_2.plot(kind='line', x='Clock', y=y_data, c=day_colors[1], ax=ax4, label='Day 2', legend=None)
        sap_flow_4_3.plot(kind='line', x='Clock', y=y_data, c=day_colors[2], ax=ax4, label='Day 3', legend=None)
        sap_flow_4_4.plot(kind='line', x='Clock', y=y_data, c=day_colors[3], ax=ax4, label='Day 4', legend=None)
        sap_flow_4_5.plot(kind='line', x='Clock', y=y_data, c=day_colors[4], ax=ax4, label='Day 5', legend=None)
        ax4.set_xticks(['0:00', '12:00'])


        sap_flow_5_1 = sap_flow_5[sap_flow_5['Sampling_Day'] == 1]
        sap_flow_5_2 = sap_flow_5[sap_flow_5['Sampling_Day'] == 2]
        sap_flow_5_3 = sap_flow_5[sap_flow_5['Sampling_Day'] == 3]
        sap_flow_5_4 = sap_flow_5[sap_flow_5['Sampling_Day'] == 4]
        sap_flow_5_5 = sap_flow_5[sap_flow_5['Sampling_Day'] == 5]
        sap_flow_5_1.plot(kind='line', x='Clock', y=y_data, c=day_colors[0], ax=ax5, label='Day 1', legend=None)
        sap_flow_5_2.plot(kind='line', x='Clock', y=y_data, c=day_colors[1], ax=ax5, label='Day 2', legend=None)
        sap_flow_5_3.plot(kind='line', x='Clock', y=y_data, c=day_colors[2], ax=ax5, label='Day 3', legend=None)
        sap_flow_5_4.plot(kind='line', x='Clock', y=y_data, c=day_colors[3], ax=ax5, label='Day 4', legend=None)
        sap_flow_5_5.plot(kind='line', x='Clock', y=y_data, c=day_colors[4], ax=ax5, label='Day 5', legend=None)
        ax5.set_xticks(['0:00', '12:00'])


        sap_flow_6_1 = sap_flow_6[sap_flow_6['Sampling_Day'] == 1]
        sap_flow_6_2 = sap_flow_6[sap_flow_6['Sampling_Day'] == 2]
        sap_flow_6_3 = sap_flow_6[sap_flow_6['Sampling_Day'] == 3]
        sap_flow_6_4 = sap_flow_6[sap_flow_6['Sampling_Day'] == 4]
        sap_flow_6_5 = sap_flow_6[sap_flow_6['Sampling_Day'] == 5]
        sap_flow_6_1.plot(kind='line', x='Clock', y=y_data, c=day_colors[0], ax=ax6, label='Day 1', legend=None)
        sap_flow_6_2.plot(kind='line', x='Clock', y=y_data, c=day_colors[1], ax=ax6, label='Day 2', legend=None)
        sap_flow_6_3.plot(kind='line', x='Clock', y=y_data, c=day_colors[2], ax=ax6, label='Day 3', legend=None)
        sap_flow_6_4.plot(kind='line', x='Clock', y=y_data, c=day_colors[3], ax=ax6, label='Day 4', legend=None)
        sap_flow_6_5.plot(kind='line', x='Clock', y=y_data, c=day_colors[4], ax=ax6, label='Day 5', legend=None)
        ax6.set_xticks(['0:00', '12:00'])

        plt.savefig('Output_Plots/sapflow_tree_split.pdf')
        plt.show()

        # col = sap_flow_1.Sampling_Day.map({1: '#41b6c4', 2: '#1d91c0', 3: '#225ea8', 4: '#253494', 5: '#081d58'})
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, sharey=True, figsize=(12, 6))
        sap_flow_1_1.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[0], ax=ax1, label='Day 1', legend=None)
        sap_flow_1_2.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[1], ax=ax1, label='Day 2', legend=None)
        sap_flow_1_3.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[2], ax=ax1, label='Day 3', legend=None)
        sap_flow_1_4.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[3], ax=ax1, label='Day 4', legend=None)
        sap_flow_1_5.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[4], ax=ax1, label='Day 5', legend=None)
        ax1.set_xticks(['0:00', '12:00'])

        sap_flow_2_1 = sap_flow_2[sap_flow_2['Sampling_Day'] == 1]
        sap_flow_2_2 = sap_flow_2[sap_flow_2['Sampling_Day'] == 2]
        sap_flow_2_3 = sap_flow_2[sap_flow_2['Sampling_Day'] == 3]
        sap_flow_2_4 = sap_flow_2[sap_flow_2['Sampling_Day'] == 4]
        sap_flow_2_5 = sap_flow_2[sap_flow_2['Sampling_Day'] == 5]
        sap_flow_2_1.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[0], ax=ax2, label='Day 1', legend=None)
        sap_flow_2_2.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[1], ax=ax2, label='Day 2', legend=None)
        sap_flow_2_3.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[2], ax=ax2, label='Day 3', legend=None)
        sap_flow_2_4.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[3], ax=ax2, label='Day 4', legend=None)
        sap_flow_2_5.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[4], ax=ax2, label='Day 5', legend=None)
        ax2.set_xticks(['0:00', '12:00'])

        sap_flow_3_1 = sap_flow_3[sap_flow_3['Sampling_Day'] == 1]
        sap_flow_3_2 = sap_flow_3[sap_flow_3['Sampling_Day'] == 2]
        sap_flow_3_3 = sap_flow_3[sap_flow_3['Sampling_Day'] == 3]
        sap_flow_3_4 = sap_flow_3[sap_flow_3['Sampling_Day'] == 4]
        sap_flow_3_5 = sap_flow_3[sap_flow_3['Sampling_Day'] == 5]
        sap_flow_3_1.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[0], ax=ax3, label='Day 1', legend=None)
        sap_flow_3_2.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[1], ax=ax3, label='Day 2', legend=None)
        sap_flow_3_3.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[2], ax=ax3, label='Day 3', legend=None)
        sap_flow_3_4.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[3], ax=ax3, label='Day 4', legend=None)
        sap_flow_3_5.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[4], ax=ax3, label='Day 5', legend=None)
        ax3.set_xticks(['0:00', '12:00'])

        sap_flow_4_1 = sap_flow_4[sap_flow_4['Sampling_Day'] == 1]
        sap_flow_4_2 = sap_flow_4[sap_flow_4['Sampling_Day'] == 2]
        sap_flow_4_3 = sap_flow_4[sap_flow_4['Sampling_Day'] == 3]
        sap_flow_4_4 = sap_flow_4[sap_flow_4['Sampling_Day'] == 4]
        sap_flow_4_5 = sap_flow_4[sap_flow_4['Sampling_Day'] == 5]
        sap_flow_4_1.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[0], ax=ax4, label='Day 1', legend=None)
        sap_flow_4_2.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[1], ax=ax4, label='Day 2', legend=None)
        sap_flow_4_3.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[2], ax=ax4, label='Day 3', legend=None)
        sap_flow_4_4.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[3], ax=ax4, label='Day 4', legend=None)
        sap_flow_4_5.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[4], ax=ax4, label='Day 5', legend=None)
        ax4.set_xticks(['0:00', '12:00'])

        sap_flow_5_1 = sap_flow_5[sap_flow_5['Sampling_Day'] == 1]
        sap_flow_5_2 = sap_flow_5[sap_flow_5['Sampling_Day'] == 2]
        sap_flow_5_3 = sap_flow_5[sap_flow_5['Sampling_Day'] == 3]
        sap_flow_5_4 = sap_flow_5[sap_flow_5['Sampling_Day'] == 4]
        sap_flow_5_5 = sap_flow_5[sap_flow_5['Sampling_Day'] == 5]
        sap_flow_5_1.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[0], ax=ax5, label='Day 1', legend=None)
        sap_flow_5_2.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[1], ax=ax5, label='Day 2', legend=None)
        sap_flow_5_3.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[2], ax=ax5, label='Day 3', legend=None)
        sap_flow_5_4.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[3], ax=ax5, label='Day 4', legend=None)
        sap_flow_5_5.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[4], ax=ax5, label='Day 5', legend=None)
        ax5.set_xticks(['0:00', '12:00'])

        sap_flow_6_1 = sap_flow_6[sap_flow_6['Sampling_Day'] == 1]
        sap_flow_6_2 = sap_flow_6[sap_flow_6['Sampling_Day'] == 2]
        sap_flow_6_3 = sap_flow_6[sap_flow_6['Sampling_Day'] == 3]
        sap_flow_6_4 = sap_flow_6[sap_flow_6['Sampling_Day'] == 4]
        sap_flow_6_5 = sap_flow_6[sap_flow_6['Sampling_Day'] == 5]
        sap_flow_6_1.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[0], ax=ax6, label='Day 1', legend=None)
        sap_flow_6_2.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[1], ax=ax6, label='Day 2', legend=None)
        sap_flow_6_3.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[2], ax=ax6, label='Day 3', legend=None)
        sap_flow_6_4.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[3], ax=ax6, label='Day 4', legend=None)
        sap_flow_6_5.plot(kind='line', x='Clock', y='SWC_Outer', c=day_colors[4], ax=ax6, label='Day 5', legend=None)
        ax6.set_xticks(['0:00', '12:00'])

        plt.savefig('Output_Plots/stem_water_tree_split.pdf')
        plt.show()

    if lwp is not None:
        lwp_means = lwp.groupby(['Tree_Number', 'Sampling_Day', 'LWP_Type'])['LWP_(MPa)'].mean()
        lwp_1 = lwp[lwp['Tree_Number'] == 1]
        lwp_2 = lwp[lwp['Tree_Number'] == 2]
        lwp_3 = lwp[lwp['Tree_Number'] == 3]
        lwp_4 = lwp[lwp['Tree_Number'] == 4]
        lwp_5 = lwp[lwp['Tree_Number'] == 5]
        lwp_6 = lwp[lwp['Tree_Number'] == 6]

        lwp_1_pd = lwp_1[lwp_1['LWP_Type'] == 'Predawn']
        lwp_1_md = lwp_1[lwp_1['LWP_Type'] == 'Midday']
        lwp_2_pd = lwp_2[lwp_2['LWP_Type'] == 'Predawn']
        lwp_2_md = lwp_2[lwp_2['LWP_Type'] == 'Midday']
        lwp_3_pd = lwp_3[lwp_3['LWP_Type'] == 'Predawn']
        lwp_3_md = lwp_3[lwp_3['LWP_Type'] == 'Midday']
        lwp_4_pd = lwp_4[lwp_4['LWP_Type'] == 'Predawn']
        lwp_4_md = lwp_4[lwp_4['LWP_Type'] == 'Midday']
        lwp_5_pd = lwp_5[lwp_5['LWP_Type'] == 'Predawn']
        lwp_5_md = lwp_5[lwp_5['LWP_Type'] == 'Midday']
        lwp_6_pd = lwp_6[lwp_6['LWP_Type'] == 'Predawn']
        lwp_6_md = lwp_6[lwp_6['LWP_Type'] == 'Midday']

        rewatering_lwp = lwp[(lwp['TIMESTAMP'] >= "2021-07-16 00:00:00") & (lwp['TIMESTAMP'] < "2021-07-17 00:00:00")]
        lwp_1_extra = rewatering_lwp[rewatering_lwp['Tree_Number'] == 1]
        lwp_2_extra = rewatering_lwp[rewatering_lwp['Tree_Number'] == 2]
        lwp_3_extra = rewatering_lwp[rewatering_lwp['Tree_Number'] == 3]
        lwp_4_extra = rewatering_lwp[rewatering_lwp['Tree_Number'] == 4]
        lwp_5_extra = rewatering_lwp[rewatering_lwp['Tree_Number'] == 5]
        lwp_6_extra = rewatering_lwp[rewatering_lwp['Tree_Number'] == 6]

        ax = plt.gca()
        lwp_1_extra.plot(kind='scatter', x='TIMESTAMP', y='LWP_(MPa)', c=tree_colors[0], ax=ax, label='Tree 1')
        lwp_2_extra.plot(kind='scatter', x='TIMESTAMP', y='LWP_(MPa)', c=tree_colors[1], ax=ax, label='Tree 2')
        lwp_3_extra.plot(kind='scatter', x='TIMESTAMP', y='LWP_(MPa)', c=tree_colors[2], ax=ax, label='Tree 3')
        lwp_4_extra.plot(kind='scatter', x='TIMESTAMP', y='LWP_(MPa)', c=tree_colors[3], ax=ax, label='Tree 4')
        lwp_5_extra.plot(kind='scatter', x='TIMESTAMP', y='LWP_(MPa)', c=tree_colors[4], ax=ax, label='Tree 5')
        lwp_6_extra.plot(kind='scatter', x='TIMESTAMP', y='LWP_(MPa)', c=tree_colors[5], ax=ax, label='Tree 6')
        plt.legend()
        plt.savefig('Output_Plots/rewatering_lwps.pdf')
        plt.show()

        #Plot with panels for each day
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True)
        ax1.plot([1.0, 1.0], [lwp_means[(1, 1, 'Midday')],lwp_means[(1, 1, 'Predawn')]], marker='o', c=tree_colors[0])
        ax1.plot([2.0, 2.0], [lwp_means[(2, 1, 'Midday')],lwp_means[(2, 1, 'Predawn')]], marker='o', c=tree_colors[1])
        ax1.plot([3.0, 3.0], [lwp_means[(3, 1, 'Midday')],lwp_means[(3, 1, 'Predawn')]], marker='o', c=tree_colors[2])
        ax1.plot([4.0, 4.0], [lwp_means[(4, 1, 'Midday')],lwp_means[(4, 1, 'Predawn')]], marker='o', c=tree_colors[3])
        ax1.plot([5.0, 5.0], [lwp_means[(5, 1, 'Midday')],lwp_means[(5, 1, 'Predawn')]], marker='o', c=tree_colors[4])
        ax1.plot([6.0, 6.0], [lwp_means[(6, 1, 'Midday')],lwp_means[(6, 1, 'Predawn')]], marker='o', c=tree_colors[5])
        ax1.axhline(y=-1.5, color='k', linestyle='--')

        ax2.plot([1.0, 1.0], [lwp_means[(1, 2, 'Midday')], lwp_means[(1, 2, 'Predawn')]], marker='o', c=tree_colors[0])
        ax2.plot([2.0, 2.0], [lwp_means[(2, 2, 'Midday')], lwp_means[(2, 2, 'Predawn')]], marker='o', c=tree_colors[1])
        ax2.plot([3.0, 3.0], [lwp_means[(3, 2, 'Midday')], lwp_means[(3, 2, 'Predawn')]], marker='o', c=tree_colors[2])
        ax2.plot([4.0, 4.0], [lwp_means[(4, 2, 'Midday')], lwp_means[(4, 2, 'Predawn')]], marker='o', c=tree_colors[3])
        ax2.plot([5.0, 5.0], [lwp_means[(5, 2, 'Midday')], lwp_means[(5, 2, 'Predawn')]], marker='o', c=tree_colors[4])
        ax2.plot([6.0, 6.0], [lwp_means[(6, 2, 'Midday')], lwp_means[(6, 2, 'Predawn')]], marker='o', c=tree_colors[5])
        ax2.axhline(y=-1.5, color='k', linestyle='--')

        ax3.plot([1.0, 1.0], [lwp_means[(1, 3, 'Midday')], lwp_means[(1, 3, 'Predawn')]], marker='o', c=tree_colors[0])
        ax3.plot([2.0, 2.0], [lwp_means[(2, 3, 'Midday')], lwp_means[(2, 3, 'Predawn')]], marker='o', c=tree_colors[1])
        ax3.plot([3.0, 3.0], [lwp_means[(3, 3, 'Midday')], lwp_means[(3, 3, 'Predawn')]], marker='o', c=tree_colors[2])
        ax3.plot([4.0, 4.0], [lwp_means[(4, 3, 'Midday')], lwp_means[(4, 3, 'Predawn')]], marker='o', c=tree_colors[3])
        ax3.plot([5.0, 5.0], [lwp_means[(5, 3, 'Midday')], lwp_means[(5, 3, 'Predawn')]], marker='o', c=tree_colors[4])
        ax3.plot([6.0, 6.0], [lwp_means[(6, 3, 'Midday')], lwp_means[(6, 3, 'Predawn')]], marker='o', c=tree_colors[5])
        ax3.axhline(y=-1.5, color='k', linestyle='--')

        ax4.plot([1.0, 1.0], [lwp_means[(1, 4, 'Midday')], lwp_means[(1, 4, 'Predawn')]], marker='o', c=tree_colors[0])
        ax4.plot([2.0, 2.0], [lwp_means[(2, 4, 'Midday')], lwp_means[(2, 4, 'Predawn')]], marker='o', c=tree_colors[1])
        ax4.plot([3.0, 3.0], [lwp_means[(3, 4, 'Midday')], lwp_means[(3, 4, 'Predawn')]], marker='o', c=tree_colors[2])
        ax4.plot([4.0, 4.0], [lwp_means[(4, 4, 'Midday')], lwp_means[(4, 4, 'Predawn')]], marker='o', c=tree_colors[3])
        ax4.plot([5.0, 5.0], [lwp_means[(5, 4, 'Midday')], lwp_means[(5, 4, 'Predawn')]], marker='o', c=tree_colors[4])
        ax4.plot([6.0, 6.0], [lwp_means[(6, 4, 'Midday')], lwp_means[(6, 4, 'Predawn')]], marker='o', c=tree_colors[5])
        ax4.axhline(y=-1.5, color='k', linestyle='--')

        ax5.plot([1.0, 1.0], [lwp_means[(1, 5, 'Midday')], lwp_means[(1, 5, 'Predawn')]], marker='o', c=tree_colors[0])
        ax5.plot([2.0, 2.0], [lwp_means[(2, 5, 'Midday')], lwp_means[(2, 5, 'Predawn')]], marker='o', c=tree_colors[1])
        ax5.plot([3.0, 3.0], [lwp_means[(3, 5, 'Midday')], lwp_means[(3, 5, 'Predawn')]], marker='o', c=tree_colors[2])
        ax5.plot([4.0, 4.0], [lwp_means[(4, 5, 'Midday')], lwp_means[(4, 5, 'Predawn')]], marker='o', c=tree_colors[3])
        ax5.plot([5.0, 5.0], [lwp_means[(5, 5, 'Midday')], lwp_means[(5, 5, 'Predawn')]], marker='o', c=tree_colors[4])
        ax5.plot([6.0, 6.0], [lwp_means[(6, 5, 'Midday')], lwp_means[(6, 5, 'Predawn')]], marker='o', c=tree_colors[5])
        ax5.axhline(y=-1.5, color='k', linestyle='--')

        plt.show()

        #Plot panels by tree
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, sharey=True, figsize=(12,6))
        ax1.plot([1.0, 1.0], [lwp_means[(1, 1, 'Midday')], lwp_means[(1, 1, 'Predawn')]], marker='_', markersize=8, c=day_colors[0])
        ax1.plot([2.0, 2.0], [lwp_means[(1, 2, 'Midday')], lwp_means[(1, 2, 'Predawn')]], marker='_', markersize=8,  c=day_colors[1])
        ax1.plot([3.0, 3.0], [lwp_means[(1, 3, 'Midday')], lwp_means[(1, 3, 'Predawn')]], marker='_', markersize=8, c=day_colors[2])
        ax1.plot([4.0, 4.0], [lwp_means[(1, 4, 'Midday')], lwp_means[(1, 4, 'Predawn')]], marker='_', markersize=8, c=day_colors[3])
        ax1.plot([5.0, 5.0], [lwp_means[(1, 5, 'Midday')], lwp_means[(1, 5, 'Predawn')]], marker='_', markersize=8, c=day_colors[4])
        lwp_1_pd.plot(kind='scatter', x='Sampling_Day', y='LWP_(MPa)', c='gray', ax=ax1, alpha=0.5, marker='o')
        lwp_1_md.plot(kind='scatter', x='Sampling_Day', y='LWP_(MPa)', c='gray', ax=ax1, alpha=0.5, marker='v')
        ax1.set_xticks([1, 2, 3, 4, 5])

        ax2.plot([1.0, 1.0], [lwp_means[(2, 1, 'Midday')], lwp_means[(2, 1, 'Predawn')]], marker='_', markersize=8, c=day_colors[0])
        ax2.plot([2.0, 2.0], [lwp_means[(2, 2, 'Midday')], lwp_means[(2, 2, 'Predawn')]], marker='_', markersize=8, c=day_colors[1])
        ax2.plot([3.0, 3.0], [lwp_means[(2, 3, 'Midday')], lwp_means[(2, 3, 'Predawn')]], marker='_', markersize=8, c=day_colors[2])
        ax2.plot([4.0, 4.0], [lwp_means[(2, 4, 'Midday')], lwp_means[(2, 4, 'Predawn')]], marker='_', markersize=8, c=day_colors[3])
        ax2.plot([5.0, 5.0], [lwp_means[(2, 5, 'Midday')], lwp_means[(2, 5, 'Predawn')]], marker='_', markersize=8, c=day_colors[4])
        lwp_2_pd.plot(kind='scatter', x='Sampling_Day', y='LWP_(MPa)', c='gray', ax=ax2, alpha=0.5, marker='o')
        lwp_2_md.plot(kind='scatter', x='Sampling_Day', y='LWP_(MPa)', c='gray', ax=ax2, alpha=0.5, marker='v')
        ax2.set_xticks([1, 2, 3, 4, 5])

        ax3.plot([1.0, 1.0], [lwp_means[(3, 1, 'Midday')], lwp_means[(3, 1, 'Predawn')]], marker='_', markersize=8, c=day_colors[0])
        ax3.plot([2.0, 2.0], [lwp_means[(3, 2, 'Midday')], lwp_means[(3, 2, 'Predawn')]], marker='_', markersize=8, c=day_colors[1])
        ax3.plot([3.0, 3.0], [lwp_means[(3, 3, 'Midday')], lwp_means[(3, 3, 'Predawn')]], marker='_', markersize=8, c=day_colors[2])
        ax3.plot([4.0, 4.0], [lwp_means[(3, 4, 'Midday')], lwp_means[(3, 4, 'Predawn')]], marker='_', markersize=8, c=day_colors[3])
        ax3.plot([5.0, 5.0], [lwp_means[(3, 5, 'Midday')], lwp_means[(3, 5, 'Predawn')]], marker='_', markersize=8, c=day_colors[4])
        lwp_3_pd.plot(kind='scatter', x='Sampling_Day', y='LWP_(MPa)', c='gray', ax=ax3, alpha=0.5, marker='o')
        lwp_3_md.plot(kind='scatter', x='Sampling_Day', y='LWP_(MPa)', c='gray', ax=ax3, alpha=0.5, marker='v')
        ax3.set_xticks([1, 2, 3, 4, 5])

        ax4.plot([1.0, 1.0], [lwp_means[(4, 1, 'Midday')], lwp_means[(4, 1, 'Predawn')]], marker='_', markersize=8, c=day_colors[0])
        ax4.plot([2.0, 2.0], [lwp_means[(4, 2, 'Midday')], lwp_means[(4, 2, 'Predawn')]], marker='_', markersize=8, c=day_colors[1])
        ax4.plot([3.0, 3.0], [lwp_means[(4, 3, 'Midday')], lwp_means[(4, 3, 'Predawn')]], marker='_', markersize=8, c=day_colors[2])
        ax4.plot([4.0, 4.0], [lwp_means[(4, 4, 'Midday')], lwp_means[(4, 4, 'Predawn')]], marker='_', markersize=8, c=day_colors[3])
        ax4.plot([5.0, 5.0], [lwp_means[(4, 5, 'Midday')], lwp_means[(4, 5, 'Predawn')]], marker='_', markersize=8, c=day_colors[4])
        lwp_4_pd.plot(kind='scatter', x='Sampling_Day', y='LWP_(MPa)', c='gray', ax=ax4, alpha=0.5, marker='o')
        lwp_4_md.plot(kind='scatter', x='Sampling_Day', y='LWP_(MPa)', c='gray', ax=ax4, alpha=0.5, marker='v')
        ax4.set_xticks([1, 2, 3, 4, 5])

        ax5.plot([1.0, 1.0], [lwp_means[(5, 1, 'Midday')], lwp_means[(5, 1, 'Predawn')]], marker='_', markersize=8, c=day_colors[0])
        ax5.plot([2.0, 2.0], [lwp_means[(5, 2, 'Midday')], lwp_means[(5, 2, 'Predawn')]], marker='_', markersize=8, c=day_colors[1])
        ax5.plot([3.0, 3.0], [lwp_means[(5, 3, 'Midday')], lwp_means[(5, 3, 'Predawn')]], marker='_', markersize=8, c=day_colors[2])
        ax5.plot([4.0, 4.0], [lwp_means[(5, 4, 'Midday')], lwp_means[(5, 4, 'Predawn')]], marker='_', markersize=8, c=day_colors[3])
        ax5.plot([5.0, 5.0], [lwp_means[(5, 5, 'Midday')], lwp_means[(5, 5, 'Predawn')]], marker='_', markersize=8, c=day_colors[4])
        lwp_5_pd.plot(kind='scatter', x='Sampling_Day', y='LWP_(MPa)', c='gray', ax=ax5, alpha=0.5, marker='o')
        lwp_5_md.plot(kind='scatter', x='Sampling_Day', y='LWP_(MPa)', c='gray', ax=ax5, alpha=0.5, marker='v')
        ax5.set_xticks([1, 2, 3, 4, 5])

        ax6.plot([1.0, 1.0], [lwp_means[(6, 1, 'Midday')], lwp_means[(6, 1, 'Predawn')]], marker='_', markersize=8, c=day_colors[0])
        ax6.plot([2.0, 2.0], [lwp_means[(6, 2, 'Midday')], lwp_means[(6, 2, 'Predawn')]], marker='_', markersize=8, c=day_colors[1])
        ax6.plot([3.0, 3.0], [lwp_means[(6, 3, 'Midday')], lwp_means[(6, 3, 'Predawn')]], marker='_', markersize=8, c=day_colors[2])
        ax6.plot([4.0, 4.0], [lwp_means[(6, 4, 'Midday')], lwp_means[(6, 4, 'Predawn')]], marker='_', markersize=8, c=day_colors[3])
        ax6.plot([5.0, 5.0], [lwp_means[(6, 5, 'Midday')], lwp_means[(6, 5, 'Predawn')]], marker='_', markersize=8, c=day_colors[4])
        lwp_6_pd.plot(kind='scatter', x='Sampling_Day', y='LWP_(MPa)', c='gray', ax=ax6, alpha=0.5, marker='o')
        lwp_6_md.plot(kind='scatter', x='Sampling_Day', y='LWP_(MPa)', c='gray', ax=ax6, alpha=0.5, marker='v')
        ax6.set_xticks([1, 2, 3, 4, 5])

        plt.savefig('Output_Plots/lwp_tree_split.pdf')
        plt.show()

    if soil_moisture is not None:
        soil_1 = soil_moisture[soil_moisture['Tree_Number'] == 1]
        soil_1_1 = soil_1[soil_1['Sampling_Day'] == 1]
        soil_1_2 = soil_1[soil_1['Sampling_Day'] == 2]
        soil_1_3 = soil_1[soil_1['Sampling_Day'] == 3]
        soil_1_4 = soil_1[soil_1['Sampling_Day'] == 4]
        soil_1_5 = soil_1[soil_1['Sampling_Day'] == 5]

        soil_2 = soil_moisture[soil_moisture['Tree_Number'] == 2]
        soil_2_1 = soil_2[soil_2['Sampling_Day'] == 1]
        soil_2_2 = soil_2[soil_2['Sampling_Day'] == 2]
        soil_2_3 = soil_2[soil_2['Sampling_Day'] == 3]
        soil_2_4 = soil_2[soil_2['Sampling_Day'] == 4]
        soil_2_5 = soil_2[soil_2['Sampling_Day'] == 5]

        soil_3 = soil_moisture[soil_moisture['Tree_Number'] == 3]
        soil_3_1 = soil_3[soil_3['Sampling_Day'] == 1]
        soil_3_2 = soil_3[soil_3['Sampling_Day'] == 2]
        soil_3_3 = soil_3[soil_3['Sampling_Day'] == 3]
        soil_3_4 = soil_3[soil_3['Sampling_Day'] == 4]
        soil_3_5 = soil_3[soil_3['Sampling_Day'] == 5]

        soil_4 = soil_moisture[soil_moisture['Tree_Number'] == 4]
        soil_4_1 = soil_4[soil_4['Sampling_Day'] == 1]
        soil_4_2 = soil_4[soil_4['Sampling_Day'] == 2]
        soil_4_3 = soil_4[soil_4['Sampling_Day'] == 3]
        soil_4_4 = soil_4[soil_4['Sampling_Day'] == 4]
        soil_4_5 = soil_4[soil_4['Sampling_Day'] == 5]

        soil_5 = soil_moisture[soil_moisture['Tree_Number'] == 5]
        soil_5_1 = soil_5[soil_5['Sampling_Day'] == 1]
        soil_5_2 = soil_5[soil_5['Sampling_Day'] == 2]
        soil_5_3 = soil_5[soil_5['Sampling_Day'] == 3]
        soil_5_4 = soil_5[soil_5['Sampling_Day'] == 4]
        soil_5_5 = soil_5[soil_5['Sampling_Day'] == 5]

        soil_6 = soil_moisture[soil_moisture['Tree_Number'] == 6]
        soil_6_1 = soil_6[soil_6['Sampling_Day'] == 1]
        soil_6_2 = soil_6[soil_6['Sampling_Day'] == 2]
        soil_6_3 = soil_6[soil_6['Sampling_Day'] == 3]
        soil_6_4 = soil_6[soil_6['Sampling_Day'] == 4]
        soil_6_5 = soil_6[soil_6['Sampling_Day'] == 5]

        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, sharey=True, sharex=True, figsize=(12,6))

        soil_1_1.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[0], ax=ax1, legend=None)
        soil_1_2.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[1], ax=ax1, legend=None)
        soil_1_3.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[2], ax=ax1, legend=None)
        soil_1_4.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[3], ax=ax1, legend=None)
        soil_1_5.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[4], ax=ax1, legend=None)
        ax1.invert_yaxis()

        soil_2_1.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[0], ax=ax2, legend=None)
        soil_2_2.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[1], ax=ax2, legend=None)
        soil_2_3.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[2], ax=ax2, legend=None)
        soil_2_4.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[3], ax=ax2, legend=None)
        soil_2_5.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[4], ax=ax2, legend=None)

        soil_3_1.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[0], ax=ax3, legend=None)
        soil_3_2.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[1], ax=ax3, legend=None)
        soil_3_3.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[2], ax=ax3, legend=None)
        soil_3_4.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[3], ax=ax3, legend=None)
        soil_3_5.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[4], ax=ax3, legend=None)

        soil_4_1.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[0], ax=ax4, legend=None)
        soil_4_2.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[1], ax=ax4, legend=None)
        soil_4_3.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[2], ax=ax4, legend=None)
        soil_4_4.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[3], ax=ax4, legend=None)
        soil_4_5.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[4], ax=ax4, legend=None)

        soil_5_1.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[0], ax=ax5, legend=None)
        soil_5_2.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[1], ax=ax5, legend=None)
        soil_5_3.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[2], ax=ax5, legend=None)
        soil_5_4.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[3], ax=ax5, legend=None)
        soil_5_5.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[4], ax=ax5, legend=None)

        soil_6_1.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[0], ax=ax6, legend=None)
        soil_6_2.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[1], ax=ax6, legend=None)
        soil_6_3.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[2], ax=ax6, legend=None)
        soil_6_4.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[3], ax=ax6, legend=None)
        soil_6_5.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=day_colors[4], ax=ax6, legend=None)

        plt.savefig('Output_Plots/soil_moisture_by_tree.pdf')
        plt.show()

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True, sharex=True, figsize=(12,6))
        soil_1_1.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[0], ax=ax1, legend=None)
        soil_2_1.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[1], ax=ax1, legend=None)
        soil_3_1.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[2], ax=ax1, legend=None)
        soil_4_1.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[3], ax=ax1, legend=None)
        soil_5_1.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[4], ax=ax1, legend=None)
        soil_6_1.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[5], ax=ax1, legend=None)
        ax1.invert_yaxis()

        soil_1_2.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[0], ax=ax2, legend=None)
        soil_2_2.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[1], ax=ax2, legend=None)
        soil_3_2.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[2], ax=ax2, legend=None)
        soil_4_2.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[3], ax=ax2, legend=None)
        soil_5_2.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[4], ax=ax2, legend=None)
        soil_6_2.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[5], ax=ax2, legend=None)

        soil_1_3.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[0], ax=ax3, legend=None)
        soil_2_3.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[1], ax=ax3, legend=None)
        soil_3_3.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[2], ax=ax3, legend=None)
        soil_4_3.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[3], ax=ax3, legend=None)
        soil_5_3.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[4], ax=ax3, legend=None)
        soil_6_3.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[5], ax=ax3, legend=None)

        soil_1_4.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[0], ax=ax4, legend=None)
        soil_2_4.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[1], ax=ax4, legend=None)
        soil_3_4.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[2], ax=ax4, legend=None)
        soil_4_4.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[3], ax=ax4, legend=None)
        soil_5_4.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[4], ax=ax4, legend=None)
        soil_6_4.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[5], ax=ax4, legend=None)

        soil_1_5.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[0], ax=ax5, legend=None)
        soil_2_5.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[1], ax=ax5, legend=None)
        soil_3_5.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[2], ax=ax5, legend=None)
        soil_4_5.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[3], ax=ax5, legend=None)
        soil_5_5.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[4], ax=ax5, legend=None)
        soil_6_5.plot(style='.-', x='VWC_Percent', y='Depth_cm', c=tree_colors[5], ax=ax5, legend=None)

        plt.savefig('Output_Plots/soil_moisture_grouped_by_day.pdf')
        plt.show()

        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, sharey=True, sharex=True, figsize=(12, 6))

        soil_1_1.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[0], ax=ax1, legend=None)
        soil_1_2.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[1], ax=ax1, legend=None)
        soil_1_3.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[2], ax=ax1, legend=None)
        soil_1_4.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[3], ax=ax1, legend=None)
        soil_1_5.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[4], ax=ax1, legend=None)
        ax1.invert_yaxis()

        soil_2_1.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[0], ax=ax2, legend=None)
        soil_2_2.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[1], ax=ax2, legend=None)
        soil_2_3.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[2], ax=ax2, legend=None)
        soil_2_4.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[3], ax=ax2, legend=None)
        soil_2_5.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[4], ax=ax2, legend=None)

        soil_3_1.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[0], ax=ax3, legend=None)
        soil_3_2.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[1], ax=ax3, legend=None)
        soil_3_3.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[2], ax=ax3, legend=None)
        soil_3_4.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[3], ax=ax3, legend=None)
        soil_3_5.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[4], ax=ax3, legend=None)

        soil_4_1.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[0], ax=ax4, legend=None)
        soil_4_2.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[1], ax=ax4, legend=None)
        soil_4_3.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[2], ax=ax4, legend=None)
        soil_4_4.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[3], ax=ax4, legend=None)
        soil_4_5.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[4], ax=ax4, legend=None)

        soil_5_1.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[0], ax=ax5, legend=None)
        soil_5_2.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[1], ax=ax5, legend=None)
        soil_5_3.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[2], ax=ax5, legend=None)
        soil_5_4.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[3], ax=ax5, legend=None)
        soil_5_5.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[4], ax=ax5, legend=None)

        soil_6_1.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[0], ax=ax6, legend=None)
        soil_6_2.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[1], ax=ax6, legend=None)
        soil_6_3.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[2], ax=ax6, legend=None)
        soil_6_4.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[3], ax=ax6, legend=None)
        soil_6_5.plot(style='.-', x='psi_soil', y='Depth_cm', c=day_colors[4], ax=ax6, legend=None)

        plt.savefig('Output_Plots/soil_water_potential_by_tree.pdf')
        plt.show()

    if isotopes is not None:
        soil_isotopes = isotopes[isotopes['Sample_Type'] == 'Soil']
        soil_isotopes = soil_isotopes.sort_values(by=['Depth'])
        plant_isotopes = isotopes[isotopes['Sample_Type'] == 'Plant']
        trans_isotopes = isotopes[isotopes['Sample_Type'] == 'Transpiration']
        trans_isotopes_sampling_days = trans_isotopes[trans_isotopes['Sampling_Day'] != 'NaN']
        petiole_isotopes = isotopes[isotopes['Sample_Type'] == 'Petiole']
        leaf_isotopes = isotopes[isotopes['Sample_Type'] == 'Leaf']

        soil_repeats = isotopes[isotopes['Sample_Type'] == 'Soil_Repeat']
        plant_repeats = isotopes[isotopes['Sample_Type'] == 'Plant_Repeat']
        plant_repeats = plant_repeats.add_suffix('_repeat')
        soil_repeats = soil_repeats.add_suffix('_repeat')
        plant_repeats = plant_repeats.rename(columns={"Sample_ID_repeat": "Sample_ID"})
        soil_repeats = soil_repeats.rename(columns={"Sample_ID_repeat": "Sample_ID"})
        soil_combo = pd.merge(soil_repeats, soil_isotopes, on='Sample_ID')
        plant_combo = pd.merge(plant_repeats, plant_isotopes, on='Sample_ID')

        ##think this is old correction??
        #plant_isotopes["D_corrected"] = plant_isotopes["del_D"]*0.7709 + 14.631
        #plant_isotopes["O_corrected"] = plant_isotopes["del_O"] * 0.7177 + 9.5903

        trial_A_isotopes = isotopes[isotopes['Sample_Type'] == 'Trial_Soil_A']
        trial_B_isotopes = isotopes[isotopes['Sample_Type'] == 'Trial_Soil_B']
        trial_C_isotopes = isotopes[isotopes['Sample_Type'] == 'Trial_Soil_C']
        trial_trans_isotopes = isotopes[isotopes['Sample_Type'] == 'Trial_Transpiration']

        trial_A_isotopes_T1 = trial_A_isotopes[trial_A_isotopes['Tree_Number'] == 11]
        trial_B_isotopes_T1 = trial_B_isotopes[trial_B_isotopes['Tree_Number'] == 11]
        trial_C_isotopes_T1 = trial_C_isotopes[trial_C_isotopes['Tree_Number'] == 11]

        trial_A_isotopes_T2 = trial_A_isotopes[trial_A_isotopes['Tree_Number'] == 12]
        trial_B_isotopes_T2 = trial_B_isotopes[trial_B_isotopes['Tree_Number'] == 12]
        trial_C_isotopes_T2 = trial_C_isotopes[trial_C_isotopes['Tree_Number'] == 12]
        trial_T1_trans = trial_trans_isotopes[trial_trans_isotopes['Tree_Number'] == 11]
        trial_T2_trans = trial_trans_isotopes[trial_trans_isotopes['Tree_Number'] == 12]


        plant_day_1 = plant_isotopes[plant_isotopes['Sampling_Day'] == 1]
        plant_day_2 = plant_isotopes[plant_isotopes['Sampling_Day'] == 2]
        plant_day_3 = plant_isotopes[plant_isotopes['Sampling_Day'] == 3]
        plant_day_4 = plant_isotopes[plant_isotopes['Sampling_Day'] == 4]
        plant_day_5 = plant_isotopes[plant_isotopes['Sampling_Day'] == 5]

        petiole_day_1 = petiole_isotopes[petiole_isotopes['Sampling_Day'] == 1]
        petiole_day_2 = petiole_isotopes[petiole_isotopes['Sampling_Day'] == 2]
        petiole_day_3 = petiole_isotopes[petiole_isotopes['Sampling_Day'] == 3]
        petiole_day_4 = petiole_isotopes[petiole_isotopes['Sampling_Day'] == 4]
        petiole_day_5 = petiole_isotopes[petiole_isotopes['Sampling_Day'] == 5]

        leaf_day_1 = leaf_isotopes[leaf_isotopes['Sampling_Day'] == 1]
        leaf_day_2 = leaf_isotopes[leaf_isotopes['Sampling_Day'] == 2]
        leaf_day_3 = leaf_isotopes[leaf_isotopes['Sampling_Day'] == 3]
        leaf_day_4 = leaf_isotopes[leaf_isotopes['Sampling_Day'] == 4]
        leaf_day_5 = leaf_isotopes[leaf_isotopes['Sampling_Day'] == 5]

        trans_day_1 = trans_isotopes_sampling_days[trans_isotopes_sampling_days['Sampling_Day'] == 1]
        trans_day_2 = trans_isotopes_sampling_days[trans_isotopes_sampling_days['Sampling_Day'] == 2]
        trans_day_3 = trans_isotopes_sampling_days[trans_isotopes_sampling_days['Sampling_Day'] == 3]
        trans_day_4 = trans_isotopes_sampling_days[trans_isotopes_sampling_days['Sampling_Day'] == 4]
        trans_day_5 = trans_isotopes_sampling_days[trans_isotopes_sampling_days['Sampling_Day'] == 5]

        Tree_1_plant = plant_isotopes[plant_isotopes['Tree_Number'] == 1]
        Tree_1_soil = soil_isotopes[soil_isotopes['Tree_Number'] == 1]
        Tree_1_trans = trans_isotopes[trans_isotopes['Tree_Number'] == 1]

        Tree_2_plant = plant_isotopes[plant_isotopes['Tree_Number'] == 2]
        Tree_2_soil = soil_isotopes[soil_isotopes['Tree_Number'] == 2]
        Tree_2_trans = trans_isotopes[trans_isotopes['Tree_Number'] == 2]

        Tree_3_plant = plant_isotopes[plant_isotopes['Tree_Number'] == 3]
        Tree_3_soil = soil_isotopes[soil_isotopes['Tree_Number'] == 3]
        Tree_3_trans = trans_isotopes[trans_isotopes['Tree_Number'] == 3]

        Tree_4_plant = plant_isotopes[plant_isotopes['Tree_Number'] == 4]
        Tree_4_soil = soil_isotopes[soil_isotopes['Tree_Number'] == 4]
        Tree_4_trans = trans_isotopes[trans_isotopes['Tree_Number'] == 4]

        Tree_5_plant = plant_isotopes[plant_isotopes['Tree_Number'] == 5]
        Tree_5_soil = soil_isotopes[soil_isotopes['Tree_Number'] == 5]
        Tree_5_trans = trans_isotopes[trans_isotopes['Tree_Number'] == 5]

        Tree_6_plant = plant_isotopes[plant_isotopes['Tree_Number'] == 6]
        Tree_6_soil = soil_isotopes[soil_isotopes['Tree_Number'] == 6]
        Tree_6_trans = trans_isotopes[trans_isotopes['Tree_Number'] == 6]

        Tree_1_soil_1 = Tree_1_soil[Tree_1_soil['Sampling_Day'] == 1]
        Tree_1_soil_2 = Tree_1_soil[Tree_1_soil['Sampling_Day'] == 2]
        Tree_1_soil_3 = Tree_1_soil[Tree_1_soil['Sampling_Day'] == 3]
        Tree_1_soil_4 = Tree_1_soil[Tree_1_soil['Sampling_Day'] == 4]
        Tree_1_soil_5 = Tree_1_soil[Tree_1_soil['Sampling_Day'] == 5]

        Tree_2_soil_1 = Tree_2_soil[Tree_2_soil['Sampling_Day'] == 1]
        Tree_2_soil_2 = Tree_2_soil[Tree_2_soil['Sampling_Day'] == 2]
        Tree_2_soil_3 = Tree_2_soil[Tree_2_soil['Sampling_Day'] == 3]
        Tree_2_soil_4 = Tree_2_soil[Tree_2_soil['Sampling_Day'] == 4]
        Tree_2_soil_5 = Tree_2_soil[Tree_2_soil['Sampling_Day'] == 5]

        Tree_3_soil_1 = Tree_3_soil[Tree_3_soil['Sampling_Day'] == 1]
        Tree_3_soil_2 = Tree_3_soil[Tree_3_soil['Sampling_Day'] == 2]
        Tree_3_soil_3 = Tree_3_soil[Tree_3_soil['Sampling_Day'] == 3]
        Tree_3_soil_4 = Tree_3_soil[Tree_3_soil['Sampling_Day'] == 4]
        Tree_3_soil_5 = Tree_3_soil[Tree_3_soil['Sampling_Day'] == 5]

        Tree_4_soil_1 = Tree_4_soil[Tree_4_soil['Sampling_Day'] == 1]
        Tree_4_soil_2 = Tree_4_soil[Tree_4_soil['Sampling_Day'] == 2]
        Tree_4_soil_3 = Tree_4_soil[Tree_4_soil['Sampling_Day'] == 3]
        Tree_4_soil_4 = Tree_4_soil[Tree_4_soil['Sampling_Day'] == 4]
        Tree_4_soil_5 = Tree_4_soil[Tree_4_soil['Sampling_Day'] == 5]

        Tree_5_soil_1 = Tree_5_soil[Tree_5_soil['Sampling_Day'] == 1]
        Tree_5_soil_2 = Tree_5_soil[Tree_5_soil['Sampling_Day'] == 2]
        Tree_5_soil_3 = Tree_5_soil[Tree_5_soil['Sampling_Day'] == 3]
        Tree_5_soil_4 = Tree_5_soil[Tree_5_soil['Sampling_Day'] == 4]
        Tree_5_soil_5 = Tree_5_soil[Tree_5_soil['Sampling_Day'] == 5]

        Tree_6_soil_1 = Tree_6_soil[Tree_6_soil['Sampling_Day'] == 1]
        Tree_6_soil_2 = Tree_6_soil[Tree_6_soil['Sampling_Day'] == 2]
        Tree_6_soil_3 = Tree_6_soil[Tree_6_soil['Sampling_Day'] == 3]
        Tree_6_soil_4 = Tree_6_soil[Tree_6_soil['Sampling_Day'] == 4]
        Tree_6_soil_5 = Tree_6_soil[Tree_6_soil['Sampling_Day'] == 5]

        soil_day1 = soil_isotopes[soil_isotopes['TIMESTAMP'] >= "2021-07-08 00:00:00"]
        soil_day1 = soil_day1[soil_day1['TIMESTAMP'] <= "2021-07-09 00:00:00"]

        plant_day1 = plant_isotopes[plant_isotopes['TIMESTAMP'] >= "2021-07-08 00:00:00"]
        plant_day1 = plant_day1[plant_day1['TIMESTAMP'] <= "2021-07-09 00:00:00"]

        soil_day2 = soil_isotopes[soil_isotopes['TIMESTAMP'] >= "2021-07-11 00:00:00"]
        soil_day2 = soil_day2[soil_day2['TIMESTAMP'] <= "2021-07-12 00:00:00"]

        plant_day2 = plant_isotopes[plant_isotopes['TIMESTAMP'] >= "2021-07-11 00:00:00"]
        plant_day2 = plant_day2[plant_day2['TIMESTAMP'] <= "2021-07-12 00:00:00"]

        soil_day3 = soil_isotopes[soil_isotopes['TIMESTAMP'] >= "2021-07-14 00:00:00"]
        soil_day3 = soil_day3[soil_day3['TIMESTAMP'] <= "2021-07-15 00:00:00"]

        plant_day3 = plant_isotopes[plant_isotopes['TIMESTAMP'] >= "2021-07-14 00:00:00"]
        plant_day3 = plant_day3[plant_day3['TIMESTAMP'] <= "2021-07-15 00:00:00"]

        soil_day4 = soil_isotopes[soil_isotopes['TIMESTAMP'] >= "2021-07-18 00:00:00"]
        soil_day4 = soil_day4[soil_day4['TIMESTAMP'] <= "2021-07-19 00:00:00"]

        plant_day4 = plant_isotopes[plant_isotopes['TIMESTAMP'] >= "2021-07-18 00:00:00"]
        plant_day4 = plant_day4[plant_day4['TIMESTAMP'] <= "2021-07-19 00:00:00"]

        soil_day5 = soil_isotopes[soil_isotopes['TIMESTAMP'] >= "2021-07-21 00:00:00"]
        soil_day5 = soil_day5[soil_day5['TIMESTAMP'] <= "2021-07-22 00:00:00"]

        plant_day5 = plant_isotopes[plant_isotopes['TIMESTAMP'] >= "2021-07-21 00:00:00"]
        plant_day5 = plant_day5[plant_day5['TIMESTAMP'] <= "2021-07-22 00:00:00"]

        first_round = soil_isotopes[soil_isotopes['TIMESTAMP'] <= "2021-07-15 00:00:00"]
        second_round = soil_isotopes[soil_isotopes['TIMESTAMP'] > "2021-07-15 00:00:00"]
        plant_iso_cols = ['#F0A58F', '#EB548C', '#Af4BCE', '#29066B']
        ax = plt.gca()
        plant_combo.plot(kind='scatter', x='del_O_repeat', y='del_O', marker="s", c=plant_iso_cols[1], ax=ax, s=40, label='Plant')
        soil_combo.plot(kind='scatter', x='del_O_repeat', y='del_O', marker="s", c=plant_iso_cols[3], ax=ax, s=40, label='Soil')
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against eachother
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='1:1 line')
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        plt.ylabel('IRIS 18O')
        plt.xlabel('IRMS 18O')
        plt.legend()
        plt.savefig('Output_Plots/isotope_repeats_18O.pdf')
        plt.show()

        ax = plt.gca()
        plant_combo.plot(kind='scatter', x='del_D_repeat', y='del_D', marker="s", c=plant_iso_cols[1], ax=ax, s=40, label='Plant')
        soil_combo.plot(kind='scatter', x='del_D_repeat', y='del_D', marker="s", c=plant_iso_cols[3], ax=ax, s=40, label='Soil')
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against eachother
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='1:1 line')
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        plt.ylabel('IRIS D')
        plt.xlabel('IRMS D')
        plt.legend()
        plt.savefig('Output_Plots/isotope_repeats_D.pdf')
        plt.show()

        # ax = plt.gca()
        # col = first_round.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                              50: '#253494', 60: '#081d58'})
        # first_round.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax, alpha=0.5)
        # col1 = plant_day1.Tree_Number.map({1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        # col2 = plant_day2.Tree_Number.map(
        #     {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        # col3 = plant_day3.Tree_Number.map(
        #     {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        # plant_day1.plot(kind='scatter', x='del_O', y='del_D', marker="s", c=col1, ax=ax, s=40)
        # plant_day2.plot(kind='scatter', x='del_O', y='del_D', marker="^", c=col2, ax=ax, s=40)
        # plant_day3.plot(kind='scatter', x='del_O', y='del_D', marker="x", c=col3, ax=ax, s=40)
        # plt.xlim(-20, 90)
        # plt.ylim(-100, 260)
        # plt.savefig('Output_Plots/isotopes_round_1.pdf')
        # plt.show()
        #
        # ax = plt.gca()
        # col = first_round.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                              50: '#253494', 60: '#081d58'})
        # first_round.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax)
        # plt.savefig('Output_Plots/isotopes_round_1_only_soil.pdf')
        # plt.show()
        #
        # ax = plt.gca()
        # col = first_round.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                              50: '#253494', 60: '#081d58'})
        # first_round.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax, alpha=0.5)
        # plt.savefig('Output_Plots/isotopes_round_1_only_soil_fade.pdf')
        # plt.show()
        #
        # ax = plt.gca()
        # col = second_round.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                               50: '#253494', 60: '#081d58'})
        # second_round.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax, alpha=0.5)
        # col3 = plant_day3.Tree_Number.map(
        #     {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        # col4 = plant_day4.Tree_Number.map(
        #     {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        # col5 = plant_day5.Tree_Number.map(
        #     {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        # plant_day3.plot(kind='scatter', x='del_O', y='del_D', marker="x", c=col3, ax=ax, s=40)
        # plant_day4.plot(kind='scatter', x='del_O', y='del_D', marker="s", c=col4, ax=ax, s=40)
        # plant_day5.plot(kind='scatter', x='del_O', y='del_D', marker="^", c=col5, ax=ax, s=40)
        # plt.savefig('Output_Plots/isotopes_round_2_plus_day3.pdf')
        # plt.show()
        #
        # ax = plt.gca()
        # col = second_round.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                               50: '#253494', 60: '#081d58'})
        # second_round.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax, alpha=0.5)
        # col3 = plant_day3.Tree_Number.map(
        #     {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        # col4 = plant_day4.Tree_Number.map(
        #     {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        # col5 = plant_day5.Tree_Number.map(
        #     {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        # #plant_day3.plot(kind='scatter', x='del_O', y='del_D', marker="x", c=col3, ax=ax, s=40)
        # plant_day4.plot(kind='scatter', x='del_O', y='del_D', marker="s", c=col4, ax=ax, s=40)
        # plant_day5.plot(kind='scatter', x='del_O', y='del_D', marker="^", c=col5, ax=ax, s=40)
        # plt.xlim(-20, 90)
        # plt.ylim(-100, 260)
        # plt.savefig('Output_Plots/isotopes_round_2_only.pdf')
        # plt.show()
        #
        # ax = plt.gca()
        # col = second_round.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                               50: '#253494', 60: '#081d58'})
        # second_round.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax)
        # plt.savefig('Output_Plots/isotopes_round_2_only_soil.pdf')
        # plt.show()
        #
        # ax = plt.gca()
        # col = second_round.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                               50: '#253494', 60: '#081d58'})
        # second_round.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax, alpha=0.5)
        # plt.savefig('Output_Plots/isotopes_round_2_only_soil_fade.pdf')
        # plt.show()
        #
        #
        #
        # col = plant_isotopes.Tree_Number.map(
        #     {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'})
        # plant_isotopes.plot(kind='scatter', x='del_O', y='del_D', c=col)
        # plt.show()
        #
        # fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, sharey=True, sharex=True, figsize=(12, 6))
        # #col = Tree_1_soil.Depth.map({0: '#d73027', 10: '#fc8d59', 20: '#fee090', 30: '#ffffbf', 40: '#e0f3f8',
        #                            #   50: '#91bfdb', 60: '#4575b4'})
        # col = Tree_1_soil.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                              50: '#253494', 60: '#081d58'})
        # Tree_1_soil.plot(kind='scatter', x='Sampling_Day', y='del_D', c=col, ax=ax1, alpha=0.8, s=60)
        # Tree_1_plant.plot(kind='scatter', x='Sampling_Day', y='del_D', c='k', ax=ax1, s=80, alpha=0.8, marker='x')
        # ax1.set_xticks([1, 2, 3, 4, 5])
        #
        # col = Tree_2_soil.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                              50: '#253494', 60: '#081d58'})
        # Tree_2_soil.plot(kind='scatter', x='Sampling_Day', y='del_D', c=col, ax=ax2, alpha=0.8, s=60)
        # Tree_2_plant.plot(kind='scatter', x='Sampling_Day', y='del_D', c='k', ax=ax2, s=80, alpha=0.8, marker='x')
        # ax2.set_xticks([1, 2, 3, 4, 5])
        #
        # col = Tree_3_soil.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                              50: '#253494', 60: '#081d58'})
        # Tree_3_soil.plot(kind='scatter', x='Sampling_Day', y='del_D', c=col, ax=ax3, alpha=0.8, s=60)
        # Tree_3_plant.plot(kind='scatter', x='Sampling_Day', y='del_D', c='k', ax=ax3, s=80, alpha=0.8, marker='x')
        # ax3.set_xticks([1, 2, 3, 4, 5])
        #
        # col = Tree_4_soil.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                              50: '#253494', 60: '#081d58'})
        # Tree_4_soil.plot(kind='scatter', x='Sampling_Day', y='del_D', c=col, ax=ax4, alpha=0.8, s=60)
        # Tree_4_plant.plot(kind='scatter', x='Sampling_Day', y='del_D', c='k', ax=ax4, s=80, alpha=0.8, marker='x')
        # ax4.set_xticks([1, 2, 3, 4, 5])
        #
        # col = Tree_5_soil.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                              50: '#253494', 60: '#081d58'})
        # Tree_5_soil.plot(kind='scatter', x='Sampling_Day', y='del_D', c=col, ax=ax5, alpha=0.8, s=60)
        # Tree_5_plant.plot(kind='scatter', x='Sampling_Day', y='del_D', c='k', ax=ax5, s=80, alpha=0.8, marker='x')
        # ax5.set_xticks([1, 2, 3, 4, 5])
        #
        # col = Tree_6_soil.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                              50: '#253494', 60: '#081d58'})
        # Tree_6_soil.plot(kind='scatter', x='Sampling_Day', y='del_D', c=col, ax=ax6, alpha=0.8, s=60)
        # Tree_6_plant.plot(kind='scatter', x='Sampling_Day', y='del_D', c='k', ax=ax6, s=80, alpha=0.8, marker='x')
        # ax6.set_xticks([1, 2, 3, 4, 5])
        #
        # plt.savefig('Output_Plots/D_time_series.pdf')
        # plt.show()
        #
        # fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, sharey=True, sharex=True, figsize=(12, 6))
        # #col = Tree_1_soil.Depth.map({0: '#d73027', 10: '#fc8d59', 20: '#fee090', 30: '#ffffbf', 40: '#e0f3f8',
        #                            #   50: '#91bfdb', 60: '#4575b4'})
        # col = Tree_1_soil.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                              50: '#253494', 60: '#081d58'})
        # Tree_1_soil.plot(kind='scatter', x='Sampling_Day', y='del_O', c=col, ax=ax1, alpha=0.8, s=60)
        # Tree_1_plant.plot(kind='scatter', x='Sampling_Day', y='del_O', c='k', ax=ax1, s=80, alpha=0.8, marker='x')
        # ax1.set_xticks([1, 2, 3, 4, 5])
        #
        # col = Tree_2_soil.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                              50: '#253494', 60: '#081d58'})
        # Tree_2_soil.plot(kind='scatter', x='Sampling_Day', y='del_O', c=col, ax=ax2, alpha=0.8, s=60)
        # Tree_2_plant.plot(kind='scatter', x='Sampling_Day', y='del_O', c='k', ax=ax2, s=80, alpha=0.8, marker='x')
        # ax2.set_xticks([1, 2, 3, 4, 5])
        #
        # col = Tree_3_soil.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                              50: '#253494', 60: '#081d58'})
        # Tree_3_soil.plot(kind='scatter', x='Sampling_Day', y='del_O', c=col, ax=ax3, alpha=0.8, s=60)
        # Tree_3_plant.plot(kind='scatter', x='Sampling_Day', y='del_O', c='k', ax=ax3, s=80, alpha=0.8, marker='x')
        # ax3.set_xticks([1, 2, 3, 4, 5])
        #
        # col = Tree_4_soil.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                              50: '#253494', 60: '#081d58'})
        # Tree_4_soil.plot(kind='scatter', x='Sampling_Day', y='del_O', c=col, ax=ax4, alpha=0.8, s=60)
        # Tree_4_plant.plot(kind='scatter', x='Sampling_Day', y='del_O', c='k', ax=ax4, s=80, alpha=0.8, marker='x')
        # ax4.set_xticks([1, 2, 3, 4, 5])
        #
        # col = Tree_5_soil.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                              50: '#253494', 60: '#081d58'})
        # Tree_5_soil.plot(kind='scatter', x='Sampling_Day', y='del_O', c=col, ax=ax5, alpha=0.8, s=60)
        # Tree_5_plant.plot(kind='scatter', x='Sampling_Day', y='del_O', c='k', ax=ax5, s=80, alpha=0.8, marker='x')
        # ax5.set_xticks([1, 2, 3, 4, 5])
        #
        # col = Tree_6_soil.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                              50: '#253494', 60: '#081d58'})
        # Tree_6_soil.plot(kind='scatter', x='Sampling_Day', y='del_O', c=col, ax=ax6, alpha=0.8, s=60)
        # Tree_6_plant.plot(kind='scatter', x='Sampling_Day', y='del_O', c='k', ax=ax6, s=80, alpha=0.8, marker='x')
        # ax6.set_xticks([1, 2, 3, 4, 5])
        #
        # plt.savefig('Output_Plots/18O_time_series.pdf')
        # plt.show()
        #
        # fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey=True, sharex=True, figsize=(12, 8))
        # col = Tree_1_plant.Sampling_Day.map({1:'#f4a582', 2:'#d6604d', 3:'#b2182b', 4:'#92c5de', 5:'#2166ac'})
        # Tree_1_plant.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax1, s=80)
        #
        # col = Tree_2_plant.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        # Tree_2_plant.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax2, s=80)
        #
        # col = Tree_3_plant.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        # Tree_3_plant.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax3, s=80)
        #
        # col = Tree_4_plant.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        # Tree_4_plant.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax4, s=80)
        #
        # col = Tree_5_plant.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        # Tree_5_plant.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax5, s=80)
        #
        # col = Tree_6_plant.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        # Tree_6_plant.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax6, s=80)
        #
        # plt.savefig('Output_Plots/plant_dual_isotope_time_series.pdf')
        # plt.show()
        #
        # fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, sharey=True, sharex=True, figsize=(12,6))
        #
        # Tree_1_soil_1.plot(style='.-', x='del_D', y='Depth', c=day_colors[0], ax=ax1, legend=None)
        # Tree_1_soil_2.plot(style='.-', x='del_D', y='Depth', c=day_colors[1], ax=ax1, legend=None)
        # Tree_1_soil_3.plot(style='.-', x='del_D', y='Depth', c=day_colors[2], ax=ax1, legend=None)
        # Tree_1_soil_4.plot(style='.-', x='del_D', y='Depth', c=day_colors[3], ax=ax1, legend=None)
        # Tree_1_soil_5.plot(style='.-', x='del_D', y='Depth', c=day_colors[4], ax=ax1, legend=None)
        # ax1.invert_yaxis()
        #
        # Tree_2_soil_1.plot(style='.-', x='del_D', y='Depth', c=day_colors[0], ax=ax2, legend=None)
        # Tree_2_soil_2.plot(style='.-', x='del_D', y='Depth', c=day_colors[1], ax=ax2, legend=None)
        # Tree_2_soil_3.plot(style='.-', x='del_D', y='Depth', c=day_colors[2], ax=ax2, legend=None)
        # Tree_2_soil_4.plot(style='.-', x='del_D', y='Depth', c=day_colors[3], ax=ax2, legend=None)
        # Tree_2_soil_5.plot(style='.-', x='del_D', y='Depth', c=day_colors[4], ax=ax2, legend=None)
        #
        # Tree_3_soil_1.plot(style='.-', x='del_D', y='Depth', c=day_colors[0], ax=ax3, legend=None)
        # Tree_3_soil_2.plot(style='.-', x='del_D', y='Depth', c=day_colors[1], ax=ax3, legend=None)
        # Tree_3_soil_3.plot(style='.-', x='del_D', y='Depth', c=day_colors[2], ax=ax3, legend=None)
        # Tree_3_soil_4.plot(style='.-', x='del_D', y='Depth', c=day_colors[3], ax=ax3, legend=None)
        # Tree_3_soil_5.plot(style='.-', x='del_D', y='Depth', c=day_colors[4], ax=ax3, legend=None)
        #
        # Tree_4_soil_1.plot(style='.-', x='del_D', y='Depth', c=day_colors[0], ax=ax4, legend=None)
        # Tree_4_soil_2.plot(style='.-', x='del_D', y='Depth', c=day_colors[1], ax=ax4, legend=None)
        # Tree_4_soil_3.plot(style='.-', x='del_D', y='Depth', c=day_colors[2], ax=ax4, legend=None)
        # Tree_4_soil_4.plot(style='.-', x='del_D', y='Depth', c=day_colors[3], ax=ax4, legend=None)
        # Tree_4_soil_5.plot(style='.-', x='del_D', y='Depth', c=day_colors[4], ax=ax4, legend=None)
        #
        # Tree_5_soil_1.plot(style='.-', x='del_D', y='Depth', c=day_colors[0], ax=ax5, legend=None)
        # Tree_5_soil_2.plot(style='.-', x='del_D', y='Depth', c=day_colors[1], ax=ax5, legend=None)
        # Tree_5_soil_3.plot(style='.-', x='del_D', y='Depth', c=day_colors[2], ax=ax5, legend=None)
        # Tree_5_soil_4.plot(style='.-', x='del_D', y='Depth', c=day_colors[3], ax=ax5, legend=None)
        # Tree_5_soil_5.plot(style='.-', x='del_D', y='Depth', c=day_colors[4], ax=ax5, legend=None)
        #
        # Tree_6_soil_1.plot(style='.-', x='del_D', y='Depth', c=day_colors[0], ax=ax6, legend=None)
        # Tree_6_soil_2.plot(style='.-', x='del_D', y='Depth', c=day_colors[1], ax=ax6, legend=None)
        # Tree_6_soil_3.plot(style='.-', x='del_D', y='Depth', c=day_colors[2], ax=ax6, legend=None)
        # Tree_6_soil_4.plot(style='.-', x='del_D', y='Depth', c=day_colors[3], ax=ax6, legend=None)
        # Tree_6_soil_5.plot(style='.-', x='del_D', y='Depth', c=day_colors[4], ax=ax6, legend=None)
        #
        # plt.savefig('Output_Plots/soil_D_profiles.pdf')
        # plt.show()
        #
        # fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, sharey=True, sharex=True, figsize=(12,6))
        #
        # Tree_1_soil_1.plot(style='.-', x='del_O', y='Depth', c=day_colors[0], ax=ax1, legend=None)
        # Tree_1_soil_2.plot(style='.-', x='del_O', y='Depth', c=day_colors[1], ax=ax1, legend=None)
        # Tree_1_soil_3.plot(style='.-', x='del_O', y='Depth', c=day_colors[2], ax=ax1, legend=None)
        # Tree_1_soil_4.plot(style='.-', x='del_O', y='Depth', c=day_colors[3], ax=ax1, legend=None)
        # Tree_1_soil_5.plot(style='.-', x='del_O', y='Depth', c=day_colors[4], ax=ax1, legend=None)
        # ax1.invert_yaxis()
        #
        # Tree_2_soil_1.plot(style='.-', x='del_O', y='Depth', c=day_colors[0], ax=ax2, legend=None)
        # Tree_2_soil_2.plot(style='.-', x='del_O', y='Depth', c=day_colors[1], ax=ax2, legend=None)
        # Tree_2_soil_3.plot(style='.-', x='del_O', y='Depth', c=day_colors[2], ax=ax2, legend=None)
        # Tree_2_soil_4.plot(style='.-', x='del_O', y='Depth', c=day_colors[3], ax=ax2, legend=None)
        # Tree_2_soil_5.plot(style='.-', x='del_O', y='Depth', c=day_colors[4], ax=ax2, legend=None)
        #
        # Tree_3_soil_1.plot(style='.-', x='del_O', y='Depth', c=day_colors[0], ax=ax3, legend=None)
        # Tree_3_soil_2.plot(style='.-', x='del_O', y='Depth', c=day_colors[1], ax=ax3, legend=None)
        # Tree_3_soil_3.plot(style='.-', x='del_O', y='Depth', c=day_colors[2], ax=ax3, legend=None)
        # Tree_3_soil_4.plot(style='.-', x='del_O', y='Depth', c=day_colors[3], ax=ax3, legend=None)
        # Tree_3_soil_5.plot(style='.-', x='del_O', y='Depth', c=day_colors[4], ax=ax3, legend=None)
        #
        # Tree_4_soil_1.plot(style='.-', x='del_O', y='Depth', c=day_colors[0], ax=ax4, legend=None)
        # Tree_4_soil_2.plot(style='.-', x='del_O', y='Depth', c=day_colors[1], ax=ax4, legend=None)
        # Tree_4_soil_3.plot(style='.-', x='del_O', y='Depth', c=day_colors[2], ax=ax4, legend=None)
        # Tree_4_soil_4.plot(style='.-', x='del_O', y='Depth', c=day_colors[3], ax=ax4, legend=None)
        # Tree_4_soil_5.plot(style='.-', x='del_O', y='Depth', c=day_colors[4], ax=ax4, legend=None)
        #
        # Tree_5_soil_1.plot(style='.-', x='del_O', y='Depth', c=day_colors[0], ax=ax5, legend=None)
        # Tree_5_soil_2.plot(style='.-', x='del_O', y='Depth', c=day_colors[1], ax=ax5, legend=None)
        # Tree_5_soil_3.plot(style='.-', x='del_O', y='Depth', c=day_colors[2], ax=ax5, legend=None)
        # Tree_5_soil_4.plot(style='.-', x='del_O', y='Depth', c=day_colors[3], ax=ax5, legend=None)
        # Tree_5_soil_5.plot(style='.-', x='del_O', y='Depth', c=day_colors[4], ax=ax5, legend=None)
        #
        # Tree_6_soil_1.plot(style='.-', x='del_O', y='Depth', c=day_colors[0], ax=ax6, legend=None)
        # Tree_6_soil_2.plot(style='.-', x='del_O', y='Depth', c=day_colors[1], ax=ax6, legend=None)
        # Tree_6_soil_3.plot(style='.-', x='del_O', y='Depth', c=day_colors[2], ax=ax6, legend=None)
        # Tree_6_soil_4.plot(style='.-', x='del_O', y='Depth', c=day_colors[3], ax=ax6, legend=None)
        # Tree_6_soil_5.plot(style='.-', x='del_O', y='Depth', c=day_colors[4], ax=ax6, legend=None)
        #
        # plt.savefig('Output_Plots/soil_O_profiles.pdf')
        # plt.show()
        #
        # fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey=True, sharex=True, figsize=(12, 8))
        # col = Tree_1_plant.Sampling_Day.map({1:'#f4a582', 2:'#d6604d', 3:'#b2182b', 4:'#92c5de', 5:'#2166ac'})
        # Tree_1_plant.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax1, s=80)
        # col = Tree_1_soil_1.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #     50: '#253494', 60: '#081d58'})
        # Tree_1_soil_1.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax1, alpha=0.5)
        # col = Tree_1_soil_2.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                                50: '#253494', 60: '#081d58'})
        # Tree_1_soil_2.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax1, alpha=0.5)
        # col = Tree_1_soil_3.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                                50: '#253494', 60: '#081d58'})
        # Tree_1_soil_3.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax1, alpha=0.5)
        #
        # col = Tree_2_plant.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        # Tree_2_plant.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax2, s=80)
        # col = Tree_2_soil_1.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #     50: '#253494', 60: '#081d58'})
        # Tree_2_soil_1.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax2, alpha=0.5)
        # col = Tree_2_soil_2.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                                50: '#253494', 60: '#081d58'})
        # Tree_2_soil_2.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax2, alpha=0.5)
        # col = Tree_2_soil_3.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                                50: '#253494', 60: '#081d58'})
        # Tree_2_soil_3.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax2, alpha=0.5)
        #
        # col = Tree_3_plant.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        # Tree_3_plant.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax3, s=80)
        # col = Tree_3_soil_1.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                                50: '#253494', 60: '#081d58'})
        # Tree_3_soil_1.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax3, alpha=0.5)
        # col = Tree_3_soil_2.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                                50: '#253494', 60: '#081d58'})
        # Tree_3_soil_2.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax3, alpha=0.5)
        # col = Tree_3_soil_3.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                                50: '#253494', 60: '#081d58'})
        # Tree_3_soil_3.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax3, alpha=0.5)
        #
        # col = Tree_4_plant.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        # Tree_4_plant.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax4, s=80)
        # col = Tree_4_soil_1.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                                50: '#253494', 60: '#081d58'})
        # Tree_4_soil_1.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax4, alpha=0.5)
        # col = Tree_4_soil_2.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                                50: '#253494', 60: '#081d58'})
        # Tree_4_soil_2.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax4, alpha=0.5)
        # col = Tree_4_soil_3.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                                50: '#253494', 60: '#081d58'})
        # Tree_4_soil_3.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax4, alpha=0.5)
        #
        # col = Tree_5_plant.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        # Tree_5_plant.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax5, s=80)
        # col = Tree_5_soil_1.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                                50: '#253494', 60: '#081d58'})
        # Tree_5_soil_1.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax5, alpha=0.5)
        # col = Tree_5_soil_2.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                                50: '#253494', 60: '#081d58'})
        # Tree_5_soil_2.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax5, alpha=0.5)
        # col = Tree_5_soil_3.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                                50: '#253494', 60: '#081d58'})
        # Tree_5_soil_3.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax5, alpha=0.5)
        #
        # col = Tree_6_plant.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        # Tree_6_plant.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax6, s=80)
        # col = Tree_6_soil_1.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                                50: '#253494', 60: '#081d58'})
        # Tree_6_soil_1.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax6, alpha=0.5)
        # col = Tree_6_soil_2.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                                50: '#253494', 60: '#081d58'})
        # Tree_6_soil_2.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax6, alpha=0.5)
        # col = Tree_6_soil_3.Depth.map({0: '#c7e9b4', 10: '#7fcdbb', 20: '#41b6c4', 30: '#1d91c0', 40: '#225ea8',
        #                                50: '#253494', 60: '#081d58'})
        # Tree_6_soil_3.plot(kind='scatter', x='del_O', y='del_D', c=col, ax=ax6, alpha=0.5)
        #
        # plt.savefig('Output_Plots/plant_dual_isotope_time_series.pdf')
        # plt.show()
        #
        # fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharey=True, sharex=True, figsize=(12, 10))
        # Tree_1_trans.plot(style='.-', x='TIMESTAMP', y='del_D', c='r', ax=ax1, legend=None)
        # ax1b = ax1.twinx()
        # Tree_1_trans.plot(style='.-', x='TIMESTAMP', y='del_O', c='b', ax=ax1b, legend=None)
        #
        # Tree_2_trans.plot(style='.-', x='TIMESTAMP', y='del_D', c='r', ax=ax2, legend=None)
        # ax2b = ax2.twinx()
        # Tree_2_trans.plot(style='.-', x='TIMESTAMP', y='del_O', c='b', ax=ax2b, legend=None)
        #
        # Tree_3_trans.plot(style='.-', x='TIMESTAMP', y='del_D', c='r', ax=ax3, legend=None)
        # ax3b = ax3.twinx()
        # Tree_3_trans.plot(style='.-', x='TIMESTAMP', y='del_O', c='b', ax=ax3b, legend=None)
        #
        # Tree_4_trans.plot(style='.-', x='TIMESTAMP', y='del_D', c='r', ax=ax4, legend=None)
        # ax4b = ax4.twinx()
        # Tree_4_trans.plot(style='.-', x='TIMESTAMP', y='del_O', c='b', ax=ax4b, legend=None)
        #
        # Tree_5_trans.plot(style='.-', x='TIMESTAMP', y='del_D', c='r', ax=ax5, legend=None)
        # ax5b = ax5.twinx()
        # Tree_5_trans.plot(style='.-', x='TIMESTAMP', y='del_O', c='b', ax=ax5b, legend=None)
        #
        # Tree_6_trans.plot(style='.-', x='TIMESTAMP', y='del_D', c='r', ax=ax6, legend=None)
        # ax6b = ax6.twinx()
        # Tree_6_trans.plot(style='.-', x='TIMESTAMP', y='del_O', c='b', ax=ax6b, legend=None)
        #
        # plt.savefig('Output_Plots/transpiration_isotope_time_series.pdf')
        # plt.show()

        #plot trial soil isotope profiles

        # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12,6))
        #
        # trial_A_isotopes_T1.plot(style='.-', x='del_D', y='Depth', c="#1b9e77", ax=ax1, legend=None)
        # trial_B_isotopes_T1.plot(style='.-', x='del_D', y='Depth', c="#d95f02", ax=ax1, legend=None)
        # trial_C_isotopes_T1.plot(style='.-', x='del_D', y='Depth', c="#7570b3", ax=ax1, legend=None)
        # ax1.invert_yaxis()
        #
        # trial_A_isotopes_T2.plot(style='.-', x='del_D', y='Depth', c="#1b9e77", ax=ax2, legend=None)
        # trial_B_isotopes_T2.plot(style='.-', x='del_D', y='Depth', c="#d95f02", ax=ax2, legend=None)
        # trial_C_isotopes_T2.plot(style='.-', x='del_D', y='Depth', c="#7570b3", ax=ax2, legend=None)
        #
        # plt.savefig('Output_Plots/trial_D_soil_profiles.pdf')
        # plt.show()
        #
        # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 6))
        #
        # trial_A_isotopes_T1.plot(style='.-', x='del_O', y='Depth', c="#1b9e77", ax=ax1, legend=None)
        # trial_B_isotopes_T1.plot(style='.-', x='del_O', y='Depth', c="#d95f02", ax=ax1, legend=None)
        # trial_C_isotopes_T1.plot(style='.-', x='del_O', y='Depth', c="#7570b3", ax=ax1, legend=None)
        # ax1.invert_yaxis()
        #
        # trial_A_isotopes_T2.plot(style='.-', x='del_O', y='Depth', c="#1b9e77", ax=ax2, legend=None)
        # trial_B_isotopes_T2.plot(style='.-', x='del_O', y='Depth', c="#d95f02", ax=ax2, legend=None)
        # trial_C_isotopes_T2.plot(style='.-', x='del_O', y='Depth', c="#7570b3", ax=ax2, legend=None)
        #
        # plt.savefig('Output_Plots/trial_O_soil_profiles.pdf')
        # plt.show()

        # fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey=True, sharex=True, figsize=(12, 8))
        # plant_iso_cols = ['#F0A58F', '#EB548C', '#Af4BCE', '#29066B']
        # plant_day_1.plot(kind='scatter', x='del_O', y='del_D', marker='o', c=plant_iso_cols[0], ax=ax1)
        # petiole_day_1.plot(kind='scatter', x='del_O', y='del_D', marker='s', c=plant_iso_cols[1], ax=ax1)
        # leaf_day_1.plot(kind='scatter', x='del_O', y='del_D', marker='d', c=plant_iso_cols[2], ax=ax1)
        # trans_day_1.plot(kind='scatter', x='del_O', y='del_D', marker='v', c=plant_iso_cols[3], ax=ax1)
        #
        # plant_day_2.plot(kind='scatter', x='del_O', y='del_D', marker='o', c=plant_iso_cols[0], ax=ax2)
        # petiole_day_2.plot(kind='scatter', x='del_O', y='del_D', marker='s', c=plant_iso_cols[1], ax=ax2)
        # leaf_day_2.plot(kind='scatter', x='del_O', y='del_D', marker='d', c=plant_iso_cols[2], ax=ax2)
        # trans_day_2.plot(kind='scatter', x='del_O', y='del_D', marker='v', c=plant_iso_cols[3], ax=ax2)
        #
        # plant_day_3.plot(kind='scatter', x='del_O', y='del_D', marker='o', c=plant_iso_cols[0], ax=ax3)
        # petiole_day_3.plot(kind='scatter', x='del_O', y='del_D', marker='s', c=plant_iso_cols[1], ax=ax3)
        # leaf_day_3.plot(kind='scatter', x='del_O', y='del_D', marker='d', c=plant_iso_cols[2], ax=ax3)
        # trans_day_3.plot(kind='scatter', x='del_O', y='del_D', marker='v', c=plant_iso_cols[3], ax=ax3)
        #
        # plant_day_4.plot(kind='scatter', x='del_O', y='del_D', marker='o', c=plant_iso_cols[0], ax=ax4)
        # petiole_day_4.plot(kind='scatter', x='del_O', y='del_D', marker='s', c=plant_iso_cols[1], ax=ax4)
        # leaf_day_4.plot(kind='scatter', x='del_O', y='del_D', marker='d', c=plant_iso_cols[2], ax=ax4)
        # trans_day_4.plot(kind='scatter', x='del_O', y='del_D', marker='v', c=plant_iso_cols[3], ax=ax4)
        #
        # plant_day_5.plot(kind='scatter', x='del_O', y='del_D', marker='o', c=plant_iso_cols[0], ax=ax5)
        # petiole_day_5.plot(kind='scatter', x='del_O', y='del_D', marker='s', c=plant_iso_cols[1], ax=ax5)
        # leaf_day_5.plot(kind='scatter', x='del_O', y='del_D', marker='d', c=plant_iso_cols[2], ax=ax5)
        # trans_day_5.plot(kind='scatter', x='del_O', y='del_D', marker='v', c=plant_iso_cols[3], ax=ax5)
        #
        # plt.savefig('Output_Plots/all_plant_isotopes_by_day.pdf')
        # plt.show()

        fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True, sharex=True, figsize=(12, 4))
        trial_T1_trans.plot(style='.-', x='TIMESTAMP', y='del_D', c='r', ax=ax1, legend=None)
        ax1b = ax1.twinx()
        trial_T1_trans.plot(style='.-', x='TIMESTAMP', y='del_O', c='b', ax=ax1b, legend=None)

        trial_T2_trans.plot(style='.-', x='TIMESTAMP', y='del_D', c='r', ax=ax2, legend=None)
        ax2b = ax2.twinx()
        trial_T2_trans.plot(style='.-', x='TIMESTAMP', y='del_O', c='b', ax=ax2b, legend=None)

        plt.savefig('Output_Plots/trial_transpiration_time_series.pdf')
        plt.show()

    if met is not None:

        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        met.plot(kind='line', x='TIMESTAMP', y='AirTC', ax=ax)
        plt.title('Air Temp (C)')
        plt.savefig('Output_Plots/air_temp_full_length.pdf')
        plt.show()

        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        met.plot(kind='line', x='TIMESTAMP', y='VPD_kPa', ax=ax)
        plt.title('Vapor Pressure Deficit (kPa)')
        plt.savefig('Output_Plots/VPD_full_length.pdf')
        plt.show()

        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        met.plot(kind='line', x='TIMESTAMP', y='SlrW_Avg', ax=ax)
        plt.title('Solar Irradiation (W/m2)')
        plt.savefig('Output_Plots/Irr_full_length.pdf')
        plt.show()

    if gasex is not None:
        A_means = gasex.groupby(['Tree_Number', 'Sampling_Day', 'Type'])['A'].mean()
        gs_means = gasex.groupby(['Tree_Number', 'Sampling_Day', 'Type'])['gsw'].mean()
        WUE_intr_means = gasex.groupby(['Tree_Number', 'Sampling_Day', 'Type'])['WUE_intr'].mean()
        WUE_inst_means = gasex.groupby(['Tree_Number', 'Sampling_Day', 'Type'])['WUE_inst'].mean()
        gasex_1 = gasex[gasex['Tree_Number'] == 1]
        gasex_2 = gasex[gasex['Tree_Number'] == 2]
        gasex_3 = gasex[gasex['Tree_Number'] == 3]
        gasex_4 = gasex[gasex['Tree_Number'] == 4]
        gasex_5 = gasex[gasex['Tree_Number'] == 5]
        gasex_6 = gasex[gasex['Tree_Number'] == 6]

        gasex_1['gs_normed'] = gasex_1.gsw/(gasex_1.gsw.max())
        gasex_2['gs_normed'] = gasex_2.gsw / (gasex_2.gsw.max())
        gasex_3['gs_normed'] = gasex_3.gsw / (gasex_3.gsw.max())
        gasex_4['gs_normed'] = gasex_4.gsw / (gasex_4.gsw.max())
        gasex_5['gs_normed'] = gasex_5.gsw / (gasex_5.gsw.max())
        gasex_6['gs_normed'] = gasex_6.gsw / (gasex_6.gsw.max())

        gasex_1['A_normed'] = gasex_1.A / (gasex_1.A.max())
        gasex_2['A_normed'] = gasex_2.A / (gasex_2.A.max())
        gasex_3['A_normed'] = gasex_3.A / (gasex_3.A.max())
        gasex_4['A_normed'] = gasex_4.A / (gasex_4.A.max())
        gasex_5['A_normed'] = gasex_5.A / (gasex_5.A.max())
        gasex_6['A_normed'] = gasex_6.A / (gasex_6.A.max())

        offset = .2
        gasex_1_am = gasex_1[gasex_1['Type'] == 'AM']
        gasex_1_am['Offset_Time'] = gasex_1_am['Sampling_Day']- offset
        gasex_1_pm = gasex_1[gasex_1['Type'] == 'PM']
        gasex_1_pm['Offset_Time'] = gasex_1_pm['Sampling_Day'] + offset
        gasex_2_am = gasex_2[gasex_2['Type'] == 'AM']
        gasex_2_am['Offset_Time'] = gasex_2_am['Sampling_Day'] - offset
        gasex_2_pm = gasex_2[gasex_2['Type'] == 'PM']
        gasex_2_pm['Offset_Time'] = gasex_2_pm['Sampling_Day'] + offset
        gasex_3_am = gasex_3[gasex_3['Type'] == 'AM']
        gasex_3_am['Offset_Time'] = gasex_3_am['Sampling_Day'] - offset
        gasex_3_pm = gasex_3[gasex_3['Type'] == 'PM']
        gasex_3_pm['Offset_Time'] = gasex_3_pm['Sampling_Day'] + offset
        gasex_4_am = gasex_4[gasex_4['Type'] == 'AM']
        gasex_4_am['Offset_Time'] = gasex_4_am['Sampling_Day'] - offset
        gasex_4_pm = gasex_4[gasex_4['Type'] == 'PM']
        gasex_4_pm['Offset_Time'] = gasex_4_pm['Sampling_Day'] + offset
        gasex_5_am = gasex_5[gasex_5['Type'] == 'AM']
        gasex_5_am['Offset_Time'] = gasex_5_am['Sampling_Day'] - offset
        gasex_5_pm = gasex_5[gasex_5['Type'] == 'PM']
        gasex_5_pm['Offset_Time'] = gasex_5_pm['Sampling_Day'] + offset
        gasex_6_am = gasex_6[gasex_6['Type'] == 'AM']
        gasex_6_am['Offset_Time'] = gasex_6_am['Sampling_Day'] - offset
        gasex_6_pm = gasex_6[gasex_6['Type'] == 'PM']
        gasex_6_pm['Offset_Time'] = gasex_6_pm['Sampling_Day'] + offset

        gasex_1_am_1 = gasex_1_am[gasex_1_am['Sampling_Day'] == 1]
        gasex_1_am_2 = gasex_1_am[gasex_1_am['Sampling_Day'] == 2]
        gasex_1_am_4 = gasex_1_am[gasex_1_am['Sampling_Day'] == 4]
        gasex_1_pm_1 = gasex_1_pm[gasex_1_pm['Sampling_Day'] == 1]
        gasex_1_pm_2 = gasex_1_pm[gasex_1_pm['Sampling_Day'] == 2]
        gasex_1_pm_4 = gasex_1_pm[gasex_1_pm['Sampling_Day'] == 4]

        gasex_2_am_1 = gasex_2_am[gasex_2_am['Sampling_Day'] == 1]
        gasex_2_am_2 = gasex_2_am[gasex_2_am['Sampling_Day'] == 2]
        gasex_2_am_4 = gasex_2_am[gasex_2_am['Sampling_Day'] == 4]
        gasex_2_pm_1 = gasex_2_pm[gasex_2_pm['Sampling_Day'] == 1]
        gasex_2_pm_2 = gasex_2_pm[gasex_2_pm['Sampling_Day'] == 2]
        gasex_2_pm_4 = gasex_2_pm[gasex_2_pm['Sampling_Day'] == 4]

        gasex_3_am_1 = gasex_3_am[gasex_3_am['Sampling_Day'] == 1]
        gasex_3_am_2 = gasex_3_am[gasex_3_am['Sampling_Day'] == 2]
        gasex_3_am_4 = gasex_3_am[gasex_3_am['Sampling_Day'] == 4]
        gasex_3_pm_1 = gasex_3_pm[gasex_3_pm['Sampling_Day'] == 1]
        gasex_3_pm_2 = gasex_3_pm[gasex_3_pm['Sampling_Day'] == 2]
        gasex_3_pm_4 = gasex_3_pm[gasex_3_pm['Sampling_Day'] == 4]

        gasex_4_am_1 = gasex_4_am[gasex_4_am['Sampling_Day'] == 1]
        gasex_4_am_2 = gasex_4_am[gasex_4_am['Sampling_Day'] == 2]
        gasex_4_am_4 = gasex_4_am[gasex_4_am['Sampling_Day'] == 4]
        gasex_4_pm_1 = gasex_4_pm[gasex_4_pm['Sampling_Day'] == 1]
        gasex_4_pm_2 = gasex_4_pm[gasex_4_pm['Sampling_Day'] == 2]
        gasex_4_pm_4 = gasex_4_pm[gasex_4_pm['Sampling_Day'] == 4]

        gasex_5_am_1 = gasex_5_am[gasex_5_am['Sampling_Day'] == 1]
        gasex_5_am_2 = gasex_5_am[gasex_5_am['Sampling_Day'] == 2]
        gasex_5_am_4 = gasex_5_am[gasex_5_am['Sampling_Day'] == 4]
        gasex_5_pm_1 = gasex_5_pm[gasex_5_pm['Sampling_Day'] == 1]
        gasex_5_pm_2 = gasex_5_pm[gasex_5_pm['Sampling_Day'] == 2]
        gasex_5_pm_4 = gasex_5_pm[gasex_5_pm['Sampling_Day'] == 4]

        gasex_6_am_1 = gasex_6_am[gasex_6_am['Sampling_Day'] == 1]
        gasex_6_am_2 = gasex_6_am[gasex_6_am['Sampling_Day'] == 2]
        gasex_6_am_4 = gasex_6_am[gasex_6_am['Sampling_Day'] == 4]
        gasex_6_pm_1 = gasex_6_pm[gasex_6_pm['Sampling_Day'] == 1]
        gasex_6_pm_2 = gasex_6_pm[gasex_6_pm['Sampling_Day'] == 2]
        gasex_6_pm_4 = gasex_6_pm[gasex_6_pm['Sampling_Day'] == 4]


        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, sharey=True, sharex=True, figsize=(12, 6))
        ax1.plot([1.2], [A_means[(1, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax1.plot([0.8], [A_means[(1, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax1.plot([2.2], [A_means[(1, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax1.plot([1.8], [A_means[(1, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax1.plot([4.2], [A_means[(1, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax1.plot([3.8], [A_means[(1, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_1_am.plot(kind='scatter', x='Offset_Time', y='A', c='gray', ax=ax1, alpha=0.5, marker='o')
        gasex_1_pm.plot(kind='scatter', x='Offset_Time', y='A', c='gray', ax=ax1, alpha=0.5, marker='v')
        ax1.set_xticks([1, 2, 3, 4, 5])
        plt.xlim(0.5,5.5)

        ax2.plot([1.2], [A_means[(2, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax2.plot([0.8], [A_means[(2, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax2.plot([2.2], [A_means[(2, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax2.plot([1.8], [A_means[(2, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax2.plot([4.2], [A_means[(2, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax2.plot([3.8], [A_means[(2, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_2_am.plot(kind='scatter', x='Offset_Time', y='A', c='gray', ax=ax2, alpha=0.5, marker='o')
        gasex_2_pm.plot(kind='scatter', x='Offset_Time', y='A', c='gray', ax=ax2, alpha=0.5, marker='v')
        ax2.set_xticks([1, 2, 3, 4, 5])

        ax3.plot([1.2], [A_means[(3, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax3.plot([0.8], [A_means[(3, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax3.plot([2.2], [A_means[(3, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax3.plot([1.8], [A_means[(3, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax3.plot([4.2], [A_means[(3, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax3.plot([3.8], [A_means[(3, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_3_am.plot(kind='scatter', x='Offset_Time', y='A', c='gray', ax=ax3, alpha=0.5, marker='o')
        gasex_3_pm.plot(kind='scatter', x='Offset_Time', y='A', c='gray', ax=ax3, alpha=0.5, marker='v')
        ax3.set_xticks([1, 2, 3, 4, 5])

        ax4.plot([1.2], [A_means[(4, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax4.plot([0.8], [A_means[(4, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax4.plot([2.2], [A_means[(4, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax4.plot([1.8], [A_means[(4, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax4.plot([4.2], [A_means[(4, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax4.plot([3.8], [A_means[(4, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_4_am.plot(kind='scatter', x='Offset_Time', y='A', c='gray', ax=ax4, alpha=0.5, marker='o')
        gasex_4_pm.plot(kind='scatter', x='Offset_Time', y='A', c='gray', ax=ax4, alpha=0.5, marker='v')
        ax4.set_xticks([1, 2, 3, 4, 5])

        ax5.plot([1.2], [A_means[(5, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax5.plot([0.8], [A_means[(5, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax5.plot([2.2], [A_means[(5, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax5.plot([1.8], [A_means[(5, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax5.plot([4.2], [A_means[(5, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax5.plot([3.8], [A_means[(5, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_5_am.plot(kind='scatter', x='Offset_Time', y='A', c='gray', ax=ax5, alpha=0.5, marker='o')
        gasex_5_pm.plot(kind='scatter', x='Offset_Time', y='A', c='gray', ax=ax5, alpha=0.5, marker='v')
        ax5.set_xticks([1, 2, 3, 4, 5])

        ax6.plot([1.2], [A_means[(6, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax6.plot([0.8], [A_means[(6, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax6.plot([2.2], [A_means[(6, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax6.plot([1.8], [A_means[(6, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax6.plot([4.2], [A_means[(6, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax6.plot([3.8], [A_means[(6, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_6_am.plot(kind='scatter', x='Offset_Time', y='A', c='gray', ax=ax6, alpha=0.5, marker='o')
        gasex_6_pm.plot(kind='scatter', x='Offset_Time', y='A', c='gray', ax=ax6, alpha=0.5, marker='v')
        ax6.set_xticks([1, 2, 3, 4, 5])

        plt.savefig('Output_Plots/A_tree_split.pdf')
        plt.show()

        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, sharey=True, sharex=True, figsize=(12, 6))
        ax1.plot([1.2], [gs_means[(1, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax1.plot([0.8], [gs_means[(1, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax1.plot([2.2], [gs_means[(1, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax1.plot([1.8], [gs_means[(1, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax1.plot([4.2], [gs_means[(1, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax1.plot([3.8], [gs_means[(1, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_1_am.plot(kind='scatter', x='Offset_Time', y='gsw', c='gray', ax=ax1, alpha=0.5, marker='o')
        gasex_1_pm.plot(kind='scatter', x='Offset_Time', y='gsw', c='gray', ax=ax1, alpha=0.5, marker='v')
        ax1.set_xticks([1, 2, 3, 4, 5])
        plt.xlim(0.5, 5.5)

        ax2.plot([1.2], [gs_means[(2, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax2.plot([0.8], [gs_means[(2, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax2.plot([2.2], [gs_means[(2, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax2.plot([1.8], [gs_means[(2, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax2.plot([4.2], [gs_means[(2, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax2.plot([3.8], [gs_means[(2, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_2_am.plot(kind='scatter', x='Offset_Time', y='gsw', c='gray', ax=ax2, alpha=0.5, marker='o')
        gasex_2_pm.plot(kind='scatter', x='Offset_Time', y='gsw', c='gray', ax=ax2, alpha=0.5, marker='v')
        ax2.set_xticks([1, 2, 3, 4, 5])

        ax3.plot([1.2], [gs_means[(3, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax3.plot([0.8], [gs_means[(3, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax3.plot([2.2], [gs_means[(3, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax3.plot([1.8], [gs_means[(3, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax3.plot([4.2], [gs_means[(3, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax3.plot([3.8], [gs_means[(3, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_3_am.plot(kind='scatter', x='Offset_Time', y='gsw', c='gray', ax=ax3, alpha=0.5, marker='o')
        gasex_3_pm.plot(kind='scatter', x='Offset_Time', y='gsw', c='gray', ax=ax3, alpha=0.5, marker='v')
        ax3.set_xticks([1, 2, 3, 4, 5])

        ax4.plot([1.2], [gs_means[(4, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax4.plot([0.8], [gs_means[(4, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax4.plot([2.2], [gs_means[(4, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax4.plot([1.8], [gs_means[(4, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax4.plot([4.2], [gs_means[(4, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax4.plot([3.8], [gs_means[(4, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_4_am.plot(kind='scatter', x='Offset_Time', y='gsw', c='gray', ax=ax4, alpha=0.5, marker='o')
        gasex_4_pm.plot(kind='scatter', x='Offset_Time', y='gsw', c='gray', ax=ax4, alpha=0.5, marker='v')
        ax4.set_xticks([1, 2, 3, 4, 5])

        ax5.plot([1.2], [gs_means[(5, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax5.plot([0.8], [gs_means[(5, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax5.plot([2.2], [gs_means[(5, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax5.plot([1.8], [gs_means[(5, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax5.plot([4.2], [gs_means[(5, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax5.plot([3.8], [gs_means[(5, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_5_am.plot(kind='scatter', x='Offset_Time', y='gsw', c='gray', ax=ax5, alpha=0.5, marker='o')
        gasex_5_pm.plot(kind='scatter', x='Offset_Time', y='gsw', c='gray', ax=ax5, alpha=0.5, marker='v')
        ax5.set_xticks([1, 2, 3, 4, 5])

        ax6.plot([1.2], [gs_means[(6, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax6.plot([0.8], [gs_means[(6, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax6.plot([2.2], [gs_means[(6, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax6.plot([1.8], [gs_means[(6, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax6.plot([4.2], [gs_means[(6, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax6.plot([3.8], [gs_means[(6, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_6_am.plot(kind='scatter', x='Offset_Time', y='gsw', c='gray', ax=ax6, alpha=0.5, marker='o')
        gasex_6_pm.plot(kind='scatter', x='Offset_Time', y='gsw', c='gray', ax=ax6, alpha=0.5, marker='v')
        ax6.set_xticks([1, 2, 3, 4, 5])

        plt.savefig('Output_Plots/gsw_tree_split.pdf')
        plt.show()

        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, sharey=True, sharex=True, figsize=(12, 6))
        ax1.plot([1.2], [WUE_intr_means[(1, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax1.plot([0.8], [WUE_intr_means[(1, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax1.plot([2.2], [WUE_intr_means[(1, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax1.plot([1.8], [WUE_intr_means[(1, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax1.plot([4.2], [WUE_intr_means[(1, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax1.plot([3.8], [WUE_intr_means[(1, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_1_am.plot(kind='scatter', x='Offset_Time', y='WUE_intr', c='gray', ax=ax1, alpha=0.5, marker='o')
        gasex_1_pm.plot(kind='scatter', x='Offset_Time', y='WUE_intr', c='gray', ax=ax1, alpha=0.5, marker='v')
        ax1.set_xticks([1, 2, 3, 4, 5])
        plt.xlim(0.5, 5.5)

        ax2.plot([1.2], [WUE_intr_means[(2, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax2.plot([0.8], [WUE_intr_means[(2, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax2.plot([2.2], [WUE_intr_means[(2, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax2.plot([1.8], [WUE_intr_means[(2, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax2.plot([4.2], [WUE_intr_means[(2, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax2.plot([3.8], [WUE_intr_means[(2, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_2_am.plot(kind='scatter', x='Offset_Time', y='WUE_intr', c='gray', ax=ax2, alpha=0.5, marker='o')
        gasex_2_pm.plot(kind='scatter', x='Offset_Time', y='WUE_intr', c='gray', ax=ax2, alpha=0.5, marker='v')
        ax2.set_xticks([1, 2, 3, 4, 5])

        ax3.plot([1.2], [WUE_intr_means[(3, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax3.plot([0.8], [WUE_intr_means[(3, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax3.plot([2.2], [WUE_intr_means[(3, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax3.plot([1.8], [WUE_intr_means[(3, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax3.plot([4.2], [WUE_intr_means[(3, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax3.plot([3.8], [WUE_intr_means[(3, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_3_am.plot(kind='scatter', x='Offset_Time', y='WUE_intr', c='gray', ax=ax3, alpha=0.5, marker='o')
        gasex_3_pm.plot(kind='scatter', x='Offset_Time', y='WUE_intr', c='gray', ax=ax3, alpha=0.5, marker='v')
        ax3.set_xticks([1, 2, 3, 4, 5])

        ax4.plot([1.2], [WUE_intr_means[(4, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax4.plot([0.8], [WUE_intr_means[(4, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax4.plot([2.2], [WUE_intr_means[(4, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax4.plot([1.8], [WUE_intr_means[(4, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax4.plot([4.2], [WUE_intr_means[(4, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax4.plot([3.8], [WUE_intr_means[(4, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_4_am.plot(kind='scatter', x='Offset_Time', y='WUE_intr', c='gray', ax=ax4, alpha=0.5, marker='o')
        gasex_4_pm.plot(kind='scatter', x='Offset_Time', y='WUE_intr', c='gray', ax=ax4, alpha=0.5, marker='v')
        ax4.set_xticks([1, 2, 3, 4, 5])

        ax5.plot([1.2], [WUE_intr_means[(5, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax5.plot([0.8], [WUE_intr_means[(5, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax5.plot([2.2], [WUE_intr_means[(5, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax5.plot([1.8], [WUE_intr_means[(5, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax5.plot([4.2], [WUE_intr_means[(5, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax5.plot([3.8], [WUE_intr_means[(5, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_5_am.plot(kind='scatter', x='Offset_Time', y='WUE_intr', c='gray', ax=ax5, alpha=0.5, marker='o')
        gasex_5_pm.plot(kind='scatter', x='Offset_Time', y='WUE_intr', c='gray', ax=ax5, alpha=0.5, marker='v')
        ax5.set_xticks([1, 2, 3, 4, 5])

        ax6.plot([1.2], [WUE_intr_means[(6, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax6.plot([0.8], [WUE_intr_means[(6, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax6.plot([2.2], [WUE_intr_means[(6, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax6.plot([1.8], [WUE_intr_means[(6, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax6.plot([4.2], [WUE_intr_means[(6, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax6.plot([3.8], [WUE_intr_means[(6, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_6_am.plot(kind='scatter', x='Offset_Time', y='WUE_intr', c='gray', ax=ax6, alpha=0.5, marker='o')
        gasex_6_pm.plot(kind='scatter', x='Offset_Time', y='WUE_intr', c='gray', ax=ax6, alpha=0.5, marker='v')
        ax6.set_xticks([1, 2, 3, 4, 5])

        plt.savefig('Output_Plots/WUE_intrinsic_tree_split.pdf')
        plt.show()

        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, sharey=True, sharex=True, figsize=(12, 6))
        ax1.plot([1.2], [WUE_inst_means[(1, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax1.plot([0.8], [WUE_inst_means[(1, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax1.plot([2.2], [WUE_inst_means[(1, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax1.plot([1.8], [WUE_inst_means[(1, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax1.plot([4.2], [WUE_inst_means[(1, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax1.plot([3.8], [WUE_inst_means[(1, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_1_am.plot(kind='scatter', x='Offset_Time', y='WUE_inst', c='gray', ax=ax1, alpha=0.5, marker='o')
        gasex_1_pm.plot(kind='scatter', x='Offset_Time', y='WUE_inst', c='gray', ax=ax1, alpha=0.5, marker='v')
        ax1.set_xticks([1, 2, 3, 4, 5])
        plt.xlim(0.5, 5.5)

        ax2.plot([1.2], [WUE_inst_means[(2, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax2.plot([0.8], [WUE_inst_means[(2, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax2.plot([2.2], [WUE_inst_means[(2, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax2.plot([1.8], [WUE_inst_means[(2, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax2.plot([4.2], [WUE_inst_means[(2, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax2.plot([3.8], [WUE_inst_means[(2, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_2_am.plot(kind='scatter', x='Offset_Time', y='WUE_inst', c='gray', ax=ax2, alpha=0.5, marker='o')
        gasex_2_pm.plot(kind='scatter', x='Offset_Time', y='WUE_inst', c='gray', ax=ax2, alpha=0.5, marker='v')
        ax2.set_xticks([1, 2, 3, 4, 5])

        ax3.plot([1.2], [WUE_inst_means[(3, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax3.plot([0.8], [WUE_inst_means[(3, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax3.plot([2.2], [WUE_inst_means[(3, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax3.plot([1.8], [WUE_inst_means[(3, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax3.plot([4.2], [WUE_inst_means[(3, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax3.plot([3.8], [WUE_inst_means[(3, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_3_am.plot(kind='scatter', x='Offset_Time', y='WUE_inst', c='gray', ax=ax3, alpha=0.5, marker='o')
        gasex_3_pm.plot(kind='scatter', x='Offset_Time', y='WUE_inst', c='gray', ax=ax3, alpha=0.5, marker='v')
        ax3.set_xticks([1, 2, 3, 4, 5])

        ax4.plot([1.2], [WUE_inst_means[(4, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax4.plot([0.8], [WUE_inst_means[(4, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax4.plot([2.2], [WUE_inst_means[(4, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax4.plot([1.8], [WUE_inst_means[(4, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax4.plot([4.2], [WUE_inst_means[(4, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax4.plot([3.8], [WUE_inst_means[(4, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_4_am.plot(kind='scatter', x='Offset_Time', y='WUE_inst', c='gray', ax=ax4, alpha=0.5, marker='o')
        gasex_4_pm.plot(kind='scatter', x='Offset_Time', y='WUE_inst', c='gray', ax=ax4, alpha=0.5, marker='v')
        ax4.set_xticks([1, 2, 3, 4, 5])

        ax5.plot([1.2], [WUE_inst_means[(5, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax5.plot([0.8], [WUE_inst_means[(5, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax5.plot([2.2], [WUE_inst_means[(5, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax5.plot([1.8], [WUE_inst_means[(5, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax5.plot([4.2], [WUE_inst_means[(5, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax5.plot([3.8], [WUE_inst_means[(5, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_5_am.plot(kind='scatter', x='Offset_Time', y='WUE_inst', c='gray', ax=ax5, alpha=0.5, marker='o')
        gasex_5_pm.plot(kind='scatter', x='Offset_Time', y='WUE_inst', c='gray', ax=ax5, alpha=0.5, marker='v')
        ax5.set_xticks([1, 2, 3, 4, 5])

        ax6.plot([1.2], [WUE_inst_means[(6, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        ax6.plot([0.8], [WUE_inst_means[(6, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        ax6.plot([2.2], [WUE_inst_means[(6, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        ax6.plot([1.8], [WUE_inst_means[(6, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        ax6.plot([4.2], [WUE_inst_means[(6, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        ax6.plot([3.8], [WUE_inst_means[(6, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_6_am.plot(kind='scatter', x='Offset_Time', y='WUE_inst', c='gray', ax=ax6, alpha=0.5, marker='o')
        gasex_6_pm.plot(kind='scatter', x='Offset_Time', y='WUE_inst', c='gray', ax=ax6, alpha=0.5, marker='v')
        ax6.set_xticks([1, 2, 3, 4, 5])

        plt.savefig('Output_Plots/WUE_instantaneous_tree_split.pdf')
        plt.show()

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey=True, sharex=True, figsize=(12, 8))
        marker_size=65.0
        # ax1.plot([gs_means[(1, 1, 'PM')]], [A_means[(1, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        # ax1.plot([gs_means[(1, 1, 'AM')]], [A_means[(1, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        # ax1.plot([gs_means[(1, 2, 'PM')]], [A_means[(1, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        # ax1.plot([gs_means[(1, 2, 'AM')]], [A_means[(1, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        # ax1.plot([gs_means[(1, 4, 'PM')]], [A_means[(1, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        # ax1.plot([gs_means[(1, 4, 'AM')]], [A_means[(1, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_1_am_1.plot(kind='scatter', x='gsw', y='A', c=day_colors[0], ax=ax1, marker='o', s=marker_size, alpha=0.6)
        gasex_1_pm_1.plot(kind='scatter', x='gsw', y='A', c=day_colors[0], ax=ax1, marker='v', s=marker_size, alpha=0.6)
        gasex_1_am_2.plot(kind='scatter', x='gsw', y='A', c=day_colors[1], ax=ax1, marker='o', s=marker_size, alpha=0.6)
        gasex_1_pm_2.plot(kind='scatter', x='gsw', y='A', c=day_colors[1], ax=ax1, marker='v', s=marker_size, alpha=0.6)
        gasex_1_am_4.plot(kind='scatter', x='gsw', y='A', c=day_colors[3], ax=ax1, marker='o', s=marker_size, alpha=0.6)
        gasex_1_pm_4.plot(kind='scatter', x='gsw', y='A', c=day_colors[3], ax=ax1, marker='v', s=marker_size, alpha=0.6)

        gasex_2_am_1.plot(kind='scatter', x='gsw', y='A', c=day_colors[0], ax=ax2, marker='o', s=marker_size, alpha=0.6)
        gasex_2_pm_1.plot(kind='scatter', x='gsw', y='A', c=day_colors[0], ax=ax2, marker='v', s=marker_size, alpha=0.6)
        gasex_2_am_2.plot(kind='scatter', x='gsw', y='A', c=day_colors[1], ax=ax2, marker='o', s=marker_size, alpha=0.6)
        gasex_2_pm_2.plot(kind='scatter', x='gsw', y='A', c=day_colors[1], ax=ax2, marker='v', s=marker_size, alpha=0.6)
        gasex_2_am_4.plot(kind='scatter', x='gsw', y='A', c=day_colors[3], ax=ax2, marker='o', s=marker_size, alpha=0.6)
        gasex_2_pm_4.plot(kind='scatter', x='gsw', y='A', c=day_colors[3], ax=ax2, marker='v', s=marker_size, alpha=0.6)

        gasex_3_am_1.plot(kind='scatter', x='gsw', y='A', c=day_colors[0], ax=ax3, marker='o', s=marker_size, alpha=0.6)
        gasex_3_pm_1.plot(kind='scatter', x='gsw', y='A', c=day_colors[0], ax=ax3, marker='v', s=marker_size, alpha=0.6)
        gasex_3_am_2.plot(kind='scatter', x='gsw', y='A', c=day_colors[1], ax=ax3, marker='o', s=marker_size, alpha=0.6)
        gasex_3_pm_2.plot(kind='scatter', x='gsw', y='A', c=day_colors[1], ax=ax3, marker='v', s=marker_size, alpha=0.6)
        gasex_3_am_4.plot(kind='scatter', x='gsw', y='A', c=day_colors[3], ax=ax3, marker='o', s=marker_size, alpha=0.6)
        gasex_3_pm_4.plot(kind='scatter', x='gsw', y='A', c=day_colors[3], ax=ax3, marker='v', s=marker_size, alpha=0.6)

        gasex_4_am_1.plot(kind='scatter', x='gsw', y='A', c=day_colors[0], ax=ax4, marker='o', s=marker_size, alpha=0.6)
        gasex_4_pm_1.plot(kind='scatter', x='gsw', y='A', c=day_colors[0], ax=ax4, marker='v', s=marker_size, alpha=0.6)
        gasex_4_am_2.plot(kind='scatter', x='gsw', y='A', c=day_colors[1], ax=ax4, marker='o', s=marker_size, alpha=0.6)
        gasex_4_pm_2.plot(kind='scatter', x='gsw', y='A', c=day_colors[1], ax=ax4, marker='v', s=marker_size, alpha=0.6)
        gasex_4_am_4.plot(kind='scatter', x='gsw', y='A', c=day_colors[3], ax=ax4, marker='o', s=marker_size, alpha=0.6)
        gasex_4_pm_4.plot(kind='scatter', x='gsw', y='A', c=day_colors[3], ax=ax4, marker='v', s=marker_size, alpha=0.6)

        gasex_5_am_1.plot(kind='scatter', x='gsw', y='A', c=day_colors[0], ax=ax5, marker='o', s=marker_size, alpha=0.6)
        gasex_5_pm_1.plot(kind='scatter', x='gsw', y='A', c=day_colors[0], ax=ax5, marker='v', s=marker_size, alpha=0.6)
        gasex_5_am_2.plot(kind='scatter', x='gsw', y='A', c=day_colors[1], ax=ax5, marker='o', s=marker_size, alpha=0.6)
        gasex_5_pm_2.plot(kind='scatter', x='gsw', y='A', c=day_colors[1], ax=ax5, marker='v', s=marker_size, alpha=0.6)
        gasex_5_am_4.plot(kind='scatter', x='gsw', y='A', c=day_colors[3], ax=ax5, marker='o', s=marker_size, alpha=0.6)
        gasex_5_pm_4.plot(kind='scatter', x='gsw', y='A', c=day_colors[3], ax=ax5, marker='v', s=marker_size, alpha=0.6)

        gasex_6_am_1.plot(kind='scatter', x='gsw', y='A', c=day_colors[0], ax=ax6, marker='o', s=marker_size, alpha=0.6)
        gasex_6_pm_1.plot(kind='scatter', x='gsw', y='A', c=day_colors[0], ax=ax6, marker='v', s=marker_size, alpha=0.6)
        gasex_6_am_2.plot(kind='scatter', x='gsw', y='A', c=day_colors[1], ax=ax6, marker='o', s=marker_size, alpha=0.6)
        gasex_6_pm_2.plot(kind='scatter', x='gsw', y='A', c=day_colors[1], ax=ax6, marker='v', s=marker_size, alpha=0.6)
        gasex_6_am_4.plot(kind='scatter', x='gsw', y='A', c=day_colors[3], ax=ax6, marker='o', s=marker_size, alpha=0.6)
        gasex_6_pm_4.plot(kind='scatter', x='gsw', y='A', c=day_colors[3], ax=ax6, marker='v', s=marker_size, alpha=0.6)

        plt.savefig('Output_Plots/A_vs_gs_tree_split.pdf')
        plt.show()

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey=True, sharex=True, figsize=(12, 8))
        marker_size=65.0
        # ax1.plot([gs_means[(1, 1, 'PM')]], [A_means[(1, 1, 'PM')]], marker='v', markersize=8, c=day_colors[0])
        # ax1.plot([gs_means[(1, 1, 'AM')]], [A_means[(1, 1, 'AM')]], marker='o', markersize=8, c=day_colors[0])
        # ax1.plot([gs_means[(1, 2, 'PM')]], [A_means[(1, 2, 'PM')]], marker='v', markersize=8, c=day_colors[1])
        # ax1.plot([gs_means[(1, 2, 'AM')]], [A_means[(1, 2, 'AM')]], marker='o', markersize=8, c=day_colors[1])
        # ax1.plot([gs_means[(1, 4, 'PM')]], [A_means[(1, 4, 'PM')]], marker='v', markersize=8, c=day_colors[3])
        # ax1.plot([gs_means[(1, 4, 'AM')]], [A_means[(1, 4, 'AM')]], marker='o', markersize=8, c=day_colors[3])
        gasex_1_am_1.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[0], ax=ax1, marker='o', s=marker_size, alpha=0.6)
        gasex_1_pm_1.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[0], ax=ax1, marker='v', s=marker_size, alpha=0.6)
        gasex_1_am_2.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[1], ax=ax1, marker='o', s=marker_size, alpha=0.6)
        gasex_1_pm_2.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[1], ax=ax1, marker='v', s=marker_size, alpha=0.6)
        gasex_1_am_4.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[3], ax=ax1, marker='o', s=marker_size, alpha=0.6)
        gasex_1_pm_4.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[3], ax=ax1, marker='v', s=marker_size, alpha=0.6)
        ax1.plot([0.0,1.0], [0.0, 1.0], color='gray', linestyle='dashed', alpha=0.6)

        gasex_2_am_1.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[0], ax=ax2, marker='o', s=marker_size, alpha=0.6)
        gasex_2_pm_1.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[0], ax=ax2, marker='v', s=marker_size, alpha=0.6)
        gasex_2_am_2.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[1], ax=ax2, marker='o', s=marker_size, alpha=0.6)
        gasex_2_pm_2.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[1], ax=ax2, marker='v', s=marker_size, alpha=0.6)
        gasex_2_am_4.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[3], ax=ax2, marker='o', s=marker_size, alpha=0.6)
        gasex_2_pm_4.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[3], ax=ax2, marker='v', s=marker_size, alpha=0.6)
        ax2.plot([0.0,1.0], [0.0, 1.0], color='gray', linestyle='dashed', alpha=0.6)

        gasex_3_am_1.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[0], ax=ax3, marker='o', s=marker_size, alpha=0.6)
        gasex_3_pm_1.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[0], ax=ax3, marker='v', s=marker_size, alpha=0.6)
        gasex_3_am_2.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[1], ax=ax3, marker='o', s=marker_size, alpha=0.6)
        gasex_3_pm_2.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[1], ax=ax3, marker='v', s=marker_size, alpha=0.6)
        gasex_3_am_4.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[3], ax=ax3, marker='o', s=marker_size, alpha=0.6)
        gasex_3_pm_4.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[3], ax=ax3, marker='v', s=marker_size, alpha=0.6)
        ax3.plot([0.0,1.0], [0.0, 1.0], color='gray', linestyle='dashed', alpha=0.6)

        gasex_4_am_1.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[0], ax=ax4, marker='o', s=marker_size, alpha=0.6)
        gasex_4_pm_1.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[0], ax=ax4, marker='v', s=marker_size, alpha=0.6)
        gasex_4_am_2.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[1], ax=ax4, marker='o', s=marker_size, alpha=0.6)
        gasex_4_pm_2.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[1], ax=ax4, marker='v', s=marker_size, alpha=0.6)
        gasex_4_am_4.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[3], ax=ax4, marker='o', s=marker_size, alpha=0.6)
        gasex_4_pm_4.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[3], ax=ax4, marker='v', s=marker_size, alpha=0.6)
        ax4.plot([0.0,1.0], [0.0, 1.0], color='gray', linestyle='dashed', alpha=0.6)

        gasex_5_am_1.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[0], ax=ax5, marker='o', s=marker_size, alpha=0.6)
        gasex_5_pm_1.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[0], ax=ax5, marker='v', s=marker_size, alpha=0.6)
        gasex_5_am_2.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[1], ax=ax5, marker='o', s=marker_size, alpha=0.6)
        gasex_5_pm_2.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[1], ax=ax5, marker='v', s=marker_size, alpha=0.6)
        gasex_5_am_4.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[3], ax=ax5, marker='o', s=marker_size, alpha=0.6)
        gasex_5_pm_4.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[3], ax=ax5, marker='v', s=marker_size, alpha=0.6)
        ax5.plot([0.0,1.0], [0.0, 1.0], color='gray', linestyle='dashed', alpha=0.6)

        gasex_6_am_1.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[0], ax=ax6, marker='o', s=marker_size, alpha=0.6)
        gasex_6_pm_1.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[0], ax=ax6, marker='v', s=marker_size, alpha=0.6)
        gasex_6_am_2.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[1], ax=ax6, marker='o', s=marker_size, alpha=0.6)
        gasex_6_pm_2.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[1], ax=ax6, marker='v', s=marker_size, alpha=0.6)
        gasex_6_am_4.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[3], ax=ax6, marker='o', s=marker_size, alpha=0.6)
        gasex_6_pm_4.plot(kind='scatter', x='gs_normed', y='A_normed', c=day_colors[3], ax=ax6, marker='v', s=marker_size, alpha=0.6)
        ax6.plot([0.0,1.0], [0.0, 1.0], color='gray', linestyle='dashed', alpha=0.6)

        plt.savefig('Output_Plots/A_vs_gs_normed_tree_split.pdf')
        plt.show()

    if leaf_area is not None:
        unknown_tree = leaf_area[leaf_area['Tree'].isna()]
        daily_sums = unknown_tree.groupby(['TIMESTAMP'])['Area_cm2'].sum()
        sums_df = daily_sums.to_frame()
        sums_df['cumulative'] = sums_df['Area_cm2'].cumsum(axis=0)

        ax = sums_df.plot(y='cumulative')
        ax.set_xlim(pd.Timestamp('2021-07-05'), pd.Timestamp('2021-07-29'))
        plt.savefig('Output_Plots/cum_leaf_area.pdf')
        plt.show()

    if root is not None:
        root = root.sort_values(by=['Depth'])
        root_1 = root[root['Tree'] == 1]
        root_4 = root[root['Tree'] == 4]
        root_5 = root[root['Tree'] == 5]

        plt.figure()
        ax = plt.gca()
        root_1.plot(style='.-', x='Root_percent', y='Depth', c=tree_colors[0], ax=ax)
        root_4.plot(style='.-', x='Root_percent', y='Depth', c=tree_colors[3], ax=ax)
        root_5.plot(style='.-', x='Root_percent', y='Depth', c=tree_colors[4], ax=ax)
        ax.invert_yaxis()

        plt.savefig('Output_Plots/root_dist_percent.pdf')
        plt.show()

        plt.figure()
        ax = plt.gca()
        root_1.plot(style='.-', x='Mass_g', y='Depth', c=tree_colors[0], ax=ax)
        root_4.plot(style='.-', x='Mass_g', y='Depth', c=tree_colors[3], ax=ax)
        root_5.plot(style='.-', x='Mass_g', y='Depth', c=tree_colors[4], ax=ax)
        ax.invert_yaxis()

        plt.savefig('Output_Plots/root_dist_mass.pdf')
        plt.show()

    if soil_moisture is not None and lwp is not None:
        psi_soil_mean = soil_moisture.groupby(['Tree_Number', 'Sampling_Day'], as_index=False)['psi_soil'].mean()
        combined_mean = pd.merge(lwp, psi_soil_mean[['Tree_Number', 'Sampling_Day', 'psi_soil']],
                          on=['Tree_Number', 'Sampling_Day'])

        psi_soil_max = soil_moisture.groupby(['Tree_Number', 'Sampling_Day'], as_index=False)['psi_soil'].max()
        combined_max = pd.merge(lwp, psi_soil_max[['Tree_Number', 'Sampling_Day', 'psi_soil']],
                                 on=['Tree_Number', 'Sampling_Day'])

        psi_soil_mean_pd = combined_mean[combined_mean['LWP_Type']=='Predawn']
        psi_soil_mean_md = combined_mean[combined_mean['LWP_Type'] == 'Midday']
        psi_soil_max_pd = combined_max[combined_max['LWP_Type'] == 'Predawn']
        psi_soil_max_md = combined_max[combined_max['LWP_Type'] == 'Midday']

        T1_pd_mean = psi_soil_mean_pd[psi_soil_mean_pd['Tree_Number'] == 1]
        T2_pd_mean = psi_soil_mean_pd[psi_soil_mean_pd['Tree_Number'] == 2]
        T3_pd_mean = psi_soil_mean_pd[psi_soil_mean_pd['Tree_Number'] == 3]
        T4_pd_mean = psi_soil_mean_pd[psi_soil_mean_pd['Tree_Number'] == 4]
        T5_pd_mean = psi_soil_mean_pd[psi_soil_mean_pd['Tree_Number'] == 5]
        T6_pd_mean = psi_soil_mean_pd[psi_soil_mean_pd['Tree_Number'] == 6]

        T1_md_mean = psi_soil_mean_md[psi_soil_mean_md['Tree_Number'] == 1]
        T2_md_mean = psi_soil_mean_md[psi_soil_mean_md['Tree_Number'] == 2]
        T3_md_mean = psi_soil_mean_md[psi_soil_mean_md['Tree_Number'] == 3]
        T4_md_mean = psi_soil_mean_md[psi_soil_mean_md['Tree_Number'] == 4]
        T5_md_mean = psi_soil_mean_md[psi_soil_mean_md['Tree_Number'] == 5]
        T6_md_mean = psi_soil_mean_md[psi_soil_mean_md['Tree_Number'] == 6]

        T1_pd_max = psi_soil_max_pd[psi_soil_max_pd['Tree_Number'] == 1]
        T2_pd_max = psi_soil_max_pd[psi_soil_max_pd['Tree_Number'] == 2]
        T3_pd_max = psi_soil_max_pd[psi_soil_max_pd['Tree_Number'] == 3]
        T4_pd_max = psi_soil_max_pd[psi_soil_max_pd['Tree_Number'] == 4]
        T5_pd_max = psi_soil_max_pd[psi_soil_max_pd['Tree_Number'] == 5]
        T6_pd_max = psi_soil_max_pd[psi_soil_max_pd['Tree_Number'] == 6]

        T1_md_max = psi_soil_max_md[psi_soil_max_md['Tree_Number'] == 1]
        T2_md_max = psi_soil_max_md[psi_soil_max_md['Tree_Number'] == 2]
        T3_md_max = psi_soil_max_md[psi_soil_max_md['Tree_Number'] == 3]
        T4_md_max = psi_soil_max_md[psi_soil_max_md['Tree_Number'] == 4]
        T5_md_max = psi_soil_max_md[psi_soil_max_md['Tree_Number'] == 5]
        T6_md_max = psi_soil_max_md[psi_soil_max_md['Tree_Number'] == 6]

        alpha_level=0.7

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 8))
        col = T1_pd_max.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T1_pd_max.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax1, s=80, alpha=alpha_level)
        ax1.plot([0.0, -1.3], [0.0, -1.3], color='gray', linestyle='dashed', alpha=0.6)

        col = T2_pd_max.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T2_pd_max.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax2, s=80, alpha=alpha_level)
        ax2.plot([0.0, -1.3], [0.0, -1.3], color='gray', linestyle='dashed', alpha=0.6)

        col = T3_pd_max.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T3_pd_max.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax3, s=80, alpha=alpha_level)
        ax3.plot([0.0, -1.3], [0.0, -1.3], color='gray', linestyle='dashed', alpha=0.6)

        col = T4_pd_max.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T4_pd_max.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax4, s=80, alpha=alpha_level)
        ax4.plot([0.0, -1.3], [0.0, -1.3], color='gray', linestyle='dashed', alpha=0.6)

        col = T5_pd_max.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T5_pd_max.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax5, s=80, alpha=alpha_level)
        ax5.plot([0.0, -1.3], [0.0, -1.3], color='gray', linestyle='dashed', alpha=0.6)

        col = T6_pd_max.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T6_pd_max.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax6, s=80, alpha=alpha_level)
        ax6.plot([0.0, -1.3], [0.0, -1.3], color='gray', linestyle='dashed', alpha=0.6)

        plt.savefig('Output_Plots/predawn_lwp_vs_max_psi_soil.pdf')
        plt.show()

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey=True, sharex=True, figsize=(12, 8))
        col = T1_pd_mean.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T1_pd_mean.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax1, s=80, alpha=alpha_level)
        ax1.plot([0.0, -1.3], [0.0, -1.3], color='gray', linestyle='dashed', alpha=0.6)

        col = T2_pd_mean.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T2_pd_mean.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax2, s=80, alpha=alpha_level)
        ax2.plot([0.0, -1.3], [0.0, -1.3], color='gray', linestyle='dashed', alpha=0.6)

        col = T3_pd_mean.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T3_pd_mean.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax3, s=80, alpha=alpha_level)
        ax3.plot([0.0, -1.3], [0.0, -1.3], color='gray', linestyle='dashed', alpha=0.6)

        col = T4_pd_mean.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T4_pd_mean.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax4, s=80, alpha=alpha_level)
        ax4.plot([0.0, -1.3], [0.0, -1.3], color='gray', linestyle='dashed', alpha=0.6)

        col = T5_pd_mean.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T5_pd_mean.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax5, s=80, alpha=alpha_level)
        ax5.plot([0.0, -1.3], [0.0, -1.3], color='gray', linestyle='dashed', alpha=0.6)

        col = T6_pd_mean.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T6_pd_mean.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax6, s=80, alpha=alpha_level)
        ax6.plot([0.0, -1.3], [0.0, -1.3], color='gray', linestyle='dashed', alpha=0.6)

        plt.savefig('Output_Plots/predawn_lwp_vs_mean_psi_soil.pdf')
        plt.show()

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey=True, sharex=True, figsize=(12, 8))
        col = T1_md_max.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T1_md_max.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax1, s=80, alpha=alpha_level, marker='v',)
        ax1.plot([0.0, -1.75], [0.0, -1.75], color='gray', linestyle='dashed', alpha=0.6)

        col = T2_md_max.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T2_md_max.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax2, s=80, alpha=alpha_level, marker='v',)
        ax2.plot([0.0, -1.75], [0.0, -1.75], color='gray', linestyle='dashed', alpha=0.6)

        col = T3_md_max.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T3_md_max.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax3, s=80, alpha=alpha_level, marker='v',)
        ax3.plot([0.0, -1.75], [0.0, -1.75], color='gray', linestyle='dashed', alpha=0.6)

        col = T4_md_max.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T4_md_max.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax4, s=80, alpha=alpha_level, marker='v',)
        ax4.plot([0.0, -1.75], [0.0, -1.75], color='gray', linestyle='dashed', alpha=0.6)

        col = T5_md_max.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T5_md_max.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax5, s=80, alpha=alpha_level, marker='v',)
        ax5.plot([0.0, -1.75], [0.0, -1.75], color='gray', linestyle='dashed', alpha=0.6)

        col = T6_md_max.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T6_md_max.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax6, s=80, alpha=alpha_level, marker='v',)
        ax6.plot([0.0, -1.75], [0.0, -1.75], color='gray', linestyle='dashed', alpha=0.6)

        plt.savefig('Output_Plots/midday_lwp_vs_max_psi_soil.pdf')
        plt.show()

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey=True, sharex=True, figsize=(12, 8))
        col = T1_md_mean.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T1_md_mean.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax1, s=80, alpha=alpha_level, marker='v',)
        ax1.plot([0.0, -1.75], [0.0, -1.75], color='gray', linestyle='dashed', alpha=0.6)

        col = T2_md_mean.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T2_md_mean.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax2, s=80, alpha=alpha_level, marker='v',)
        ax2.plot([0.0, -1.75], [0.0, -1.75], color='gray', linestyle='dashed', alpha=0.6)

        col = T3_md_mean.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T3_md_mean.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax3, s=80, alpha=alpha_level, marker='v',)
        ax3.plot([0.0, -1.75], [0.0, -1.75], color='gray', linestyle='dashed', alpha=0.6)

        col = T4_md_mean.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T4_md_mean.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax4, s=80, alpha=alpha_level, marker='v',)
        ax4.plot([0.0, -1.75], [0.0, -1.75], color='gray', linestyle='dashed', alpha=0.6)

        col = T5_md_mean.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T5_md_mean.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax5, s=80, alpha=alpha_level, marker='v',)
        ax5.plot([0.0, -1.75], [0.0, -1.75], color='gray', linestyle='dashed', alpha=0.6)

        col = T6_md_mean.Sampling_Day.map({1: '#f4a582', 2: '#d6604d', 3: '#b2182b', 4: '#92c5de', 5: '#2166ac'})
        T6_md_mean.plot(kind='scatter', x='psi_soil', y='LWP_(MPa)', c=col, ax=ax6, s=80, alpha=alpha_level, marker='v',)
        ax6.plot([0.0, -1.75], [0.0, -1.75], color='gray', linestyle='dashed', alpha=0.6)

        plt.savefig('Output_Plots/midday_lwp_vs_mean_psi_soil.pdf')
        plt.show()

    if gasex_curves is not None:
        ACi_Curves = gasex_curves[gasex_curves['Curve_Type'] == 'ACi']
        Light_Curves = gasex_curves[gasex_curves['Curve_Type'] == 'Light']

        colLight = Light_Curves.Curve_Rep.map(
             {1: '#003f5c', 2: '#58508d', 3: '#bc5090', 4: '#ff6361', 5: '#ffa600'})
        colACi = ACi_Curves.Curve_Rep.map(
            {1: '#003f5c', 2: '#bc5090', 3: '#ffa600'})

        ACi_Curves.plot(kind='scatter', x='Ci', y='A', c=colACi, s=80, alpha=0.8, marker='.')
        plt.savefig('Output_Plots/ACi_Curves.pdf')
        plt.show()

        Light_Curves.plot(kind='scatter', x='Qin', y='A', c=colLight, s=80, alpha=0.8, marker='.')
        plt.savefig('Output_Plots/Light_Curves.pdf')
        plt.show()



def isotope_mixing(isotopes_df):
    #uses scipy optimize routine to solve for contributions from 6 10-cm soil layers to water uptake based on linear mixing
    sampling_days = [1, 2, 3, 4, 5]
    tree_nums = [1, 2, 3, 4, 5, 6]

    water_fractions = []
    ###unweighted 6 source two isotopes
    for t in tree_nums:
        tree_set = isotopes_df[isotopes_df['Tree_Number'] == t]
        for s in sampling_days:
            day_set = tree_set[tree_set['Sampling_Day'] == s]
            day_soil = day_set[day_set['Sample_Type'] == 'Soil']
            D_1 = day_soil[day_soil['Depth'] <= 10.0]['del_D'].mean()
            D_2 = day_soil[(day_soil['Depth'] >= 10.0) & (day_soil['Depth'] <= 20.0)]['del_D'].mean()
            D_3 = day_soil[(day_soil['Depth'] >= 20.0) & (day_soil['Depth'] <= 30.0)]['del_D'].mean()
            D_4 = day_soil[(day_soil['Depth'] >= 30.0) & (day_soil['Depth'] <= 40.0)]['del_D'].mean()
            D_5 = day_soil[(day_soil['Depth'] >= 40.0) & (day_soil['Depth'] <= 50.0)]['del_D'].mean()
            D_6 = day_soil[day_soil['Depth'] >= 50.0]['del_D'].mean()
            O_1 = day_soil[day_soil['Depth'] <= 10.0]['del_O'].mean()
            O_2 = day_soil[(day_soil['Depth'] >= 10.0) & (day_soil['Depth'] <= 20.0)]['del_O'].mean()
            O_3 = day_soil[(day_soil['Depth'] >= 20.0) & (day_soil['Depth'] <= 30.0)]['del_O'].mean()
            O_4 = day_soil[(day_soil['Depth'] >= 30.0) & (day_soil['Depth'] <= 40.0)]['del_O'].mean()
            O_5 = day_soil[(day_soil['Depth'] >= 40.0) & (day_soil['Depth'] <= 50.0)]['del_O'].mean()
            O_6 = day_soil[day_soil['Depth'] >= 50.0]['del_O'].mean()
            tree_D = day_set[day_set['Sample_Type'] == 'Plant']['del_D'].mean()
            tree_O = day_set[day_set['Sample_Type'] == 'Plant']['del_O'].mean()
            O_depths = [O_1, O_2, O_3, O_4, O_5, O_6]
            D_depths = [D_1, D_2, D_3, D_4, D_5, D_6]

            def objective(x):  # fractions sum to 1
                return abs(1.0 - x[0] - x[1] - x[2] - x[3] - x[4] - x[5])

            def constraint1(x):  # D values
                sum_eq = tree_D
                for i in range(6):
                    sum_eq = sum_eq - D_depths[i] * x[i]
                return sum_eq

            def constraint2(x):  # 18O values
                sum_eq = tree_O
                for i in range(6):
                    sum_eq = sum_eq - O_depths[i] * x[i]
                return sum_eq

            # initial guesses - assume equal contributions
            n = 6
            x0 = np.zeros(n)
            x0[0] = 1.0 / n
            x0[1] = 1.0 / n
            x0[2] = 1.0 / n
            x0[3] = 1.0 / n
            x0[4] = 1.0 / n
            x0[5] = 1.0 / n

            # optimize
            b = (0.0, 1.0)
            bnds = (b, b, b, b, b, b)
            con1 = {'type': 'eq', 'fun': constraint1}
            con2 = {'type': 'eq', 'fun': constraint2}
            cons = ([con1, con2])
            solution = minimize(objective, x0, method='SLSQP', \
                                bounds=bnds, constraints=cons)
            [f_1, f_2, f_3, f_4, f_5, f_6] = solution.x

            water_fractions.append(
                {
                    'Tree_Number': t,
                    'Sampling_Day': s,
                    'f_1': f_1,
                    'f_2': f_2,
                    'f_3': f_3,
                    'f_4': f_4,
                    'f_5': f_5,
                    'f_6': f_6
                }
            )
    water_fractions_df = pd.DataFrame(water_fractions)
    water_fractions_df['Mean_Uptake_Depth'] = (water_fractions_df['f_1'] * 5.0 + water_fractions_df['f_2'] * 15.0 + \
                                               water_fractions_df['f_3'] * 25.0 + water_fractions_df['f_4'] * 35.0 + \
                                               water_fractions_df['f_5'] * 45.0 + water_fractions_df['f_6'] * 55.0) / \
                                              (water_fractions_df['f_1'] + water_fractions_df['f_2'] +
                                               water_fractions_df['f_3'] + water_fractions_df['f_4'] +
                                               water_fractions_df['f_5'] + water_fractions_df['f_6'])
    return water_fractions_df


