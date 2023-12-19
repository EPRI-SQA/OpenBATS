import pandas as pd
import numpy as np 
import os 

import plotly.graph_objs as go 
import plotly.offline as pyo 
import plotly.express as px 
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt 

from darts import TimeSeries 
import test_darts_forecast_v3 as forecaster

import torch 

#simulate v3: vis
#   convert data to hourly data for day, and discharge according to higher cost hours
#   keep 15 min interval data, and discharge according to higher cost qurters hours with hourly-price split into quarter for simulation 


#simulate v2: vis
#   convert data to hourly data for day, and discharge according to higher cost hours
#   issue: PG&e peak pricing is 8:30 to 12:30 summer and therefor cannot align this price in simulation correctly 

#batt model linear regression delta_soc = 1.66 * x + 0.5 , where x = kw in 15 min sampling



class GLOBS():
    # Old simulate.py GLOBS: 
    #filepath_historical = "./data/df_historical.csv" 
    #filepath_model = "./.darts/checkpoints/2021-10-02_21.15.59.427539_torch_model_run_8836/checkpoint_19.pth.tar"

    #filepath_X = "./X_scaled.P" 
    #filepath_X_alt = "./X_altered_scaled.P" 

    #start_ts = pd.to_datetime("2021-05-02") 
    #n_startup = 24*45 
    #forecast_horizon = 4

    # New simulate.py GLOBS: 
    start_date = "2021-04-20" # TODO: Add Winter Months to training data (or warmer summer...?) (DONE) 
    end_date = "2021-05-30"
    plot_start = "2021-05-15" 
    plot_end = "2021-05-19" 

    # eval_darts_forecast GLOBS: 
    filepath_in = "./data/df_historical_20210101-20210601.csv"
    filepath_tstat_alt = "./data/test_data/darts_counterfactual_tstat.csv" 
    # trained model from Corey's machine
    #file_trained_model = ".darts/checkpoints/2021-10-09_14.01.57.041023_torch_model_run_1228/checkpoint_49.pth.tar" 
    # trained model from run on Vis machine
    file_trained_model = ".darts/checkpoints/2022-01-05_16.35.18.059790_torch_model_run_27084/checkpoint_49.pth.tar" 
    pred_dates = [
        "2021-05-01", "2021-05-02", "2021-05-03", "2021-05-04",
        "2021-05-05", "2021-05-06", "2021-05-07", "2021-05-08",
        "2021-05-09", "2021-05-10", 
        "2021-05-11", "2021-05-12", "2021-05-13", "2021-05-14", 
        "2021-05-15", "2021-05-16", "2021-05-17", "2021-05-18", 
        "2021-05-19", "2021-05-20", "2021-05-21", "2021-05-22", 
        "2021-05-23", "2021-05-24", "2021-05-25", "2021-05-26", 
        "2021-05-27", "2021-05-28", "2021-05-29", "2021-05-30",
        "2021-05-31"
    ]
    forecast_horizon = 24*4
    t_res_minutes = 15 

    # Storage Operational Configuration 
    # Uncomment the battery_test_case declaration for the appropriate simulation test case. 
    # Currently implemented for "TUNE HISTORICAL BASELINE" and "X_d_max SENSITIVITY ANALYSIS" 
    #battery_test_case = "TUNE HISTORICAL BASELINE" 
    #battery_test_case = "X_d_max SENSITIVITY ANALYSIS" 
    battery_test_case = "bill_reduction" 

    if battery_test_case == "TUNE HISTORICAL BASELINE": 
        # Battery Params (tuned to historical usage from 2021-05-11 through 2021-05-14)
        batt_cap_kwh = 52 # kWh 
        battery_soc_max = 63 # SOC% maximum setting 
        x_soc_min = 30 
        X_d_max = [2.7] # kW 
        chg_kw_max = 29 
        dchg_kw_max = 29 
    elif battery_test_case == "X_d_max SENSITIVITY ANALYSIS": 
        # Grid Seach optimization 
        batt_cap_kwh = 52 # Obtained from 
        battery_soc_max = 100 
        x_soc_min = 30 
        X_d_max = [2,4,5,6,7,8,9,10,11,12] # kW 
        chg_kw_max = 29 
        dchg_kw_max = 29 
    elif battery_test_case == "bill_reduction":
        # new algorithm using batt to reduce high price ranges
        batt_cap_kwh = 52 # Obtained from 
        battery_soc_max = 100 
        x_soc_min = 30 
        X_d_max = 2.7 # kW 
        chg_kw_max = 29 
        dchg_kw_max = 29

    start_charge_limit = -4 # kW 

    price_offpeak = 0.14181 # $/kWh 
    price_onpeak = 0.22501 # $/kWh  
    price_partpeak = 0.16988 #$kwh
    price_export = 0.18 # $/kwh     
    price_peak = 19.85 # $/kW*month (demand charge)
    customer_charge = 4.59959 # charge / day according to bill
    price_arr = []
    price_arr.extend( (np.ones(34) * price_offpeak).tolist() )  # 12:00 Am to 8:30 Am
    price_arr.extend( (np.ones(14) * price_partpeak).tolist() ) # 8:30 AM to 12:00 PM
    price_arr.extend( (np.ones(24) * price_onpeak).tolist() )   # 12:00 PM to 6:00 PM
    price_arr.extend( (np.ones(14) * price_partpeak).tolist() ) # 6:00 PM to 9:30 PM
    price_arr.extend( (np.ones(10) * price_offpeak).tolist() )  # 9:30 PM to 12:00 AM
    price_tou = dict(enumerate(price_arr))
    
    price_arr = []
    price_arr.extend( (np.ones(96) * price_offpeak).tolist() )  # 12:00 Am to 12:00 Am next day
    price_tou_hday = dict(enumerate(price_arr))
 


if __name__ == "__main__": 
    # TODO: Take the eval_darts_forecast main, and expand to incorporate battery operation response to 
    # HVAC alternate operation (counterfactual operation) 
    f = forecaster.Forecaster(
        forecaster.GLOBS.y_var, 
        forecaster.GLOBS.X_vars
    )
    f.import_data(GLOBS.filepath_in) 
    f.df_raw = f.prep_features(f.df_raw) 
    f.prep_data(start_date=GLOBS.start_date, end_date=GLOBS.end_date)  
    f.split_scale_data(forecaster.GLOBS.train_val_split_date) 

    f.prep_altered_data(GLOBS.filepath_tstat_alt) 
    f.import_trained_model(GLOBS.file_trained_model) 

    # Run Episodic Predictions
    f.get_episodic_preds(
        future_covariates=f.X_scaled, 
        future_covariates_altered=f.X_altered_scaled, 
        pred_dates = GLOBS.pred_dates, 
        forecast_horizon = GLOBS.forecast_horizon 
    )
    #f.plot_episodic_preds(fig_height = 1600) 


    # Simulate Battery Operation accounting for simulated HVAC 
    df_sim_baseline = f.prep_sim_df(
        df_hist = f.df_raw.set_index("Timestamp")[["G_solar", "D_nonflex", "D_battery_eg"]],
        df_hvac = f.y_ep_pred, 
        t_res = "{}min".format(GLOBS.t_res_minutes), 
    )

    if GLOBS.battery_test_case == "X_d_max SENSITIVITY ANALYSIS":

        df_sim_baseline = f.simulate_batt(
            df_sim_baseline, 
            x_d_max = 2.7, # kW 
            x_soc_min = GLOBS.x_soc_min, # percent, 
            batt_cap_kwh = GLOBS.batt_cap_kwh, # kWh, TODO: update with actual values 
            battery_soc_max = GLOBS.battery_soc_max, 
            t_res_minutes = GLOBS.t_res_minutes, # TODO: moving to func args...time resolution of simulation in minutes 
            chg_kw_max = GLOBS.chg_kw_max, # TODO: moving to func args...max charge (kW) 
            dchg_kw_max = GLOBS.dchg_kw_max # TODO: moving to func args...max discharge (kW) 
            # cap_kwh_max = 20 # TODO: moving to func args...Max batter kWh, appears not to be used 
        )
        f.plot_battery_sim(
            df_sim_baseline,
            fig_title = "Simulated Baseline", 
            savepath = "timeseries-plot_sim-baseline.png", 
            start = GLOBS.plot_start, 
            end = GLOBS.plot_end 
        ) 

        # Alternate battery setting 
        df_sim_alt = f.prep_sim_df(
            df_hist = f.df_raw.set_index("Timestamp")[["G_solar", "D_nonflex"]],
            df_hvac = f.y_ep_pred, 
            t_res = "{}min".format(GLOBS.t_res_minutes), 
        )
        
        df_sim_alt = f.simulate_batt(
            df_sim_alt, 
            x_d_max=7.5, 
            x_soc_min = GLOBS.x_soc_min, 
            batt_cap_kwh = GLOBS.batt_cap_kwh, 
            battery_soc_max = GLOBS.battery_soc_max, 
            t_res_minutes=GLOBS.t_res_minutes, 
            chg_kw_max=GLOBS.chg_kw_max, 
            dchg_kw_max=GLOBS.dchg_kw_max 
        )
        f.plot_battery_sim(
            df_sim_alt,
            fig_title = "Simulated Alternate Battery Setting", 
            savepath = "timeseries-plot_sim-altbatt.png", 
            start = GLOBS.plot_start, 
            end = GLOBS.plot_end 
        )

        # Simulate x_d_max sensitivity analysis 
        results = {} 
        bills = {} 
        bills_tou = {} 
        peaks = {} 
        for x_d_max in GLOBS.X_d_max: 
            results[x_d_max] = f.calc_bill(
                f.simulate_batt(
                    df_sim_alt, 
                    x_d_max=x_d_max, 
                    x_soc_min = GLOBS.x_soc_min, 
                    batt_cap_kwh=GLOBS.batt_cap_kwh, 
                    battery_soc_max = GLOBS.battery_soc_max, 
                    t_res_minutes=GLOBS.t_res_minutes, 
                    chg_kw_max=GLOBS.chg_kw_max, 
                    dchg_kw_max=GLOBS.dchg_kw_max
                )
            )
            bills[x_d_max] = results[x_d_max]['bill_total_permonth'].sum() 
            bills_tou[x_d_max] = results[x_d_max]['bill_tou_permonth'].sum() 
            peaks[x_d_max] = results[x_d_max]['D_net_mains'].max() 
            print("Dem Limit: {} kW, TOU Bill: {:.2f}, Peak Dem: {:.2f}, Bill: ${:.2f}".format(x_d_max, bills_tou[x_d_max], peaks[x_d_max], bills[x_d_max]))
            # Export Results to CSV 
            results[x_d_max].to_csv("timeseries-results_{}kw.csv".format(x_d_max)) 
        pd.Series(bills).plot() 
        plt.xlabel("Net Load Demand Limit (kW)") 
        plt.ylabel("Monthly Bill ($/month)") 
        plt.savefig('sensitivity-analysis.png', dpi=200) 
        plt.close() 

        # Investigate Timeseries Plot Outputs 
        
        # 6 kW Limit Plots 
        results[6][['bill_import', 'bill_export']].plot(figsize=[12,4])
        plt.title("6 kW Limit")
        plt.savefig('timeseries-bills_6kw.png')
        plt.close() 

        results[6][["D_net_mains"]].plot(figsize=[12,4])
        plt.title("6 kW Limit")
        plt.savefig('timeseries-netmains_6kw.png') 
        plt.close() 

        # 8 kW Limit Plots 

        # Plot Historical Baseline 
        df_hist_baseline = f.df_raw.set_index("Timestamp")
        df_hist_baseline = df_hist_baseline[(df_hist_baseline.index>=df_sim_baseline.index.min()) & (df_hist_baseline.index <= df_sim_baseline.index.max())]
        df_hist_baseline["D_tot"] = df_hist_baseline["D_mains"] - df_hist_baseline["G_solar"] - df_hist_baseline["D_battery_eg"]
        df_hist_baseline["SOC_kwh"] = df_hist_baseline["SOC"] * GLOBS.batt_cap_kwh / 100 
        #df_hist_baseline[["D_tot", "G_solar", "D_mains", "D_battery_eg", "SOC_kwh"]].plot() 
        #plt.title("Historical Baseline") 

        print('Done %s' % GLOBS.battery_test_case)

    elif GLOBS.battery_test_case == "bill_reduction":
        # Timestamp is set as index, later on in the computation, it is converted to use hourly data, so specifically setting Timestamp
        df_sim_baseline['Timestamp'] = df_sim_baseline.index
        df_sim_baseline = df_sim_baseline.reset_index(drop=True)
        
        df_sim_hr = f.simulate_batt_sch_with_cost(
            df_sim_baseline, 
            x_d_max=7.5, 
            x_soc_min = GLOBS.x_soc_min, 
            batt_cap_kwh = GLOBS.batt_cap_kwh, 
            battery_soc_max = GLOBS.battery_soc_max, 
            t_res_minutes= 15, 
            chg_kw_max=GLOBS.chg_kw_max*4, 
            dchg_kw_max=GLOBS.dchg_kw_max*4,
            price_tou_dict=GLOBS.price_tou,
            price_export = GLOBS.price_export,
            price_peak = GLOBS.price_peak,
            price_hday = GLOBS.price_tou_hday
        )
        df_sim_hr.set_index('Timestamp')

        print('     ')

        df_sim_baseline_hr = f.prepare_baseline_qhourly(df_sim_baseline, 
                                                       cols = ["G_solar", "D_hvac", "D_battery_eg", "D_nonflex"],
                                                       t_res_minutes=15,
                                                       price_tou_dict=GLOBS.price_tou,
                                                       price_export = GLOBS.price_export,
                                    price_peak = GLOBS.price_peak,
                                    price_hday = GLOBS.price_tou_hday)

        #df_sim_hr[['D_batt', 'G_solar', "D_net_mains", 'D_tot_mains', 'D_hvac', 'soc']].plot()


        df_sim_baseline_hr.rename(
            columns={"G_solar":"G_solar_b", "D_hvac":"D_hvac_b",
                        "D_nonflex":"D_nonflex_b",
                        "price": "price_b", 
                        "consumption": "consumption_b",
                        "consumption_kwh": "consumption_kwh_b",
                        "kw_import": "kw_import_b",
                        "kw_export" : "kw_export_b",
                        "kw_peak" : "kw_peak_b",
                        "kwh_import": "kwh_import_b",
                        "kwh_export" : "kwh_export_b",
                        "bill_import" : "bill_import_b",
                        "bill_export" : "bill_export_b"
                        }
                  ,inplace=True)

        df_cost_sim = f.process_cost(df_sim_hr)
        peak_sim = df_sim_hr.kw_peak.max()
        df_cost_baseline = f.process_cost(df_sim_baseline_hr, import_key='bill_import_b', export_key = "bill_export_b")
        peak_baseline = df_sim_baseline_hr.kw_peak_b.max()
        
        print("Baseline (based on computed HVAC for simulated rabge")
        print(df_cost_baseline )
        print("peak kw = ", peak_baseline)
        tmp_import_cost = df_cost_baseline.import_cost.sum()
        tmp_export_cost = df_cost_baseline.export_cost.sum()
        total_baseline_cost = GLOBS().price_peak * peak_baseline + tmp_import_cost + tmp_export_cost
        print("import -> $", tmp_import_cost, "  export -> $", tmp_export_cost)
        print("total bill -> $", total_baseline_cost)
        
        print("Simulation: (based on computed HVAC for simulated rabge")
        print(df_cost_sim )
        print("peak kw = ", peak_sim)
        tmp_import_cost = df_cost_sim.import_cost.sum()
        tmp_export_cost = df_cost_sim.export_cost.sum()
        total_sim_cost = GLOBS().price_peak * peak_sim + tmp_import_cost + tmp_export_cost
        print("import -> $", tmp_import_cost, "  export -> $", tmp_export_cost)
        print("total bill -> $", total_sim_cost)
       

        df_sim_baseline_hr.set_index('Timestamp', inplace=True)
        df_sim_hr.set_index('Timestamp', inplace=True)
        df_sim_hr.drop(['index', 'level_0', 'Hour', 'Minute'], inplace=True, axis=1)
        df_sim_baseline_hr.drop('index', inplace=True, axis=1)        
        
        tmpdf = df_sim_hr.join(df_sim_baseline_hr)
        tmpdf.to_csv('./temp_sim_comp_hr.csv')


        # df_sim_hr_bill = f.calc_bill(df_sim_hr, net_kw_col='D_net_mains', t_res_minutes=15)
        # df_sim_baseline_hr_bill = f.calc_bill(df_sim_baseline_hr, net_kw_col='D_net_mains_b', t_res_minutes=15)

        # bills = df_sim_hr_bill['bill_total_permonth'].sum() 
        # bills_tou = df_sim_hr_bill['bill_tou_permonth'].sum() 
        # peaks = df_sim_hr_bill['D_net_mains'].max() 
        # print('Simulated')
        # print("TOU Bill: {:.2f}, Peak Dem: {:.2f}, Bill: ${:.2f}".format(bills_tou, peaks, bills))


        # bills = df_sim_baseline_hr_bill['bill_total_permonth'].sum() 
        # bills_tou = df_sim_baseline_hr_bill['bill_tou_permonth'].sum() 
        # peaks = df_sim_baseline_hr_bill['D_net_mains_b'].max() 
        # print('baseline')
        # print("TOU Bill: {:.2f}, Peak Dem: {:.2f}, Bill: ${:.2f}".format(bills_tou, peaks, bills))

        #df_sim_baseline['D_net_mains'] = df_sim_baseline[["G_solar", "D_hvac", "D_nonflex"]].sum(axis=1)
        #df_sim_baseline['D_tot_mains'] = df_sim_baseline[["D_hvac", "D_nonflex" ]].sum(axis=1)

        #df_sim_baseline.set_index('Timestamp')
        #df_sim_baseline[['G_solar', "D_net_mains", 'D_tot_mains', 'D_hvac']].plot()

        df_sim_hr.reset_index(inplace=True)
        df_sim_baseline_hr.reset_index(inplace=True)


        traces = [] 
        #for col in ['D_batt', 'G_solar', "D_net_mains", 'D_tot_mains', 'D_hvac', 'soc']:
        for col in ['D_batt', 'G_solar', "D_net_mains", 'D_tot_mains', 'D_hvac', "kw_peak", 
                   "kwh_import", "bill_import", "kwh_export", "bill_export"]:
            trace = go.Scatter(
                x = df_sim_hr.Timestamp, 
                y = df_sim_hr[col], 
                name = col, 
                opacity = 0.7
            )
            traces.append(trace) 
        for col in ['D_net_mains_b', 'D_tot_mains_b', "kw_peak_b",
                    "kwh_import_b", "bill_import_b", "kwh_export_b", "bill_export_b"]:
            trace = go.Scatter(
                x = df_sim_baseline_hr.Timestamp, 
                y = df_sim_baseline_hr[col], 
                name = col, 
                opacity = 0.7
            )
            traces.append(trace)
        trace = go.Scatter(
            x = df_sim_baseline_hr.Timestamp, 
            y = df_sim_baseline_hr['D_battery_eg'], 
            name = "D_batt_b", 
            opacity = 0.7
        )
        traces.append(trace)
                
        fig = go.Figure(data=traces, layout={'title': "Sim Schedule"}) 
        pyo.plot(fig, 'all_data') 
        