from datetime import timedelta
from lzma import CHECK_UNKNOWN
#from sqlite3 import Timestamp
import pandas as pd
import numpy as np 
import os 
import calendar 

import plotly.graph_objs as go 
import plotly.offline as pyo 
import plotly.express as px 
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt 

from darts import TimeSeries 
from darts.models import NBEATSModel
from darts.models import RegressionModel
from darts.models import RNNModel 
from darts.models import BlockRNNModel 
from darts.dataprocessing.transformers import Scaler

import torch 

class GLOBS():
    filepath_in = "./data/df_historical_20210101-20210601.csv"
    filepath_tstat_alt = "./data/test_data/darts_counterfactual_tstat.csv" 
    # TODO: Add multiple sample series inputs to model fitting (non-contiguous time series)  
    start_date = "2021-04-20" # TODO: Add Winter Months to training data (or warmer summer...?) (DONE) 
    end_date = "2021-06-01" 
    train_val_split_date = "2021-05-20" 
    #n_val_samples = 480 # Deprecated 
    train_frac = 0.8
    n_epochs = 50
    forecast_horizon = 24*4
    pred_dates = [
        "2021-05-21", "2021-05-22", "2021-05-23", "2021-05-24", 
        "2021-05-25", "2021-05-26", "2021-05-27"
    ]
    #pred_dates = [
    #    "2021-05-01", "2021-05-02", "2021-05-03", "2021-05-04", 
    #    "2021-05-05", "2021-05-06", "2021-05-07", "2021-05-08"
    #]
    training_pred_dates = [
        "2021-05-14", "2021-05-15", "2021-05-16", "2021-05-17", 
        "2021-05-18", "2021-05-19", "2021-05-20"
    ]
    n_rnn_layers = 2

    #y_var = "D_hvac" # need to include the correlated targets as well (DONE) 
    y_var = ["D_hvac", "T_in_3", "T_in_5", "T_in_6", "T_in_7"] # TODO: Try adding zone 7 and D_hvac to the model targets. (DONE) 
    # TODO: Try a RandomForestRegressor (or other future variates model), maybe will overfit less than RNN 
    X_vars = [
        "T_out", "t_hr", "t_dow", #"t_isweekday", #"G_solar", #"T_hd65", "T_cd65", 
        "T_csp_out_diff_3", "T_hsp_out_diff_3", "T_csp_3", "T_hsp_3", 
        "T_csp_out_diff_5", "T_hsp_out_diff_5", "T_csp_5", "T_hsp_5", #TODO: Figure out how to account for the non-SP conforming behavior (DONE) 
        "T_csp_out_diff_7", "T_hsp_out_diff_7", "T_csp_7", "T_hsp_7", 
        "T_csp_out_diff_6", "T_hsp_out_diff_6", "T_csp_6", "T_hsp_6" # no setpoint changes occur in 2021
    ] 
    # TODO: Write code for counterfactual thermostat schedules that imports a csv table
    # that maps ZONE, HOUR, DAYOFWEEK, to a setpoint that will override the historical 
    # default value 

class GLOBS_V2():
    filepath_in = "./data/df_historical_20210101-20210601.csv"
    filepath_tstat_alt = "./data/test_data/darts_counterfactual_tstat.csv" 
    # TODO: Add multiple sample series inputs to model fitting (non-contiguous time series)  
    start_date = "2021-04-20" # TODO: Add Winter Months to training data (or warmer summer...?) (DONE) 
    end_date = "2021-06-01" 
    train_val_split_date = "2021-05-20" 
    #n_val_samples = 480 # Deprecated 
    train_frac = 0.8
    n_epochs = 50
    forecast_horizon = 24*4
    pred_dates = [
        "2021-05-21", "2021-05-22", "2021-05-23", "2021-05-24", 
        "2021-05-25", "2021-05-26", "2021-05-27"
    ]
    #pred_dates = [
    #    "2021-05-01", "2021-05-02", "2021-05-03", "2021-05-04", 
    #    "2021-05-05", "2021-05-06", "2021-05-07", "2021-05-08"
    #]
    training_pred_dates = [
        "2021-05-14", "2021-05-15", "2021-05-16", "2021-05-17", 
        "2021-05-18", "2021-05-19", "2021-05-20"
    ]
    n_rnn_layers = 2

    #y_var = "D_hvac" # need to include the correlated targets as well (DONE) 
    y_var = ["D_hvac", "T_in_3", "T_in_5", "T_in_6", "T_in_7"] # TODO: Try adding zone 7 and D_hvac to the model targets. (DONE) 
    # TODO: Try a RandomForestRegressor (or other future variates model), maybe will overfit less than RNN 
    X_vars = [
        "T_out", "t_hr", "t_dow", #"t_isweekday", #"G_solar", #"T_hd65", "T_cd65", 
        "T_csp_out_diff_3", "T_hsp_out_diff_3", "T_csp_3", "T_hsp_3", 
        "T_csp_out_diff_5", "T_hsp_out_diff_5", "T_csp_5", "T_hsp_5", #TODO: Figure out how to account for the non-SP conforming behavior (DONE) 
        "T_csp_out_diff_7", "T_hsp_out_diff_7", "T_csp_7", "T_hsp_7", 
        "T_csp_out_diff_6", "T_hsp_out_diff_6", "T_csp_6", "T_hsp_6" # no setpoint changes occur in 2021
    ] 
    # TODO: Write code for counterfactual thermostat schedules that imports a csv table
    # that maps ZONE, HOUR, DAYOFWEEK, to a setpoint that will override the historical 
    # default value

    price_offpeak = 0.14181 # $/kWh 
    price_onpeak = 0.22501 # $/kWh  
    price_partpeak = 0.16988 #$kwh
    price_export = 0.18 # $/kwh     
    price_peak = 19.85 # $/kW*month (demand charge)
    customer_charge = 4.59959 # charge / day according to bill
    price_arr = []
    price_arr.append(np.ones(34) * price_offpeak )  # 12:00 Am to 8:30 Am
    price_arr.append(np.ones(14) * price_partpeak ) # 8:30 AM to 12:00 PM
    price_arr.append(np.ones(24) * price_onpeak )   # 12:00 PM to 6:00 PM
    price_arr.append(np.ones(14) * price_partpeak ) # 6:00 PM to 9:30 PM
    price_arr.append(np.ones(10) * price_offpeak )  # 9:30 PM to 12:00 AM
    price_tou = dict(enumerate(price_arr))
 
    price_arr = []
    price_arr.extend( (np.ones(96) * price_offpeak).tolist() )  # 12:00 Am to 12:00 Am next day
    price_tou_hday = dict(enumerate(price_arr))
   
 

class Forecaster():
    def __init__(self, y_var, X_vars):
        self.y_var = y_var 
        self.X_vars = X_vars
        self.n_thermal_zones = 11 

    def import_data(self, filepath):
        self.df_raw = pd.read_csv(filepath) 
        self.df_raw["Timestamp"] = pd.to_datetime(self.df_raw["Timestamp"]) 
        #self.df_raw["T_in_lag1_diff_3"] = s

    def import_trained_model(self, filepath): 
        """
        Import trained model from prior torch_model_run *.pth.tar
        """
        if torch.cuda.is_available():
            self.model = torch.load(filepath)
        else:
            self.model = torch.load(filepath, map_location=torch.device("cpu")) 
        #self.model = torch.load(filepath) 

    def prep_features(self, df): 
        #self.df_raw["T_diff_3"] = self.df_raw["T_out"] - self.df_raw["T_csp_3"]
        #df = self.df_raw.copy() 
                # update_indoor_temp features: 
        df["t_isweekday"] = df["Timestamp"].dt.dayofweek < 5 
        df["t_isweekday"] = df["t_isweekday"].map({True: 1, False: 0})

        # Add a non-flexible load feature for simulation 
        df["D_nonflex"] = df["D_mains"] - df["G_solar"] - df["D_hvac"] - df["D_battery_eg"]  

        for zone in range(1,self.n_thermal_zones): 
            # Outdoor-SP temp diff (this is not as good as indoor-sp diff, but this feature is not cooupled to the target or the decision variable) 
            df["T_csp_out_diff_{}".format(zone)] = (df["T_out"] - df["T_csp_{}".format(zone)]).clip(lower=0)
            df["T_hsp_out_diff_{}".format(zone)] = (df["T_hsp_{}".format(zone)] - df["T_out"]).clip(lower=0) 

            # Indoor-SP temp diff 
            df["T_csp_in_diff_{}".format(zone)] = (df["T_in_{}".format(zone)] - df["T_csp_{}".format(zone)]).clip(lower=0)
            df["T_hsp_in_diff_{}".format(zone)] = (df["T_hsp_{}".format(zone)] - df["T_in_{}".format(zone)]).clip(lower=0)

            # CD65 & HD65 
            df["T_cd65".format(zone)] = (df["T_out"] - 65).clip(lower=0) 
            df["T_hd65".format(zone)] = (65 - df["T_out"]).clip(lower=0) 

            # Targets: 
            df["T_in_lag1_diff_{}".format(zone)] = df["T_in_{}".format(zone)].diff() 

            # One more feature... 
            df["T_in_lag1_diff_lag1_{}".format(zone)] = df["T_in_lag1_diff_{}".format(zone)].shift() 
        #self.df_raw = df.copy() 
        return df 

    def prep_data(self, idx_col="Timestamp", start_date="2000-01-01", end_date="2050-01-01", t_res="15min"):
        """
        Remove non-model variable columns
        """
        # Calculate the "fudged" setpoints to make training data more robust 
        df = self.df_raw.copy() 
        # If T_csp_* - T_in > 4 --> T_csp_* += uniform(-2,2) 
        # If T_in - T_hsp_* > 4 --> T_hsp_* += uniform(-2,2) 
        for z in range(1,self.n_thermal_zones):
            df["T_csp_{}".format(z)] += (df["T_csp_{}".format(z)]-df["T_in_{}".format(z)] >= 1)*np.random.uniform(-0.5, 0.5, df.shape[0]) 
            df["T_hsp_{}".format(z)] += (df["T_in_{}".format(z)]-df["T_hsp_{}".format(z)] >= 1)*np.random.uniform(-0.5, 0.5, df.shape[0]) 
            df["T_csp_{}".format(z)] += (df["T_csp_{}".format(z)]-df["T_in_{}".format(z)] >= 3)*np.random.uniform(-2, 1, df.shape[0]) 
            df["T_hsp_{}".format(z)] += (df["T_in_{}".format(z)]-df["T_hsp_{}".format(z)] >= 3)*np.random.uniform(-1, 2, df.shape[0]) 
            df["T_csp_{}".format(z)] += (df["T_csp_{}".format(z)]-df["T_in_{}".format(z)] >= 7)*np.random.uniform(-5, 4, df.shape[0]) 
            df["T_hsp_{}".format(z)] += (df["T_in_{}".format(z)]-df["T_hsp_{}".format(z)] >= 7)*np.random.uniform(-4, 5, df.shape[0]) 
            df["T_csp_{}".format(z)] += (df["T_csp_{}".format(z)]-df["T_in_{}".format(z)] >= 10)*np.random.uniform(-7, 6, df.shape[0])  
            

        if isinstance(self.y_var, list):
            cols = [idx_col] + self.y_var + self.X_vars 
        else: 
            cols = [idx_col, self.y_var] + self.X_vars 
        #self.df_in = self.df_raw[["Timestamp", "D_mains", "T_out"]]
        self.df_in = df[cols].copy() 
        self.df_in = self.df_in[self.df_in.Timestamp >= start_date] 
        self.df_in = self.df_in[self.df_in.Timestamp < end_date] 
        self.df_in = self.df_in.set_index("Timestamp").resample(t_res).mean()
        #self.df_in = self.df_in.interpolate(method="spline", order=3) 
        self.df_in = self.df_in.interpolate(method="linear").reset_index()  
        #self.df_in = self.df_in[self.df_in.Timestamp.dt.dayofweek < 5] # Forecast does not work 

    def prep_altered_data(self, file_tstat_alt=None): 
        df = self.df_raw.copy() 

        # Import Altered Setpoints and Overwrite from File 
        if file_tstat_alt is not None: 
            setpoints_alt = pd.read_csv(file_tstat_alt)\
                .drop_duplicates(subset=["zone", "t_hr", "t_dow"])\
                .pivot(index=["t_hr", "t_dow"], columns="zone", values=["T_hsp", "T_csp"])\
                .T.reset_index() 
            setpoints_alt["col"] = setpoints_alt["level_0"].astype(str) + "_" + setpoints_alt["zone"].astype(str) 
            setpoints_alt = setpoints_alt\
                .set_index("col")\
                .drop(["level_0", "zone"], axis=1).T\
                .reset_index()\
                .set_index(["t_hr", "t_dow"]) 

            df = df.set_index(["t_hr", "t_dow"])
            df.update(setpoints_alt)
            df = df.reset_index()  
        # More hack-y way of doing it...should deprecate this after testing import+overwrite method 
        else:
            setpoints_alt_diff = {
                "T_csp_1": 0, 
                "T_csp_2": 0, 
                "T_csp_3": +2, 
                "T_csp_4": 0, 
                "T_csp_5": 0, 
                "T_csp_6": 0, 
                "T_csp_7": -2, 
                "T_csp_8": 0, 
                "T_csp_9": 0, 
                "T_csp_10": 0, 
                "T_hsp_1": 0, 
                "T_hsp_2": 0, 
                "T_hsp_3": +2, 
                "T_hsp_4": 0, 
                "T_hsp_5": 0, 
                "T_hsp_6": 0, 
                "T_hsp_7": -3, 
                "T_hsp_8": 0, 
                "T_hsp_9": 0, 
                "T_hsp_10": 0,             
            }
            altered_hours = [9,10,11,12,13,14] 
            df["mask"] = df["t_hr"].isin(altered_hours).map({True:1, False:0})
            for col in setpoints_alt_diff.keys():
                df[col] = df[col] + setpoints_alt_diff[col]*df["mask"] 

        self.df_altered = self.prep_features(df) 

        self.X_altered = TimeSeries.from_dataframe(self.df_altered.reset_index(), "Timestamp", self.X_vars) 
        self.X_altered_scaled = self.scaler_X.transform(self.X_altered)

    def init_model(self): 
        '''
        # for 15 min resolution 
        self.model = NBEATSModel(
            input_chunk_length = 48*4, 
            output_chunk_length = 24*4, 
            n_epochs = 50, 
            random_state = 0 
        )
        '''
        # 1-hr model 
        self.model = NBEATSModel(
            input_chunk_length = 48, 
            output_chunk_length = 24, 
            n_epochs = 5, 
            random_state = 0 
        )

    def init_rnn_model(self, n_rnn_layers=2):
        self.model = RNNModel(
            model="RNN", 
            input_chunk_length = 24*1, 
            training_length = 24*2,
            save_checkpoints = True,
            n_rnn_layers = n_rnn_layers
        )

    def init_blockrnn_model(self, n_epochs=50):
        self.model = BlockRNNModel(
            model = "LSTM", 
            input_chunk_length = 24*4, 
            output_chunk_length = 24*4, 
            n_epochs = n_epochs, 
            random_state = 0 
        )

    def fit_rnn_model(self, epochs):
        self.model.fit(
            self.train_y, 
            future_covariates = self.train_X, # Set as bast or future 
            epochs = epochs, 
            verbose = True 
        )   

    def fit_blockrnn_model(self):
        self.model.fit(
            self.train_y, 
            past_covariates=self.train_X, 
            verbose = True 
        )
    
    def init_regr_model(self):
        self.model = RegressionModel(
            lags=None, 
            lags_past_covariates=[-24,-12,-2,-1], 
            lags_future_covariates=[-24,-12,-2,-1,0], 
        )

    def fit_regr_model(self):
        self.model.fit(
            self.train_y, 
            past_covariates = self.train_X["T_out"], 
            future_covariates = self.train_X["T_csp_3"]
        )

    def split_scale_data(self, split_date): 
        self.y = TimeSeries.from_dataframe(self.df_in, "Timestamp", self.y_var)
        self.X = TimeSeries.from_dataframe(self.df_in, "Timestamp", self.X_vars) 

        # Add noise 
        x_shape = self.X.pd_dataframe().shape
        y_shape = self.y.pd_dataframe().shape
        x_noise = np.random.normal(0, self.X.pd_dataframe().std()*0.09, (x_shape[0], x_shape[1])) 
        y_noise = np.random.normal(0, self.y.pd_dataframe().std()*0.09, (y_shape[0], y_shape[1])) 
        x_noise = TimeSeries.from_dataframe(pd.DataFrame(x_noise, index=self.X.pd_dataframe().index, columns=self.X.pd_dataframe().columns))
        y_noise = TimeSeries.from_dataframe(pd.DataFrame(y_noise, index=self.y.pd_dataframe().index, columns=self.y.pd_dataframe().columns))
        self.X_wNoise = self.X + x_noise
        self.y_wNoise = self.y + y_noise 

        self.scaler_y = Scaler().fit(self.y_wNoise) 
        self.scaler_X = Scaler().fit(self.X_wNoise) 
        self.y_scaled = self.scaler_y.transform(self.y_wNoise) 
        self.X_scaled = self.scaler_X.transform(self.X_wNoise) 

        self.train_y = self.y_scaled.drop_after(pd.to_datetime(split_date))#[:-n_val_samples]
        self.val_y = self.y_scaled.drop_before(pd.to_datetime(split_date))#[-n_val_samples:]
        self.train_X = self.X_scaled.drop_after(pd.to_datetime(split_date))#[:-n_val_samples]
        self.val_X = self.X_scaled.drop_before(pd.to_datetime(split_date))#[-n_val_samples:] 

    def eval_model(self, past_covariates=None, future_covariates=None, forecast_horizon=10, start=GLOBS.train_frac, test_name=None):
        if test_name is None: 
            test_name= "backtest (n={})".format(forecast_horizon)
        else: 
            test_name = test_name 
        # Backtest the model on the last 20% of the self.train_y series, with a horizon of 24 steps 
        self.backtest = self.model.historical_forecasts(
            series = self.y_scaled, 
            past_covariates=past_covariates, 
            future_covariates=future_covariates, 
            start=start, 
            retrain=False, 
            verbose=True, 
            forecast_horizon=forecast_horizon
        )
        #self.y[-len(self.backtest)-150:].plot() 
        self.y.plot(alpha=0.8) 
        b_inv = self.scaler_y.inverse_transform(self.backtest) 
        b_inv.plot(
            label = test_name, #label='backtest (n={})'.format(forecast_horizon), 
            alpha = 0.8
        ) 
        self.df_raw.set_index("Timestamp")[["T_out", "T_in_3", "T_csp_3", "T_hsp_3"]].plot()

        

        plt.legend()
        plt.show() 

    def eval_model_altered(self, past_covariates=None, past_covariates_altered=None, future_covariates=None, future_covariates_altered=None, forecast_horizon=10, start=GLOBS.train_frac):
        # Backtest the model on the last 20% of the self.train_y series, with a horizon of 24 steps 
        self.backtest = self.model.historical_forecasts(
            series = self.y_scaled, 
            past_covariates=past_covariates, 
            future_covariates=future_covariates, 
            start=start, 
            retrain=False, 
            verbose=True, 
            forecast_horizon=forecast_horizon
        )
        #self.y[-len(self.backtest)-150:].plot() 
        #self.y.plot(alpha=0.8) 
        b_inv = self.scaler_y.inverse_transform(self.backtest) 
        #b_inv.plot(
        #    label='backtest (n={})'.format(forecast_horizon), 
        #    alpha = 0.8
        #) 

        # Run backtest on altered data 
        if past_covariates_altered is not None: 
            self.backtest_altered = self.model.historical_forecasts(
                series = self.y_scaled, 
                past_covariates = past_covariates_altered, 
                future_covariates = future_covariates_altered, 
                start = start, 
                retrain = False, 
                verbose = True, 
                forecast_horizon = forecast_horizon
            )
        elif future_covariates_altered is not None: 
            self.backtest_altered = self.model.historical_forecasts(
                series = self.y_scaled, 
                past_covariates = past_covariates_altered, 
                future_covariates = future_covariates_altered,
                start = start, 
                retrain = False, 
                verbose = True, 
                forecast_horizon = forecast_horizon
            )
            b_inv_altered = self.scaler_y.inverse_transform(self.backtest_altered)
            #b_inv_altered.plot(
            #    label = "altered X", 
            #    alpha = 0.8 
            #)

        # Plot temperature data 
        y = self.y.pd_dataframe()
        
        y_pred = b_inv.pd_dataframe() # .rename(columns={0:"b_inv"}) 
        y_pred_altered = b_inv_altered.pd_dataframe() # .rename(columns={0:"counterfactual preds"}) 

        y.columns = self.y_var
        y_pred.columns = self.y_var 
        y_pred_altered.columns = self.y_var 

        
        if isinstance(self.y_var, list):
            row_idx = 1 
            traces = [] 
            trace_rows = [] 
            trace_cols = [] 
            df_raw_15min = self.df_raw.set_index("Timestamp").resample("15min").mean() 
            for col in self.y_var:

                trace_y = go.Scatter(x=pd.to_datetime(y.index), y=y[col], name=col, opacity=0.8)
                trace_y_pred = go.Scatter(x=pd.to_datetime(y_pred.index), y=y_pred[col], name=col+"_pred", opacity=0.8) 
                trace_y_altered = go.Scatter(x=pd.to_datetime(y_pred_altered.index), y=y_pred_altered[col], name=col+"_counterfactual", opacity=0.8) 

                traces.append(trace_y) 
                traces.append(trace_y_pred) 
                traces.append(trace_y_altered) 
                trace_rows += [row_idx, row_idx, row_idx] 
                trace_cols += [1,1,1] 

                if "T_in_" in col: 
                    # TODO: Need to only include the CSP/HSP traces if the target variable is an indoor temperature 
                    zone_idx = col.split("_")[-1] 
                    hsp_col = "T_hsp_{}".format(zone_idx) 
                    csp_col = "T_csp_{}".format(zone_idx) 
                    trace_csp = go.Scatter(x=pd.to_datetime(df_raw_15min.index), y=df_raw_15min[csp_col], name=csp_col, opacity=0.8) 
                    trace_hsp = go.Scatter(x=pd.to_datetime(df_raw_15min.index), y=df_raw_15min[hsp_col], name=hsp_col, opacity=0.8) 
                    trace_csp_alt = go.Scatter(x=pd.to_datetime(self.df_altered["Timestamp"]), y=self.df_altered[csp_col], name=csp_col+"_counterfactual", opacity=0.6) 
                    trace_hsp_alt = go.Scatter(x=pd.to_datetime(self.df_altered["Timestamp"]), y=self.df_altered[hsp_col], name=hsp_col+"_counterfactual", opacity=0.6) 
                    traces.append(trace_csp) 
                    traces.append(trace_hsp)
                    traces.append(trace_csp_alt) 
                    traces.append(trace_hsp_alt) 
                    trace_rows += [row_idx, row_idx, row_idx, row_idx]
                    trace_cols += [1,1,1,1] 

                row_idx += 1 

        else:
            row_idx = 1 
            trace_y = go.Scatter(x=pd.to_datetime(y.index), y=y, name=self.y_var, opacity=0.8)
            trace_y_pred = go.Scatter(x = pd.to_datetime(y_pred.index), y=y_pred, name=self.y_var+"_pred", opacity=0.8) 
            trace_y_pred_altered = go.Scatter(x=pd.to_datetime(y_pred_altered.index), y=y_pred_altered, name=self.y_var+"_counterfactual", opacity=0.8)
        

            trace_rows = [row_idx, row_idx, row_idx]
            trace_cols = [1,1,1] 
            traces = [trace_y, trace_y_pred, trace_y_pred_altered] 

        
        #for col in ["T_out", "T_csp_3", "T_hsp_3", "T_in_3"]: 
        for col in ["T_out"]: 
            trace = go.Scatter(
                x = pd.to_datetime(df_raw_15min.index), 
                y = df_raw_15min[col], 
                name = col, 
                opacity = 0.8 
            )
            traces.append(trace) 
            trace_rows.append(row_idx) 
            trace_cols.append(1) 
        
        fig = make_subplots(rows=row_idx, shared_xaxes=True) 
        fig.add_traces(traces, rows=trace_rows, cols=trace_cols) 
        pyo.plot(fig) 


        #self.df_raw.set_index("Timestamp")[["T_out", "T_csp_3", "T_hsp_3", "T_out"]] 
        #plt.legend()
        #plt.show() 

    def get_episodic_preds(self, pred_dates, past_covariates=None, past_covariates_altered=None, future_covariates=None, future_covariates_altered=None, forecast_horizon=10, start=GLOBS.train_frac): 
        """
        Run multiple 24-hr predictions 
        Args: 
            pred_dates: list of dates to run a 24-hr forecast on 
            forecast_horizon = 24*4 (15-min intervals) 
        """
        self.y_ep_pred = pd.DataFrame() 
        self.y_ep_pred_alt = pd.DataFrame() 
        for pred_date in pred_dates: 
            print(pred_date)
            y_pred = self.model.predict(
                n = 24*4, # Predict a full 24 hrs 
                series = self.y_scaled.drop_after(pd.to_datetime(pred_date)), 
                future_covariates=future_covariates, 
                past_covariates=past_covariates
            )
            y_pred_alt = self.model.predict(
                n = 24*4, 
                series = self.y_scaled.drop_after(pd.to_datetime(pred_date)), 
                future_covariates=future_covariates_altered, 
                past_covariates = past_covariates_altered
            )

            y_pred = self.scaler_y.inverse_transform(y_pred) 
            y_pred_alt = self.scaler_y.inverse_transform(y_pred_alt) 

            self.y_ep_pred = self.y_ep_pred.append(y_pred.pd_dataframe())
            self.y_ep_pred_alt = self.y_ep_pred_alt.append(y_pred_alt.pd_dataframe()) 

    def prep_sim_df(self, df_hist, df_hvac, t_res): 
        """
        Merges the historical and simulated HVAC operation into a single dataframe 
        :param df_hist: dataframe of historical data needed for the simulation
        :param df_hvac: dataframe of simulated hvac operation
        :param t_res: time resolution of the simulation. 
        :return df_sim: the merged dataframe, with only the inner join of simulated timestamps 
        """
        df_hist = df_hist.resample(t_res).mean().interpolate(method="linear") 
        df_hvac = df_hvac.resample(t_res).mean().interpolate(method="linear")  
        df_sim = pd.concat([df_hist, df_hvac], axis=1, join="inner") 
        return df_sim 

    def simulate_batt(self, df_sim_in, x_d_max, x_soc_min, batt_cap_kwh, battery_soc_max=100, t_res_minutes=15, chg_kw_max=29, dchg_kw_max=-22): 
        """
        Simulates battery operation 
        :param df_sim: merged dataframe with only the inner join of simulated timestamp 
        :param x_d_max: net load demand limite that battery discharge profile maintains with available capacity 
        :param x_soc_min: minimum SOC of battery reserved for backup and other services 
        :param batt_cap_kwh: total kwh of capacity 

        dchg_kw_max = 7 # kW 
        chg_kw_max = 7 # kW 
        # Assumes SOC starts between 0-100% 
        available_dchg_kwh = SOC * batt_cap_kwh / 100    # kWh available for discharging (til 0%) 
        available_chg_kwh = batt_cap_kwh - SOC_kwh       # kWh available for charging (til 100%)

        # Convert to kW 
        available_dchg_kw = available_dchg_kwh * 60 / t_res_minutes 
        available_chg_kw = available_chg_kwh * 60 / t_res_minutes 

        # Constrained by max chg/dchg rates of battery 
        available_dchg_kw = - min(available_dchg_kw, dchg_kw_max) # Adds negative sign for chg/dchg polarity 
        available_chg_kw = min(available_chg_kw, chg_kw_max) 
        
        # Battery kW needed 
        chg_needed_kw = -G_solar - D_nonflex - D_hvac - x_d_max

        # Contrained battery kW dispatched 
        chg_dispatched_kw = max(min(chg_needed_kw, available_chg_kw), available_dchg_kw) 
        D_batt = chg_dispatched_kw 

        # Update SOC with dispatched battery chg/dchg rate (kW) 
        SOC += (D_batt * t_res_minutes/60) / battery_capacity * 100 
        """
        #t_res_minutes = 15 # TODO: moving to func args...time resolution of simulation in minutes 
        #chg_kw_max = 7 # TODO: moving to func args...max charge (kW) 
        #dchg_kw_max = -22 # TODO: moving to func args...max discharge (kW) 
        # cap_kwh_max = 20 # TODO: moving to func args...Max batter kWh, appears not to be used 
        #df_sim["X_soc_min"] = x_soc_min 
        df_sim = df_sim_in.copy() 
        df_sim["soc"] =np.nan
        #df_sim.iloc[0,-1] = 5 
        df_sim["D_batt"] = np.nan
        #df_sim.iloc[0,-1] = 0 
        

        # Intial conditions for SOC and battery kW 
        soc_next = x_soc_min
        #d_batt_next = 0 
        # Step through storage operation 
        for i in df_sim.index:
            df_sim.loc[i,"soc"] = soc_next
            #df_sim.loc[i,"D_batt"] = d_batt_next
            row = df_sim.loc[i,:].copy() 
            soc_next, d_batt = update_battery(
                row["soc"], row["G_solar"], row["D_nonflex"], row["D_hvac"], 
                x_d_max, x_soc_min, battery_soc_max, 
                batt_cap_kwh, t_res_minutes, 
                chg_kw_max, dchg_kw_max
            )
            df_sim.loc[i,"D_batt"] = d_batt
        df_sim["D_net_mains"] = df_sim[["G_solar", "D_hvac", "D_nonflex", "D_batt"]].sum(axis=1) 
        df_sim["D_tot_mains"] = df_sim[["D_hvac", "D_nonflex"]].sum(axis=1) 
        df_sim["soc_kwh"] = df_sim["soc"] * batt_cap_kwh / 100 
        return df_sim


    def simulate_compute_batt_soc(self, current_soc, kw_15mins):
        # kw_15mins could be a float or an array of flaots
        #based on batt data for month of May 2021, linear fit using excel
        # gives the equation for delta_soc = 1.98 * energy_kw + 1.12
        tmp_kw_arr = []
        updated_soc = current_soc
        if type(kw_15mins) == float:
            tmp_kw_arr.append(kw_15mins)
        elif type(kw_15mins) == list:
            tmp_kw_arr.append(kw_15mins)
        else: 
            print ("expecting float or list of floats, got")
            print (type(kw_15mins))

        if len(tmp_kw_arr) > 0:
            for x in tmp_kw_arr:
                delta_soc = 1.98 * x + 1.12
                updated_soc = updated_soc + delta_soc
        
        return updated_soc
            
        
    def process_cost(self, df_sim_in, import_key = "bill_import", export_key = "bill_export"):
        df_sim = df_sim_in.copy()
        
        # df_sim_cost = pd.DataFrame()
        #split the data rows into rows for each day
        
        next_day_index = 0
        day_import = []
        day_export = []
        date_data = []
        date_str = []
        weekday_name = []
        while next_day_index < df_sim.shape[0]:
            current_sim_day = df_sim.iloc[next_day_index]["Timestamp"].date()
            next_sim_day = current_sim_day + timedelta(days = 1)
            # extracting all rows for the current_sim_day
            rows_for_day = df_sim[ ( df_sim["Timestamp"] >= str(current_sim_day) ) & (df_sim["Timestamp"] < str(next_sim_day) )] 
            next_day_index = next_day_index + rows_for_day.index[-1] + 1
            if rows_for_day.size > 0:
                #date_data.append(rows_for_day.iloc[0]['Timestamp'].to_pydatetime())
                date_data.append(current_sim_day)
                date_str.append(current_sim_day.strftime("%m/%d/%Y"))
                weekday_name.append(calendar.day_name[current_sim_day.weekday()])
                day_import.append( rows_for_day[import_key].sum() ) 
                day_export.append( rows_for_day[export_key].sum() ) 
        df_cost = pd.DataFrame({
                                "Date": date_str,
                                "Weekday" : weekday_name,
                                "import_cost": day_import,
                                "export_cost": day_export
                             })
        return df_cost
        
    
    
    def prepare_baseline_qhourly(self, df_sim_in, cols = ["G_solar", "D_hvac", "D_battery_eg", "D_nonflex"], t_res_minutes = 15,
                                price_tou_dict=GLOBS_V2.price_tou, price_export = 0.18, price_peak = 19.85,
                                price_hday = GLOBS_V2.price_tou_hday):
        
        df_sim = df_sim_in.copy()

        df_sim_comp = pd.DataFrame()
        #split the data rows into rows for each day
        next_day_index = 0
        while next_day_index < df_sim.shape[0]:
            current_sim_day = df_sim.iloc[next_day_index]["Timestamp"].date()
            next_sim_day = current_sim_day + timedelta(days = 1)
            # extracting all rows for the current_sim_day
            rows_for_day = df_sim[ ( df_sim["Timestamp"] >= str(current_sim_day) ) & (df_sim["Timestamp"] < str(next_sim_day) )] 
            next_day_index = rows_for_day.index[-1] + 1
            if rows_for_day.size > 0:
                tmp_arr = ['Timestamp'] + cols
                tmp_rows_day = rows_for_day[tmp_arr].copy()
                tmp_rows_day['consumption'] = tmp_rows_day[['D_hvac', 'D_nonflex']].sum(axis=1)
                tmp_rows_day['consumption_kwh'] = tmp_rows_day['consumption']*(t_res_minutes/60)
                tmp_rows_day['Hour'] = tmp_rows_day['Timestamp'].dt.hour
                tmp_rows_day['Minute'] = tmp_rows_day['Timestamp'].dt.minute
                tmp_rows_day = tmp_rows_day.reset_index()
                
                # tmp_dict = {}
                # for x in cols:
                #     tmp_dict[x] = 'sum'
                # tmp_dict['consumption'] = 'sum'
                # tmp_dict['consumption_kwh'] = 'sum'
                # hourly_consumption = tmp_rows_day.resample('H', on='Timestamp').agg(tmp_dict)
                # hourly_consumption['price'] = price_tou_dict.values()
                # hourly_consumption = hourly_consumption.reset_index()
                wday_int = tmp_rows_day.loc[0, 'Timestamp'].to_pydatetime().weekday()
                if wday_int >= 0 and wday_int < 5:
                    tmp_rows_day['price'] = price_tou_dict.values()                
                else:
                    tmp_rows_day['price'] = price_hday.values()
                
                tmp_rows_day['D_net_mains_b'] = tmp_rows_day[cols].sum(axis=1)
                tmp_rows_day['D_tot_mains_b'] = tmp_rows_day[["D_hvac", "D_nonflex" ]].sum(axis=1)
                print("max = ", tmp_rows_day.D_net_mains_b.max(), "min = ", tmp_rows_day.D_net_mains_b.min())
                
                #temporarily define parameeters
                net_kw_col = "D_net_mains_b"
                #price_export = 0.18 # $/kwh     
                #price_peak = 19.5 # $/kW*month 
                tmp_rows_day.loc[tmp_rows_day[net_kw_col] >= 0, "kw_import"] = tmp_rows_day.loc[tmp_rows_day[net_kw_col] >= 0, net_kw_col]
                tmp_rows_day.loc[tmp_rows_day[net_kw_col] <= 0, "kw_export"] = tmp_rows_day.loc[tmp_rows_day[net_kw_col] <= 0, net_kw_col] 
                tmp_rows_day["kwh_import"] = tmp_rows_day["kw_import"].fillna(0) * t_res_minutes/60
                tmp_rows_day["kwh_export"] = tmp_rows_day["kw_export"].fillna(0) * t_res_minutes/60 
                tmp_rows_day["bill_import"] = tmp_rows_day["kwh_import"] * tmp_rows_day["price"] 
                tmp_rows_day["bill_export"] = tmp_rows_day["kwh_export"] * price_export
                tmp_peak_idx = tmp_rows_day[["kw_import"]].idxmax() 
                tmp_rows_day['kw_peak'] = 0
                tmp_rows_day.at[tmp_peak_idx, 'kw_peak'] = tmp_rows_day.iloc[tmp_peak_idx]["kw_import"]
                
                df_sim_comp = pd.concat([df_sim_comp, tmp_rows_day])
                print('adjusted hourly for ',  current_sim_day)
                
        return df_sim_comp
        
        

    def simulate_batt_sch_with_cost(self, df_sim_in, x_d_max, x_soc_min, batt_cap_kwh, battery_soc_max=100, 
                                    t_res_minutes=15, chg_kw_max=29, dchg_kw_max=-22, 
                                    price_tou_dict=GLOBS_V2.price_tou, price_export = 0.18,
                                    price_peak = 19.85,
                                    solar_threshold_kw_batt_charge = 2.25,
                                    price_hday = GLOBS_V2.price_tou_hday): 
        """
        charge: when power available > GLOBS_V2.start_charge_limit till it reaches max_soc
        discharge:
            identify max cost for the day = consumption * TOU cost for 1 hour intervals
            discharge battery to reduce the cost to the next tier, repeat till battery soc reaches min

        # Update SOC with dispatched battery chg/dchg rate (kW) 
        SOC += (D_batt * t_res_minutes/60) / battery_capacity * 100 
        """
        df_sim = df_sim_in.copy() 
        df_sim["D_batt"] = np.nan
        df_sim['batt_action'] = 'none'        

        df_sch = pd.DataFrame()

        # Intial conditions for SOC and battery kW 
        batt_soc = x_soc_min
        #d_batt_next = 0 

        #split the data rows into rows for each day
        next_day_index = 0
        while next_day_index < df_sim.shape[0]:
            current_sim_day = df_sim.iloc[next_day_index]["Timestamp"].date()
            next_sim_day = current_sim_day + timedelta(days = 1)
            # extracting all rows for the current_sim_day
            rows_for_day = df_sim[ ( df_sim["Timestamp"] >= str(current_sim_day) ) & (df_sim["Timestamp"] < str(next_sim_day) )] 
            next_day_index = rows_for_day.index[-1] + 1
            if rows_for_day.size > 0:
                tmp_rows_day = rows_for_day[['Timestamp', 'D_hvac', 'D_nonflex', "G_solar", 'batt_action', 'D_batt']].copy()
                tmp_rows_day['consumption'] = tmp_rows_day[['D_hvac', 'D_nonflex']].sum(axis=1)
                tmp_rows_day['consumption_kwh'] = tmp_rows_day['consumption']*(t_res_minutes/60)
                tmp_rows_day['Hour'] = tmp_rows_day['Timestamp'].dt.hour
                tmp_rows_day['Minute'] = tmp_rows_day['Timestamp'].dt.minute
                tmp_rows_day['soc'] = batt_soc
                tmp_rows_day.loc[ (-tmp_rows_day.G_solar - tmp_rows_day.consumption) > solar_threshold_kw_batt_charge, 'batt_action'] = 'charge'
                tmp_rows_day = tmp_rows_day.reset_index()
                
                for ii in tmp_rows_day.loc[tmp_rows_day['batt_action'] == 'charge'].index:
                    tmp_rate = -tmp_rows_day.iloc[ii]['G_solar'] - tmp_rows_day.iloc[ii]['consumption']
                    tmp_soc, tmp_d_batt = update_battery_v2( batt_soc, 
                        tmp_rate, x_soc_min, battery_soc_max, batt_cap_kwh, t_res_minutes, 
                        chg_kw_max, dchg_kw_max, 1
                    )
                    tmp_rows_day.at[ii, 'soc'] = tmp_soc
                    batt_soc = tmp_soc
                    tmp_rows_day.at[ii, 'D_batt'] = tmp_d_batt
                    if batt_soc >= battery_soc_max:
                        break
                wday_int = tmp_rows_day.loc[0, 'Timestamp'].to_pydatetime().weekday()
                if wday_int >= 0 and wday_int < 5:
                    tmp_rows_day['price'] = price_tou_dict.values()                
                else:
                    tmp_rows_day['price'] = price_hday.values()
                tmp_rows_day = tmp_rows_day.reset_index()
                
                tmp_rows_day['computed_cost'] = 0.0
                tmp_rows_day['discharge_kwh'] = 0.0
                tmp_batt_soc = batt_soc
                batt_kwh_available = ((tmp_batt_soc-x_soc_min)/100) * batt_cap_kwh
                while batt_kwh_available > 0:
                    cond_mask = (tmp_rows_day['batt_action'] != 'charge')
                    h_c_selec = tmp_rows_day[cond_mask]
                    tmp_rows_day.loc[cond_mask, 'computed_cost'] = (h_c_selec['consumption_kwh'] - h_c_selec['discharge_kwh']  ) * h_c_selec['price']
                    tmp_max_cost = (tmp_rows_day.sort_values('computed_cost', ascending=False))['computed_cost'].copy()
                    if max(tmp_max_cost) < 0.01:
                        break
                    arr_max_cost = []
                    arr_max_cost.append(tmp_max_cost.index[0])
                    jj = 1
                    while (jj < tmp_max_cost.shape[0] ) and (abs(tmp_max_cost.iloc[jj] -  tmp_max_cost.iloc[0] )  < 0.15) :
                        arr_max_cost.append(tmp_max_cost.index[jj])
                        jj = jj + 1
                    idx_2max_cost = tmp_max_cost.index[jj]
                    batt_kwh_req = 0
                    batt_kwh_req_arr = []
                    for kk in arr_max_cost:
                        tmp_dchg_kwh = (tmp_rows_day.iloc[kk]['computed_cost'] - tmp_rows_day.iloc[idx_2max_cost]['computed_cost']) / tmp_rows_day.iloc[kk]['price']
                        tmp_val =  tmp_rows_day.loc[kk]['discharge_kwh']
                        tmp_rows_day.at[kk,'discharge_kwh'] = tmp_val + tmp_dchg_kwh
                        batt_kwh_req = batt_kwh_req + tmp_dchg_kwh
                        batt_kwh_req_arr.append(tmp_dchg_kwh)
                        tmp_rows_day.at[kk,'batt_action'] = 'discharge'
                        tmp_rows_day.at[kk,'D_batt'] = -tmp_rows_day.iloc[kk]['discharge_kwh'] * 60 / t_res_minutes
                        #print(kk, batt_kwh_req, hourly_consumption.iloc[kk]['computed_cost'] - hourly_consumption.iloc[idx_2max_cost]['computed_cost'])
                    tmp_val = ( (dchg_kw_max * t_res_minutes)/60)
                    tmp_rate = min(batt_kwh_available, tmp_val)
                    if batt_kwh_req > tmp_rate:
                        tmp_sum = sum(batt_kwh_req_arr)
                        for tt in range(len(arr_max_cost)):
                            kk = arr_max_cost[tt]
                            tmp_val =  tmp_rows_day.loc[kk]['discharge_kwh'] - batt_kwh_req_arr[tt]
                            tmp_rows_day.at[kk,'discharge_kwh'] = tmp_val + (tmp_rate*batt_kwh_req_arr[tt] )/ tmp_sum
                            tmp_rows_day.at[kk,'D_batt'] = -tmp_rows_day.iloc[kk]['discharge_kwh'] * 60 / t_res_minutes
                        batt_kwh_req = tmp_rate
                        
                    batt_kwh_available = batt_kwh_available - batt_kwh_req

                # # compute soc for discharge sets
                # for ii in hourly_consumption.loc[hourly_consumption['discharge_kwh'] > 0 ].index:
                #     tmp_rate = hourly_consumption.loc[ii]['discharge_kwh'] * t_res_minutes / 60 # hourly obtained from combining four 1`5 mins intervvals
                #     tmp_soc, tmp_d_batt = update_battery_v2( batt_soc, 
                #         tmp_rate, x_soc_min, battery_soc_max, batt_cap_kwh, 60, 
                #         chg_kw_max, dchg_kw_max, -1
                #     )
                #     hourly_consumption.at[ii, 'soc'] = tmp_soc
                #     batt_soc = tmp_soc
                #     hourly_consumption.at[ii, 'D_batt'] = tmp_d_batt
                #     if batt_soc <= x_soc_min:
                #         break
                #if batt_kwh_available > 0, then batt_soc would be > x_soc_min
                batt_soc = x_soc_min # used up all kwh in battery to support high cost regions
                
                tmp_rows_day['D_batt'] = tmp_rows_day['D_batt'].fillna(0)
                tmp_rows_day['D_net_mains'] = tmp_rows_day[["G_solar", "D_hvac", "D_nonflex", "D_batt"]].sum(axis=1)
                tmp_rows_day['D_tot_mains'] = tmp_rows_day[["D_hvac", "D_nonflex" ]].sum(axis=1)
                print("max = ", tmp_rows_day.D_net_mains.max(), "min = ", tmp_rows_day.D_net_mains.min())

                #temporarily define parameeters
                net_kw_col = "D_net_mains"
                #price_export = 0.18 # $/kwh     
                #price_peak = 19.5 # $/kW*month 
                tmp_rows_day.loc[tmp_rows_day[net_kw_col] >= 0, "kw_import"] = tmp_rows_day.loc[tmp_rows_day[net_kw_col] >= 0, net_kw_col]
                tmp_rows_day.loc[tmp_rows_day[net_kw_col] < 0, "kw_export"] = tmp_rows_day.loc[tmp_rows_day[net_kw_col] <= 0, net_kw_col] 
                tmp_rows_day["kwh_import"] = tmp_rows_day["kw_import"].fillna(0) * t_res_minutes/60
                tmp_rows_day["kwh_export"] = tmp_rows_day["kw_export"].fillna(0) * t_res_minutes/60 
                tmp_rows_day["bill_import"] = tmp_rows_day["kwh_import"] * tmp_rows_day["price"] 
                tmp_rows_day["bill_export"] = tmp_rows_day["kwh_export"] * price_export
                tmp_peak_idx = tmp_rows_day[["kw_import"]].idxmax() 
                tmp_rows_day['kw_peak'] = 0
                tmp_rows_day.at[tmp_peak_idx, 'kw_peak'] = tmp_rows_day.iloc[tmp_peak_idx]["kw_import"]
                
                df_sch = pd.concat([df_sch, tmp_rows_day])
                print('completed sim for the ',  current_sim_day)
                
        return df_sch




    def plot_battery_sim(self, df_sim, savepath=None, fig_title="", start="2000-01-01", end="2100-01-01"): 
        plot_cols = ["D_tot_mains", "G_solar", "D_net_mains", "D_batt", "soc_kwh"]
        df_plot = df_sim.loc[(df_sim.index>=start) & (df_sim.index <= end), plot_cols]
        df_plot.plot() 
        plt.title(fig_title) 
        #plt.show() 
        if savepath is not None: 
            plt.savefig(savepath, dpi=200) 
            plt.close() 
        


    def calc_bill(self, df_sim_in, net_kw_col="D_net_mains", t_res_minutes=15, peak_startup=24):
        """
        Calculates a TOU plus Demand Charge Electricity Bill given a kW time series input
        :param ts_kw: pandas Series with datetime index of average kW values 
            ts_kw must be a uniform timeseries with atleast 1-hour resolution. 
        :param price_tou: dict of $/kwh price of electricity mapped to each hour of the day 
        :param price_peak: $/kw*month price of peak electricity demand that is applied on a per month basis 
        :return bill: electricity bill normalized to $/month
        """
        # TODO: move price_tou into args
        # TODO: move price_peak into args 
        # TODO: need to figure out the actual TOU import and price_export $/kWh values. 
        price_offpeak = 0.14 # $/kWh 
        price_onpeak = 0.22 # $/kWh     
        price_export = 0.18 # $/kwh     
        price_peak = 19.5 # $/kW*month 
        price_tou = {
            0: price_offpeak, 1: price_offpeak, 2: price_offpeak, 
            3: price_offpeak, 4: price_offpeak, 5: price_offpeak, 
            6: price_offpeak, 7: price_offpeak, 8: price_offpeak, 
            9: price_offpeak, 10: price_offpeak, 11: price_offpeak, 
            12: price_onpeak, 13: price_onpeak, 14: price_onpeak, 
            15: price_onpeak, 
            16: price_onpeak, 17: price_onpeak, 18: price_onpeak, 19: price_offpeak, 20: price_offpeak, 
            21: price_offpeak, 22: price_offpeak, 23: price_offpeak
        }

        df_sim = df_sim_in.copy() 
        df_sim.loc[df_sim[net_kw_col] >= 0, "kw_import"] = df_sim.loc[df_sim[net_kw_col] >= 0, net_kw_col]
        df_sim.loc[df_sim[net_kw_col] <= 0, "kw_export"] = df_sim.loc[df_sim[net_kw_col] <= 0, net_kw_col] 

        d_peak = df_sim.tail(-peak_startup)[net_kw_col].max() 
        df_sim.loc[df_sim.tail(-peak_startup)[net_kw_col].idxmax(), "kw_peak"] = df_sim.loc[df_sim.tail(-peak_startup)[net_kw_col].idxmax(), net_kw_col]

        df_sim["kwh_import"] = df_sim["kw_import"].fillna(0) * t_res_minutes/60
        df_sim["kwh_export"] = df_sim["kw_export"].fillna(0) * t_res_minutes/60 

        df_sim["price_tou"] = df_sim.index.hour.map(price_tou) 

        df_sim["bill_import"] = df_sim["kwh_import"] * df_sim["price_tou"] 
        df_sim["bill_export"] = df_sim["kwh_export"] * price_export
        df_sim["bill_peak"] = df_sim["kw_peak"] * price_peak

        n_sims_permonth = pd.Timedelta(days=30) / (df_sim.index.max()-df_sim.index.min()) # Number of simulation periods per 30 days 
        df_sim["bill_import_permonth"] = df_sim["bill_import"] * n_sims_permonth
        df_sim["bill_export_permonth"] = df_sim["bill_export"] * n_sims_permonth
        df_sim["bill_peak_permonth"] = df_sim["bill_peak"].fillna(0) 
        df_sim["bill_tou_permonth"] = df_sim[["bill_import_permonth", "bill_export_permonth"]].sum(axis=1) 
        df_sim["bill_total_permonth"] = df_sim[["bill_import_permonth", "bill_export_permonth", "bill_peak_permonth"]].sum(axis=1) 

        return df_sim 


    def plot_episodic_preds(self, fig_height=None): 
        # Convert TimeSeries Objects to DataFrames 
        y = self.y.pd_dataframe()
        y_pred = self.y_ep_pred.copy() # b_inv.pd_dataframe() # .rename(columns={0:"b_inv"}) 
        y_pred_altered = self.y_ep_pred_alt.copy() # b_inv_altered.pd_dataframe() # .rename(columns={0:"counterfactual preds"}) 

        # Name DataFrame Columns 
        y.columns = self.y_var
        y_pred.columns = self.y_var 
        y_pred_altered.columns = self.y_var 

        
        if isinstance(self.y_var, list):
            row_idx = 1 
            traces = [] 
            trace_rows = [] 
            trace_cols = [] 
            df_raw_15min = self.df_raw.set_index("Timestamp").resample("15min").mean() 
            for col in self.y_var:

                trace_y = go.Scatter(
                    x=pd.to_datetime(y.index), 
                    y=y[col], 
                    name=col, opacity=0.8, 
                    marker=dict(color=px.colors.qualitative.G10[0])
                )
                trace_y_pred = go.Scatter(
                    x=pd.to_datetime(y_pred.index), 
                    y=y_pred[col], 
                    name=col+"_pred", opacity=0.8, 
                    marker=dict(color=px.colors.qualitative.G10[1])
                ) 
                trace_y_altered = go.Scatter(
                    x=pd.to_datetime(y_pred_altered.index), 
                    y=y_pred_altered[col], 
                    name=col+"_counterfactual", opacity=0.8, 
                    marker=dict(color=px.colors.qualitative.G10[2])
                ) 

                traces.append(trace_y) 
                traces.append(trace_y_pred) 
                traces.append(trace_y_altered) 
                trace_rows += [row_idx, row_idx, row_idx] 
                trace_cols += [1,1,1] 

                if "T_in_" in col: 
                    # TODO: Need to only include the CSP/HSP traces if the target variable is an indoor temperature 
                    zone_idx = col.split("_")[-1] 
                    hsp_col = "T_hsp_{}".format(zone_idx) 
                    csp_col = "T_csp_{}".format(zone_idx) 
                    trace_csp = go.Scatter(x=pd.to_datetime(df_raw_15min.index), y=df_raw_15min[csp_col], name=csp_col, opacity=0.8, marker=dict(color="black"))
                    trace_hsp = go.Scatter(x=pd.to_datetime(df_raw_15min.index), y=df_raw_15min[hsp_col], name=hsp_col, opacity=0.8, marker=dict(color="black"))
                    trace_csp_alt = go.Scatter(x=pd.to_datetime(self.df_altered["Timestamp"]), y=self.df_altered[csp_col], name=csp_col+"_counterfactual", opacity=0.5, line=dict(color="black", dash="dot"))
                    trace_hsp_alt = go.Scatter(x=pd.to_datetime(self.df_altered["Timestamp"]), y=self.df_altered[hsp_col], name=hsp_col+"_counterfactual", opacity=0.5, line=dict(color="black", dash="dot"))
                    trace_t_out = go.Scatter(x=pd.to_datetime(df_raw_15min.index), y=df_raw_15min["T_out"], name="T_out", opacity=0.4, marker=dict(color="black"))
                    traces.append(trace_csp) 
                    traces.append(trace_hsp)
                    traces.append(trace_csp_alt) 
                    traces.append(trace_hsp_alt) 
                    traces.append(trace_t_out)
                    trace_rows += [row_idx, row_idx, row_idx, row_idx, row_idx]
                    trace_cols += [1,1,1,1,1] 

                row_idx += 1 

        else:
            row_idx = 1 
            trace_y = go.Scatter(x=pd.to_datetime(y.index), y=y, name=self.y_var, opacity=0.8, marker=dict(color=px.colors.qualitative.G10[0]))
            trace_y_pred = go.Scatter(x = pd.to_datetime(y_pred.index), y=y_pred, name=self.y_var+"_pred", opacity=0.8, marker=dict(color=px.colors.qualitative.G10[1])) 
            trace_y_pred_altered = go.Scatter(x=pd.to_datetime(y_pred_altered.index), y=y_pred_altered, name=self.y_var+"_counterfactual", opacity=0.8, marker=dict(color=px.colors.qualitative.G10[2]))
        

            trace_rows = [row_idx, row_idx, row_idx]
            trace_cols = [1,1,1] 
            traces = [trace_y, trace_y_pred, trace_y_pred_altered] 

        """
        #for col in ["T_out", "T_csp_3", "T_hsp_3", "T_in_3"]: 
        for col in ["T_out"]: 
            trace = go.Scatter(
                x = pd.to_datetime(df_raw_15min.index), 
                y = df_raw_15min[col], 
                name = col, 
                opacity = 0.8, 
                marker = dict(color="black") 
            )
            traces.append(trace) 
            trace_rows.append(row_idx) 
            trace_cols.append(1) 
        """
        
        
        fig = make_subplots(rows=row_idx, shared_xaxes=True) 
        fig.add_traces(traces, rows=trace_rows, cols=trace_cols) 
        fig.update_xaxes(showticklabels=True)
        if fig_height is not None: 
            fig.update_layout(height=fig_height) 
        pyo.plot(fig) 
        self.fig = fig

    def fit_model(self): 
        series_list = [self.train_y] + [self.train_X[c] for c in self.train_X.components] 
        self.model.fit(series_list, verbose=True) 

    def pred_model(self, n_pred_samples): 
        self.pred = self.model.predict(n=n_pred_samples, series=self.train_y) 
        self.pred = self.scaler_y.inverse_transform(self.pred) 

    def plot_results(self, fig_title="", savepath=None):
        pred = self.pred.pd_dataframe() 
        actual = self.y.pd_dataframe() 
        pred["Timestamp"] = pd.to_datetime(pred.index)
        
        pred = pred.reset_index(drop=True)\
            .rename(columns={self.y_var: "forecast"})\
            .set_index("Timestamp") 

        actual["Timestamp"] = pd.to_datetime(actual.index)

        actual = actual.reset_index(drop=True)\
            .rename(columns={self.y_var:"actual"})\
            .set_index("Timestamp")

        df = pd.concat([actual, pred], axis=1) 

        traces = [] 
        for col in df: 
            trace = go.Scatter(
                x = df.index, 
                y = df[col], 
                name = col, 
                opacity = 0.7
            )
            traces.append(trace) 
        fig = go.Figure(data=traces) 
        pyo.plot(fig) 
        self.fig = fig

def update_battery(soc, G_solar, D_nonflex, D_hvac, x_d_max, x_soc_min, x_soc_max, batt_cap_kwh, t_res_minutes, chg_kw_max, dchg_kw_max):
    #dchg_kw_max = 20 # kW, TODO: move to fn args
    #chg_kw_max = 7 # kW, TODO: move to fn args 
    # Assumes soc starts between 0-100% 
    available_dchg_kwh = (soc - x_soc_min)/100 * batt_cap_kwh  # kWh available for discharging (til x_soc_min%) 
    #available_chg_kwh = batt_cap_kwh - available_dchg_kwh       # kWh available for charging (til 100%)
    available_chg_kwh = (x_soc_max - soc)/100 * batt_cap_kwh

    # Convert to kW 
    available_dchg_kw = available_dchg_kwh * 60 / t_res_minutes 
    available_chg_kw = available_chg_kwh * 60 / t_res_minutes 

    # Constrained by max chg/dchg rates of battery 
    available_dchg_kw = - min(available_dchg_kw, dchg_kw_max) # Adds negative sign for chg/dchg polarity 
    available_chg_kw = min(available_chg_kw, chg_kw_max) 
    
    # Battery kW needed 
    chg_needed_kw = -G_solar - D_nonflex - D_hvac + x_d_max

    # Contrained battery kW dispatched 
    chg_dispatched_kw = max(min(chg_needed_kw, available_chg_kw), available_dchg_kw) 
    D_batt = chg_dispatched_kw 

    # Update soc with dispatched battery chg/dchg rate (kW) 
    soc += (D_batt * t_res_minutes/60) / batt_cap_kwh * 100 
    return soc, D_batt


def update_battery_v2(soc, x_d_max, x_soc_min, x_soc_max, batt_cap_kwh, t_res_minutes, chg_kw_max, dchg_kw_max, perform_action):
    # av: if perform_action == 1, charge at x_d_max, if available
    #        else if it is -1, discharge at x_d_max if available

    #dchg_kw_max = 20 # kW, TODO: move to fn args
    #chg_kw_max = 7 # kW, TODO: move to fn args 
    # Assumes soc starts between 0-100% 
    available_dchg_kwh = (soc - x_soc_min)/100 * batt_cap_kwh  # kWh available for discharging (til x_soc_min%) 
    available_chg_kwh = (x_soc_max - soc)/100 * batt_cap_kwh

    # Convert to kW 
    available_dchg_kw = available_dchg_kwh * 60 / t_res_minutes 
    available_chg_kw = available_chg_kwh * 60 / t_res_minutes 

    # Constrained by max chg/dchg rates of battery 
    available_dchg_kw = min(available_dchg_kw, dchg_kw_max) # Adds negative sign for chg/dchg polarity 
    available_chg_kw = min(available_chg_kw, chg_kw_max) 

    if perform_action == 1:
        D_batt = min(x_d_max, available_chg_kw)
    elif perform_action == -1:
        D_batt = -min(x_d_max, abs(available_dchg_kw))
    else:
        D_batt = 0

    # Update soc with dispatched battery chg/dchg rate (kW) 
    soc = soc + ( (D_batt * t_res_minutes/60) / batt_cap_kwh * 100 )
    return soc, D_batt


if __name__ == "__main__": 
    d = Forecaster(
        GLOBS.y_var, 
        GLOBS.X_vars
    )
    d.import_data(GLOBS.filepath_in) 
    d.df_raw = d.prep_features(d.df_raw) 
    d.prep_data(start_date=GLOBS.start_date, end_date=GLOBS.end_date)  
    d.split_scale_data(GLOBS.train_val_split_date) 
    d.prep_altered_data(GLOBS.filepath_tstat_alt) 
   
   # Fit RNN Model with Future Covariates 
    d.init_rnn_model(n_rnn_layers=GLOBS.n_rnn_layers)
    d.fit_rnn_model(GLOBS.n_epochs) 
    #d.eval_model_altered(future_covariates=d.X_scaled, future_covariates_altered=d.X_altered_scaled, forecast_horizon=GLOBS.forecast_horizon) 

    # Run Episodic Predictions
    d.get_episodic_preds(
        future_covariates=d.X_scaled, 
        future_covariates_altered=d.X_altered_scaled, 
        pred_dates = GLOBS.pred_dates, 
        forecast_horizon = GLOBS.forecast_horizon 
    )
    d.plot_episodic_preds(fig_height = 1600) 

    # earlier predictions during the training timeperiod 
    #d.get_episodic_preds(
    #    future_covariates=d.X_scaled, 
    #    future_covariates_altered=d.X_altered_scaled, 
    #    pred_dates = GLOBS.training_pred_dates, 
    #    forecast_horizon = GLOBS.forecast_horizon 
    #)
    #d.plot_episodic_preds(fig_height = 1600) 

    # Fit Block RNN Model with Past Covariates 
    #d.init_blockrnn_model(n_epochs=GLOBS.n_epochs) 
    #d.fit_blockrnn_model() 
    #d.eval_model_altered(past_covariates=d.X_scaled, past_covariates_altered=d.X_altered_scaled, forecast_horizon=GLOBS.forecast_horizon)
    


    # Run NBEATS moel 
    #d.init_model() 
    #d.fit_model() 
    #d.eval_model(fig_title="NBEATS") 

    #d.pred_model(GLOBS.n_val_samples)
    #d.plot_results()  

    print("Test DONE!") 
