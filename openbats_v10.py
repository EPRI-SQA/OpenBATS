# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:19:08 2022

@author: pvan002
"""

# openbats GUI to run the simulation in the background

import threading
import time
import os.path
import PySimpleGUI as sg

#import simulate_v3gui as sim_ob
import simulate_v4gui as sim_ob


def simulation_thread(window, model_file = None, data_file = None):
    #for ii in range(10):
    #    time.sleep(2)
    #    print("in step " + str(ii))
    #file_trained_model = "C:/Users/pvan002/Documents/jupyternotebooks/demos/epri_sim_env_ananconda/data/model4sim.pth.tar"
    #data_in_file = "C:\\Users\\pvan002\\Documents\\jupyternotebooks\\demos\\epri_sim_env_ananconda\\data\\df_historical_20210101-20210601.csv"
    if model_file is None or data_file is None:
        print(f'model file => {model_file}')
        print(f'data file => {data_file}')
        print("both data file and model file are required for optimizaton")
        window.write_event_value('-SIMULATION FAILED-', "")
    else:
        sim_ob.simulate_from_gui(file_model = model_file, data_in_file = data_file)
        window.write_event_value('-SIMULATION DONE-', '')

def do_simulation(model_file = None, data_file = None):
    threading.Thread(target=simulation_thread, args=(window, model_file, data_file), daemon=True).start()


def make_window(theme):
    sg.theme(theme)
    
    layout_sim = [
        [sg.Text("OpenBats Simulation Tool",size=(38, 1), justification='center', font=("Helvetica", 16), relief=sg.RELIEF_RIDGE, k='-TEXT HEADING-', enable_events=True)],
        [sg.Text("Choose Input Data File for Simulation (*.csv)")],
        [sg.Combo(sg.user_settings_get_entry('-filenames-data-', []), default_value=sg.user_settings_get_entry('-last filename-data-', ''), size=(50, 1), key='-FILENAME-DATA-'),
         sg.FileBrowse()],
        [sg.Text("Input Variables")],
        [sg.Input("T_out, t_hr, t_dow, T_csp_out_diff_3, T_hsp_out_diff_3, T_csp_3, T_hsp_3, T_csp_out_diff_5, T_hsp_out_diff_5, T_csp_5, T_hsp_5, T_csp_out_diff_7, T_hsp_out_diff_7, T_csp_7, T_hsp_7, T_csp_out_diff_6, T_hsp_out_diff_6, T_csp_6, T_hsp_6",
                  key='-input-vars-', expand_x = True, expand_y = True)],
        [sg.Text('Output Variables')],
        [sg.Input("D_hvac, T_in_3, T_in_5, T_in_6, T_in_7", key='-output-vars-', expand_x = True, expand_y = True)],
        [sg.Text("Select File that contains the trained model")],        
        [sg.Combo(sg.user_settings_get_entry('-filenames-model-', []), default_value=sg.user_settings_get_entry('-last filename-model-', ''), size=(50, 1), key='-FILENAME-MODEL-'),
         sg.FileBrowse()],
        [sg.Text("Optimization >> "), 
             sg.Combo(values=('Price Reduction', 'Reduce Peak Energy'), default_value='Price Reduction', 
             readonly=True, k='-OPT-COMBO-')],
        [sg.Button("Run Optimization"), sg.Button('Exit')],
        [sg.Frame("Output", [[sg.Multiline(size=(75,15), font='Courier 8', expand_x=True, expand_y=True, write_only=True,
                                    reroute_stdout=True, reroute_stderr=True, echo_stdout_stderr=True, autoscroll=True, auto_refresh=True)]])]
        ]
    
    window = sg.Window('OpenBats Simulation Tool 1.0', layout_sim, finalize=True, keep_on_top=True)
    return window

sg.theme('LightGrey1')
window = make_window(sg.theme())

while True:
    event, values = window.read()
    print(event)
    print(values)
    print(type(event))
    if event == sg.WIN_CLOSED or event == 'Exit':
        break 
    elif event is None:
        break
    elif event == 'Run Optimization':
        print ("Running Optimization")
        dat_file = values['-FILENAME-DATA-']
        mod_file = values['-FILENAME-MODEL-']
        tx_vars = values['-input-vars-']
        x_vars= tx_vars.split(',') if tx_vars else []
        x_vars = [s.strip() for s in x_vars]
        ty_vars = values['-output-vars-']
        y_vars = ty_vars.split(',') if ty_vars else []
        y_vars = [s.strip() for s in y_vars]
        print (f' xvars = {x_vars}')
        print (f' yvars = {y_vars}')
        if dat_file is None or mod_file is None or len(dat_file) == 0 or len(mod_file) == 0:
            print("Cannot do optimizaton, both data and model file required")
            print(f'model file => {mod_file}')
            print(f'data file => {dat_file}')
        else:
            sg.user_settings_set_entry('-filenames-data-', list(set(sg.user_settings_get_entry('-filenames-data-', []) + [values['-FILENAME-DATA-'], ])))
            sg.user_settings_set_entry('-last filename-data-', values['-FILENAME-DATA-'])
            sg.user_settings_set_entry('-filenames-model-', list(set(sg.user_settings_get_entry('-filenames-model-', []) + [values['-FILENAME-MODEL-'], ])))
            sg.user_settings_set_entry('-last filename-model-', values['-FILENAME-MODEL-'])
            do_simulation(mod_file, dat_file)
    elif event == '-SIMULATION DONE-':
        print ("Optimization done")
        
        
window.close()
    