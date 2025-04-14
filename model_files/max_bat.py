#%% Initialisation
import calliope

calliope.set_log_verbosity('INFO') #sets the level of verbosity of Calliope's operations


#%% 
'''
--------------------------------------------------------
---------------Parallel batch of SPORES ----------------
--------------------------------------------------------
'''

'''
Model creation, run and saving each SPORE individually to netCDF
'''

model = calliope.Model('model_files/model.yaml', scenario='res_3h,spores_group_constraint,spores_run_config,children_spores,max_battery,max_mode') 
model.run()

# %%
