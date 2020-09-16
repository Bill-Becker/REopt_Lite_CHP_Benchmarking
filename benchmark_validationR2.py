# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:32:54 2020

@author: dolis
"""
import os
#import glob
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import rcParams
from scipy.stats import linregress

#benchmark_jsons_dir ='C:/Users/dolis/Box/CHP Benchmarking/EightPercentGap/'
#benchmark_jsons_dir ='C:/Users/dolis/Box/CHP Benchmarking/Decomposition, Sept 3-4 2020/OnePctConstFullOneIter/'
#benchmark_jsons_dir ='C:/Users/dolis/Box/CHP Benchmarking/Decomposition, Sept 3-4 2020/ThreePctConstFullOneIterFixedCool/'
#benchmark_jsons_dir ='C:/Users/dolis/Box/CHP Benchmarking/Decomposition, Sept 3-4 2020/FivePctGapBothTESReportGap/'
#benchmark_jsons_dir ='C:/Users/dolis/Box/CHP Benchmarking/Monolith/ThreePctMonolithConstFullOneIter/'
#benchmark_jsons_dir ='C:/Users/dolis/Box/CHP Benchmarking/Monolith/HalfPctMonolithConstFullOneIter/'
benchmark_jsons_dir ='C:/Users/dolis/Box/CHP Benchmarking/Monolith/MonoOnePct900SecsConst/'
#benchmark_jsons_dir ='C:/Users/dolis/Box/CHP Benchmarking/FivePercentGap/'
outdir = 'C:/Users/dolis/Documents/REopt-REO/AMO CHP Lab call 2018/API validation/Benchmarking/'
           
files = [f for f in os.listdir(benchmark_jsons_dir) if os.path.isfile(os.path.join(benchmark_jsons_dir, f)) and f.endswith('.json')]
scenario = benchmark_jsons_dir.split("/")[-2]
summary = [] 
z = []
for file in range(0,len(files)):
#for file in range(0,3):
   #Load the JSON into a dictionary
   d = json.loads(eval(open(benchmark_jsons_dir + files[file]).read()))
   #d = json.loads(eval(open(benchmark_jsons_dir + files[file]).read()))
   #d = json.loads(eval(open(benchmark_jsons_dir + 'chp_hottes_SanFrancisco_Hospital_5cb742345457a31e119b6ec3_13_0.json').read()))
   urdb_label = d['inputs']['Scenario']['Site']['ElectricTariff']['urdb_label']
   natural_gas_price = d['inputs']['Scenario']['Site']['FuelTariff']['boiler_fuel_blended_annual_rates_us_dollars_per_mmbtu']
   status = d['outputs']['Scenario']['status']
   try: lower_bound = d['outputs']['Scenario']['lower_bound']
   except: lower_bound = None
   try: optimality_gap = d['outputs']['Scenario']['optimality_gap']
   except: optimality_gap = None
   reopt_seconds = d['outputs']['Scenario']['Profile']['reopt_seconds']
   lcc_bau = d['outputs']['Scenario']['Site']['Financial']['lcc_bau_us_dollars']
   lcc = d['outputs']['Scenario']['Site']['Financial']['lcc_us_dollars']
   npv = d['outputs']['Scenario']['Site']['Financial']['npv_us_dollars'] 
   year_one_electric_energy_produced_kwh = d['outputs']['Scenario']['Site']['CHP']['year_one_electric_energy_produced_kwh']
   load_year = str(d['inputs']['Scenario']['Site']['LoadProfile']['year'])
   dti = pd.date_range(load_year +'-01-01', periods=8760, freq='H')
   #bau_grid_load_hrly = pd.DataFrame(d['inputs']['Scenario']['Site']['LoadProfile']['loads_kw'], index = dti)
   #bau_grid_load_peak = np.max(bau_grid_load_hrly)
   #bau_grid_load_mean = np.mean(bau_grid_load_hrly)
   #bau_grid_load_total = np.sum(bau_grid_load_hrly)
   bau_boiler_thermal_load_mmbtu_hrly = pd.DataFrame(d['outputs']['Scenario']['Site']['LoadProfileBoilerFuel']['year_one_boiler_thermal_load_series_mmbtu_per_hr'], index = dti)
   bau_therm_load_peak = np.max(bau_boiler_thermal_load_mmbtu_hrly).get(0) #units of MMBtu heat, not fuel
   bau_therm_load_mean = np.mean(bau_boiler_thermal_load_mmbtu_hrly).get(0)
   bau_therm_load_total = np.sum(bau_boiler_thermal_load_mmbtu_hrly).get(0)
   pv_min_kw = d['inputs']['Scenario']['Site']['PV']['min_kw']
   pv_max_kw = d['inputs']['Scenario']['Site']['PV']['max_kw']
   pv_size = d['outputs']['Scenario']['Site'].get('PV',{}).get('size_kw')
   bess_min_kw = d['inputs']['Scenario']['Site']['Storage']['min_kwh']
   bess_max_kw = d['inputs']['Scenario']['Site']['Storage']['max_kwh']
   bess_size_kW = d['outputs']['Scenario']['Site'].get('Storage',{}).get('size_kw')
   bess_min_kwh = d['inputs']['Scenario']['Site']['Storage']['min_kw']
   bess_max_kwh = d['inputs']['Scenario']['Site']['Storage']['max_kw']
   bess_size_kWh = d['outputs']['Scenario']['Site'].get('Storage',{}).get('size_kwh')
   CWTES_min_gal = d['inputs']['Scenario']['Site']['ColdTES']['min_gal']
   CWTES_max_gal = d['inputs']['Scenario']['Site']['ColdTES']['max_gal']
   CWTES_size = d['outputs']['Scenario']['Site'].get('ColdTES',{}).get('size_gal')
   HWTES_min_gal = d['inputs']['Scenario']['Site']['HotTES']['min_gal']
   HWTES_max_gal = d['inputs']['Scenario']['Site']['HotTES']['max_gal']
   HWTES_size = d['outputs']['Scenario']['Site'].get('HotTES',{}).get('size_gal')
   chp_min_kw = d['inputs']['Scenario']['Site']['CHP']['min_kw']
   chp_max_kw = d['inputs']['Scenario']['Site']['CHP']['max_kw']
   chp_elec_eff_full_load = d['inputs']['Scenario']['Site']['CHP']['elec_effic_full_load']
   chp_elec_eff_half_load = d['inputs']['Scenario']['Site']['CHP']['elec_effic_half_load']
   chp_therm_eff_full_load = d['inputs']['Scenario']['Site']['CHP']['thermal_effic_full_load']
   chp_therm_eff_half_load = d['inputs']['Scenario']['Site']['CHP']['thermal_effic_half_load']
   chp_therm_to_elec_full_load = chp_therm_eff_full_load/chp_elec_eff_full_load if chp_elec_eff_full_load is not None else None
   chp_size = d['outputs']['Scenario']['Site'].get('CHP',{}).get('size_kw')
   chp_year_one_electric_energy_produced_kwh = d['outputs']['Scenario']['Site']['CHP']['year_one_electric_energy_produced_kwh']
   chp_year_one_electric_production_series_kw = pd.DataFrame(d['outputs']['Scenario']['Site']['CHP']['year_one_electric_production_series_kw'], index = dti)
   chp_year_one_to_battery_series_kw = pd.DataFrame(d['outputs']['Scenario']['Site']['CHP']['year_one_to_battery_series_kw'], index = dti)
   chp_year_one_to_load_series_kw = pd.DataFrame(d['outputs']['Scenario']['Site']['CHP']['year_one_to_load_series_kw'], index = dti)
   chp_year_one_to_grid_series_kw = pd.DataFrame(d['outputs']['Scenario']['Site']['CHP']['year_one_to_grid_series_kw'], index = dti)
   chp_elec_total_series_kW = chp_year_one_to_battery_series_kw + chp_year_one_to_load_series_kw + chp_year_one_to_grid_series_kw
   chp_year_one_electric_production_series_total = np.sum(chp_year_one_electric_production_series_kw).get(0)
   chp_year_one_to_battery_total = np.sum(chp_year_one_to_battery_series_kw).get(0)
   chp_year_one_to_load_total = np.sum(chp_year_one_to_load_series_kw).get(0)
   chp_year_one_to_grid_total = np.sum(chp_year_one_to_grid_series_kw).get(0)
   chp_therm_to_load_hrly = pd.DataFrame(d['outputs']['Scenario']['Site']['CHP']['year_one_thermal_to_load_series_mmbtu_per_hour'], index = dti)
   chp_therm_to_TES_hrly = pd.DataFrame(d['outputs']['Scenario']['Site']['CHP']['year_one_thermal_to_tes_series_mmbtu_per_hour'], index = dti)
   chp_therm_total_hrly = chp_therm_to_load_hrly + chp_therm_to_TES_hrly
   chp_fuel_annual = d['outputs']['Scenario']['Site']['CHP']['year_one_fuel_used_mmbtu']
   #chp_elec_effic = chp_year_one_electric_energy_produced_kwh/(chp_fuel_annual * 1E6/3412.0) if chp_size is not None else None
   try: chp_total_effic = (chp_year_one_electric_energy_produced_kwh + (np.sum(chp_therm_total_hrly)).get(0) * 1E6/3412.0)/(chp_fuel_annual * 1E6/3412.0) if chp_size is not None else None
   except: chp_total_effic = None
   try: chp_elec_effic = chp_year_one_electric_energy_produced_kwh/(chp_fuel_annual * 1E6/3412.0) if chp_size is not None else None
   except ZeroDivisionError: chp_elec_effic = 0
   try: chp_elec_cf = chp_year_one_electric_energy_produced_kwh/(chp_size*8760) if chp_size is not None else None
   except ZeroDivisionError: chp_elec_cf = 0
   #chp_elec_cf = chp_year_one_electric_energy_produced_kwh/(chp_size*8760) if chp_size is not None else None
   chp_avg_load_when_on = chp_elec_total_series_kW[chp_elec_total_series_kW>1].mean().get(0)
   chp_therm_total_hrly = chp_therm_to_load_hrly + chp_therm_to_TES_hrly
   chp_annual_runhrs = np.sum(chp_elec_total_series_kW>1).get(0)
   try: chp_elec_cf_when_on = chp_avg_load_when_on/chp_size if chp_size is not None else None
   #except ZeroDivisionError: chp_elec_cf_when_on = 0
   except: chp_elec_cf_when_on = 0
   chp_annual_heatinghrs = np.sum(chp_therm_total_hrly>0.01).get(0)
   #chp_capacity_elec_to_load_peak = chp_size/bau_grid_load_peak
   #chp_capacity_elec_to_load_mean = chp_size/bau_grid_load_mean
   #chp_capacity_therm_to_load_peak = chp_size*chp_therm_to_elec_full_load/(bau_therm_load_peak*1e6/3412) if bau_therm_load_peak > 0 else None
   #chp_capacity_therm_to_load_mean = chp_size*chp_therm_to_elec_full_load/(bau_therm_load_mean*1e6/3412) if bau_therm_load_mean > 0 else None
   #load_hours_less_than_chp_capacity = np.sum(bau_grid_load_hrly < chp_size)
   #find base load of bau_grid_load_hrly. Using value with 85% exceedance as described in ASHRAE 'Combined Heat and Power Design Guide' 
   #base_load_site_elec = bau_grid_load_hrly.sort_values(0, ascending = False).iloc[int(8760*.85)-1]
   #base_load_site_therm = bau_boiler_thermal_load_mmbtu_hrly.sort_values(0, ascending = False).iloc[int(8760*.85)-1]
   #chp_capacity_elec_to_base_load = chp_size/base_load_site_elec
   #chp_capacity_therm_to_base_load = chp_size*chp_therm_to_elec_full_load/(base_load_site_therm*1e6/3412) if base_load_site_therm > 0 else None
   #chp_gen_elec_to_load = chp_elec_total_hrly.resample('Y').sum()/bau_grid_load_hrly.resample('Y').sum()
   #chp_gen_therm_to_load = chp_therm_total_hrly.resample('Y').sum()/(bau_boiler_thermal_load_mmbtu_hrly.resample('Y').sum()) if bau_boiler_thermal_load_mmbtu_hrly.resample('Y').sum() > 0 else None
   
   result = {'file':files[file],
             'urdb_label':urdb_label,
             'natural_gas_price':natural_gas_price,
             'status':status,
             'lower_bound':lower_bound,
             'optimality_gap':optimality_gap,
             'reopt_seconds':reopt_seconds,
             'lcc_bau':lcc_bau,
             'lcc':lcc,
             'npv':npv,
             'pv_min_kw':pv_min_kw,
             'pv_max_kw':pv_max_kw,
             'pv_size':pv_size,
             'bess_min_kw':bess_min_kw,
             'bess_max_kw':bess_max_kw,
             'bess_size_kW':bess_size_kW,
             'bess_min_kwh':bess_min_kwh,
             'bess_max_kwh':bess_max_kwh,
             'bess_size_kWh':bess_size_kWh,
             'CWTES_min_gal':CWTES_min_gal,
             'CWTES_max_gal':CWTES_max_gal,
             'CWTES_size':CWTES_size,
             'HWTES_min_gal':HWTES_min_gal,
             'HWTES_max_gal':HWTES_max_gal,
             'HWTES_size':HWTES_size,
             'chp_min_kw':chp_min_kw,
             'chp_max_kw':chp_max_kw,
             'chp_size':chp_size,
             'chp_year_one_electric_energy_produced_kwh':chp_year_one_electric_energy_produced_kwh,
             'chp_year_one_electric_production_series_total':chp_year_one_electric_production_series_total,
             'chp_year_one_to_battery_total':chp_year_one_to_battery_total,
             'chp_year_one_to_load_total':chp_year_one_to_load_total,
             'chp_year_one_to_grid_total':chp_year_one_to_grid_total,
             'chp_elec_effic':chp_elec_effic,
             'chp_total_effic':chp_total_effic,
             'chp_elec_cf':chp_elec_cf,
             'chp_avg_load_when_on':chp_avg_load_when_on,
             'chp_annual_runhrs':chp_annual_runhrs,
             'chp_annual_heatinghrs':chp_annual_heatinghrs
             #'chp_capacity_elec_to_load_peak':chp_capacity_elec_to_load_peak,
             #'chp_capacity_elec_to_load_mean':chp_capacity_elec_to_load_mean,
             #'chp_capacity_therm_to_load_peak':chp_capacity_therm_to_load_peak,
             #'chp_capacity_therm_to_load_mean':chp_capacity_therm_to_load_mean
            }
   z.append(result)
df = pd.DataFrame(z)
#df.to_csv(outdir +'results_mono_const_1prcntR2'+scenario+'.csv')

timeCHP = df['reopt_seconds'][(df['chp_size']>1)]

#Identify tech options in each run
is_CHP_option = np.array([1 if x > 1 else 0 for x in df['chp_max_kw']])
is_PV_option = np.array([1 if x > 1 else 0 for x in df['pv_max_kw']])
is_BESS_option = np.array([1 if x > 1 else 0 for x in df['bess_max_kwh']])
is_CWTES_option = np.array([1 if x > 1 else 0 for x in df['CWTES_max_gal']])
is_HWTES_option = np.array([1 if x > 1 else 0 for x in df['HWTES_max_gal']])

#Identify tech selected in each run
tech_list = ['CHP', 'PV', 'BESS', 'CWTES', 'HWTES']
tech_size_column = ['chp_size', 'pv_size', 'bess_size_kWh', 'CWTES_size', 'HWTES_size']
blankdata = []*len(df)
df_is_tech_selected = pd.DataFrame(blankdata, index = None)
for tech in range(0,len(tech_list)):
    dftechis = pd.DataFrame([1 if x > 1 else 0 for x in df[tech_size_column[tech]]])
    dftechis.columns = ['is_' + tech_list[tech] + '_selected']
    df_is_tech_selected = pd.concat([df_is_tech_selected, dftechis], axis = 1) #axis = 1 joins df to dfDispatch by adding columns

count_storage_option = is_BESS_option + is_CWTES_option + is_HWTES_option
count_storage_option = pd.DataFrame(count_storage_option, columns = ['count_storage_option'], index = None)
count_tech_option = is_CHP_option + is_PV_option + is_BESS_option + is_CWTES_option + is_HWTES_option
count_tech_option = pd.DataFrame(count_tech_option, columns = ['count_tech_option'], index = None)
#count_tech_option = pd.DataFrame(count_tech_option).rename('count_tech_option')
df = pd.concat([df, count_storage_option, count_tech_option, df_is_tech_selected], axis = 1) 

df.to_csv(outdir +'results_mono_const_1prcntR3'+scenario+'.csv')

#solve times by tech count
time_1tech = df.reopt_seconds[(df.status == 'optimal') & (df.count_tech_option == 1)]
time_2tech = df.reopt_seconds[(df.status == 'optimal') & (df.count_tech_option == 2)]
time_3tech = df.reopt_seconds[(df.status == 'optimal') & (df.count_tech_option == 3)]
time_4tech = df.reopt_seconds[(df.status == 'optimal') & (df.count_tech_option == 4)]
time_5tech = df.reopt_seconds[(df.status == 'optimal') & (df.count_tech_option == 5)]

#solve times for runs with storage in the tech options
time_1storage = df.reopt_seconds[(df.status == 'optimal') & (df.count_storage_option == 1)]
time_2storage = df.reopt_seconds[(df.status == 'optimal') & (df.count_storage_option == 2)]
time_3storage = df.reopt_seconds[(df.status == 'optimal') & (df.count_storage_option == 3)]
time_CWTES = df.reopt_seconds[(df.status == 'optimal') & (df.CWTES_max_gal > 0) & (df.HWTES_max_gal == 0) & (df.bess_max_kwh == 0)]
time_HWTES = df.reopt_seconds[(df.status == 'optimal') & (df.CWTES_max_gal == 0) & (df.HWTES_max_gal > 0) & (df.bess_max_kwh == 0)]
time_bothTES = df.reopt_seconds[(df.status == 'optimal') & (df.CWTES_max_gal > 0) & (df.HWTES_max_gal > 0) & (df.bess_max_kwh == 0)]
time_BESS_CWTES = df.reopt_seconds[(df.status == 'optimal') & (df.CWTES_max_gal > 0) & (df.HWTES_max_gal == 0) & (df.bess_max_kwh > 0)]
time_BESS_HWTES = df.reopt_seconds[(df.status == 'optimal') & (df.CWTES_max_gal == 0) & (df.HWTES_max_gal > 0) & (df.bess_max_kwh > 0)]


#histogram of CF data
CFdata = df['chp_elec_cf'][(df['chp_size']>1)]
bins = np.arange(0, 1.0, 0.05)
n, bins, patches = plt.hist(CFdata, bins=bins)
plt.title('Histogram of CHP Capacity Factor for Solutions with CHP')
plt.xlabel('CHP Capacity Factor Bin')
plt.ylabel('Count')
plt.xticks(bins, size = 12, rotation = 90)
plt.text
plt.show()

#histogram of CWTES sizes
bins = np.arange(0, df['CWTES_size'].max(), 10000)
n, bins, patches = plt.hist(df['CWTES_size'][(df['CWTES_size']>1)], bins=bins)
plt.title('Histogram of CWTES Sizes')
plt.xlabel('CWTES Size [gallons]')
plt.ylabel('Count')
plt.xticks(size = 12)
plt.text
plt.show()

#histogram of HWTES sizes
#bins= np.arange(0, df['HWTES_size'].max(), 10000)
n, bins, patches = plt.hist(df['HWTES_size'][(df['HWTES_size']>1)])
plt.title('Histogram of HWTES Sizes')
plt.xlabel('HWTES Size [gallons]')
plt.ylabel('Count')
plt.xticks(size = 12)
plt.text
plt.show()

#plot NPV v. CF
NPVwCHP = df['npv'][(df['chp_size']>1)]
CHPsize = df['chp_size'][(df['chp_size']>1)]
fig, ax = plt.subplots()
plt.scatter(CFdata, NPVwCHP, s = CHPsize, alpha = 0.5)
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick) 
plt.title('NPV vs CHP Capacity Factor for Solutions with CHP')
plt.xlabel('CHP Capacity Factor')
plt.ylabel('Net Present Value ($)')
plt.xticks(np.arange(0, 1.1, 0.1),size = 12)
plt.show()

fig, ax = plt.subplots()
plt.boxplot([time_1tech, time_2tech, time_3tech, time_4tech, time_5tech])
plt.title('Solve Time for Number of Techs Screened')
plt.ylabel('REopt Solve Time (sec)')
plt.xticks(size = 12)
plt.show()

fig, ax = plt.subplots()
data = [time_1storage, time_2storage, time_3storage, time_CWTES, time_HWTES, time_bothTES, time_BESS_CWTES, time_BESS_HWTES]
labels = ['1 storage', '2 storages', '3 storages', 'CWTES', 'HWTES', 'CW & HWTES', 'CWTES & BESS', 'HWTES & BESS']
plt.boxplot(data, labels = labels)
plt.title('Solve Time for Storage Options Screened')
plt.xticks(size = 12, rotation = 90)
plt.ylabel('REopt Solve Time (sec)')
plt.show()



#print("CHP size =", '%.1f' %chp_size, "kW")
#print("Annual CHP Electric Efficiency =", '%.3f' %chp_elec_effic)
#print("Annual CHP Total Efficiency =", '%.3f' %chp_total_effic)
#print("Annual CHP Elec Capacity Factor =", '%.2f' %chp_elec_cf)
#print("Annual CHP Elec Capacity Factor when running =", '%.2f' %chp_elec_cf_when_on)
#print("Annual CHP run hours =", '%.0f' %chp_annual_runhrs, "(fraction",'%.2f'%(chp_annual_runhrs/8760),")")
#print("Annual CHP hours of useful heat =", '%.0f' %chp_annual_heatinghrs, "(fraction",'%.2f'%(chp_annual_heatinghrs/8760),")")
#print("Annual CHP mean power output when on =", '%.0f' %chp_avg_load_when_on, "kW")
#print("Elec peak load =", '%.0f' %bau_grid_load_peak, "kW")
#print("Elec mean load =", '%.0f' %bau_grid_load_mean, "kW")
#print("Elec base load =", '%.0f' %base_load_site_elec, "kW (85% exceedance value)")
#print("Heating peak load =", '%.2f' %bau_therm_load_peak, "MMBtu")
#print("Heating mean load =", '%.2f' %bau_therm_load_mean, "MMBtu")
#print("Heating base load =", '%.2f' %base_load_site_therm, "MMBtu (85% exceedance value)")
#print("Ratio of CHP elec capacity to peak elec load =", '%.3f' %chp_capacity_elec_to_load_peak)
#print("Ratio of CHP elec capacity to mean elec load =", '%.3f' %chp_capacity_elec_to_load_mean)
#print("Ratio of CHP elec capacity to base elec load =", '%.2f' %chp_capacity_elec_to_base_load)
#print("Ratio of CHP therm capacity to peak heat load =", '%.3f' %chp_capacity_therm_to_load_peak)
#print("Ratio of CHP therm capacity to mean heat load =", '%.3f' %chp_capacity_therm_to_load_mean)
#print("Ratio of CHP therm capacity to base heat load =", '%.3f' %chp_capacity_therm_to_base_load)
#print("Useful heat recovery ratio =", '%.3f' )
#print("Annual fraction of site elec load served by CHP =", '%.3f' %chp_gen_elec_to_load.iloc[0][0])
#print("Annual fraction of site thermal load served by CHP =", '%.3f' %chp_gen_therm_to_load.iloc[0][0])


