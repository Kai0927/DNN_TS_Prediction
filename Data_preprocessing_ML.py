# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:45:33 2023

@author: user
"""
# Import and read raw data
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import datetime
import glob

import csv
filename ='D:/D_file\\Research assistance\\Machine learning\\OBS_STATION_DATA.csv'
import pandas as pd
data_id = pd.read_csv(filename, delimiter= ',')

dict_OBS = {}
for i in range(len(data_id['id'])):
    dict_OBS[data_id['id'][i]] = [data_id['lon'][i],data_id['lat'][i],data_id['alt'][i]]
    
filename_1 = glob.glob("D:/D_file\\Research assistance\\Machine learning\\OBS_DATA\\test\\*.txt")

data_f = []
for i in range(len(filename_1)):
    data_f = data_f
    with open(filename_1[i], 'r') as f:
        data = f.readlines()
    #if (int(filename_1[i][56:60]) < 2019):
    #    data = data[77::]
    #else:
        data = data[78::]
    data_f = np.concatenate((data_f, data),axis=0)
out=[]
for line in data_f:
    line = str(line)
    line=line.strip('\n')
    if (line!=''):
        out.append(line.split(' '))
        
station_id1=[];time=[];pre=[];ws=[];wd=[];t=[];p=[];rh=[];lat=[];lon=[];alt=[];date=[]
for i in range(len(out)):
    if (out[i][0] != "C0E860"):
        if (23<float(dict_OBS[out[i][0]][1])<24.25 and 120<float(dict_OBS[out[i][0]][0])<120.9):
            print(i)
            station_id1.append(out[i][0])
            date.append(out[i][1][0:8])
            time.append(out[i][1][8:10])
            p.append(data_f[i][18:24])
            t.append(data_f[i][27:32])
            rh.append(data_f[i][36:38])
            pre.append(data_f[i][52:59])
            wd.append(data_f[i][45:52])
            ws.append(data_f[i][38:45])
            lat.append(dict_OBS[out[i][0]][1])
            lon.append(dict_OBS[out[i][0]][0])
            alt.append(dict_OBS[out[i][0]][2])   
#%% reshape to per day data
len_days = int(len(date)/24)
station_id = np.asarray(station_id1).reshape((len_days,24))[:,0]
time =  np.asarray(time).reshape((len_days,24))#[:,0] 
date =  np.asarray(date).reshape((len_days,24))[:,0]   
lat_sta = np.asarray(lat,dtype='float').reshape((len_days,24))[:,0] 
lon_sta = np.asarray(lon,dtype='float').reshape((len_days,24))[:,0]
alt_sta = np.asarray(alt,dtype='float').reshape((len_days,24))[:,0]
pre_sta = np.asarray(pre,dtype='float').reshape((len_days,24))
wd_sta = np.asarray(wd,dtype='float').reshape((len_days,24))  
ws_sta = np.asarray(ws,dtype='float').reshape((len_days,24))  
t_sta = np.asarray(t,dtype='float').reshape((len_days,24))  
p_sta = np.asarray(p,dtype='float').reshape((len_days,24))  
rh_sta = np.asarray(rh,dtype='float').reshape((len_days,24))  

missing_value = np.nan
pre_sta[pre_sta<-9990]=missing_value
wd_sta[wd_sta<-9990]=missing_value
ws_sta[ws_sta<-9990]=missing_value
t_sta[t_sta<-9990]=missing_value
t_sta[t_sta>99]=missing_value
p_sta[p_sta<-9990]=missing_value
rh_sta[rh_sta<-9990]=missing_value
rh_sta[rh_sta==0]=missing_value

td_sta = t_sta-(100-rh_sta)/5
es = 611.21*np.exp((22.587*t_sta)/(t_sta+273.86))/100
w_s = 621.97*(es/(p_sta-es))
M_sta = (rh_sta*w_s)/100
delta_td = t_sta-td_sta
#%% make dictionary to recognize weak synoptic
filename ='D:/D_file\\Research assistance\\Machine learning\\OBS_DATA\\2022_OBS_DATA_FILTER.csv'
sta_status = pd.read_csv(filename, delimiter= ',')

date_csv = np.asarray(sta_status['date'])[np.where(sta_status['y/n']=="y")]

dict_STA_STATUS = {}
for i in range(len(sta_status['date'])):
    dict_STA_STATUS[sta_status['date'][i]] = sta_status['y/n'][i]
#%% Pre process data
station_id_out=[];date_out=[];time_out=[];lat_out=[];lon_out=[];alt_out=[];
p_out=[];t_out=[];dtd_out=[];rh_out=[];M_out=[];wd_out=[];ws_out=[];pre_out=[]
for i in range(len(time)):
    if (dict_STA_STATUS[int(date[i])]=='y'):
        print(i)
        station_id_out.append(station_id[i])
        date_out.append(date[i])
        time_out.append(time[i])
        lat_out.append(lat_sta[i])
        lon_out.append(lon_sta[i])
        alt_out.append(alt_sta[i])
        p_out.append(p_sta[i])
        t_out.append(t_sta[i])
        dtd_out.append(delta_td[i])
        rh_out.append(rh_sta[i])
        M_out.append(M_sta[i])
        wd_out.append(wd_sta[i])
        ws_out.append(ws_sta[i])
        pre_out.append(pre_sta[i]) 
len_days_TC = len(station_id_out)        
station_id_out = np.asarray(station_id_out)
time_out =  np.asarray(time_out)
date_out =  np.asarray(date_out)
lat_out = np.asarray(lat_out,dtype='float')
lon_out = np.asarray(lon_out,dtype='float')
alt_out = np.asarray(alt_out,dtype='float')
wd_out = np.asarray(wd_out,dtype='float')  
ws_out = np.asarray(ws_out,dtype='float')  
t_out = np.asarray(t_out,dtype='float') 
p_out = np.asarray(p_out,dtype='float') 
rh_out = np.asarray(rh_out,dtype='float') 
dtd_out = np.asarray(dtd_out,dtype='float')
M_out = np.asarray(M_out,dtype='float')        
pre_out = np.asarray(pre_out)
date_out = np.asarray(date_out)
date_TC = []
dict_TC = {}
for i in range(len(date_csv)):
    P = pre_out[np.where(date_out==str(date_csv[i]))]
    P_12_22 = P[:,11:21]
    P_1_11 = P[:,0:10]
    a = np.where(P_12_22>=0.5, P_12_22, 0)
    b =  np.where(P_1_11>=0.2, P_1_11, 0)
    condition = 100*np.nansum(P_12_22)/np.nansum(P) >= 80 and 100*np.nansum(P_1_11)/np.nansum(P) <= 10 and np.nansum(P) >= 30 and np.count_nonzero(a) >= 3 #and np.count_nonzero(b) <= 3
    if (condition == True):
        print(date_csv[i])
        date_TC.append(date_csv[i])
        dict_TC[date_csv[i]] = 1
    else:
        dict_TC[date_csv[i]] = 0
date_TC = np.asarray(date_TC)

grid_TC_or_NTC = []
for i in range(len(pre_out)):
    grid_P = pre_out[i]
    grid_P_12_22 = grid_P[11:21]
    grid_P_1_11 = grid_P[0:10]
    condition = 100*np.nansum(grid_P_12_22)/np.nansum(grid_P) >= 80 and 100*np.nansum(grid_P_1_11)/np.nansum(grid_P) <= 10 and np.nansum(grid_P) >= 10
    if(condition == True):
        grid_TC_or_NTC.append(1)
        print(i)
    else:
        grid_TC_or_NTC.append(0)
#%%
TC_or_NTC = []
for i in range(len(time)):
    if (dict_STA_STATUS[int(date[i])]=='y'):
        TC_or_NTC.append(dict_TC[int(date[i])])
TC_or_NTC_out = np.asarray(TC_or_NTC).reshape((len_days_TC,1))
station_id_out = np.asarray(station_id_out).reshape((len_days_TC,1))
#time_out =  np.asarray(time_out).reshape((len_days_TC,1))
date_out =  np.asarray(date_out).reshape((len_days_TC,1)) 
lat_out = np.asarray(lat_out,dtype='float').reshape((len_days_TC,1)) 
lon_out = np.asarray(lon_out,dtype='float').reshape((len_days_TC,1))
alt_out = np.asarray(alt_out,dtype='float').reshape((len_days_TC,1))
grid_TC_or_NTC = np.asarray(grid_TC_or_NTC).reshape((len_days_TC,1))    
#%% random select data
data_cat = np.concatenate((lat_out, lon_out, alt_out, p_out[:,8:14], t_out[:,8:14], dtd_out[:,8:14], rh_out[:,8:14], M_out[:,8:14], wd_out[:,8:14], ws_out[:,8:14],grid_TC_or_NTC), axis=1)
data_cat_final = data_cat[~np.isnan(data_cat).any(axis=1)]
#data_cat_final = data_cat_final[np.where(data_cat_final[:,2]>=100)]
#%% 
import random
state = 'valid'
if state == 'valid':
    indices =  [i for i, val in enumerate(data_cat_final[:,45]) if val == 0]
    num_grids_to_remove = int(len(indices)*(3/4))
    random_indices = random.sample(indices, num_grids_to_remove)
    data_cat_final_1 = np.delete((data_cat_final),random_indices,axis=0)
#%% output
header = [ 'lat', 'lon', 'alt', 
           'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 
           't8', 't9', 't10', 't11', 't12', 't13', 
           'dtd8', 'dtd9', 'dtd10', 'dtd11', 'dtd12', 'dtd13', 
           'rh8', 'rh9', 'rh10', 'rh11', 'rh12', 'rh13', 
           'M8', 'M9', 'M10', 'M11', 'M12', 'M13', 
           'wd8', 'wd9', 'wd10', 'wd11', 'wd12', 'wd13', 
           'ws8', 'ws9', 'ws10', 'ws11', 'ws12', 'ws13',
           'TC_or_NTC'
         ]
data_output = data_cat_final_1.copy()

with open('D:/D_file\\Research assistance\\Machine learning\\OBS_DATA\\OBS_DATA_FOR_ML.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)
    # write multiple rows
    writer.writerows(data_output)
#%%
filename ='D:/D_file\\Research assistance\\Machine learning\\OBS_DATA\\test\\New_pred.csv'
pre = pd.read_csv(filename, delimiter= ',')
#data_cat = np.concatenate((lat_out, lon_out, alt_out, p_out[:,0:12], t_out[:,0:12], dtd_out[:,0:12], rh_out[:,0:12], M_out[:,0:12], wd_out[:,0:12], ws_out[:,0:12], grid_TC_or_NTC), axis=1)
data_cat_final = data_cat[~np.isnan(data_cat).any(axis=1)] 
obs_TC = data_cat_final[:,45]#[np.where((data_cat_final[:,2]>=100) & (data_cat_final[:,1]<120.9) & (data_cat_final[:,0]<24))]   
pre_TC = np.asarray(pre['TC/NTC (1/0)'])#[np.where((data_cat_final[:,2]>=100) & (data_cat_final[:,1]<120.9) & (data_cat_final[:,0]<24))]
data_cat_final = data_cat_final#[np.where((data_cat_final[:,2]>=100) & (data_cat_final[:,1]<120.9) & (data_cat_final[:,0]<24))  ]

a = ((pre_TC==1) & (obs_TC==1)).sum()    
b = ((pre_TC==1) & (obs_TC==0)).sum()    
c = ((pre_TC==0) & (obs_TC==1)).sum()    
d = ((pre_TC==0) & (obs_TC==0)).sum()    

POD = a/(a+c)
bias = (a+b)/(a+c)
false = b/(a+b)
relation = a/(a+b+c)
print(POD, bias,false,relation)
#%%
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import NaturalEarthFeature

lat_TC_obs =  data_cat_final[:,0][np.where(obs_TC==1)]; lon_TC_obs =  data_cat_final[:,1][np.where(obs_TC==1)]
lat_TC_pre =  data_cat_final[:,0][np.where((pre_TC==1) & (obs_TC==1))]; lon_TC_pre =  data_cat_final[:,1][np.where((pre_TC==1) & (obs_TC==1))]
lat_all =  data_cat_final[:,0]; lon_all =  data_cat_final[:,1]
alt_pre = data_cat_final[:,2][np.where(obs_TC==1)]

fig = plt.figure(figsize=(12,6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cf.BORDERS)
states = NaturalEarthFeature(category="cultural", scale="10m",
                             facecolor="none",
                             name="admin_1_states_provinces_shp")
#ax.add_feature(states, linewidth=.5, edgecolor="black")
ax.coastlines('10m', linewidth=1)
lon_lim = [119.9,122.]
lat_lim = [22.8,24.6]
ax.set_extent((lon_lim[0],lon_lim[1],lat_lim[0],lat_lim[1]),crs=ccrs.PlateCarree())


#plt.scatter(x=lon_all, y=lat_all,color="k",s=70,alpha=0.1,transform=ccrs.PlateCarree()) ## Important
plt.scatter(x=lon_TC_obs, y=lat_TC_obs,color="b",s=50,alpha=0.3,transform=ccrs.PlateCarree()) ## Important
plt.scatter(x=lon_TC_pre, y=lat_TC_pre,color="r",s=10,alpha=0.5,transform=ccrs.PlateCarree()) ## Important

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='grey', alpha=0.8, linestyle='--')
gl.xlabels_top = False
gl.ylabels_left = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

plt.show()
