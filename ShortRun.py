import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.stats import cauchy

### Define functions ###

def getCauchy(size, mag):
  num = cauchy.rvs(size=1,loc=0, scale=2)
  if num > 25 or num < -25:
    while num > 25 or num < -25:
      num = cauchy.rvs(size=1,loc=0, scale=1)
    return num * mag
  else:
    return num * mag

def PRW_constantvelocity(tp, v_mag, dt, T):
  t = 0
  time = [t]
  pos_x = 0
  pos_y = 0 
  all_pos_x = [pos_x]
  all_pos_y = [pos_y]

  vc = getCauchy(1,v_mag)[0]

  prev_theta = np.random.uniform(-np.pi,np.pi)
  r = vc * dt

  pos_x += r * np.sin(prev_theta)
  pos_y += r * np.cos(prev_theta)
  t += dt
  time.append(t)
  all_pos_x.append(pos_x)
  all_pos_y.append(pos_y)

  std_dev = np.sqrt(2*dt/tp)

  while t<T:
    vc = getCauchy(1,v_mag)[0]
    r = vc * dt
    d_theta = np.random.normal(0,std_dev)
    theta = prev_theta + d_theta
    pos_x += r * np.sin(theta)
    pos_y += r * np.cos(theta)
    t += dt
    time.append(t)
    all_pos_x.append(pos_x)
    all_pos_y.append(pos_y)
    prev_theta = theta
    
  return all_pos_x, all_pos_y, time

def calc_vel(arr):
  series = pd.Series(arr)
  series_smooth = series.rolling(3,center=True).mean()
  series_smooth_dx = series_smooth.diff()
  vel = series_smooth_dx.rolling(2).mean().shift(-1)
  vel[np.isnan(vel)] = 0
  return vel

def calc_window(track, window_size, param):
  length = (track_length)//window_size

  vx = []

  begin = 2
  end = window_size
  for j in range(length):
    vx.append(np.average((track[param]).tolist()[begin:end]))
    begin = end
    end += window_size

  return vx

def xcorr(dfraw):
  df = dfraw.dropna()
  v1 = np.asarray(df.iloc[:,0])  
  v2 = np.asarray(df.iloc[:,1])
  v1 = v1 - v1.mean()
  v2 = v2 - v2.mean()
  length = len(df)
  poslagsmean=[]
  neglagsmean=[]
  Nposlags=[]
  Nneglags=[]
  for lag in range(length):
    poslags =  v2[lag:]*v1[0:length-lag] 
    neglags =  v2[0:length-lag]*v1[lag:] 
    poslagsmean.append(np.nanmean(poslags))
    neglagsmean.append(np.nanmean(neglags))
    Nposlags.append(sum(~np.isnan(poslags)))
    Nneglags.append(sum(~np.isnan(neglags)))

  return np.asarray(poslagsmean[0:track_length-2])/poslagsmean[0], np.asarray(Nposlags[0:track_length-2]), np.asarray(neglagsmean[0:track_length-2])/neglagsmean[0], np.asarray(Nneglags[0:track_length-2])

def make_comb_df(vx, vy):
  d = {'vx': vx, 'vy': vy}
  vel_df=pd.DataFrame(data=d)
  combined = pd.concat([vel_df[['vx','vy']].reset_index(drop=True), vel_df[['vx','vy']].reset_index(drop=True)], axis = 1 )
  return combined

### Define parameter values ###

#persistence time (in hours)
tp = 1

#std dev of vel from normal distribution for vx and vy
v_mag = 15

#time interval (in hours)
dt = 0.1

#total time (in hours)
T = 3

#number of walkers 
Nwalkers = 113

track_length = int(T/dt)

num_experiments = 1000

### run sim for specified number of experiments ###

all_data = [] #length is number of experiments, and within each index is a list of length number of cells that contains dataframes of each cell's position and speed
for j in range (0,num_experiments):
  data = [] #list for each experiment that is length of the number of cells and contains dataframe for each cell
  for i in range(0,Nwalkers):
    x,y,time = PRW_constantvelocity(tp,v_mag,dt,T)
    x_vel = calc_vel(x)
    y_vel = calc_vel(y)
    vel = np.sqrt(x_vel**2 + y_vel**2)
    onewalker = {'x': x, 'y': y, 'vx': x_vel, 'vy': y_vel, 'v': vel}
    onewalker_df = pd.DataFrame(data=onewalker)
    data.append(onewalker_df)

  all_data.append(data)

### find averages across experiments ###

avg_data = [] #length is number of cells and it contains dataframes of averages across each experiment
for j in range(0,Nwalkers):
  cellnum_data = [] #list of length number of experiments that will store dataframe of cell iteration for each experiment
  for i in range(0,num_experiments):
    cellnum_data.append(all_data[i][j])
  df_concat = pd.concat(cellnum_data)
  by_row_index = df_concat.groupby(df_concat.index)
  df_means = by_row_index.mean()
  avg_data.append(df_means)

# ### velocity histogram of all experiments ###
# for arr in all_data:
#   v = []
#   for i in range(len(arr[i])):
#     v.append(arr[i]['v'].tolist()[2:track_length-2])

#   v = np.concatenate(v).ravel()
#   #plt.figure()
#   plt.hist(v,bins=50)
#   plt.title('velocity')
# #plt.savefig('/content/cellsim_stationarity/vel_hist_short.png')

### velocity histogram averaged across experiments ###
v = []
for i in range(len(avg_data)):
  v.append(avg_data[i]['v'].tolist()[2:track_length-2])

v = np.concatenate(v).ravel()

plt.hist(v,bins=50)
plt.title('velocity')
plt.savefig('/content/cellsim_stationarity/vel_hist_short.png')

# ### vx histogram of all experiments ###
# for arr in all_data:
#   vx = []
#   for i in range(len(arr[i])):
#     vx.append(np.array(arr[i]['vx'].tolist()[2:track_length-2]))

#   vx = np.concatenate(vx).ravel()

#   plt.hist(vx,bins=50)
#   plt.title('vx')
# #plt.savefig('/content/cellsim_stationarity/velx_hist_short.png')

### vx and vy histogram averaged across experiments ###
vx = []
vy = []
for i in range(len(avg_data)):
  vx.append(np.array(avg_data[i]['vx'].tolist()[2:track_length-2]))
  vy.append(np.array(avg_data[i]['vy'].tolist()[2:track_length-2]))

vx = np.concatenate(vx).ravel()
vy = np.concatenate(vy).ravel()

plt.hist(vx,bins=50)
plt.title('vx')
plt.savefig('/content/cellsim_stationarity/velx_hist_short.png')

plt.hist(vy,bins=50)
plt.title('vy')
plt.savefig('/content/cellsim_stationarity/vely_hist_short.png')

# ### vy histogram of all experiments ###
# for arr in all_data:
#   vy = []
#   for i in range(len(arr[i])):
#     vy.append(np.array(arr[i]['vy'].tolist()[2:track_length-2]))

#   vy = np.concatenate(vy).ravel()

#   plt.hist(vy,bins=50)
#   plt.title('vy')
# #plt.savefig('/content/cellsim_stationarity/vely_hist_short.png')

# ### dx histogram of all experiments ###
# for arr in all_data:
#   dx = []
#   for i in range(len(arr)):
#     dx.append(np.diff(np.array(arr[i]['x'].tolist())))

#   dx = np.concatenate(dx).ravel()

#   plt.hist(dx,bins=50)
#   plt.title('dx')
# #plt.savefig('/content/cellsim_stationarity/dx_hist_short.png')

# ### dy historam of all experiments ###
# for arr in all_data:
#   dy = []
#   for i in range(len(arr)):
#     dy.append(np.diff(np.array(arr[i]['y'].tolist())))

#   dy = np.concatenate(dy).ravel()

#   plt.hist(dy,bins=50)
#   plt.title('dy')
# #plt.savefig('/content/cellsim_stationarity/dy_hist_short.png')

### dx and dy histogram averaged across experiments ###
dx = []
dy = []
for i in range(len(avg_data)):
  dx.append(np.diff(np.array(avg_data[i]['x'].tolist())))
  dy.append(np.diff(np.array(avg_data[i]['y'].tolist())))

dx = np.concatenate(dx).ravel()
dy = np.concatenate(dy).ravel()

plt.hist(dx,bins=50)
plt.title('dx')
plt.savefig('/content/cellsim_stationarity/dx_hist_short.png')

plt.hist(dy,bins=50)
plt.title('dy')
plt.savefig('/content/cellsim_stationarity/dy_hist_short.png')

# ### theta histogram of all experiments ###
# for arr in all_data:
#   theta = []

#   for i in range(len(arr)):
#     theta_val = []
#     for j in range(len(arr[i])):
#       theta_val.append(math.atan2(arr[i]['y'][j],arr[i]['x'][j]))
#     theta.append(theta_val)

#   theta = np.concatenate(theta).ravel()

#   plt.hist(theta,bins=50)
#   plt.title('theta')
# #plt.savefig('/content/cellsim_stationarity/theta_hist_short.png')

### theta histogram averaged across experiments ###
theta = []

for i in range(len(avg_data)):
  theta_val = []
  for j in range(len(avg_data[i])):
    theta_val.append(math.atan2(avg_data[i]['y'][j],avg_data[i]['x'][j]))
  theta.append(theta_val)

theta = np.concatenate(theta).ravel()

plt.hist(theta,bins=50)
plt.title('theta')
plt.savefig('/content/cellsim_stationarity/theta_hist_short.png')

### average speed_x over time ###
vxlag=[]
for df in avg_data:
  vxlag.append((df['vx']).tolist()[2:track_length-2])

plt.plot(np.average(vxlag,axis=0))
plt.title('Average speed_x over time')
plt.xlabel('time (10 min)')
plt.ylabel('average speed_x')
plt.savefig('/content/cellsim_stationarity/velx_avg_short.png')

### average speed_y over time ###
vylag=[]
for df in avg_data:
  vylag.append((df['vy']).tolist()[2:track_length-2])

plt.plot(np.average(vylag,axis=0))
plt.title('Average speed_y over time')
plt.xlabel('time (10 min)')
plt.ylabel('average speed_y')
plt.savefig('/content/cellsim_stationarity/vely_avg_short.png')

### average speed over time ###
vlag=[]
for df in avg_data:
  vlag.append((df['v']).tolist()[2:track_length-2])

plt.plot(np.average(vlag,axis=0))
plt.title('Average speed over time')
plt.xlabel('time (10 min)')
plt.ylabel('average speed')
plt.savefig('/content/cellsim_stationarity/vel_avg_short.png')

### speed_x window averaged over time ###
window_size = 3

vx_winsize3 = []

for df in avg_data:
  vx = calc_window(df, window_size, 'vx')
  vx_winsize3.append(vx)
  plt.plot(vx)

plt.xlabel(r'integer of window number')
plt.ylabel('mean of vx with window size 3')
plt.savefig('/content/cellsim_stationarity/velx_winmean_short.png')

#average of average means for lag
plt.plot(np.average(vx_winsize3,axis=0))
plt.ylabel('cell avgeraged mean of vx of window size 3')
plt.xlabel(r'Integer of window number')
plt.savefig('/content/cellsim_stationarity/velx_avgwinmean_short.png')

### speed_y window averaged over time ###
window_size = 3

vy_winsize3 = []

for df in avg_data:
  vy = calc_window(df, window_size, 'vy')
  vy_winsize3.append(vy)
  plt.plot(vy)

plt.xlabel(r'integer of window number')
plt.ylabel('mean of vy with window size 3')
plt.savefig('/content/cellsim_stationarity/vely_winmean_short.png')

#average of average means for lag
plt.plot(np.average(vy_winsize3,axis=0))
plt.ylabel('cell avgeraged mean of vy of window size 3')
plt.xlabel(r'Integer of window number')
plt.savefig('/content/cellsim_stationarity/vely_avgwinmean_short.png')

### speed window averaged over time ###
window_size = 3

v_winsize3 = []

for df in avg_data:
  v = calc_window(df, window_size, 'v')
  v_winsize3.append(v)
  plt.plot(v)

plt.xlabel(r'integer of window number')
plt.ylabel('mean of v with window size 3')
plt.savefig('/content/cellsim_stationarity/vel_winmean_short.png')

#average of average means for lag
plt.plot(np.average(v_winsize3,axis=0))
plt.ylabel('cell avgeraged mean of v of window size 3')
plt.xlabel(r'Integer of window number')
plt.savefig('/content/cellsim_stationarity/vel_avgwinmean_short.png')

### speed acf plot ###
poslagaverage = np.zeros(300)
Nposlagtotal = np.zeros(300)
all_ac = []
for df in avg_data:
  track=df
  combined = make_comb_df(track['v'].to_list()[2:track_length-2],track['v'].to_list()[2:track_length-2])
  combined = combined.dropna()
  poslagsmean, Nposlags, neglagsmean, Nneglags = xcorr(combined)

  #remove nans here
  poslagsmean[np.isnan(poslagsmean)] = 0
  all_ac.append(poslagsmean)
  poslagaverage[0:len(poslagsmean)] += poslagsmean #Nposlags*poslagsmean
  #Nposlagtotal[0:len(Nposlags)] += Nposlags
poslagaverage /= len(data)# Nposlagtotal 

std_err = np.std(all_ac,axis=0)/np.sqrt(np.shape(all_ac)[0])

#plt.plot(poslagaverage,label = "positive lag")
plt.hlines(y=0,xmin=0,xmax=100)
plt.xlim(0,track_length-2)
plt.ylim(-1,1)
plt.errorbar(np.arange(0,track_length-4),poslagaverage[0:track_length-4],yerr=std_err)
plt.xlabel("time lag")
plt.title(" Autocorrelation speed")
plt.savefig('/content/cellsim_stationarity/speed_acf_avg_short.png')

### speed_x acf plot ###
poslagaverage = np.zeros(300)
Nposlagtotal = np.zeros(300)
all_ac = []
for df in avg_data:
  track=df
  combined = make_comb_df(track['vx'].to_list()[2:track_length-2],track['vx'].to_list()[2:track_length-2])
  combined = combined.dropna()
  poslagsmean, Nposlags, neglagsmean, Nneglags = xcorr(combined)

  #remove nans here
  poslagsmean[np.isnan(poslagsmean)] = 0
  all_ac.append(poslagsmean)
  poslagaverage[0:len(poslagsmean)] += poslagsmean #Nposlags*poslagsmean
  #Nposlagtotal[0:len(Nposlags)] += Nposlags
poslagaverage /= len(data)# Nposlagtotal 

std_err = np.std(all_ac,axis=0)/np.sqrt(np.shape(all_ac)[0])

#plt.plot(poslagaverage,label = "positive lag")
plt.hlines(y=0,xmin=0,xmax=100)
plt.xlim(0,track_length-2)
plt.ylim(-1,1)
plt.errorbar(np.arange(0,track_length-4),poslagaverage[0:track_length-4],yerr=std_err)
plt.xlabel("time lag")
plt.title(" Autocorrelation speed_x")
plt.savefig('/content/cellsim_stationarity/speedx_acf_avg_short.png')

### speed_y acf plot ###
poslagaverage = np.zeros(300)
Nposlagtotal = np.zeros(300)
all_ac = []
for df in avg_data:
  track=df
  combined = make_comb_df(track['vy'].to_list()[2:track_length-2],track['vy'].to_list()[2:track_length-2])
  combined = combined.dropna()
  poslagsmean, Nposlags, neglagsmean, Nneglags = xcorr(combined)

  #remove nans here
  poslagsmean[np.isnan(poslagsmean)] = 0
  all_ac.append(poslagsmean)
  poslagaverage[0:len(poslagsmean)] += poslagsmean #Nposlags*poslagsmean
  #Nposlagtotal[0:len(Nposlags)] += Nposlags
poslagaverage /= len(data)# Nposlagtotal 

std_err = np.std(all_ac,axis=0)/np.sqrt(np.shape(all_ac)[0])

#plt.plot(poslagaverage,label = "positive lag")
plt.hlines(y=0,xmin=0,xmax=100)
plt.xlim(0,track_length-2)
plt.ylim(-1,1)
plt.errorbar(np.arange(0,track_length-4),poslagaverage[0:track_length-4],yerr=std_err)
plt.xlabel("time lag")
plt.title(" Autocorrelation speed_y")
plt.savefig('/content/cellsim_stationarity/speedy_acf_avg_short.png')