from imd import PCP, flyby_diff, PCPSearch3, Planet_Info, pcp_search3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import math


#PCP dates given
E_date_1 = 2453714.5
E_date_2 = 2453794.5
J_date1_1 = 2454129.5
J_date1_2 = 2454239.5
J_date2_1 = 2454129.5
J_date2_2 = 2454239.5
P_date_1 = 2456917.5
P_date_2 = 2457517.5

#launch - JGA PCP
tic = time.perf_counter()
res = 5.0
Earth = Planet_Info('Earth', np.arange(E_date_1, E_date_2, res))
Jupiter1 = Planet_Info('Jupiter', np.arange(J_date1_1, J_date1_2, res))
EJ_contour_info = [range(100,250,10), np.arange(10,20,.5), range(200,600,25)]
EJ_transfer = PCP(Earth, Jupiter1,1, EJ_contour_info)

#JGA - arrival PCP
res = 10.0
Jupiter2 = Planet_Info('Jupiter', np.arange(J_date2_1, J_date2_2, res))
Pluto = Planet_Info('Pluto', np.arange(P_date_1, P_date_2, res))
JP_contour_info = [np.arange(1,30,.5), np.arange(1,30,.5), range(2500, 4000, 50)]
JP_transfer = PCP(Jupiter2, Pluto, 2, JP_contour_info)
toc = time.perf_counter()
print(f'Time: {tic-toc:0.4f} s')

## 2, refining

#dates
E_date_1 = pd.Timestamp(year=2006, month=1, day=9).to_julian_date()
E_date_2 = E_date_1
J_date1_1 = 2454129.5
J_date1_2 = 2454239.5
J_date2_1 = 2454129.5
J_date2_2 = 2454239.5
P_date_1 = 2456917.5
P_date_2 = 2457517.5

leg_dates = np.array([[E_date_1,E_date_2],[J_date1_1,J_date1_2],[P_date_1,P_date_2]])

#flyby planet specifications
Jrad = 71492 
muJ = 1.266865361e8

C3_max = 180
conditions = [0.5,0]
vinf_max = 14.5
planet_strings = ['Earth','Jupiter','Pluto']

#do low res
valid_traj = pcp_search3(leg_dates,planet_strings,C3_max,conditions,vinf_max,5)
plt.show()

new_dates = np.array([  [np.min(valid_traj[:, valid_traj.shape[1]/2+1]), np.max(valid_traj[:, valid_traj.shape[1]/2+1])],
                        [np.min(valid_traj[:, valid_traj.shape[1]/2+2]), np.max(valid_traj[:, valid_traj.shape[1]/2+2])],
                        [np.min(valid_traj[:, valid_traj.shape[1]/2+3]), np.max(valid_traj[:, valid_traj.shape[1]/2+3])]])
plt.figure(3)
plt.subplot(3,1,1)
plt.scatter(valid_traj[3,:],valid_traj[0,:], marker='.')
plt.subplot(3,1,2)
plt.scatter(valid_traj[4,:],valid_traj[1,:], marker='.')
plt.subplot(3,1,3)
plt.scatter(valid_traj[5,:],valid_traj[2,:], marker='.')
        
#do high res
new_con = conditions
new_con[0] = 0.1
new_valid_traj = pcp_search3(new_dates,planet_strings,C3_max,new_con,vinf_max,1)
plt.figure(4)
ax1 = plt.subplot(3,1,1)
plt.scatter(new_valid_traj[3,:],new_valid_traj[0,:], marker='.')
plt.title('Earth Departure')
plt.xticks(np.linspace(np.min(new_valid_traj[3,:]), np.max(new_valid_traj[3,:]),5))
ax1.set_xticklabels(str(pd.to_datetime(np.linspace(np.min(new_valid_traj[3,:]),np.max(new_valid_traj[3,:]),5), unit='D', origin='julian')))
plt.xlabel('Date')
plt.ylabel(r'$C_3 (km^2/s^2)$')
ax2 = plt.subplot(3,1,2)
plt.scatter(new_valid_traj[4,:],new_valid_traj[1,:], marker='.')
plt.title('Jupiter Flyby')
plt.xticks(np.linspace(min(new_valid_traj[4,:]),max(new_valid_traj[4,:]),5))
ax2.set_xticklabels(str(pd.to_datetime(np.linspace(min(new_valid_traj[4,:]),max(new_valid_traj[4,:]),5), unit='D', origin='julian')))
plt.xlabel('Date')
plt.ylabel(r'$|\DeltaV_{\infty,diff}| (km/s)$')
ax3 = plt.subplot(3,1,3)
plt.scatter(new_valid_traj[5,:],new_valid_traj[2,:], marker='.')
plt.title('Pluto Arrival')
plt.xticks(np.linspace(min(new_valid_traj[5,:]),max(new_valid_traj[5,:]),5))
ax3.set_xticklabels(str(pd.to_datetime(np.linspace(min(new_valid_traj[5,:]),max(new_valid_traj[5,:]),5), unit='D', origin='julian')))
plt.xlabel('Date')
plt.ylabel(r'$V_\infty^- (km/s)$')

#find the minimums for the question
C3min = min(new_valid_traj[0,:])
arr_min = pd.to_datetime(min(new_valid_traj[:,-1]), unit='D', origin='julian')
vinf_min = min(new_valid_traj[:,new_valid_traj.shape[1]/2])

#do a cost function of the trajectories
def costfn(weights,trajectories):

    #split the trajectories into the weighted components
    C3 = trajectories[:,0]
    dv_inf = sum(trajectories[:,2:trajectories.shape[1]/2-1],2)
    vinf_arr = trajectories[:,trajectories.shape[1]/2]
    TOF = trajectories[:,-1] - trajectories[:,trajectories[1]/2+1]
    factors = [C3,dv_inf,vinf_arr,TOF]

    #find the costby multiplying by weight and normalizing
    cost = (factors/max(factors))@weights.transpose()

    #concatenate
    traj_cost = np.concatenate([cost,trajectories], axis=0)

    #sort rows
    traj_cost = traj_cost[:, traj_cost[0, :].argsort()]

    #turn julian dates to datetime
    values = traj_cost[:,1:math.ceil(traj_cost.shape[1]/2)]
    dates = pd.to_datetime(traj_cost[:, math.ceil(traj_cost.shape[1]/2)+1:-1], unit='D', origin='julian')
    return values, dates
weights = [3,4,2,1]
[cost_vals,dates] = costfn(weights,valid_traj)

#report the minimum cost
min_cost = cost_vals[1,:]
min_cost_dates = dates[1,:]

#want the actual vinf_in and vinf_out values
[vinf_in,vinf_out] = flyby_diff(planet_strings,pd.to_julian_date(min_cost_dates))
toc = time.perf_counter()