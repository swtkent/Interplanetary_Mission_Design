from astro_constants import Meeus, PLANETS
from imd import ephem, ls
import numpy as np
from math import nan
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Set


class Planet_Info:
    def __init__(self, name, date_range: Set):
        planet_info = PLANETS[name]
        self.radius = planet_info['radius']
        self.sgp = planet_info['mu']
        self.name = planet_info['name']
        self.period = planet_info['period']
        self.set_dates(date_range)

    def set_dates(self, date_range):
        # can be used to update date range as needed
        self.dates = date_range


class Planet_Transfer:
    '''
    Used for Porkchop plot and final trajectory design to have all information needed for creating plots and telling if transfers are valid
    '''
    def __init__(self, planet1, planet2, leg_num):
        self.planet1 = planet1
        self.planet2 = planet2
        self.name = f'{self.planet1.name}-{self.planet2.name}'
        self.leg_number = leg_num

    def reduce_dates_by_date(self):
        first_date_l1 = min(self.planet1.date_range)
        last_date_l2 = max(self.planet2.date_range)
        self.planet1.date_range = [i for i in self.planet1.date_range if i < last_date_l2]
        self.planet2.date_range = [i for i in self.planet1.date_range if i > first_date_l1]
        assert self.planet1.date_range, f"{self.planet1.name} has no dates before {self.planet2.name}'s latest"
        assert self.planet2.date_range, f"{self.planet2.name} has no dates after {self.planet1.name}'s earliest"

    def set_post_search(self, transfer_tof, vinfp_C3, vinfm):
        self.tof = transfer_tof
        self.vinf_minus = vinfm
        if self.leg_number != 1:
            self.vinf_plus = vinfp_C3
        else:
            self.C3 = vinfp_C3


class Multi_Planet_Transfer:
    '''
    Used to find valid transfers when combining multiple transfers
    '''
    def __init__(self, planet_list: List[Planet_Info], max_c3=None,
                 max_dvinf=1, max_vinfm=None, max_tof=None):
        # TODO: check if that 1 is m/s or km/s
        self.planet_transfers = []
        for i, (planet1, planet2) in enumerate(zip(self.planet_list[:-1], self.planet_list[1:])):
            self.planet_transfers.append(Planet_Transfer(planet1, planet2, i))
        self.max_c3 = max_c3
        self.max_dvinf = max_dvinf
        self.max_vinfm = max_vinfm
        self.max_tof = max_tof

    def reduce_dates_by_tof(self):
        for pt in self.planet_transfers:
            pass

    def reduce_dates_by_final_vinf(self):
        if self.max_vinfm is not None:
            pass

    def reduce_dates_by_c3(self):
        # find what dates work for the max c3
        pass


def pcp(planet1, planet2, leg_number, contour_info, num_ticks=11,
        n_revs=0, plot=True):
    '''
    initial departure day, how many days after want values, initial arrival
    date, how many days after want values, planet leaving from, planet
    arriving to, how often want a measurement
    assume everything already in JD
    '''
    C3 = np.empty(shape=[len(planet2.dates), len(planet1.dates)])
    C3[:] = nan  # C3 of departure
    v_inf = np.copy(C3)  # V_infinity at arrival
    tof = np.copy(C3)  # time of flight in days

    # loop departure
    for d, dep_date in enumerate(planet1.dates):
        # loop arrival
        for a, arr_date in enumerate(planet2.dates):
            # find time of flight
            tof[a, d] = arr_date-dep_date

            # ignore bad departure and arrivals
            if tof[a, d] > 0:
                # solve for transfer
                r1_vec, v1_vec = ephem(planet1.name, dep_date)
                r2_vec, v2_vec = ephem(planet2.name, arr_date)
                v0_vec, vf_vec = ls(r1_vec, r2_vec, tof[a, d], n_revs)

                # collect the mission parameters
                C3[a, d] = np.linalg.norm(v1_vec-v0_vec)**2
                v_inf[a, d] = np.linalg.norm(v2_vec-vf_vec)
    dep_dates, arr_dates = np.meshgrid(planet1.dates, planet2.dates)

    # choose proper terminology for plotting
    if leg_number > 1:
        dep_con = r'$V_\infty^+ (km/s)$'
        trans = 'Transfer'
        C3 = np.sqrt(C3)
    else:
        dep_con = r'$C_3 (km^2/s^2)$'
        trans = 'Launch'

    # actual plot
    if plot:
        C3_con, vinf_con, tof_con = contour_info
        fig = plt.figure()
        fig.set_label(f'Leg {leg_number}')
        fig.canvas.manager.set_window_title(f'Leg {leg_number}')
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        c3_lines = ax.contour(planet1.dates, planet2.dates, C3, C3_con,
                              levels=contour_info[0], colors='r')
        ax.clabel(c3_lines, c3_lines.levels, inline=True)
        h_c3, _ = c3_lines.legend_elements()
        vinf_lines = ax.contour(planet1.dates, planet2.dates, v_inf, vinf_con,
                                levels=contour_info[1], colors='b')
        ax.clabel(vinf_lines, vinf_lines.levels, inline=True)
        h_vinf, _ = vinf_lines.legend_elements()
        tof_lines = ax.contour(planet1.dates, planet2.dates, tof, tof_con,
                               levels=contour_info[2], colors='k')
        ax.clabel(vinf_lines, vinf_lines.levels, inline=True)
        h_tof, _ = tof_lines.legend_elements()

        plt.xlabel(f'{planet1.name} Departure Date')
        plt.ylabel(f'{planet2.name} Arrival Date')
        plt.xticks(np.linspace(min(planet1.dates), max(planet1.dates), num_ticks), rotation=45)
        ax.set_xticklabels(pd.to_datetime(
            np.linspace(min(planet1.dates), max(planet1.dates), num_ticks),
            unit='D', origin='julian').strftime("%m/%d/%Y"))
        plt.yticks(
            np.linspace(min(planet2.dates), max(planet2.dates), num_ticks))
        ax.set_yticklabels(pd.to_datetime(
            np.linspace(min(planet2.dates), max(planet2.dates), num_ticks),
            unit='D', origin='julian').strftime("%m/%d/%Y"))
        ax.legend([h_c3[0], h_vinf[0], h_tof[0]],
                  [dep_con+f' @ {planet1.name}',
                   r'$V_\infty^-$'+f'(km/s) @ {planet2.name}', 'TOF (days)'])
        plt.grid
        plt.title(f'{planet1.name}-{planet2.name} ' + trans)
        planet_transfer = Planet_Transfer(planet1, planet2).set_pos_search(
            transfer_tof=tof, leg_num=leg_number, vinfp_C3=C3, vinfm=v_inf)
    return planet_transfer
