'''
File to hold all functions used for Interplanetary Mission Design
'''

#imports
from astro_constants import MU_SUN_KM, AU_KM, Meeus, PLANETS, EARTH_MOON_DIST_KM, EARTH_PERIOD_DAYS
import numpy as np
from scipy.optimize import fminbound
import matplotlib.pyplot as plt
import math
import pandas as pd

#aliases
sind = lambda degrees: np.sin(np.deg2rad(degrees))
cosd = lambda degrees: np.cos(np.deg2rad(degrees))


class Planet_Info:
    def __init__(self, name, date_range):
        planet_info = PLANETS[name]
        self.radius = planet_info['radius']
        self.sgp = planet_info['mu']
        self.name = planet_info['name']
        self.period = planet_info['period']
        self.set_dates(date_range)

    def set_dates(self, date_range):
        #can be used to update date range as needed
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

    def set_post_search(self, transfer_tof, vinfp_C3, vinfm):
        self.tof = transfer_tof
        self.vinf_minus = vinfm
        if self.leg_number!=1:
            self.vinf_plus = vinfp_C3
        else:
            self.C3 = vinfp_C3



def sv_from_coe(oe, mu=MU_SUN_KM):
    '''
    ------------------------------------------------------------

    This function computes the classical orbital elements (coe)
    from the state vector (R,V) using Algorithm 4.1, but in reverse.

    mu - gravitational parameter (km**3/s**2)
    R - position vector in the geocentric equatorial frame
    (km)
    V - velocity vector in the geocentric equatorial frame
    (km)
    r, v - the magnitudes of R and V
    vr - radial velocity component (km/s)
    H - the angular momentum vector (km�2/s)
    h - the magnitude of H (km�2/s)
    incl - inclination of the orbit (rad)
    N - the node line vector (km�2/s)
    n - the magnitude of N
    cp - cross product of N and R
    RA - right ascension of the ascending node (rad)
    E - eccentricity vector
    e - eccentricity (magnitude of E)
    eps - a small number below which the eccentricity is
    considered to be zero
    w - argument of perigee (rad)
    TA - true anomaly (rad)
    a - semimajor axis (km)
    coe - vector of orbital elements [h e RA incl w TA a]
    ------------------------------------------------------------
    '''
    #unpack
    [a, e, RA, incl, w, TA] = [el for el in oe]

    #calculate h
    h = np.sqrt(mu*(a*(1-e**2)))

    #...Equations 4.37 and 4.38 (rp and vp are column vectors):
    rp = (h**2/mu) * (1/(1 + e*np.cos(TA))) * (np.cos(TA)*np.vstack((1,0,0)) + np.sin(TA)*np.vstack((0,1,0)))
    vp = (mu/h) * (-np.sin(TA)*np.vstack((1,0,0)) + (e + np.cos(TA))*np.vstack((0,1,0)))
    #...Equation 4.39:
    R3_W = np.array([[ np.cos(RA), np.sin(RA), 0.0],
    [-np.sin(RA), np.cos(RA), 0.0],
    [0.0, 0.0, 1.0]])
    #...Equation 4.40:
    R1_i = np.array([[1.0, 0.0, 0.0],
                    [0.0, np.cos(incl), np.sin(incl)],
                    [0.0, -np.sin(incl), np.cos(incl)]])
    #...Equation 4.41:
    R3_w = np.array([[np.cos(w), np.sin(w), 0.0],
                    [-np.sin(w), np.cos(w), 0.0],
                    [0.0, 0.0, 1.0]])
    #...Equation 4.44:
    Q_pX = R3_W.transpose() @ R1_i.transpose() @ R3_w.transpose()
    #...Equations 4.46 (r and v are column vectors):
    r = Q_pX @ rp
    v = Q_pX @ vp
    return r, v



def coe_from_sv(R,V,mu=MU_SUN_KM):
    #TOOK DIRECTLY FROM Curtis's Orbital Mechanics
    #only change was no global mu
    '''
    ------------------------------------------------------------

    This function computes the classical orbital elements (coe)
    from the state vector (R,V) using Algorithm 4.1.

    mu - gravitational parameter (km**3/s**2)
    R - position vector in the geocentric equatorial frame
    (km)
    V - velocity vector in the geocentric equatorial frame
    (km)
    r, v - the magnitudes of R and V
    vr - radial velocity component (km/s)
    H - the angular momentum vector (km�2/s)
    h - the magnitude of H (km�2/s)
    incl - inclination of the orbit (rad)
    N - the node line vector (km�2/s)
    n - the magnitude of N
    cp - cross product of N and R
    RA - right ascension of the ascending node (rad)
    E - eccentricity vector
    e - eccentricity (magnitude of E)
    eps - a small number below which the eccentricity is
    considered to be zero
    w - argument of perigee (rad)
    TA - true anomaly (rad)
    a - semimajor axis (km)
    coe - vector of orbital elements [h e RA incl w TA a]
    ------------------------------------------------------------
    '''
    eps = 1.e-10
    r = np.linalg.norm(R)
    v = np.linalg.norm(V)
    vr = np.dot(R.transpose(),V)/r
    H = np.cross(R,V, axisa=0, axisb=0)
    h = np.linalg.norm(H)
    #...Equation 4.7:
    incl = np.arccos(H[2]/h)
    #...Equation 4.8:
    N = np.cross(np.vstack((0,0,1)), H, axisa=0, axisb=0).transpose()
    n = np.linalg.norm(N)
    #...Equation 4.9:
    if n != 0:
        ra = np.arcnp.cos(N[0][0]/n)
        if N[1][0] < 0:
            ra = 2*np.pi - ra
    else:
        ra = 0
    #...Equation 4.10:
    E = 1/mu*((v**2 - mu/r)*R - r*vr*V)
    e = np.linalg.norm(E)
    #...Equation 4.12 (incorporating the case e = 0):
    if n != 0:
        if e > eps:
            w = np.arccos(np.dot(N.transpose(),E)/n/e)
            if E[2][0] < 0:
                w = 2*np.pi - w
        else:
            w = 0
    else:
        w = 0
    #...Equation 4.13a (incorporating the case e = 0):
    if e > eps:
        ta = np.arccos(np.dot(E.transpose(),R)/e/r)
        if vr < 0:
            ta = 2*np.pi - ta
    else:
        cp = np.cross(N,R, axisa=0, axisb=0).transpose()
        if cp[2][0] >= 0:
            ta = np.arccos(np.dot(N.transpose(),R)/n/r)
        else:
            ta = 2*np.pi - np.arccos(np.dot(N,R)/n/r)
    #...Equation 2.61 (a < 0 for a hyperbola):
    a = h**2/mu/(1 - e**2)

    return [a,e,incl,ra,w,ta]


def Ephem(planet_str,JDE,Frame='Helio'):
    '''
    Finds the ephemeris for a planet given a Julian date using the Meeus algorithm
    '''
    #give the planet, the element, and the JDE

    #get the proper element coefficients
    planet = Meeus[planet_str]

    #get the T value
    T = (JDE - 2451545.0)/36525
    Ts = np.array([1, T, T**2, T**3])

    #get the element value at the specific time
    planet_val = planet @ Ts
    L = np.deg2rad(planet_val[0])
    a = planet_val[1]*AU_KM
    e = planet_val[2]
    inc = np.deg2rad(planet_val[3])
    O = np.deg2rad(planet_val[4])
    Pi = np.deg2rad(planet_val[5])

    #AOP
    w = Pi - O

    #TA in 
    M = L-Pi
    Cen = (2*e - e**3/4 + 5/96*e**5)*np.sin(M) +\
    (5/4*e**2 - 11/24*e**4)*np.sin(2*M) + \
    (13/12*e**3 - 43/64*e**5)*np.sin(3*M) +\
    103/96*e**4*np.sin(4*M) + 1097/960*e**5*np.sin(5*M)
    nu = M+Cen

    #Cartesian Coordinates
    r_vec, v_vec = sv_from_coe([a,e,O,inc,w,nu])

    # Convert to EME2000 if necessary
    if str.lower(Frame)==str.lower('EME2000'):
        theta = 23.4393*np.pi/180
        C = np.array([[1, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            0.0, np.sin(theta), np.cos(theta)])
        r_vec = C @ r_vec
        v_vec = C @ v_vec
    
    return r_vec, v_vec


def psi_TOF(psi, r0_vec, rf_vec, mu=MU_SUN_KM):
    '''
    Find the TOF associated with a psi value
    psi is decided as (2*n*pi)**2 <= psi <= (2*(n+1)*pi)**2, where n=number of revolutions
    '''
    
    tol = 1e-6
    
    #get the real cos(delta nu)
    r0 = np.linalg.norm(r0_vec)
    rf = np.linalg.norm(rf_vec)
    #get Direction of Motion
    cosdnu = np.dot(r0_vec.transpose(), rf_vec)/(r0*rf)
    nu0 = np.arctan2(r0_vec[1][0],r0_vec[0][0])
    nuf = np.arctan2(rf_vec[1][0],rf_vec[0][0])
    dnu = nuf-nu0 # 2*np.pi
    DM=1 if dnu<np.pi else -1
    #calulate A
    A = DM*np.sqrt(r0*rf*(1+cosdnu))

    #case for hyperbolic/parabolic/elliptical orbit transfer
    if psi>tol:
        c2 = (1-np.cos(np.sqrt(psi)))/psi
        c3 = (np.sqrt(psi)-np.sin(np.sqrt(psi)))/np.sqrt(psi**3)
    elif psi<-tol:
        c2 = (1-np.cosh(np.sqrt(-psi)))/psi
        c3 = (np.sinh(np.sqrt(-psi))-np.sqrt(-psi))/np.sqrt((-psi)**3)
    else:
        c2 = 1./2
        c3 = 1./6
    
    #compute y
    y = r0 + rf + A*(psi*c3-1)/np.sqrt(c2)
    
    if A>0 and y<0:
        while y<0:
            psi=psi+0.1
            y = r0 + rf + A*(psi*c3-1)/np.sqrt(c2)
    
    #get the new dt
    chi = np.sqrt(y/c2)
    dt = (chi**3*c3+A*np.sqrt(y))/np.sqrt(mu)
    return dt/86400.


def multirev_minflight(planet_names, dates, n_revs=0):
    '''
    Find the possible TOF based on the psi value for hyperbolic through multi-rev elliptical orbits
    '''
    #starting position
    p1r, p1v = Ephem(planet_names[0], dates[0])

    #collect the 
    p2r, p2v = Ephem(planet_names[1], dates[1])

    # get the time of flight for each psi, split by rev number and plot
    plt.figure()
    for n in range(n_revs+1):
        psi_low = (2*n*np.pi)**2 if n!=0 else -4*np.pi #covers hyperbolic case
        psi_high = (2*(n+1)*np.pi)**2
        psis = np.linspace(psi_low+1e-6, psi_high-1e-6)
        tof = []
        for psi in psis:
            tof_p = psi_TOF(psi, p1r, p2r)
            tof.append(tof_p)
        plt.plot(psis, tof, label=f'{n} Revolutions')
        plt.legend()
        plt.grid()
        plt.xlabel(r'$\psi (rad^2)$')
        plt.ylabel('TOF (days)')
        plt.title('Time of Flight vs Psi')
        plt.xlim(-4*np.pi, psi_high)
        plt.ylim(0,10000)

    if n_revs>0:
        min_psi = fminbound(psi_TOF, psi_low, psi_high, args=(p1r,p2r))
        min_TOF = psi_TOF(min_psi, p1r, p2r)
        print(f'Minimum Transfer Possibility: {min_TOF} days')
        print(f'Minimum Psi: {min_psi} rad^2')


def ls(r0_vec, rf_vec, dt0_days, n_revs=0, DM=None, mu=MU_SUN_KM):
    '''
    Lambert's Problem Solver
    Given two position vectors and the TOF, finds the velocities at departure/arrival
    '''    
    #convert TOF to seconds
    dt0 = dt0_days*86400

    if np.linalg.norm(rf_vec-r0_vec)/dt0<100: #bad case where they won't really find a valid solution
        
        #find TA diff and thus motion direction
        nu0 = np.arctan2(r0_vec[1][0],r0_vec[0][0])
        nuf = np.arctan2(rf_vec[1][0],rf_vec[0][0])
        dnu = (nuf-nu0) % (2*np.pi)
        if DM is None:
            DM=1 if dnu<np.pi else -1
        
        #get the real np.cos(delta nu)
        r0 = np.linalg.norm(r0_vec)
        rf = np.linalg.norm(rf_vec)
        cosdnu = np.asscalar(np.dot(r0_vec.transpose(),rf_vec))/abs(r0*rf)
        
        #calulate A
        A = DM*np.sqrt(r0*rf*(1+cosdnu))
        
        #set psi bounds based on which side of the minimum TOF you're on
        psi_up = (2*(n_revs+1)*np.pi)**2
        if n_revs==0:
            psi_low = -4*np.pi
        else:
            psi_low = (2*n_revs*np.pi)**2
            psi_min = fminbound(psi_TOF, psi_low, psi_up, args=(r0_vec,rf_vec))
            if dnu>np.pi:
                psi_low = psi_min
            else:
                psi_up = psi_min
        psi = (psi_low+psi_up)/2
        
        #initialize while loop
        tol = 1e-6
        dt = 0 #strictly initializing
        max_it = 5e3
        k=0
        while abs(dt-dt0)>tol and k<max_it:
            
            #update c vals
            if psi>tol:
                c2 = (1-np.cos(np.sqrt(psi)))/psi
                c3 = (np.sqrt(psi)-np.sin(np.sqrt(psi)))/np.sqrt(psi**3)
            elif psi<-tol:
                c2 = (1-np.cosh(np.sqrt(-psi)))/psi
                c3 = (np.sinh(np.sqrt(-psi))-np.sqrt(-psi))/np.sqrt((-psi)**3)
            else:
                c2 = 1./2
                c3 = 1./6
            
            #compute y
            y = r0 + rf + A*(psi*c3-1)/np.sqrt(c2)
            
            if A>0 and y<0:
                while y<0:
                    psi=psi+0.1
                    y = r0 + rf + A*(psi*c3-1)/np.sqrt(c2)
            
            #get the new dt
            chi = np.sqrt(y/c2)
            dt = (chi**3*c3+A*np.sqrt(y))/np.sqrt(mu)
            
            #compute new psi
            if n_revs>0 and dnu<np.pi:
                if dt>dt0:
                    psi_low = psi
                else:
                    psi_up = psi
            else:
                if dt<dt0:
                    psi_low = psi
                else:
                    psi_up = psi
            psi = (psi_low+psi_up)/2
            
            #update iteration number
            k+=1
        
        #compute the other functions
        f = 1 - y/r0
        gdot = 1 - y/rf
        g = A*np.sqrt(y/mu)
        
        #compute outputs
        v0_vec = (rf_vec-f*r0_vec)/g
        vf_vec = (gdot*rf_vec - r0_vec)/g

    else:
        v0_vec = [math.nan]*3
        vf_vec = v0_vec
        dnu = math.nan
        psi= math.nan      

    return v0_vec, vf_vec


def flyby(vinf_in,vinf_out,mu):
    '''
    Calculates the radius of periapsis at which the flyby happens and the 
    '''
    #rp = radius of periapsis
    #psi = turn angle

    #calculate the turn angle
    psi = np.asscalar(np.arccos(np.dot(vinf_in.transpose(),vinf_out)/(np.linalg.norm(vinf_in)*np.linalg.norm(vinf_out))))

    #calculate the radius of periapsis
    vinf = np.mean([np.linalg.norm(vinf_in),np.linalg.norm(vinf_out)])
    rp = mu/vinf**2 * (1/np.cos((np.pi-psi)/2) - 1)

    return rp, psi


def flyby_diff(planet_strings,dates):
    '''
    Calculates the values of the excess velocity at arrival and departure of a planet given the information of the first, second, and third planet in the string
    '''
    #get the planet values
    r1_vec, _ = Ephem(planet_strings[0],dates[0])
    r2_vec, v2_vec = Ephem(planet_strings[1],dates[1])
    r3_vec, _ = Ephem(planet_strings[2],dates[2])

    #do the Lambert's of each and only save the velocities
    _, v2_vec1 = ls(r1_vec,r2_vec,dates[1]-dates[0])
    v2_vec2, _ = ls(r2_vec,r3_vec,dates[2]-dates[1])

    #get the vinfs of each
    vinf_in = np.linalg.norm(v2_vec1 - v2_vec)
    vinf_out = np.linalg.norm(v2_vec2 - v2_vec)
    return vinf_in, vinf_out


def res_orb(vinfm_GA1,GA1_rv,vinfp_GA2,GA2_rv, planet, min_r, XY, plot=False):
    '''
    Find the valid turn angles and radius of periapses for the two flybies of a resonant orbit
    ---------------INPUTS---------------
    vinf_GA1 = vinf_in of GA1 of planet
    GA1_rv = r_vec and v_vec of planet at GA1
    vinf_GA2 = vinf_out of GA2 of planet
    GA2_rv = r_vec and v_vec of planet at GA2
    planet - dictionary of planetary parameters needed
    min_r = planet's radius plus any extra height
    XY = ratio of planetary ervolutions per spacecraft revolutions
    Period = planet's orbital period in days
    plot_yn
    --------------OUTPUTS---------------
    rp1_acc = acceptable rp for the first flyby
    rp2_acc = acceptable rp for second flyby
    phi_acc = acceptable phi for the two
    '''

    #split up the GA rv's
    r1_vec = GA1_rv[:3,:]
    v1_vec = GA1_rv[3:,:]
    v2_vec = GA2_rv[3:,:]


    #turn Period into seconds
    Period = planet['period']*86400

    #calculate resonant orbit's semi-major axis
    P = Period*XY
    a = ((P/(2*np.pi))**2*MU_SUN_KM)**(1./3)

    #get the velocity of the spacecraft during the orbi
    #speed required to enter resonant orbit
    r1 = np.linalg.norm(r1_vec)
    vsc = np.sqrt(MU_SUN_KM*(2.0/r1 - 1.0/a))

    #get theta
    vp = np.linalg.norm(v1_vec)
    vinf = np.linalg.norm(vinfm_GA1)
    theta = np.arccos((vinf**2 + vp**2 - vsc**2) / (2*vinf*vp))

    #compute the VNC to the ecliptic
    V = v1_vec/vp
    N = np.cross(r1_vec, v1_vec, axisa=0, axisb=0).transpose()/np.linalg.norm(np.cross(r1_vec,v1_vec, axisa=0, axisb=0))
    C = np.cross(V,N, axisa=0, axisb=0).transpose()
    T = np.hstack((V,N,C))

    #phi can be anything for now
    phis = np.arange(0, 360, 0.1)
    rp1 = []
    rp2 = []
    rp1_acc = []
    rp2_acc = []
    phi_acc = []
    vinfp1_acc = np.array([], dtype=np.float64).reshape(3,0)
    vinfm2_acc = np.array([], dtype=np.float64).reshape(3,0)
    for phi in phis:
        
        #get the vnc vinf after flyby
        vinfp_GA1_vnc = vinf*np.vstack((np.cos(np.pi-theta), np.sin(np.pi-theta)*cosd(phi), -np.sin(np.pi-theta)*sind(phi)))
        
        #turn the vinf after GA1 into the ecliptic frame
        vinfp_GA1 = T @ vinfp_GA1_vnc
        
        #get the vinf before GA2
        vinfm_GA2 = vinfp_GA1 + v1_vec - v2_vec
        
        #find the radius of periapsis from these GA's
        rp1_dum = flyby(vinfm_GA1,vinfp_GA1,planet['mu'])[0]
        rp1.append(rp1_dum)
        rp2_dum = flyby(vinfm_GA2,vinfp_GA2,planet['mu'])[0]
        rp2.append(rp2_dum)

        #collect those viable
        if rp1_dum>min_r and rp2_dum>min_r:
            phi_acc.append(phi)
            rp1_acc.append(rp1_dum)
            rp2_acc.append(rp2_dum)
            vinfp1_acc = np.hstack((vinfp1_acc,vinfp_GA1))
            vinfm2_acc = np.hstack((vinfm2_acc,vinfm_GA1))

    #make plot of all of the data just created
    if plot is True:
        plt.figure()
        plt.fill_between([min(phi_acc), max(phi_acc)], [max([max(rp1_acc),max(rp2_acc)]),max([max(rp1_acc),max(rp2_acc)])], color='y')
        plt.plot(phis,rp1,label='Flyby 1', color='b')
        plt.plot(phis,rp2,label='Flyby2', color='r')
        plt.plot(phis,[min_r for _ in range(len(phis))], color='k', linestyle='--', label=r'$Min r_p$')
        plt.text(np.median(phi_acc)-(max(phi_acc)-min(phi_acc))/4,min_r-2000,r'Valid $\phi$'+'\nRange')
        plt.xlabel(r'$\phi (deg)$')
        plt.ylabel('Perigee Radius (km)')
        plt.legend()
        plt.xlim(0,360)
        plt.ylim(0,max((max(rp1_acc),max(rp2_acc))))

    return rp1_acc, rp2_acc, phi_acc, vinfp1_acc, vinfm2_acc


def bplane2(r_vec,v_vec,mu):
    '''
    Calculate the B-plane parameters of a flyby
    '''
    #Method 2 B-Plane Targeting
    # r_vec = position vector
    # v_vec = velocity vector
    # mu = standard gravitational parameter

    #math behind the hats
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    a = -mu/v**2
    e_vec = 1/mu*((v**2-mu/r)*r_vec - np.dot(r_vec.transpose(),v_vec)*v_vec)
    e = np.linalg.norm(e_vec)
    p = np.arccos(1/e)

    #all the hats
    h_hat = np.cross(r_vec,v_vec, axisa=0, axisb=0).transpose()/np.linalg.norm(np.cross(r_vec,v_vec, axisa=0, axisb=0))
    k_hat = [0,0,1]
    S_hat = e_vec/e*np.cos(p) + np.cross(h_hat,e_vec, axisa=0, axisb=0).transpose()/np.linalg.norm(np.cross(h_hat,e_vec, axisa=0, axisb=0))*np.sin(p)
    T_hat = np.cross(S_hat,k_hat, axisa=0, axisb=0).transpose()/np.linalg.norm(np.cross(S_hat,k_hat, axisa=0, axisb=0))
    R_hat = np.cross(S_hat,T_hat, axisa=0, axisb=0).transpose()
    Bhat = np.cross(S_hat,h_hat, axisa=0, axisb=0).transpose()

    #outputs
    b = abs(a)*np.sqrt(e**2-1)
    B = b*Bhat
    Bt = np.dot(B.transpose(),T_hat).item()
    Br = np.dot(B.transpose(),R_hat).item()
    theta = np.arccos(np.dot(T_hat.transpose(),B))
    theta = 2*np.pi - theta if Br<0 else theta
    return Bt, Br, b, theta


def bplane_correction(r_vec, v_vec, mu, Bt_desired, Br_desired, pert):
    '''
    Correct the B-Plane estimate to the desired B-Plane parameters
    '''
    #keep calculating until within the tolerance
    tol = 1e-6
    TCM = np.vstack((0., 0.))
    delVi = np.vstack((1e3,1e3))
    dB = np.vstack((1e5, 1e5))
    k=0
    v_orig = np.copy(v_vec)
    while np.any(abs(delVi)>tol) and np.any(abs(dB)>tol):
        k+=1
        #new nominal B parameters
        Bt_nom, Br_nom, *_ = bplane2(r_vec, v_vec, mu)

        #perturb the velocity
        Vpx = v_vec + np.vstack((pert,0,0))
        Vpy = v_vec + np.vstack((0,pert,0))
        Btpx,Brpx,*_ = bplane2(r_vec,Vpx,mu)
        Btpy,Brpy,*_ = bplane2(r_vec,Vpy,mu)
        
        #get the B-Plane partials
        dBtdVx = (Btpx-Bt_nom)/pert
        dBtdVy = (Btpy-Bt_nom)/pert
        dBrdVx = (Brpx-Br_nom)/pert
        dBrdVy = (Brpy-Br_nom)/pert
        
        #delta B
        dB = np.array([[Bt_desired-Bt_nom], [Br_desired-Br_nom]])
        
        #delta Vi
        delVi = np.linalg.inv(np.array([[dBtdVx,dBtdVy], [dBrdVx,dBrdVy]])) @ dB
        
        #update velocity vector
        v_vec += np.vstack((delVi,0.))
        
        #update the TCM movement
        TCM += delVi

    #final outputs
    return v_vec-v_orig, v_vec


def rot2inert(rot_coords, non_tspan, L=EARTH_MOON_DIST_KM):
    '''
    Convert from CR3BP to pseudo inertial frame
    '''
    thetadot = 1
    inert_coords = np.empty_like(rot_coords)
    for k, theta in enumerate(non_tspan):
        Tir = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
        Tirdot = thetadot*np.array([[-np.sin(theta), -np.cos(theta), 0],
                                    [np.cos(theta), -np.sin(theta), 0],
                                    [0, 0, 0]])
        inert_coord_pos = Tir@rot_coords[:3, k] * L
        inert_coord_vel = (Tirdot@rot_coords[:3, k] + Tir@rot_coords[3:, k]) * L
        inert_coord = np.concatenate((inert_coord_pos, inert_coord_vel), axis=0)
        inert_coords[:,k] = inert_coord
    return inert_coords


def inert2rot(inert_coords, non_tspan, L=EARTH_MOON_DIST_KM):
    '''
    convert from inertial frame to rotating frame
    '''
    thetadot = 1
    rot_coords = np.empty_like(inert_coords)
    for k, theta in enumerate(non_tspan):
        Tir = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
        Tirdot = thetadot*np.array([[-np.sin(theta), -np.cos(theta), 0],
                                    [np.cos(theta), -np.sin(theta), 0],
                                    [0, 0, 0]])
        rot_coord_pos = Tir.transpose()@inert_coords[:3, k] / L
        rot_coord_vel = (Tirdot.transpose()@inert_coords[:3, k] + Tir.transpose()@inert_coords[3:, k]) / L
        rot_coord = np.vstack((rot_coord_pos, rot_coord_vel))
        rot_coords[:,k] = rot_coord
    return rot_coords


def tisserand(planet_numbers, contours, x_limits, y_axis='ra'):
    '''
    Plots the Tisserand plot given the planets wanting to look at, the vInf, and the value wanting to compare on the Y axis
    '''

    # Planetary Info
    semiMajor = [.387, .723, 1, 1.524, 5.203, 9.537, 19.191, 30.069, 39.482]
    planet_names = ['Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune','Pluto']


	# Vinf Contours
    Vp = np.zeros_like(planet_numbers)
    alphas = np.arange(0, 180,1)
    vInfMags = contours
    plottings = np.zeros(shape=(len(alphas), len(vInfMags), len(planet_numbers)))
    rps = np.zeros(shape=(len(alphas), len(vInfMags), len(planet_numbers)))
    
    colors = ['c','r','b','g','m','y','c']
    
    plt.figure()

	
    
    #do each planet
    for k, planet_num in enumerate(planet_numbers):
        
        #do each contour line
        for j, vInfMag in enumerate(vInfMags):
		
			#get the planet's info
            R = semiMajor[planet_num]*AU_KM
            Vp = np.sqrt(MU_SUN_KM/R)
            
            for i, alpha in enumerate(alphas):
				
				#get the outgoing velocity
                vinfOut = np.vstack([vInfMag*cosd(alpha), vInfMag*sind(alpha)])
                vOut = np.vstack((Vp, 0.)) + vinfOut
				
				#get a and e for this alpha
                a,e,*_ = coe_from_sv(np.vstack((0.,R,0.)),np.vstack((vOut,np.array(0.))),MU_SUN_KM)
				
				#find the rp and ra for this alpha
                if e < 1:
					#plt.figure() out which thing to collect
                    if y_axis=='ra':
                        plotting_val = a*(1+e)/AU_KM
                    elif y_axis=='period':
                        plotting_val = 2*np.pi*np.sqrt(a**3/MU_SUN_KM)/86400 #now in days
                        if max(planet_numbers)>4:
                            plotting_val = plotting_val/EARTH_PERIOD_DAYS #now in years
                        elif y_axis=='C3':
                            plotting_val = np.linalg.norm(vinfOut)**2
                    
                    rp_val = a*(1-e)/AU_KM
                
                else:
                    plotting_val = math.nan
                    rp_val = math.nan
                plottings[i,j,k] = plotting_val
                rps[i,j,k] = rp_val
			
	#plot the contours second
    for k in range(len(planet_numbers)):
        plt.semilogy(rps[:,:,k], plottings[:,:,k], color=colors[(k%len(colors)-1)+1])
	
	#add the things that only make sense for doing rp vs ra
    if y_axis=='ra':
		#plot the planets and their names
        for j, planet_num in enumerate(planet_numbers):
            plt.scatter(semiMajor[planet_num],semiMajor[planet_num],color=colors[(j%len(colors)-1)+1],marker='x')
            plt.text(semiMajor[planet_num]+0.025,semiMajor[planet_num],planet_names[planet_num],color=colors[(j%len(colors)-1)+1])
		
		
		#also add grey dashed lines for finding planetary intersections better
        min_planet = semiMajor[planet_numbers[1]]
        max_planet = semiMajor[planet_numbers[-1]]
		#vertical lines
        for planet_num in planet_numbers:
            plt.plot(np.hstack((semiMajor[planet_num],semiMajor[planet_num])),[semiMajor[planet_num],max_planet],color='k',linestyle=':',linewidth=1)
        
		#horizontal lines
        for planet_num in planet_numbers:
            plt.plot([min_planet,semiMajor[planet_num]],np.hstack((semiMajor[planet_num],semiMajor[planet_num])),color='k',linestyle=':',linewidth=1)
    else:
        for k in range(planet_numbers):
            plt.text(semiMajor[planet_num]+0.025,min(plottings[:,:,k]),planet_names[planet_num],color=colors[(k%len(colors)-1)+1])

	#finish the plot's look
    plt.grid
    if str.lower(y_axis)!='ra':
        axis_limits = [x_limits,min(plottings),100]
    plt.xlabel(r'$R_p$ (AU)')
    plt.title([f'v_\infty shown are {contours} km/s'])
    if str.lower(y_axis)=='ra':
        plt.ylabel(r'$R_a$ (AU)')
    elif str.lower(y_axis)=='period':
        if max(planet_numbers)>4:
            plt.ylabel('Period (years)')
        else:
            plt.ylabel('Period (days)')
    elif str.lower(y_axis)=='c3':
        plt.ylabel(r'$C_3 (km^2/s^2)$')


def tisserand_path(planet_num,vInfMag):
    '''
    Calculates the path to take given the planet number and the v_inf magnitude to make it happen
    '''
    
    semiMajor = [.387, .723, 1, 1.524, 5.203, 9.537, 19.191, 30.069, 39.482]
    planet_names = ['Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune','Pluto']


    R = semiMajor[planet_num-1]*AU_KM
    Vp = np.sqrt(MU_SUN_KM/R)

    alphas = range(180)
    rp = []
    ra = []

    for alpha in alphas:
        
        #get the outgoing velocity
        vinfOut = np.vstack((vInfMag*cosd(alpha), vInfMag*sind(alpha)))
        vOut = np.vstack((Vp, 0.)) + vinfOut
        
        #get a and e for this alpha
        a, e, *_ = coe_from_sv(np.vstack((0., R, 0.)), np.vstack((vOut, np.array(0.))), mu=MU_SUN_KM)
        
        #find the rp and ra for this alpha
        if e < 1:
            
            #figure out which thing to collect
            ra.append(a*(1+e)/AU_KM)
            
            rp.append(a*(1-e)/AU_KM)
            
        else:
            ra.append(math.nan)
            rp.append(math.nan)
    return rp, ra


def pcp(planet1, planet2, leg_number, contour_info, num_ticks=11, n_revs=0, plot=True):
    #initial departure day, how many days after want values, initial arrival
    #date, how many days after want values, planet leaving from, planet
    #arriving to, how often want a measuremm
    #assume everything already in JD

    C3 = np.zeros(shape=[len(planet2.dates),len(planet1.dates)])
    v_inf = np.zeros(shape=[len(planet2.dates),len(planet1.dates)])
    tof = np.zeros(shape=[len(planet2.dates),len(planet1.dates)])


    #loop departure
    for d, dep_date in enumerate(planet1.dates):
        
        #loop arrival
        for a, arr_date in enumerate(planet2.dates):
            
            #find time of flight
            tof[a,d] = arr_date-dep_date
            
            #ignore bad departure and arrivals
            if tof[a,d]<=0:
                C3[a,d] = math.nan
                v_inf[a,d] = math.nan
                
            else:
                
                #find planets' ephemeris data
                r1_vec, v1_vec = Ephem(planet1.name,dep_date)
                r2_vec, v2_vec = Ephem(planet2.name,arr_date)
                
                #do Lambert's
                v0_vec,vf_vec = ls(r1_vec,r2_vec,tof[a,d], n_revs)
                
                #collect the mission parameters
                C3[a,d] = np.linalg.norm(v1_vec-v0_vec)**2
                v_inf[a,d] = np.linalg.norm(v2_vec-vf_vec)

    dep_dates, arr_dates = np.meshgrid(planet1.dates,planet2.dates)

    #choose proper terminology for plotting
    if leg_number>1:
        dep_con = r'$V_\infty^+ (km/s)$'
        trans = 'Transfer'
        C3 = np.sqrt(C3)
    else:
        dep_con = r'$C_3 (km^2/s^2)$'
        trans = 'Launch'

    #actual plot
    if plot==True:
        C3_con, vinf_con, tof_con = contour_info
        fig = plt.figure()
        fig.set_label(f'Leg {leg_number}')
        fig.canvas.manager.set_window_title(f'Leg {leg_number}')
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        c3_lines = ax.contour(planet1.dates, planet2.dates, C3, C3_con, levels=contour_info[0], colors='r')
        ax.clabel(c3_lines, c3_lines.levels, inline=True)
        h_c3,_ = c3_lines.legend_elements()
        vinf_lines = ax.contour(planet1.dates, planet2.dates, v_inf, vinf_con, levels=contour_info[1], colors='b')
        ax.clabel(vinf_lines, vinf_lines.levels, inline=True)
        h_vinf,_ = vinf_lines.legend_elements()
        tof_lines = ax.contour(planet1.dates, planet2.dates, tof, tof_con, levels=contour_info[2], colors='k')
        ax.clabel(vinf_lines, vinf_lines.levels, inline=True)
        h_tof,_ = tof_lines.legend_elements()    

        plt.xlabel(f'{planet1.name} Departure Date')
        plt.ylabel(f'{planet2.name} Arrival Date')
        plt.xticks(np.linspace(min(planet1.dates),max(planet1.dates),num_ticks), rotation = 45)
        ax.set_xticklabels(pd.to_datetime(np.linspace(min(planet1.dates),max(planet1.dates),num_ticks), unit='D', origin='julian').strftime("%m/%d/%Y"))
        plt.yticks(np.linspace(min(planet2.dates),max(planet2.dates),num_ticks))
        ax.set_yticklabels(pd.to_datetime(np.linspace(min(planet2.dates),max(planet2.dates),num_ticks), unit='D', origin='julian').strftime("%m/%d/%Y"))
        ax.legend([h_c3[0], h_vinf[0], h_tof[0]], [dep_con+f' @ {planet1.name}', r'$V_\infty^-$'+f'(km/s) @ {planet2.name}', 'TOF (days)'])
        plt.grid
        plt.title(f'{planet1.name}-{planet2.name} ' + trans)
        planet_transfer = Planet_Transfer(planet1, planet2, transfer_tof=tof, leg_num=leg_number, vinfp_C3=C3, vinfm=v_inf)
    
    return  planet_transfer


'''
NEED TO FIGURE OUT WHAT THESE FUNCTIONS ACTUALLY DO AND THEN WRITE UP A DOC STRING FOR EACH
'''
def lscheck(planet1_str,dep_date,planet2_str,arr_date):

    #check the TOF
    tof = arr_date - dep_date

    #ignore bad departure and arrivals
    if tof<=0: #|| TOF(a,d)>=500
        v_inf_out = math.nan
        v_inf_in = math.nan
        
    else:
        
        #find planets' ephemeris data
        r1_vec, v1_vec = Ephem(planet1_str,dep_date)
        r2_vec, v2_vec = Ephem(planet2_str,arr_date)
        
        #do Lambert's
        v0_vec, vf_vec = ls(r1_vec,r2_vec,tof)
        
        #collect the mission parameters
        v_inf_out = v0_vec-v1_vec
        v_inf_in = vf_vec-v2_vec
        
    return v_inf_out, v_inf_in


def legcheck(d1, planet_transfers, saving_info, C3_max, stopping_conditions, vinf_out, vinf_in, vinf_max, leg_num):
    #go day to day
    daterange = planet_transfers[leg_num].planet2.date_range
    for d2 in daterange:
        
        #do Lambert's of the leg
        vio, vii = lscheck(planet_transfers[leg_num].planet1.name,d1,planet_transfers[leg_num].planet2.name,d2)
        vinf_out[leg_num,:] = vio
        vinf_in[leg_num,:] = vii
        
        #check if first leg
        if leg_num!=0:            
            #put the stopping conditions in the right order
            dv_diff = stopping_conditions[leg_num-1][0]
            rp_min = stopping_conditions[leg_num-1][1]
            
            #check that the vinf_out isn't out of range of vinf_in at this date
            if abs(np.linalg.norm(vinf_out[leg_num,:]) - np.linalg.norm(vinf_in[leg_num-1,:]))<dv_diff:
                #check if the vinf's match and if it gets too close
                rp = flyby(vinf_in[leg_num-1,:], vinf_out[leg_num,:], mu=planet_transfers[leg_num].planet2.sgp)
                if rp>rp_min:
                    dv = abs(np.linalg.norm(vinf_out[leg_num,:]) - np.linalg.norm(vinf_in[leg_num-1,:]))
                    saving_info[-1,leg_num] = dv
                    saving_info[-1,len(planet_transfers)+leg_num] = d1

                    #continue if not the final leg of the mission
                    if leg_num!=planet_transfers[-1].leg_number:
                        #go to next leg_num
                        saving_info = legcheck(d2, planet_transfers, saving_info, C3_max, stopping_conditions, vinf_out, vinf_in, vinf_max, leg_num+1)
                    else:
                        #check that final arrival vinf_in isn't too big
                        if np.linalg.norm(vinf_in[leg_num,:])<vinf_max:
                            saving_info[-1,len(planet_transfers)] = np.linalg.norm(vinf_in[leg_num,:])
                            saving_info[-1,-1] = d2
                            #start with the same info as was there just before
                            saving_info[-1+1,:] = saving_info[-1,:]            
        else:
            #replace the vinf_out with C3
            C3 = np.linalg.norm(vinf_out)**2
            #check that the C3 isn't too big
            if C3<C3_max:
                #go to next leg_num
                saving_info[-1,1] = C3
                saving_info[-1,1+len(planet_transfers)] = d1
                saving_info = legcheck(d2, planet_transfers, saving_info, C3_max, stopping_conditions, vinf_out, vinf_in, vinf_max, leg_num+1)
    
    return saving_info


def pcp_search3(planet_transfers,C3_max,conditions,vinf_max):

    #reset the saved info
    vinf_out = np.zeros(shape=(len(planet_transfers)-1,3))
    vinf_in = vinf_out
    dv = np.zeros(shape=(1,vinf_in.shape[0]-1))
    dates = np.zeros(shape=(1,len(planet_transfers)))
    worthy_transfers = []

    #do the work for each leg
    date_range = planet_transfers[0].planet1.date_range
    for d1 in date_range:
        worthy_transfers = legcheck(d1, planet_transfers, worthy_transfers, C3_max, conditions, vinf_out, vinf_in, vinf_max, 1)
    return worthy_transfers
