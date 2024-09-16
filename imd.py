'''
File to hold all functions used for Interplanetary Mission Design
'''
from astro_constants import (MU_SUN_KM, AU_KM, Meeus, PLANETS,
                             EARTH_MOON_DIST_KM, EARTH_PERIOD_DAYS)
import numpy as np
from scipy.optimize import fminbound
import matplotlib.pyplot as plt
from math import nan


def sind(degrees):
    return np.sin(np.deg2rad(degrees))


def cosd(degrees):
    return np.cos(np.deg2rad(degrees))


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
    # unpack
    a, e, RA, incl, w, TA = oe

    # calculate
    h = np.sqrt(mu*(a*(1-e**2)))

    # Write initial position along (perigee?)
    rp = (h**2/mu) * (1/(1 + e*np.cos(TA))) * (np.cos(TA)*np.array((1, 0, 0))
                                                + np.sin(TA)*np.array((0, 1, 0)))
    vp = (mu/h) * (-np.sin(TA)*np.array((1, 0, 0))
                    + (e + np.cos(TA))*np.array((0, 1, 0)))

    # Rotations based on RAAN, inclination, and AOP
    R3_W = np.array([[np.cos(RA), np.sin(RA), 0.0],
                    [-np.sin(RA), np.cos(RA), 0.0],
                    [0.0, 0.0, 1.0]])
    R1_i = np.array([[1.0, 0.0, 0.0],
                    [0.0, np.cos(incl), np.sin(incl)],
                    [0.0, -np.sin(incl), np.cos(incl)]])
    R3_w = np.array([[np.cos(w), np.sin(w), 0.0],
                    [-np.sin(w), np.cos(w), 0.0],
                    [0.0, 0.0, 1.0]])
    # Combined rotation due to orbit's angles
    Q_pX = R3_W.T @ R1_i.T @ R3_w.T
    # Rotate r and v as column vectors
    r = Q_pX @ rp
    v = Q_pX @ vp
    return r, v


def coe_from_sv(R, V, mu=MU_SUN_KM):
    '''
    ------------------------------------------------------------
    This function computes the classical orbital elements (coe)
    from the state vector (R,V)

    mu - gravitational parameter (km**3/s**2)
    R - position vector in the geocentric equatorial frame
    (km)
    V - velocity vector in the geocentric equatorial frame
    (km)
    r, v - the magnitudes of R and V
    vr - radial velocity component (km/s)
    H - the angular momentum vector (km^2/s)
    h - the magnitude of H (km^2/s)
    incl - inclination of the orbit (rad)
    N - the node line vector (km^2/s)
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
    vr = np.dot(R, V)/r
    H = np.cross(R, V).flatten()
    h = np.linalg.norm(H)
    # ...Equation 4.7:
    incl = np.arccos(H[2]/h)
    # ...Equation 4.8:
    N = np.cross(np.array((0, 0, 1)), H)
    n = np.linalg.norm(N)
    # ...Equation 4.9:
    if n != 0:
        ra = np.arcnp.cos(N[0]/n)
        if N[1][0] < 0:
            ra = 2*np.pi - ra
    else:
        ra = 0
    # ...Equation 4.10:
    E = 1/mu*((v**2 - mu/r)*R - r*vr*V)
    e = np.linalg.norm(E)
    # in case e=0
    if n != 0:
        if e > eps:
            w = np.arccos(np.dot(N, E)/n/e)
            if E[2][0] < 0:
                w = 2*np.pi - w
        else:
            w = 0
    else:
        w = 0
    # in case e=0
    if e > eps:
        ta = np.arccos(np.dot(E, R)/e/r)
        if vr < 0:
            ta = 2*np.pi - ta
    else:
        cp = np.cross(N, R, axisa=0, axisb=0).flatten()
        if cp[2] >= 0:
            ta = np.arccos(np.dot(N, R)/n/r)
        else:
            ta = 2*np.pi - np.arccos(np.dot(N, R)/n/r)
    # a<0 for a hyperbola
    a = h**2/mu/(1 - e**2)

    return [a, e, incl, ra, w, ta]


def ephem(planet_str, JDE, Frame='Helio'):
    '''
    Finds the ephemeris for a planet given a Julian date using the
    Meeus algorithm
    '''

    # get the proper element coefficients
    planet = Meeus[planet_str]

    # get the T value
    T = (JDE - 2451545.0)/36525
    Ts = np.array([1, T, T**2, T**3])

    # get the element value at the specific time
    planet_val = planet @ Ts
    L = np.deg2rad(planet_val[0])
    a = planet_val[1]*AU_KM
    e = planet_val[2]
    inc = np.deg2rad(planet_val[3])
    raan = np.deg2rad(planet_val[4])
    Pi = np.deg2rad(planet_val[5])

    # AOP
    w = Pi - raan

    # TA in
    M = L-Pi
    Cen = (2*e - e**3/4 + 5/96*e**5)*np.sin(M) +\
        (5/4*e**2 - 11/24*e**4)*np.sin(2*M) + \
        (13/12*e**3 - 43/64*e**5)*np.sin(3*M) +\
        103/96*e**4*np.sin(4*M) + 1097/960*e**5*np.sin(5*M)
    nu = M+Cen

    # Cartesian Coordinates
    r_vec, v_vec = sv_from_coe([a, e, raan, inc, w, nu])

    # Convert to EME2000 if necessary
    if str.lower(Frame) == str.lower('EME2000'):
        theta = 23.4393*np.pi/180
        C = np.array([[1, 0.0, 0.0],
                      [0.0, np.cos(theta), -np.sin(theta)],
                      [0.0, np.sin(theta), np.cos(theta)]])
        r_vec = C @ r_vec
        v_vec = C @ v_vec
    return r_vec, v_vec


def psi_TOF(psi, r0_vec, rf_vec, mu=MU_SUN_KM):
    '''
    Find the TOF associated with a psi value
    psi is decided as (2*n*pi)**2 <= psi <= (2*(n+1)*pi)**2,
        where n=number of revolutions
    '''
    tol = 1e-6

    # get the real cos(delta nu)
    r0 = np.linalg.norm(r0_vec)
    rf = np.linalg.norm(rf_vec)
    # get Direction of Motion
    cosdnu = np.dot(r0_vec, rf_vec)/(r0*rf)
    nu0 = np.arctan2(r0_vec[1], r0_vec[0])
    nuf = np.arctan2(rf_vec[1], rf_vec[0])
    dnu = nuf-nu0  # 2*np.pi
    DM = 1 if dnu < np.pi else -1
    # calulate A
    A = DM*np.sqrt(r0*rf*(1+cosdnu))

    # case for hyperbolic/parabolic/elliptical orbit transfer
    if psi > tol:
        c2 = (1-np.cos(np.sqrt(psi)))/psi
        c3 = (np.sqrt(psi)-np.sin(np.sqrt(psi)))/np.sqrt(psi**3)
    elif psi < -tol:
        c2 = (1-np.cosh(np.sqrt(-psi)))/psi
        c3 = (np.sinh(np.sqrt(-psi))-np.sqrt(-psi))/np.sqrt((-psi)**3)
    else:
        c2 = 1./2
        c3 = 1./6

    # compute y
    y = r0 + rf + A*(psi*c3-1)/np.sqrt(c2)

    if A > 0 and y < 0:
        while y < 0:
            psi += 0.1
            y = r0 + rf + A*(psi*c3-1)/np.sqrt(c2)

    # get the new dt
    chi = np.sqrt(y/c2)
    dt = (chi**3*c3+A*np.sqrt(y))/np.sqrt(mu)
    return dt/86400


def multirev_minflight(planet_names, dates, n_revs=0):
    '''
    Find the possible TOF based on the psi value for hyperbolic through
    multi-rev elliptical orbits
    '''
    # starting position
    p1r, _ = ephem(planet_names[0], dates[0])

    # ending position
    p2r, _ = ephem(planet_names[1], dates[1])

    # get the time of flight for each psi, split by rev number and plot
    plt.figure()
    for n in range(n_revs+1):
        psi_low = (2*n*np.pi)**2 if n != 0 else -4*np.pi  # covers hyperbolic case
        psi_high = (2*(n+1)*np.pi)**2
        psis = np.linspace(psi_low+1e-6, psi_high-1e-6)
        tof = np.array([psi_TOF(psi, p1r, p2r) for psi in psis])
        plt.plot(psis, tof, label=f'{n} Revolutions')
        plt.legend()
        plt.grid()
        plt.xlabel(r'$\psi$ ($rad^2$)')
        plt.ylabel('TOF (days)')
        plt.title('Time of Flight vs Psi')
        plt.xlim(-4*np.pi, psi_high)
        plt.ylim(0, 1e4)

    if n_revs > 0:
        min_psi = fminbound(psi_TOF, psi_low, psi_high, args=(p1r, p2r))
        min_TOF = psi_TOF(min_psi, p1r, p2r)
        print(f'Minimum Transfer Possibility: {min_TOF:.3f} days')
        print(fr'Minimum Psi: {min_psi:.3f} $rad^2$')


def ls(r0_vec, rf_vec, dt0_days, n_revs=0, DM=None, mu=MU_SUN_KM):
    '''
    Lambert's Problem Solver
    Given two position vectors and the TOF, finds the velocities at departure/arrival
    '''
    # convert TOF to seconds
    dt0 = dt0_days*86400

    # pre-define variables
    v0_vec = [nan]*3
    vf_vec = v0_vec
    dnu = nan
    psi = nan

    # bad case where they won't really find a valid solution
    if np.linalg.norm(rf_vec-r0_vec)/dt0 < 100:
        # find TA diff and thus motion direction
        nu0 = np.arctan2(r0_vec[1], r0_vec[0])
        nuf = np.arctan2(rf_vec[1], rf_vec[0])
        dnu = (nuf-nu0) % (2*np.pi)
        if DM is None:
            DM = 1 if dnu < np.pi else -1

        # get the real np.cos(delta nu)
        r0 = np.linalg.norm(r0_vec)
        rf = np.linalg.norm(rf_vec)
        cosdnu = np.dot(r0_vec, rf_vec)/abs(r0*rf)

        # calulate A
        A = DM*np.sqrt(r0*rf*(1+cosdnu))

        # set psi bounds based on which side of the minimum TOF you're on
        psi_up = (2*(n_revs+1)*np.pi)**2
        if n_revs == 0:
            psi_low = -4*np.pi
        else:
            psi_low = (2*n_revs*np.pi)**2
            psi_min = fminbound(psi_TOF, psi_low, psi_up, args=(r0_vec, rf_vec))
            if dnu > np.pi:
                psi_low = psi_min
            else:
                psi_up = psi_min
        psi = (psi_low+psi_up)/2

        # initialize while loop
        tol = 1e-6
        dt = 0  # strictly initializing
        max_it = 5e3
        k = 0
        while abs(dt-dt0) > tol and k < max_it:
            # update c vals
            if psi > tol:
                c2 = (1 - np.cos(np.sqrt(psi))) / psi
                c3 = (np.sqrt(psi) - np.sin(np.sqrt(psi))) / np.sqrt(psi**3)
            elif psi < -tol:
                c2 = (1 - np.cosh(np.sqrt(-psi))) / psi
                c3 = (np.sinh(np.sqrt(-psi)) - np.sqrt(-psi)) / np.sqrt((-psi)**3)
            else:
                c2 = 1/2
                c3 = 1/6

            # compute y
            y = r0 + rf + A*(psi*c3-1)/np.sqrt(c2)

            if A > 0 and y < 0:
                while y < 0:
                    psi += 0.1
                    y = r0 + rf + A*(psi*c3 - 1) / np.sqrt(c2)

            # get the new dt
            chi = np.sqrt(y/c2)
            dt = (chi**3*c3 + A*np.sqrt(y)) / np.sqrt(mu)

            # compute new psi
            if n_revs > 0 and dnu < np.pi:
                if dt > dt0:
                    psi_low = psi
                else:
                    psi_up = psi
            else:
                if dt < dt0:
                    psi_low = psi
                else:
                    psi_up = psi
            psi = (psi_low+psi_up) / 2

            # update iteration number
            k += 1

        # compute the other functions
        f = 1 - y/r0
        gdot = 1 - y/rf
        g = A*np.sqrt(y/mu)

        # compute outputs
        v0_vec = (rf_vec - f*r0_vec) / g
        vf_vec = (gdot*rf_vec - r0_vec) / g
    return v0_vec, vf_vec


def flyby(vinf_in, vinf_out, mu):
    '''
    Calculates the radius of periapsis at which the flyby happens
        and the turn angle to make it happen
    '''
    # rp = radius of periapsis
    # psi = turn angle

    # calculate the turn angle
    psi = np.arccos(np.dot(vinf_in, vinf_out).item() /
                    (np.linalg.norm(vinf_in)*np.linalg.norm(vinf_out)))

    # calculate the radius of periapsis
    vinf = np.mean([np.linalg.norm(vinf_in), np.linalg.norm(vinf_out)])
    rp = mu/vinf**2 * (1/np.cos((np.pi-psi)/2) - 1)
    return rp, psi


def flyby_diff(planet_strings, dates):
    '''
    Calculates the values of the excess velocity at arrival and departure of a
        planet given the information of the first, second, and third planet in
        the string
    '''
    # get the planet values
    r1_vec, _ = ephem(planet_strings[0], dates[0])
    r2_vec, v2_vec = ephem(planet_strings[1], dates[1])
    r3_vec, _ = ephem(planet_strings[2], dates[2])

    # do the Lambert's of each and only save the velocities
    _, v2_vec1 = ls(r1_vec, r2_vec, dates[1]-dates[0])
    v2_vec2, _ = ls(r2_vec, r3_vec, dates[2]-dates[1])

    # get the vinfs of each
    vinf_in = np.linalg.norm(v2_vec1 - v2_vec)
    vinf_out = np.linalg.norm(v2_vec2 - v2_vec)
    return vinf_in, vinf_out


def res_orb(vinfm_GA1, GA1_rv, vinfp_GA2, GA2_rv, planet, min_r, XY, plot=False):
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

    # split up the GA rv's
    r1_vec = GA1_rv[:3]
    v1_vec = GA1_rv[3:]
    v2_vec = GA2_rv[3:]

    # turn Period into seconds
    Period = planet['period']*86400

    # calculate resonant orbit's semi-major axis
    P = Period*XY
    a = ((P/(2*np.pi))**2*MU_SUN_KM)**(1/3)

    # get the velocity of the spacecraft during the orbit
    # speed required to enter resonant orbit
    r1 = np.linalg.norm(r1_vec)
    vsc = np.sqrt(MU_SUN_KM*(2.0/r1 - 1.0/a))

    # get theta
    vp = np.linalg.norm(v1_vec)
    vinf = np.linalg.norm(vinfm_GA1)
    theta = np.arccos((vinf**2 + vp**2 - vsc**2) / (2*vinf*vp))

    # compute the VNC to the ecliptic
    V = v1_vec/vp
    N = np.cross(r1_vec, v1_vec) / np.linalg.norm(np.cross(r1_vec, v1_vec))
    C = np.cross(V, N)
    T = np.column_stack((V, N, C))

    # phi can be anything for now
    phis = np.arange(0, 360, 0.1)
    rp1 = []
    rp2 = []
    rp1_acc = []
    rp2_acc = []
    phi_acc = []
    vinfp1_acc = np.array([], dtype=np.float64).reshape(3, 0)
    vinfm2_acc = np.array([], dtype=np.float64).reshape(3, 0)
    for phi in phis:
        # get the vnc vinf after flyby
        vinf_pointing = np.array((np.cos(np.pi-theta),
                                   np.sin(np.pi-theta)*cosd(phi),
                                   -np.sin(np.pi-theta)*sind(phi)))
        vinfp_GA1_vnc = vinf*vinf_pointing

        # turn the vinf after GA1 into the ecliptic frame
        vinfp_GA1 = T @ vinfp_GA1_vnc

        # get the vinf before GA2
        vinfm_GA2 = vinfp_GA1 + v1_vec - v2_vec

        # find the radius of periapsis from these GA's
        rp1_dum = flyby(vinfm_GA1, vinfp_GA1, planet['mu'])[0]
        rp1.append(rp1_dum)
        rp2_dum = flyby(vinfm_GA2, vinfp_GA2, planet['mu'])[0]
        rp2.append(rp2_dum)

        # collect those viable
        if rp1_dum > min_r and rp2_dum > min_r:
            phi_acc.append(phi)
            rp1_acc.append(rp1_dum)
            rp2_acc.append(rp2_dum)
            vinfp1_acc = np.hstack((vinfp1_acc, vinfp_GA1.reshape(3,1)))
            vinfm2_acc = np.hstack((vinfm2_acc, vinfm_GA1.reshape(3,1)))

    # make plot of all of the data just created
    if plot is True:
        plt.figure()
        plt.axvspan(min(phi_acc), max(phi_acc), color='y')
        plt.plot(phis, rp1, label='Flyby 1', color='b')
        plt.plot(phis, rp2, label='Flyby2', color='r')
        plt.plot(phis, [min_r for _ in range(len(phis))],
                 color='k', linestyle='--', label=r'Min $r_p$')
        plt.text(np.median(phi_acc) - (max(phi_acc) - min(phi_acc))/4,
                 min_r-2000, r'Valid $\phi$'+'\nRange')
        plt.xlabel(r'$\phi (deg)$')
        plt.ylabel('Perigee Radius (km)')
        plt.legend()
        plt.xlim(0, 360)
        plt.ylim(0, max((max(rp1_acc), max(rp2_acc))))
    return rp1_acc, rp2_acc, phi_acc, vinfp1_acc, vinfm2_acc


def bplane2(r_vec, v_vec, mu):
    '''
    Calculate the B-plane parameters of a flyby
    '''
    # Method 2 B-Plane Targeting
    # r_vec = position vector
    # v_vec = velocity vector
    # mu = standard gravitational parameter

    # math behind the hats
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    a = -mu/v**2
    e_vec = 1/mu*((v**2-mu/r)*r_vec - np.dot(r_vec, v_vec)*v_vec)
    e = np.linalg.norm(e_vec)
    p = np.arccos(1/e)

    # all the hats
    h_hat = np.cross(r_vec, v_vec, axisa=0, axisb=0).T /\
            np.linalg.norm(np.cross(r_vec, v_vec, axisa=0, axisb=0))
    k_hat = [0, 0, 1]
    S_hat = e_vec/e*np.cos(p) +\
            np.cross(h_hat, e_vec, axisa=0, axisb=0).T /\
            np.linalg.norm(np.cross(h_hat, e_vec, axisa=0, axisb=0)) * np.sin(p)
    T_hat = np.cross(S_hat, k_hat, axisa=0, axisb=0).T /\
            np.linalg.norm(np.cross(S_hat, k_hat, axisa=0, axisb=0))
    R_hat = np.cross(S_hat, T_hat, axisa=0, axisb=0).T
    Bhat = np.cross(S_hat, h_hat, axisa=0, axisb=0).T

    # outputs
    b = abs(a)*np.sqrt(e**2-1)
    B = b*Bhat
    Bt = np.dot(B, T_hat).item()
    Br = np.dot(B, R_hat).item()
    theta = np.arccos(np.dot(T_hat, B)).item()
    theta = 2*np.pi - theta if Br < 0 else theta
    return Bt, Br, b, theta


def bplane_correction(r_vec, v_vec, mu, Bt_desired, Br_desired, pert):
    '''
    Correct the B-Plane estimate to the desired B-Plane parameters
    '''
    # keep calculating until within the tolerance
    tol = 1e-6
    TCM = np.array((0.0, 0.0))
    delVi = np.array((1e3, 1e3))
    dB = np.array((1e5, 1e5))
    k = 0
    v_new = np.copy(v_vec)
    while np.any(abs(delVi) > tol) and np.any(abs(dB) > tol):
        k += 1
        # new nominal B parameters
        Bt_nom, Br_nom, *_ = bplane2(r_vec, v_new, mu)

        # perturb the velocity
        Vpx = v_vec + np.array((pert, 0, 0))
        Vpy = v_vec + np.array((0, pert, 0))
        Btpx, Brpx, *_ = bplane2(r_vec, Vpx, mu)
        Btpy, Brpy, *_ = bplane2(r_vec, Vpy, mu)

        # get the B-Plane partials
        dBtdVx = (Btpx-Bt_nom)/pert
        dBtdVy = (Btpy-Bt_nom)/pert
        dBrdVx = (Brpx-Br_nom)/pert
        dBrdVy = (Brpy-Br_nom)/pert

        # delta B
        dB = np.array([[Bt_desired-Bt_nom], [Br_desired-Br_nom]])

        # delta Vi
        delVi = (np.linalg.inv(np.array([[dBtdVx, dBtdVy], [dBrdVx, dBrdVy]])) 
            @ dB).flatten()

        # update velocity vector and TCM
        v_new += np.append(delVi, 0)
        TCM += delVi
    return v_new-v_vec, v_new


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
        inert_coords[:, k] = inert_coord
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
        rot_coord_pos = Tir.T@inert_coords[:3, k] / L
        rot_coord_vel = (Tirdot.T @ inert_coords[:3, k] +
                         Tir.T @ inert_coords[3:, k]) / L
        rot_coord = np.vstack((rot_coord_pos, rot_coord_vel))
        rot_coords[:, k] = rot_coord
    return rot_coords


def tisserand(planet_numbers, contours, x_limits, y_axis='ra'):
    '''
    Plots the Tisserand plot given the planets wanting to look at, the vInf,
    and the value wanting to compare on the Y axis
    '''

    # Planetary Info
    semiMajor = [.387, .723, 1, 1.524, 5.203, 9.537, 19.191, 30.069, 39.482]
    planet_names = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter',
                    'Saturn', 'Uranus', 'Neptune', 'Pluto']

    # Vinf Contours
    Vp = np.zeros_like(planet_numbers)
    alphas = np.arange(0, 180, 1)
    vInfMags = contours
    plottings = np.zeros(shape=(len(alphas), len(vInfMags), len(planet_numbers)))
    rps = np.zeros(shape=(len(alphas), len(vInfMags), len(planet_numbers)))

    colors = ['c', 'r', 'b', 'g', 'm', 'y', 'c']
    plt.figure()

    # do each planet
    for k, planet_num in enumerate(planet_numbers):
        # do each contour line
        for j, vInfMag in enumerate(vInfMags):
            # get the planet's info
            R = semiMajor[planet_num]*AU_KM
            Vp = np.sqrt(MU_SUN_KM/R)

            for i, alpha in enumerate(alphas):
                # get the outgoing velocity
                vinfOut = np.array([vInfMag*cosd(alpha), vInfMag*sind(alpha)])
                vOut = np.array((Vp, 0.0)) + vinfOut

                # get a and e for this alpha
                a, e, *_ = coe_from_sv(np.array((0.0, R, 0.0)),
                                     np.append(vOut, 0.0),
                                     MU_SUN_KM)

                # find the rp and ra for this alpha
                if e < 1:
                    # figure out which thing to collect
                    if y_axis == 'ra':
                        plotting_val = a*(1+e)/AU_KM
                    elif y_axis == 'period':
                        plotting_val = 2*np.pi*np.sqrt(a**3/MU_SUN_KM)/86400  # now in days
                        if max(planet_numbers) > 4:
                            plotting_val /= EARTH_PERIOD_DAYS  # now in years
                    elif y_axis == 'C3':
                        plotting_val = np.linalg.norm(vinfOut)**2
                    rp_val = a*(1-e)/AU_KM
                else:
                    plotting_val = nan
                    rp_val = nan
                plottings[i, j, k] = plotting_val
                rps[i, j, k] = rp_val

    # plot the contours second
    for k in range(len(planet_numbers)):
        plt.semilogy(rps[:, :, k], plottings[:, :, k],
                     color=colors[(k % len(colors)-1)+1])

    # add the things that only make sense for doing rp vs ra
    if y_axis == 'ra':
        # plot the planets and their names
        for j, planet_num in enumerate(planet_numbers):
            plt.scatter(semiMajor[planet_num], semiMajor[planet_num],
                        color=colors[(j % len(colors)-1)+1], marker='x')
            plt.text(semiMajor[planet_num]+0.025, semiMajor[planet_num],
                     planet_names[planet_num],
                     color=colors[(j % len(colors)-1)+1])

        # also add grey dashed lines for finding planetary intersections better
        min_planet = semiMajor[planet_numbers[1]]
        max_planet = semiMajor[planet_numbers[-1]]
        # vertical lines
        for planet_num in planet_numbers:
            plt.plot(np.hstack((semiMajor[planet_num], semiMajor[planet_num])),
                     [semiMajor[planet_num], max_planet],
                     color='k', linestyle=':', linewidth=1)

        # horizontal lines
        for planet_num in planet_numbers:
            plt.plot([min_planet, semiMajor[planet_num]],
                     np.hstack((semiMajor[planet_num], semiMajor[planet_num])),
                     color='k', linestyle=':', linewidth=1)
    else:
        for k in range(planet_numbers):
            plt.text(semiMajor[planet_num]+0.025, min(plottings[:, :, k]),
                     planet_names[planet_num], color=colors[(k % len(colors)-1)+1])

    # finish the plot's look
    plt.grid
    if str.lower(y_axis) != 'ra':
        plt.ylim([min(plottings), 100])
    plt.xlabel(r'$R_p$ (AU)')
    plt.title(fr'$v_\infty$ shown are {contours} km/s')
    if str.lower(y_axis) == 'ra':
        plt.ylabel(r'$R_a$ (AU)')
    elif str.lower(y_axis) == 'period':
        if max(planet_numbers) > 4:
            plt.ylabel('Period (years)')
        else:
            plt.ylabel('Period (days)')
    elif str.lower(y_axis) == 'c3':
        plt.ylabel(r'$C_3 (km^2/s^2)$')
    plt.xlim(x_limits)


def tisserand_path(planet_num, vInfMag):
    '''
    Calculates the path to take given the planet number and the v_inf magnitude to make it happen
    '''
    semiMajor = [.387, .723, 1, 1.524, 5.203, 9.537, 19.191, 30.069, 39.482]
    planet_names = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn',
                    'Uranus', 'Neptune', 'Pluto']
    R = semiMajor[planet_num-1]*AU_KM
    Vp = np.sqrt(MU_SUN_KM/R)
    rp = []
    ra = []

    for alpha in range(180):
        # get the outgoing velocity
        vinfOut = np.array((vInfMag*cosd(alpha), vInfMag*sind(alpha)))
        vOut = np.array((Vp, 0)) + vinfOut

        # get a and e for this alpha
        a, e, *_ = coe_from_sv(np.array((0, R, 0)),
                               np.append(vOut, 0),
                               mu=MU_SUN_KM)

        # find the rp and ra for this alpha
        if e < 1:
            # figure out which thing to collect
            ra.append(a*(1+e)/AU_KM)
            rp.append(a*(1-e)/AU_KM)
        else:
            ra.append(nan)
            rp.append(nan)
    return rp, ra


'''
NEED TO FIGURE OUT WHAT THESE FUNCTIONS ACTUALLY DO AND THEN WRITE UP A DOC STRING FOR EACH
'''


def ls_check(planet1_str, dep_date, planet2_str, arr_date):
    # check the TOF
    tof = arr_date - dep_date

    # ignore bad departure and arrivals
    if tof <= 0: #|| TOF(a,d)>=500
        v_inf_out = nan
        v_inf_in = nan
    else:
        # solve Lambert's problem
        r1_vec, v1_vec = ephem(planet1_str, dep_date)
        r2_vec, v2_vec = ephem(planet2_str, arr_date)
        v0_vec, vf_vec = ls(r1_vec, r2_vec, tof)
        
        # collect the mission parameters
        v_inf_out = v0_vec-v1_vec
        v_inf_in = vf_vec-v2_vec
    return v_inf_out, v_inf_in


def leg_check(d1, planet_transfers, saving_info, C3_max, stopping_conditions,
              vinf_out, vinf_in, vinf_max, leg_num):
    # go day to day
    daterange = planet_transfers[leg_num].planet2.date_range
    for d2 in daterange:
        # do Lambert's of the leg
        vio, vii = ls_check(planet_transfers[leg_num].planet1.name, d1,
                            planet_transfers[leg_num].planet2.name, d2)
        vinf_out[leg_num, :] = vio
        vinf_in[leg_num, :] = vii

        # check if first leg
        if leg_num != 0:
            # put the stopping conditions in the right order
            dv_diff = stopping_conditions[leg_num-1][0]
            rp_min = stopping_conditions[leg_num-1][1]
            
            # check that the vinf_out isn't out of range of vinf_in at this date
            if abs(np.linalg.norm(vinf_out[leg_num, :]) - np.linalg.norm(vinf_in[leg_num-1, :])) < dv_diff:
                # check if the vinf's match and if it gets too close
                rp = flyby(vinf_in[leg_num-1, :], vinf_out[leg_num, :],
                           mu=planet_transfers[leg_num].planet2.sgp)
                if rp > rp_min:
                    dv = abs(np.linalg.norm(vinf_out[leg_num, :]) -
                             np.linalg.norm(vinf_in[leg_num-1, :]))
                    saving_info[-1][leg_num] = dv
                    saving_info[-1][len(planet_transfers)+leg_num] = d1

                    # continue if not the final leg of the mission
                    if leg_num != planet_transfers[-1].leg_number:
                        # go to next leg_num
                        saving_info = leg_check(d2, planet_transfers,
                                                saving_info, C3_max,
                                                stopping_conditions,
                                                vinf_out, vinf_in,
                                                vinf_max, leg_num+1)
                    else:
                        # check that final arrival vinf_in isn't too big
                        if np.linalg.norm(vinf_in[leg_num, :]) < vinf_max:
                            saving_info[-1][len(planet_transfers)] = np.linalg.norm(vinf_in[leg_num, :])
                            saving_info[-1][-1] = d2
                            # start with the same info as was there just before
                            saving_info.append(saving_info[-1][:])
        else:
            # replace the vinf_out with C3
            C3 = np.linalg.norm(vinf_out)**2
            # check that the C3 isn't too big
            if C3 <= C3_max:
                # go to next leg_num
                saving_info[-1][1] = C3
                saving_info[-1][1+len(planet_transfers)] = d1
                saving_info = leg_check(d2, planet_transfers, saving_info,
                                        C3_max, stopping_conditions, vinf_out,
                                        vinf_in, vinf_max, leg_num+1)
    return saving_info


def pcp_search3(planet_transfers, C3_max, conditions, vinf_max):
    # reset the saved info
    vinf_out = np.zeros(shape=(len(planet_transfers)-1, 3))
    vinf_in = vinf_out
    dv = np.zeros(shape=(1, vinf_in.shape[0]-1))
    dates = np.zeros(shape=(1, len(planet_transfers)))
    worthy_transfers = []

    # do the work for each leg
    date_range = planet_transfers[0].planet1.date_range
    for d1 in date_range:
        worthy_transfers = leg_check(d1, planet_transfers, worthy_transfers,
                                     C3_max, conditions, vinf_out, vinf_in,
                                     vinf_max, 1)
    return worthy_transfers
