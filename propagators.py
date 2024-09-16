'''
File to hold all propagators that may be used
For now will make use of scipy.integrate.solve_ivp to do all integration
'''
from astro_constants import EARTH_MOON_CENTER
from scipy.integrate import solve_ivp
import numpy as np


def ode_solve(fun, tspan, init_state, args=None, events=None, RTOL=1e-13, ATOL=1e-13, event_choices=None):
    '''
    Main way of integrating all integrators within this file, it will handle
        the function title setup and the kwarg values unless
        specified otherwise
    fun = function to solve ODE
    tspan = list/array of tspan values wanting to evaluate at
    init_state = ndarray of states looking to propagate
    args = optional variables to pass into function not wanting to propagate
    events = function to have it decide if needing to stop or anything else
        (look at solve_ivp documentation)
    RTOL = relative tolerance
    ATOL = absolute tolerance
    '''
    if isinstance(args, float) or isinstance(args, int):
        args = list([args])
    init_state = init_state.flatten()
    '''
    if events!=None:
        for event, choices in zip(events, event_choices):
            event.terminal = choices['terminal']
            event.direction = choices['direction']
    '''
    return solve_ivp(fun, [tspan[0], tspan[-1]], y0=init_state, t_eval=tspan,
                     events=events, args=args, rtol=RTOL, atol=ATOL)


def twoBody(t, Z, mu):
    '''
    2-Body propagator with no perturbations
    '''
    states = np.split(Z, range(6, len(Z), 6))
    return_states = np.empty(0)
    for state in states:
        r_vec, v_vec = np.split(state, 2)
        r = np.linalg.norm(r_vec)
        a_vec = -mu*r_vec/r**3
        return_state = np.concatenate((v_vec, a_vec), axis=None)
        return_states = np.concatenate((return_states, return_state),
                                       axis=None)
    return return_states


def nBody(t, Z, mus):
    '''
    N-body propagator built for one s/c with multiple large bodies also
    orbiting around one larger body and affecting the s/c
    '''
    mu = mus[0]
    mus2 = mus[1:]
    try:
        large_bodies = np.split(Z[len(Z)-len(mus2)*6::], len(mus2))
    except ValueError:
        print('The number of large bodies doesn''t match the number of SGP''s')
    states = np.split(Z, range(6, len(Z), 6))
    return_states = np.empty(0)
    for state in states:
        r_vec, v_vec = np.split(state, 2)
        r = np.linalg.norm(r_vec)
        a_vec = -mu*r_vec/r**3
        for num, body in enumerate(large_bodies):
            L_vec, _ = np.split(body, 2)
            rL_vec = r_vec - L_vec
            rL = np.linalg.norm(rL_vec)
            aL = mus2[num]*rL_vec/rL**3 if rL > 1 else 0
            a_vec -= aL
        return_state = np.concatenate((v_vec, a_vec), axis=None)
        return_states = np.concatenate((return_states, return_state), axis=None)
    return return_states


def CR3BP(t, Z, mu=EARTH_MOON_CENTER):
    '''
    3-body propagator built for one small s/c orbiting around two much larger objects in space in a nondimensional rotating frame
    '''
    # sate what vals is
    x, y, z, xdot, ydot, zdot = Z

    # define R1 and R2
    R1 = np.linalg.norm([x+mu, y, z])
    R2 = np.linalg.norm([x-1+mu, y, z])

    # velocity change
    xddot = 2*ydot + x - (1-mu)*(x+mu)/R1**3 - mu*(x-1+mu)/R2**3
    yddot = -2*xdot + y - (1-mu)*y/R1**3 - mu*y/R2**3
    zddot = -(1-mu)*z/R1**3 - mu*z/R2**3

    # collect the derivatives
    v_vec = np.array([xdot, ydot, zdot])
    a_vec = np.array([xddot, yddot, zddot])
    return np.concatenate((v_vec, a_vec), axis=None)


def halomaker(tspan, pos_vel_phi, mu=EARTH_MOON_CENTER):
    '''
    CR3BP but with a State Transition Matrix for corrections
    '''
    # get the position and velocity out
    rvec = pos_vel_phi[:3]
    x, y, z = rvec
    vvec = np.hstack(pos_vel_phi[3:6])
    phi_vec = pos_vel_phi[6:]
    phi_mat = np.reshape(phi_vec, newshape=(6, 6))

    # define the radius
    # r = np.linalg.norm(rvec)

    # define the STM
    A = np.array([
        [                                                                                                                                                                                                                                                       0,                                                                                                                                                                                                             0,                                                                                                                                                                                                         0,  1, 0, 0],
        [                                                                                                                                                                                                                                                       0,                                                                                                                                                                                                             0,                                                                                                                                                                                                         0,  0, 1, 0],
        [                                                                                                                                                                                                                                                       0,                                                                                                                                                                                                             0,                                                                                                                                                                                                         0,  0, 0, 1],
        [ (mu - 1)/((mu + x)**2 + y**2 + z**2)**(3/2) - mu/(y**2 + z**2 + abs(mu + x - 1)**2)**(3/2) - (3*(2*mu + 2*x)*(mu + x)*(mu - 1))/(2*((mu + x)**2 + y**2 + z**2)**(5/2)) + (3*mu*(mu + x - 1)*(mu + x - 1))/(y**2 + z**2 + abs(mu + x - 1)**2)**(5/2) + 1,                                                                                  (3*mu*y*(mu + x - 1))/(y**2 + z**2 + abs(mu + x - 1)**2)**(5/2) - (3*y*(mu + x)*(mu - 1))/((mu + x)**2 + y**2 + z**2)**(5/2),                                                                              (3*mu*z*(mu + x - 1))/(y**2 + z**2 + abs(mu + x - 1)**2)**(5/2) - (3*z*(mu + x)*(mu - 1))/((mu + x)**2 + y**2 + z**2)**(5/2),  0, 2, 0],
        [                                                                                                                    (3*mu*y*(mu + x - 1))/(y**2 + z**2 + abs(mu + x - 1)**2)**(5/2) - (3*y*(2*mu + 2*x)*(mu - 1))/(2*((mu + x)**2 + y**2 + z**2)**(5/2)), (mu - 1)/((mu + x)**2 + y**2 + z**2)**(3/2) - mu/(y**2 + z**2 + abs(mu + x - 1)**2)**(3/2) - (3*y**2*(mu - 1))/((mu + x)**2 + y**2 + z**2)**(5/2) + (3*mu*y**2)/(y**2 + z**2 + abs(mu + x - 1)**2)**(5/2) + 1,                                                                                                (3*mu*y*z)/(y**2 + z**2 + abs(mu + x - 1)**2)**(5/2) - (3*y*z*(mu - 1))/((mu + x)**2 + y**2 + z**2)**(5/2), -2, 0, 0],
        [                                                                                                                    (3*mu*z*(mu + x - 1))/(y**2 + z**2 + abs(mu + x - 1)**2)**(5/2) - (3*z*(2*mu + 2*x)*(mu - 1))/(2*((mu + x)**2 + y**2 + z**2)**(5/2)),                                                                                                    (3*mu*y*z)/(y**2 + z**2 + abs(mu + x - 1)**2)**(5/2) - (3*y*z*(mu - 1))/((mu + x)**2 + y**2 + z**2)**(5/2), (mu - 1)/((mu + x)**2 + y**2 + z**2)**(3/2) - mu/(y**2 + z**2 + abs(mu + x - 1)**2)**(3/2) - (3*z**2*(mu - 1))/((mu + x)**2 + y**2 + z**2)**(5/2) + (3*mu*z**2)/(y**2 + z**2 + abs(mu + x - 1)**2)**(5/2),  0, 0, 0]])

    # define R1 and R2
    R1 = np.linalg.norm([x+mu, y, z])
    R2 = np.linalg.norm([x-1+mu, y, z])

    # velocity change
    xdot = vvec[0]
    ydot = vvec[1]
    xddot = 2*ydot + x - (1-mu)*(x+mu)/R1**3 - mu*(x-1+mu)/R2**3
    yddot = -2*xdot + y - (1-mu)*y/R1**3 - mu*y/R2**3
    zddot = -(1-mu)*z/R1**3 - mu*z/R2**3

    # get the derivatives of everything
    accel = np.hstack([xddot, yddot, zddot])
    phidot_mat = A@phi_mat
    phidot_vec = np.reshape(phidot_mat, newshape=(36,))
    dvals = np.concatenate([vvec, accel, phidot_vec], axis=0)
    return dvals


'''
--------------------------------------------------------------------------------
EVENT FUNCTIONS BELOW
--------------------------------------------------------------------------------
'''
# ode exit function
# Event function
def yzero(t, y): return y[1]  # detect value at zero


def yzero2(t, y):
    val = y[1] if t > 1 else 100
    return val


def upward_cannon(t, y): return [y[1], -0.5]


def hit_ground(t, y): return y[0]
