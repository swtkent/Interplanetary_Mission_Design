'''
This file is strictly to hold constants of any and all astrodynamics to be
called upon. This will allow for all to be the same across files and not need
to constantly need to go searching for them. All units will be given in meters
and seconds unless specified elsewhere for higher accuracy in propagation
'''
from numpy import array

# Standard Gravitational Parameters
MU_SUN_KM = 1.32712440018e11
MU_VENUS_KM = 3.24858599e5
MU_EARTH_KM = 3.986004415e5
MU_MARS_KM = 4.28283100e4
MU_JUPITER_KM = 1.266865361e8
MU_SATURN_KM = 3.7931208e7
MU_URANUS_KM = 5.7939513e6
MU_NEPTUNE_KM = 6.835100e6
MU_PLUTO_KM = 870.0

# Planetary Radii
R_VENUS_KM = 6051.8
R_EARTH_KM = 6378.1363
R_MARS_KM = 3396.19
R_JUPITER_KM = 71492.0
R_SATURN_KM = 60268.0
R_URANUS_KM = 25559.0
R_NEPTUNE_KM = 24764.0
R_PLUTO_KM = 1188.0


# orbit length of planets
VENUS_PERIOD_DAYS = 224.695
EARTH_PERIOD_DAYS = 365.24189
MARS_PERIOD_DAYS = 686.973
JUPITER_PERIOD_DAYS = 4330.595
SATURN_PERIOD_DAYS = 10746.94
URANUS_PERIOD_DAYS = 30588.740
NEPTUNE_PERIOD_DAYS = 59799.9
PLUTO_PERIOD_DAYS = 90560.


'''
Meeus Algorithm Data
all in order [L,a,e,i,O,Pi]
'''
# Venus
Venus = array([[181.979801, 58517.8156760, 0.00000165, -0.000000002],
               [0.72332982, 0.0, 0.0, 0.0],
               [0.00677188, -0.000047766, 0.0000000975, 0.00000000044],
               [3.394662, -0.0008568, -0.00003244, 0.000000010],
               [76.679920, -0.2780080, -0.00014256, -0.000000198],
               [131.563707, 0.0048646, -0.00138232, -0.000005332]])

# Earth
Earth = array([[100.466449, 35999.3728519, -0.00000568, 0.0],
               [1.000001018, 0.0, 0.0, 0.0],
               [0.01670862, -0.000042037, -0.0000001236, 0.00000000004],
               [0.0, 0.0130546, -0.00000931, -0.000000034],
               [174.873174, -0.2410908, 0.00004067, -0.000001327],
               [102.937348, 0.3225557, 0.00015026, 0.000000478]])

# Mars
Mars = array([[355.433275, 19140.2993313, 0.00000261, -0.000000003],
              [1.523679342, 0.0, 0.0, 0.0],
              [0.09340062, 0.000090483, -0.0000000806, -0.00000000035],
              [1.849726, -0.0081479, -0.00002255, -0.000000027],
              [49.558093, -0.2949846, -0.00063993, -0.000002143],
              [336.060234, 0.4438898, -0.00017321, 0.000000300]])

# Jupiter
Jupiter = array([[34.351484, 3034.9056746, -0.00008501, 0.000000004],
                 [5.202603191, 0.0000001913, 0.0, 0.0],
                 [0.04849485, 0.000163244, -0.0000004719, -0.00000000197],
                 [1.303270, -0.0019872, 0.00003318, 0.000000092],
                 [100.464441, 0.1766828, 0.00090387, -0.000007032],
                 [14.331309, 0.2155525, 0.00072252, -0.000004590]])

# Saturn
Saturn = array([[50.077471, 1222.1137943, 0.00021004, -0.000000019],
                [9.554909596, -0.0000021389, 0.0, 0.0],
                [0.05550862, -0.000346818, -0.0000006456, 0.00000000338],
                [2.488878, 0.0025515, -0.00004903, 0.000000018],
                [113.665524, -0.2566649, -0.00018345, 0.000000357],
                [93.056787, 0.5665496, 0.00052809, 0.000004882]])

# Uranus
Uranus = array([[314.055005, 429.8640561, 0.00030434, 0.000000026],
                [19.218446062, -0.0000000372, 0.00000000098, 0.0],
                [0.04629590, -0.000027337, 0.0000000790, 0.00000000025],
                [0.773196, 0.0007744, 0.00003749, -0.000000092],
                [74.005947, 0.5211258, 0.00133982, 0.000018516],
                [173.005159, 1.4863784, 0.0021450, 0.000000433]])

# Neptune
Neptune = array([[304.348665, 219.8833092, 0.00030926, 0.000000018],
                 [30.110386869, -0.0000001663, 0.00000000069, 0.0],
                 [0.00898809, 0.000006408, -0.0000000008, -0.00000000005],
                 [1.769952, -0.0093082, -0.00000708, 0.000000028],
                 [131.784057, 1.1022057, 0.00026006, -0.000000636],
                 [48.123691, 1.4262677, 0.00037918, -0.000000003]])

# Pluto
Pluto = array([[238.92903833, 145.20780515, 0.0, 0.0],
               [39.48211675, -0.00031596, 0.0, 0.0],
               [0.24882730, 0.00005170, 0.0, 0.0],
               [17.14001206, 0.00004818, 0.0, 0.0],
               [110.30393684, -0.01183482, 0.0, 0.0],
               [224.06891629, -0.04062942, 0.0, 0.0]])

# dictionary of this information
Meeus = {'Venus': Venus, 'Earth': Earth, 'Mars': Mars, 'Jupiter': Jupiter,
         'Saturn': Saturn, 'Uranus': Uranus, 'Neptune': Neptune,
         'Pluto': Pluto}


# Constants for a given planet
VENUS = {'mu': MU_VENUS_KM, 'radius': R_VENUS_KM,
         'period': VENUS_PERIOD_DAYS, 'name': 'Venus'}
EARTH = {'mu': MU_EARTH_KM, 'radius': R_EARTH_KM,
         'period': EARTH_PERIOD_DAYS, 'name': 'Earth'}
MARS = {'mu': MU_MARS_KM, 'radius': R_MARS_KM,
        'period': MARS_PERIOD_DAYS, 'name': 'Mars'}
JUPITER = {'mu': MU_JUPITER_KM, 'radius': R_JUPITER_KM,
           'period': JUPITER_PERIOD_DAYS, 'name': 'Jupiter'}
URANUS = {'mu': MU_URANUS_KM, 'radius': R_URANUS_KM,
          'period': URANUS_PERIOD_DAYS, 'name': 'Uranus'}
NEPTUNE = {'mu': MU_NEPTUNE_KM, 'radius': R_NEPTUNE_KM,
           'period': NEPTUNE_PERIOD_DAYS, 'name': 'Neptune'}
PLUTO = {'mu': MU_PLUTO_KM, 'radius': R_PLUTO_KM,
         'period': PLUTO_PERIOD_DAYS, 'name': 'Pluto'}
PLANETS = {'Venus': VENUS, 'Earth': EARTH, 'Mars': MARS, 'Jupiter': JUPITER,
           'Uranus': URANUS, 'Neptune': NEPTUNE, 'Pluto': PLUTO}


# other terms
AU_KM = 1.49597870700e8
EARTH_MOON_DIST_KM = 384747.962856037
EARTH_MOON_CENTER = 0.012150585609624
