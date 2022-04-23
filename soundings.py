import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import collections as c
import seaborn as sns
import random
sns.set_theme()

sys.path.append(os.environ['CLASS4GL'])

from interface_multi import stations,stations_iterator, records_iterator,get_record_yaml,get_records

data_dir = "./data"

def select_sites(n):
    """select the sites that have more than N data available.

For translating site IDS into human readable names, see:

/home/adam/org/doc/igra2-station-list.txt
"""
    path = os.environ['CLASS4GL_DATA'] + '/forcing/IGRA_PAIRS_20190515/'
    path_output = os.environ['CLASS4GL_DATA'] + '/experiments/IGRA_PAIRS_20190515/BASE/'
    all_stations = stations(path, suffix='ini', refetch_stations=False)
    all_stations_select = pd.DataFrame(all_stations.table)
    records = get_records(all_stations_select,\
                          path,\
                          subset='ini',\
                          refetch_records=False,\
                          )

    grouped = records.groupby(level='STNID').apply(lambda _df: _df.shape[0])

    subset = grouped[grouped >= n]
    return subset

FORCING_PATH = os.environ['CLASS4GL_DATA'] + '/forcing/IGRA_PAIRS_20190515/'

# below are all sites with >= 500 records (e.g., select_sites(500)
STATION_IDS = {
    'milano' : 16080,
    'idar_oberstein' : 10618, # west germany  (near luxemborg/belgium)
    'lindenberg' : 10393, # germany, near berlin by coodinates
    'elko' : 72582, # NV
    'riverton': 72672, # WY
    'spokane' : 72786,
    'bergen' : 10238, # northern germany near hamberg
    'great_falls': 72776, # MT
    'kelowna' : 71203, # BC
    'flagstaff' : 72376,
    'las_vegas' : 72388,
    'quad_city' : 74455,
    }


def generic_path(station_name, suffix='yaml'):
    """Generate a path given a STATION_NAME and SUFFIX (e.g., yaml or pkl)."""
    return "%s/%05d_%s.%s" % (FORCING_PATH, STATION_IDS[station_name], 'ini', suffix)

def load_record(f, record):
    """Load the record given a file handle F and a RECORD from a *_ini.pkl dataframe"""
    c4gli = get_record_yaml(f,
                            record.index_start,
                            record.index_end,
                            mode='model_input')

    # calculate averages in the troposphere; this code yanked from simulations/simulations.py
    seltropo = (c4gli.air_ac.p > c4gli.air_ac.p.iloc[-1]+ 3000.*(- 1.2 * 9.81 ))
    profile_tropo = c4gli.air_ac[seltropo]
    for var in ['advt', 'advq', 'advu', 'advv']:
        if var[:3] == 'adv':
            mean_adv_tropo = np.mean(profile_tropo[var+'_x']+profile_tropo[var+'_y'] )
            c4gli.update(source='era-interim',pars={var+'_tropo':mean_adv_tropo})
    return c4gli


def load_records(station_name):
    """Load all records for STATION_NAME"""

    f = open(generic_path(station_name, suffix='pkl'), 'rb')
    df = pd.read_pickle(f)
    f.close()

    f = open(generic_path(station_name, suffix='yaml'), 'r')

    records = c.deque()

    for (_station_id,_chunk,_index),record in df.iterrows():
        class_input = load_record(f, record)
        records.append(class_input)
    f.close()

    return records

def maybe_grab_key(x, key):
    """if KEY is in X, return X[KEY}, otherwise nan

NOTE: this was kind of a hack; really every time slice seems
like it should have the same keys, so I might want to look
into why that isn't he case."""
    try:
        return x.pars.__dict__[key]
    except KeyError:
        return np.nan


def grab_record(key, records):
    """Grab a list of objects from RECORDS corresponding to KEY"""
    return [maybe_grab_key(x, key) for x in records]

# where are 'zi0', 'wsls', ug, ug
# also confirm all equivalencies are right

def dataframe_from_records(keys, records):
    """Generate a pandas datafrae from KEYS in RECORDS.

If KEYS is false, then return a dataframe of every key in records."""
    if not keys:
        keys = records[0].pars.__dict__.keys()
    df = pd.DataFrame(dict([(key, grab_record(key, records)) for key
                            in keys]))
    df.set_index('datetime', inplace=True, drop=False)
    df['w_average'] = 0.5 * (df.w2 + df.wg)
    return df


def fraction_mode(ds):
    """Return the fraction of observations that are equal to ds's mode

This either returns the fraction (floating point), or an error"""
    try:
        mode = ds.mode()[0]
    except KeyError:
        return KeyError
    try:
        modecount = ds[ds == mode].size
    except TypeError:
        return TypeError
    return float(modecount) / float(ds.size)


def generate_variability_keys(df_input):
    """Generate all keys that vary, don't vary, and are filled with None
values (e.g., no mode for the dataseries, or cannot compare with mode)
from a dataframe DF_INPUT
    """
    variability_keys = c.deque()
    constant_keys = c.deque()
    none_keys = c.deque()
    for (label, ds) in df_input.iteritems():
        mode_fraction = fraction_mode(ds)
        if (mode_fraction == TypeError) or (mode_fraction == KeyError):
            none_keys.append(label)
        elif mode_fraction >= 0.99:
            constant_keys.append(label)
        else:
            variability_keys.append(label)
    return (set(variability_keys), set(constant_keys), set(none_keys))

def load_dataframes():
    """load all station dataframes in STATION_IDS into a dict keyed by the keys in STATION_IDS"""
    data_frames = dict()
    for station_id in STATION_IDS.keys():
        data_frame = dataframe_from_records(False, load_records(station_id))
        data_frames[station_id] = data_frame
    return data_frames

def data_frame_size(_df):
    """return the number of samples in a dataframe"""
    return _df.shape[0]

def monthly_counts(_df):
    """return the monthly counts of sample in a dataframe"""
    return _df.groupby(pd.Grouper(freq='M')).apply(data_frame_size)

def max_monthly_counts(dfs):
    """take the output of laod_dataframes, and calculate the max monthly sample sizes"""
    monthly_counts_dict = dict()
    for (station_id, _df) in dfs.items():
        monthly_counts_dict[station_id] = monthly_counts(_df).max()
    return monthly_counts_dict

def ppm_to_ppb(x):
    """convert ppm to ppb"""
    return x * 1000.0

def pa_to_hpa(x):
    """convert Pa to hPa"""
    return x / 100.0

def station_longitude(x):
    """convert a -180-180 degree longitude to a 0-360 degree longitude"""
    if x < 0.0:
        x = 360 + x
    return x

def kg_to_grams(x):
    """convert kg to g"""
    return x * 1000.0

def ls_type_to_Ags_flag(x):
    """convert a ls_type ('js' or 'Ags') to bools for use in 'mxlch-lrsAgs' and 'mxlch-lCO2Ags'"""
    if x == 'Ags':
        return ".true."
    elif x == 'js':
        return ".false."
    else:
        raise

def fortran_print_bool(x):
    if x:
        return '.true.'
    else:
        return '.false.'

def set_minimum_wind(x):
    """set a minimum wind speed (0.1 m/s), because model will articially exit if windspeed less that 1 cm/s"""
    if x == 0.0:
        x = 0.1
    elif abs(x) < 0.1:
        x = 0.1*x/abs(x)
    return x


def set_minimum_ABL_height(x):
    return max(x, 40.0)

def runtime(tstart):
    """Calculate a runtime in seconds from tstart in hours.

Note simulation time finish will always be 1600 local,
because at Kelowna this always ended at 16.049722, so we are conisstent with
that forumulation.

REDO: bump it up to 22 hrs local to get nighttime response."""
    return '%d' % round(np.ceil((3600.0 * (22.049722 - tstart))))

def print_int(value):
    """Print an integer value, converting from float if necessary"""
    return '%d' % round(value)

elisp_conversion_functions = {
    'CO2' : ppm_to_ppb,
    'h' : set_minimum_ABL_height,
    'dCO2' : ppm_to_ppb,
    'gammaCO2' : ppm_to_ppb,
    'wCO2' : ppm_to_ppb,
    'Ps' : pa_to_hpa,
    'lon' : station_longitude,
    'q' : kg_to_grams,
    'dq' : kg_to_grams,
    'gammaq' : kg_to_grams,
    'wq' : kg_to_grams,
    'advq_tropo' : kg_to_grams,
    'ls_type' : ls_type_to_Ags_flag,
    'sw_fixft' : fortran_print_bool,
    'sw_ls' : fortran_print_bool,
    'sw_rad' : fortran_print_bool,
    'sw_shearwe' : fortran_print_bool,
    'sw_sl' : fortran_print_bool,
    'sw_cu' : fortran_print_bool,
    'u' : set_minimum_wind,
    'v' : set_minimum_wind,
    'doy' : print_int,
    'dt' : print_int
    }

elisp_conversion_functions_elisp_key = {
    'mxlch-time' : runtime }

elisp_conversion = {
    'mxlch-C1sat' : 'C1sat',
    'mxlch-C2ref' : 'C2ref',
    'mxlch-CGsat' : 'CGsat',
    'mxlch-cm0' : 'CO2',
    'mxlch-LAI' : 'LAI',
    'mxlch-pressure' : 'Ps',
    'mxlch-pressure_ft' : 'Ps',
    'mxlch-T2' : 'T2',
    'mxlch-Ts' : 'Ts',
    'mxlch-Tsoil' : 'Tsoil',
    'mxlch-Wl' : 'Wl',
    'mxlch-CLa' : 'a',
    'mxlch-CLb' : 'b',
    'mxlch-CLc' : 'p',
    'mxlch-albedo' : 'alpha',
    'mxlch-beta' : 'beta',
    'mxlch-cc' : 'cc',
    'mxlch-cveg' :'cveg',
    'mxlch-dc0': 'dCO2',
    'mxlch-DeltaFlw' : 'dFz',
    'mxlch-day' : 'doy',
    'mxlch-dq0' :'dq',
    'mxlch-dtime' : 'dt',
    'mxlch-dtheta0' : 'dtheta',
    'mxlch-gD' : 'gD',
    'mxlch-gammac' : 'gammaCO2',
    'mxlch-gammaq' : 'gammaq',
    'mxlch-gamma' : 'gammatheta',
    'mxlch-gammau' : 'gammau',
    'mxlch-gammav' : 'gammav',
    'mxlch-zi0' : 'h',
    'mxlch-latt' : 'lat',
    'mxlch-long' : 'lon',
    'mxlch-lrsAgs' : 'ls_type',
    'mxlch-lCO2Ags' : 'ls_type',
    'mxlch-qm0' : 'q',
    'mxlch-rsmin' : 'rsmin',
    'mxlch-rssoilmin' : 'rssoilmin',
    'mxlch-time' : 'tstart',
    'mxlch-lscu' : 'sw_cu',
    'mxlch-lfixedtroposphere' : 'sw_fixft',
    'mxlch-llandsurface' : 'sw_ls',
    'mxlch-lradiation' : 'sw_rad',
    'mxlch-lenhancedentrainment' : 'sw_shearwe',
    'mxlch-lsurfacelayer' : 'sw_sl',
    'mxlch-thetam0' : 'theta',
    'mxlch-hour' : 'tstart',
    'mxlch-um0' : 'u',
    'mxlch-vm0' : 'v',
    'mxlch-ug' : 'u',
    'mxlch-vg' : 'v',
    'mxlch-w2' : 'w_average',
    'mxlch-wg' :'w_average',
    'mxlch-wfc' : 'wfc',
    'mxlch-wsat' : 'wsat',
    'mxlch-wwilt' : 'wwilt',
    'mxlch-z0h' : 'z0h',
    'mxlch-z0m' : 'z0m',
    'mxlch-advtheta' : 'advt_tropo',
    'mxlch-advq' : 'advq_tropo'
    }

VARIABLE_ELISP_KEYS = {
    'mxlch-LAI',
    'mxlch-pressure',
    'mxlch-pressure_ft',
    'mxlch-T2',
    'mxlch-Ts',
    'mxlch-Tsoil',
    'mxlch-cc',
    'mxlch-day',
    'mxlch-zi0',
    'mxlch-qm0',
    'mxlch-time',
    'mxlch-thetam0',
    'mxlch-hour',
    'mxlch-um0',
    'mxlch-vm0',
    'mxlch-ug',
    'mxlch-vg',
    'mxlch-w2',
    'mxlch-wg',
    'mxlch-advtheta',
    'mxlch-advq'
}

VARIABLE_PANDAS_KEYS = set(map(lambda x: elisp_conversion[x], VARIABLE_ELISP_KEYS))

def elisp_print_dispatcher(key, series):
    """Print the value corresponding to elisp variable KEY in class input pandas SERIES

Constants is a dictionary of key values that are constant across a dataset, but vary at each
site.

TODO: make constants"""
    pandas_key = elisp_conversion[key]
    try:
        f = elisp_conversion_functions[pandas_key]
    except KeyError:
        try:
            f = elisp_conversion_functions_elisp_key[key]
        except KeyError:
            f = lambda x : x
    return f(series[pandas_key])

def write_variable(key, series, f):
    """write a variable kiven an elisp variable KEY, and a filehandle F"""

    f.write("(setq %s \"%s\")\n" % (key, elisp_print_dispatcher(key, series)))
    return True

def write_row(prefix_f, series, elisp_keys):
    """write a row of input data given SERIES.

PREFIX_F is a function that takes SERIES as argument and returns a string of a filepath

e.g., lamda x: '%d_%04d_%03d/variables.el' % (series.STNID, series.datetime.year, series.doy)

ELISP_KEYS is a list of the elisp keys that we want to write to a file.

"""
    el_path = 'data/%s'% (prefix_f(series))
    os.makedirs(os.path.dirname(el_path), exist_ok=True)
    if (not os.path.exists(el_path)):
        f = open(el_path, 'w')
        for key in elisp_keys:
            write_variable(key, series, f)
        f.close()
    return True

def write_experiment(prefix_f, df):
    """write a station's input data given a PREFIX_F (see `write_row') and its dataframe DF"""
    for (i, series) in df.iterrows():
        write_row(prefix_f, series, VARIABLE_ELISP_KEYS)
    return True

def randomize_soil_moisture(df):
    """generate a set of data where soil moisture is randomly sampled
indedpendent of the synoptics

    """
    random.seed(a=1)
    dfout = df.copy()
    dfout['w_average'] = [df['w_average'][random.randrange(df.shape[0])]
                          for _i in range(df.shape[0])]
    return dfout

def slope_experiment(_ds):
    """Take a _ds and make a causal experiment where we reduce and increase SM by 0.01"""
    _ds_neg = _ds.copy()
    _ds_neg['w_average'] = _ds.w_average - 0.01
    _ds_neg['experiment'] = -1
    _ds_pos = _ds.copy()
    _ds_pos['w_average'] = _ds.w_average + 0.01
    _ds_pos['experiment'] = 1
    _ds_out = _ds.copy()
    _ds_out['experiment'] = 0
    return pd.DataFrame(data=[_ds_neg, _ds_out, _ds_pos],
                        index=pd.MultiIndex.from_product([[_ds.datetime.year],
                                                          [_ds.doy],
                                                          [-1, 0, 1]],
                                                         names=['year',
                                                                'doy',
                                                                'experiment']))

def pandas_mapappend(_df, f):
    """Apply F to every row of _DF and concatenate the results"""
    dfs = c.deque()
    for (i, _ds) in _df.iterrows():
        dfs.append(f(_ds))
    return pd.concat(dfs)

def input_generation(site_key):
    """generate all input data and directory structure for SITE_KEY

SITE_KEY is a human name and must be a key in STATION_IDS

"""
    df = dataframe_from_records(False, load_records(site_key))

    (df_var_keys, df_constant_keys, df_none_keys) = generate_variability_keys(df)

    constant_keys = set(elisp_conversion.values()) - VARIABLE_PANDAS_KEYS

    should_be_constant = constant_keys - df_constant_keys

    df_mode = df.mode().iloc[0]

    os.makedirs('data', exist_ok=True)
    f = open('data/%s-WARNINGS.txt' % site_key, 'w')
    for key in should_be_constant:
        f.write('Even though it is not constant, using mode for site-level constant variable:\n***%s***\nStd dev: %f\nFraction of obs at mode: %f\n\n' % (key, df[key].std(), fraction_mode(df[key])))
    if (df_mode.du != 0.0):
        f.write('du not equal to zero so we should be setting geostrophic wind to something other than mixed layer wind.\ndu: %f\n\n' % df_mode.du)
    if (df_mode.dv != 0.0):
        f.write('dv not equal to zero so we should be setting geostrophic wind to something other than mixed layer wind.\ndv: %f\n\n' % df_mode.dv)
    f.close()

    write_row(lambda _df: '%s-constants.el' % site_key, df_mode,
              (set(elisp_conversion.keys()) - VARIABLE_ELISP_KEYS))


    write_experiment(lambda _df: '%s-reality-slope/%s_%04d_%03d_SM%d/variables.el'\
                     % (site_key, site_key, _df.datetime.year, _df.doy, _df.experiment),
                     pandas_mapappend(df, slope_experiment))

    write_experiment(lambda _df: '%s-randomized/%s_%04d_%03d_SM%d/variables.el'\
                     % (site_key, site_key, _df.datetime.year, _df.doy, _df.experiment),
                     pandas_mapappend(randomize_soil_moisture(df),
                                      slope_experiment))

    return True

for site in STATION_IDS.keys():
    print('*****Working on %s***\n' % site)
    input_generation(site)

f = open('%s/stations.pkl' % data_dir, 'wb')
pickle.dump(STATION_IDS, f)
f.close()
