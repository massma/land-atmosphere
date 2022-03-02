import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import collections as c
import seaborn as sns

sns.set_theme()

sys.path.append(os.environ['CLASS4GL'])

from interface_multi import stations,stations_iterator, records_iterator,get_record_yaml,get_records

FORCING_PATH = os.environ['CLASS4GL_DATA'] + '/forcing/IGRA_PAIRS_20190515/'

STATION_IDS = {
    'albany' : 72518,
    'milano' : 16080,
    'spokane' : 72786,
    'lindenberg' : 10393,
    'idar_oberstein' : 10618,
    'kelowna' : 71203,
    'bergen' : 10238,
    'prince_george' : 71908,
    'quad_city' : 74455,
    'aberdeen' : 72659,
    'norman' : 72357,
    'shreveport' : 72248,
    'chanhassen' : 72649,
    'edwards' : 72381,
    'lincoln' : 74560,
    'gaylord' : 72634,
    'phoenix' : 74626,
    'sterling' : 72403,
    'greensboro' : 72317,
    'birmingham' : 72230,
    'peachtree' : 72215,
    'pittsburgh' : 72520,
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
    return df

input_keys = ['gammatheta', 'theta', 'dtheta',\
              'advtheta', 'gammaq', 'q', 'dq', 'advq',
              'gammau', 'gammav', 'u', 'v', 'cc',
              'Tsoil', 'advt_tropo', 'advq_tropo',
              'w2', 'wg']


def generate_variability_keys(df_input):
    """Generate all keys that vary, don't vary, and are string (or other
non-floating data) from a dataframe DF_INPUT"""
    variability_keys = c.deque()
    constant_keys = c.deque()
    not_number_keys = c.deque()
    for key in df_input.columns:
        try:
           std_dev = df_input[key].std()
           if std_dev > (abs(0.001 * df_input[key].mean())):
               variability_keys.append(key)
           else:
               constant_keys.append(key)
        except TypeError:
            not_number_keys.append(key)
    return (variability_keys, constant_keys, not_number_keys)

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

def write_station(station_id):
    """write a station's input data given a STATION_ID"""
    df = dataframe_from_records(False, load_records('kelowna'))
    for (i, series) in kelowna.iterrows():
        write_row(station_id, series)
    return True

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
        return True
    elif x == 'js':
        return False
    else:
        raise

def fortran_print_bool(x):
    if x:
        return '.true.'
    else:
        return '.false.'

elisp_conversion_functions = {
    'CO2' : ppm_to_ppb,
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
    }

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
    'mxlch-Wl' : 'Wl', # this doesn't vary and is set to default, we could remove
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
    'mxlch-gD', : 'gD',
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
    'mxlch-time' : 'runtime',
    'mxlch-lscu' : 'sw_cu',
    'mxlch-lfixedtroposphere' : 'sw_fixft',
    'mxlch-llandsurface' : 'sw_ls',
    'mxlch-lradiation' : 'sw_rad',
    'mxlch-lenhancedentrainment' : 'sw_shearwe',
    'mxlch-lsurfacelayer' : 'sw_sl',
    'mxlch-thetam0' : 'theta',
    'mxlch-hour' : 'tstart',
    'mxlch-um0' : 'u',
    'mxlch-uv0' : 'v',
    'mxlch-ug' : 'u',
    'mxlch-vg' : 'v',
    'mxlch-w2' : 'w2',
    'mxlch-wcsmax' : 'wCO2', # same comment as wtheta; these are used slightly differently
    'mxlch-wfc' : 'wfc',
    'mxlch-wg' :'wg',
    'mxlch-wqsmax' : 'wq', # same comment as wtheta; these are used
                           # slightly differently; we may also want to
                           # set these as offsets and use a constant
                           # flux function??? we really need to
                           # understand how 'wq' gets sued, and how
                           # the offset functions get used. so onl
    'mxlch-wsat' : 'wsat',
    'mxlch-wthetasmax' : 'wtheta', # this might not be right
    'mxlch-wwilt' : 'wwilt',
    'mxlch-z0h' : 'z0h',
    'mxlch-z0m' : 'z0m',
    'mxlch-advtheta' : 'advt_tropo',
    'mxlch-advq' : 'advq_tropo',

    }

def write_row(station_id, series):
    """write a row of input data given STATION_ID and SERIES"""
    # checks
    if ((series.du != 0.0) or (series.dv != 0.0)):
        raise RuntimeWarning("du/dv is not euqal to zero so we ahve to set geostrophic wind to
somethign other than mixed layer wind")
    pwd = "data/%s_%d_%03d" % (station_id, series.STNID, series.doy)
    os.makedirs(pwd)
    f = open("%s/input.el" % pwd, 'w')
    for key in elisp_conversion.keys():
        write_variable(key, series, f)
    f.close()
    return True

def write_variable(key, series, f):
    """write a variable kiven a KEY to SERIES, and a filehandle F"""
    pandas_key = elisp_conversion[key]
    f.write("(setq %s \"%s\")\n" % (key, elisp_conversion_functions[pandas_key](
        series[pandas_key])))
    return True

kelowna = dataframe_from_records(False, load_records('kelowna'))

for (i, series) in kelowna.iterrows():
    print(series)

for i in kelowna:
    print(i)
