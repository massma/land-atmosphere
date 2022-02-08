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

ALBANY_ID = 72518


MILANO_ID = 16080
# spokane has 915 reocords
SPOKANE_ID = 72786
# near berlin 974 records
LINDENBERG_ID = 10393
# west germany near belguim/luxemborg 1014 records
IDAR_OBERSTEIN_ID = 10618
# slightly norther british columbia, prob similar to spokane
KELOWNA_ID = 71203
# northern germany
BERGEN_ID = 10238
# northern britich columbia
PRINCE_GEORGE_ID = 71908
# Iowa
QUAD_CITY_ID = 74455
# south dakota 472 records
ABERDEEN_ID = 72659
# oklahoma 441 reocrds
NORMAN_ID = 72357
# loisiana 383 records
SHREVEPORT_ID = 72248
# minetota 323 records
CHANHASSEN_ID = 72649
# califronia (315 records)
EDWARDS_ID = 72381
# illinois (237 records)
LINCOLN_ID = 74560
GAYLORD_ID = 72634
PHOENIX_ID = 74626
STERLING_ID = 72403
GREENSBORO_ID = 72317
BIRMINGHAM_ID = 72230
PEACHTREE_ID = 72215
PITTSBURGH_ID = 72520

STATION_ID = SPOKANE_ID

def generic_path(station_id, suffix='yaml'):
    """Generate a path given a STATION_ID and SUFFIX (e.g., yaml or pkl)."""
    return "%s/%05d_%s.%s" % (FORCING_PATH, station_id, 'ini', suffix)

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


f = open(generic_path(STATION_ID, suffix='pkl'), 'rb')
df = pd.read_pickle(f)
f.close()

f = open(generic_path(STATION_ID, suffix='yaml'), 'r')

records = c.deque()

for (_station_id,_chunk,_index),record in df.iterrows():
    class_input = load_record(f, record)
    records.append(class_input)
f.close()

print("\n****Length****:\n%10d" % len(records))

def grab_record(key, records):
    """Grab a list of objects from RECORDS corresponding to KEY"""
    return [x.pars.__dict__[key] for x in records]

# where are 'zi0', 'wsls', ug, ug
# also confirm all equivalencies are right

def pandas_dataframe(keys, records):
    """Generate a pandas datafrae from KEYS in RECORDS"""
    return pd.DataFrame(dict([(key, grab_record(key, records)) for key
                              in keys]))

input_keys = ['gammatheta', 'theta', 'dtheta',\
              'advtheta', 'gammaq', 'q', 'dq', 'advq',
              'gammau', 'gammav', 'u', 'v', 'cc',
              'Tsoil', 'advt_tropo', 'advq_tropo',
              'w2', 'wg']


variability_keys = ['theta', 'q', 'u', 'v', 'cc',
                    'Tsoil', 'advt_tropo', 'advq_tropo',
                    'w2', 'wg']

df_input = pandas_dataframe(variability_keys, records)

corr = df_input.corr()

for key in df_input.columns:
    std_dev = df_input[key].std()
    if std_dev > (abs(0.001 * df_input[key].mean())):
        print("\n%s std: %15f" % (key, std_dev))
    else:
        print("no variability for %s" % key)


for key in corr.columns:
    print("\n*****%s*****" % key)
    print(corr[key])
