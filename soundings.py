import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import collections as c

sys.path.append(os.environ['CLASS4GL'])

from interface_multi import stations,stations_iterator, records_iterator,get_record_yaml,get_records

FORCING_PATH = os.environ['CLASS4GL_DATA'] + '/forcing/IGRA_PAIRS_20190515/'

ALBANY_ID = 72518

SPOKANE_ID = 72786
MILANO_ID = 16080
NORMAN_ID = 72357
ABERDEEN_ID = 72659
SHREVEPORT_ID = 72248
EDWARDS_ID = 72381
LINCOLN_ID = 74560
GAYLORD_ID = 72634
PHOENIX_ID = 74626
STERLING_ID = 72403
GREENSBORO_ID = 72317
BIRMINGHAM_ID = 72230
PEACHTREE_ID = 72215
PITTSBURGH_ID = 72520

STATION_ID = PITTSBURGH_ID

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
    records.append(load_record(f, record))

f.close()

print("\n****Length****:\n%10d" % len(records))
