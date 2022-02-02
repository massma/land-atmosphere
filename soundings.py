import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append(os.environ['CLASS4GL'])

from interface_multi import stations,stations_iterator, records_iterator,get_record_yaml,get_records

path = os.environ['CLASS4GL_DATA'] + '/forcing/IGRA_PAIRS_20190515/'
path_output = os.environ['CLASS4GL_DATA'] + '/experiments/IGRA_PAIRS_20190515/BASE/'


# loading station information
# second argument is which subset of statiosn to initialize with; e.g., 'ini' or 'morning'
# all_stations.table has a table (dataframe) of all the statiosn int he dataset
all_stations = stations(path, suffix='ini', refetch_stations=False)

stations_iter = stations_iterator(all_stations)
STNID,run_station = stations_iter.set_STNID(STNID=int(72518))

# this is just a dataframe of all the stations we want to work on, we
# could alternatiely load all stations with
# pd.DataFrame(all_stations.table)
all_stations_select = pd.DataFrame([run_station])

# records morning is just a dataframe, which includes the indices of
# each available sounding (index_start and index_end)
# we could check and see if we can just get that index information from the _ini.pickle
# file/dataframe
records = get_records(all_stations_select,\
                              path,\
                              subset='ini',\
                              refetch_records=False,\
                              )
records_station = records.query('STNID == ' + str(72518))
path = "%s/%05d_%s.yaml" % (path, 72518, 'ini')
f = open(path, 'r')

for (STNID,chunk,index),record in records_station.iterrows():
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


f.close()
