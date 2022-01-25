import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt


path = os.environ['HOME'] + "/class/data/forcing/IGRA_PAIRS_20190515"
# largest file in database
# even the largest files don't ahve any variation in dthetea, gamma theta, etc.
# file= open(path + "/11520_diag.pkl", "rb")

# albany
# file = open(path + "/72518_diag.pkl", "rb")
# file = open(path + "/72518_ini.pkl", "rb")
# file = open(path + "/72518_end.pkl", "rb")
# new york city
file = open(path + "/72501_diag.pkl", "rb")
df = pd.read_pickle(file)
file.close()

# we only want morning soundings, and those with realistic temperature for spring,summer,fall
morning = df[(df.tstart < 12.0) & (df.theta > 240.0)]

def print_stats(_df):
    for i in _df.columns:
        try:
            mean = _df[i].mean()
            std = _df[i].std()
            if std > abs((0.01*mean)):
                print("\n%s" % i)
                print(mean)
                print(std)
            else:
                print("\nNO VARIABILITY in: %s" % i)
        except:
            True

print_stats(morning)

# define spring as march april may;
spring = morning[((morning.doy >= 60) & (morning.doy < 152))]

# define summer as june july august
summer = morning[((morning.doy >= 152) & (morning.doy < 244))]

# define fall as sep oct nov
fall = morning[((morning.doy >= 244) & (morning.doy < 335))]

print(morning.advtheta.std())
print(morning.gammatheta.std())

# for _df,title in zip([spring, summer, fall], ["spring", "summer", "fall"]):
#     # tc = _df.theta - 273.15
#     for var in ['theta', 'q', 'h', 'u', 'v', 'cc' 'z0h', 'z0m']:
#         plt.figure()
#         _df[var].hist(bins=20)
#         plt.title("%s %s" % (title, var))

# plt.show()


######
## do some stuff with albany

# albany
# file = open(path + "/72518_diag.pkl", "rb")
# file = open(path + "/72518_ini.pkl", "rb")
# file = open(path + "/72518_end.pkl", "rb")
# albany = pd.read_pickle(file)
# file.close()

# print_stats(albany)
