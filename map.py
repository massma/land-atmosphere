#! /usr/bin/python3
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import matplotlib.pyplot as plt
import numpy as np

sites = { \
          'Bergen' : (52.8153, 9.9247),
          'Idar Oberstein' :(49.6928, 7.3264),
          'Lindenberg' : (52.2167, 14.1167),
          'Milano' :( 45.4614, 9.2831),
          'Kelowna' : (49.9408, -119.4003),
          'Quad City' : (41.6114, -90.5817),
          'Spokane' : (47.6806, -117.6267),
          'Flagstaff' : (35.23, -111.8217),
          'Elko' : (40.8600, -115.7422),
          'Las Vegas' : (36.05, -115.1833),
          'Riverton' : (43.0647, -108.4767),
          'Great Falls' : (47.4614, -111.3847),
          }

europe = {'Bergen', 'Idar Oberstein', 'Lindenberg', 'Milano'}
north_america = { 'Kelowna',
                  'Quad City',
                  'Spokane',
                  'Flagstaff',
                  'Elko',
                  'Las Vegas',
                  'Riverton',
                  'Great Falls',
                 }

fig = plt.figure()
fig.set_figheight(fig.get_figheight()*2.0)

# europe
lats = [sites.get(site)[0] for site in europe]
lons = [sites.get(site)[1] for site in europe]
proj = ccrs.Orthographic(central_latitude=np.average(lats),
                         central_longitude=np.average(lons))
transform = ccrs.Geodetic()
ax = fig.add_subplot(211, projection=proj)
ax.set_extent([np.floor(min(lons))-18,
               np.ceil(max(lons))+10,
               np.floor(min(lats))-10,
               np.ceil(max(lats))+10])
# ax.gridlines()
ax.coastlines(resolution='50m')
ax.plot(lons, lats, 'k.', transform=transform)
for (lat, lon, label) in zip(lats, lons, europe):
    if label == 'Lindenberg':
        ax.text(lon, lat, label, transform=transform,
                verticalalignment='top')
    else:
        ax.text(lon, lat, label, transform=transform)
# below kind of expensive, only save for publications
ax.stock_img()
ax.set_title('Sites in Europe')

# north_america
lats = [sites.get(site)[0] for site in north_america]
lons = [sites.get(site)[1] for site in north_america]
proj = ccrs.Orthographic(central_latitude=np.average(lats),
                         central_longitude=np.average(lons))
transform = ccrs.Geodetic()
ax = fig.add_subplot(212, projection=proj)
ax.set_extent([np.floor(min(lons))-10,
               np.ceil(max(lons))+5,
               np.floor(min(lats))-3,
               np.ceil(max(lats))+3])
# ax.gridlines()
ax.coastlines(resolution='50m')
ax.plot(lons, lats, 'k.', transform=transform)
for (lat, lon, label) in zip(lats, lons, north_america):
    if label in ['Flagstaff', 'Great Falls']:
        ax.text(lon, lat, label, transform=transform,
                verticalalignment='top')
    else:
        ax.text(lon, lat, label, transform=transform)

# below kind of expensive, only save for publications
ax.stock_img()
ax.set_title('Sites in North America')
plt.savefig('figs/map.pdf', bbox_inches='tight')
# plt.show()
