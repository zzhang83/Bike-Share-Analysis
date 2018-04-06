import numpy as np
from pandas import DataFrame
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import seaborn as sns
STATION = pd.read_csv('station.csv')
station = STATION.copy(deep=True)
WEATHER= pd.read_csv('weather.csv')
weather = WEATHER.copy(deep=True)
TRIP = pd.read_csv('trip.csv',
                    parse_dates=['starttime', 'stoptime'],
                    infer_datetime_format=True)
trip = TRIP.copy(deep=True)

trip['minutes'] = trip.tripduration / 60
a = pd.DatetimeIndex(trip.starttime)
trip['date'] = a.date.astype('datetime64')
trip['hour'] = a.hour
#%%
# summary of the data set:
len(trip)    #236065 trips


a = np.sum(trip.tripduration)/60    # total riding time 4731577.523566707 hours

start = pd.to_datetime(trip.starttime)

end = pd.to_datetime(trip.stoptime)

duration = end-start

np.mean(duration)        # average riding time 20:02

start[236064] - start[0] 

len(trip[trip.usertype=='Member'])/len(trip)



# In those 688 days, there are 236065 trips, total riding time:  4731577.523566707 hours, average 20.02 minutes per trip

#%%  calculate distance (mile) between stations 

#  This is the shortest possible distance between stations, which is a lower bound on the actual distance ridden on each trip. 

# build matrix using station data set:

from geopy.distance import great_circle
distance= np.zeros((58,58))
for j in range(len(station)):
    for i in range(len(station)):
        origin = (station.lat[j],station.long[j])  
        destination = (station.lat[i],station.long[i])
        distance[j][i]=great_circle(origin,destination).miles
        
names = station.station_id        
distance = pd.DataFrame(distance, columns = names)
distance = distance.set_index([names])

# 3*3 section of distance matrix looks like:

distance.iloc[:6, :6]

'''
station_id     BT-01     BT-03     BT-04
station_id                              
BT-01       0.000000  0.210994  0.486713
BT-03       0.210994  0.000000  0.348127
BT-04       0.486713  0.348127  0.000000
'''
# distance is 0 when the two stations are the same
stacked = distance.stack()
stacked.name = 'distance'

#combine the distance with trip data set, then we now the distance for each trip, we can further calculate the speed

trip = trip.join(stacked, on=['from_station_id', 'to_station_id'])

# Visualization: 

# trip distances by usertype

fig, ax = plt.subplots(figsize=(8,4))
trip.groupby('usertype')['distance'].hist(bins=np.linspace(0, 6, 40),
            alpha=0.5, ax=ax);
plt.title('Distance of Trips',fontsize=20)
plt.xlabel('Distance between start & end (miles)',fontsize=15)
plt.ylabel('relative frequency',fontsize=15)
plt.legend(['Annual Members', 'Short-term Pass'],fontsize=15);


fig, ax = plt.subplots(figsize=(12, 4))
trip.groupby('gender')['distance'].hist(bins=np.linspace(0, 6.99, 50),
                                           alpha=0.5, ax=ax);
a = trip['distance'].groupby(trip.usertype).mean()
a.plot('hist')
(0.850041-0.778983)/0.778983

# speed by usertype
trip['speed'] = trip.distance * 60 / trip.minutes
trip.groupby('usertype')['speed'].hist(bins=np.linspace(0, 13, 50), alpha=0.5, normed=True);
plt.xlabel('riding speed (MPH)',fontsize=15)
plt.ylabel('relative frequency',fontsize=15)
plt.title('Rider Speed Distribution',fontsize=20)
plt.legend(['Annual Members', 'Short-term Pass'],fontsize=15);                                          
                                         
#%%
# Add speed column to trip data set                                         
trip['speed'] = trip.distance * 60 / trip.minutes

trip.speed.groupby(trip.usertype).mean()
# want to predict gender based on speed using support vector machine
from sklearn import svm
from sklearn import preprocessing

x = {'speed':trip.speed, 'gender':trip.gender}
x = DataFrame(x, columns=['speed','gender'])
x = x.sample(5000)
combine = x[x.speed>0]

d1 = combine[combine.gender=='Female']
d2 = combine[combine.gender=='Male'][1:650]
d3 = combine[combine.gender=='Other']

frames = [d3,d1,d2]
train = pd.concat(frames)

encoder= preprocessing.LabelEncoder()
train.gender = encoder.fit_transform(train.gender)

trainx = list(train.speed)
for i in range(0,len(trainx)):
    trainx[i] = [trainx[i]]
trainy = list(train.gender)

# svm
clf = svm.SVC(C = 5.0,kernel = 'poly',degree=3)
clf.fit(trainx, trainy) 

x = {'speed':trip.speed, 'gender':trip.gender}
x = DataFrame(x, columns=['speed','gender'])
inds = pd.isnull(x).any(1).nonzero()[0]
miss = list()
for i in inds:
    miss.append(x.ix[i].speed)
    
for i in range(0,len(miss)):
    miss[i] = [miss[i]]

predict = clf.predict(miss)

# fill in nan
for i in range(len(inds)):
    index = inds[i]
    trip.gender[index]=predict[i]

#%%
# try to find most popular stations by counting the number of occurrence

starting = trip.trip_id.groupby(trip.from_station_name).count()
starting.sort(['trip_id'],ascending=False)

destinations = trip.trip_id.groupby(trip.to_station_name).count()
destinations.sort(['trip_id'],ascending=False)

# most popular starting station
'''
Pier 69 / Alaskan Way & Clay St           13054
E Pine St & 16th Ave                      11392
3rd Ave & Broad St                        10934
2nd Ave & Pine St                         10049
Westlake Ave & 6th Ave                     9994
E Harrison St & Broadway Ave E             9639
Cal Anderson Park / 11th Ave & Pine St     9468
REI / Yale Ave N & John St                 8382
2nd Ave & Vine St                          8168
15th Ave E & E Thomas St                   7680
'''
# most popular destination
'''
2nd Ave & Pine St                                       13784
Pier 69 / Alaskan Way & Clay St                         13736
Westlake Ave & 6th Ave                                  10962
3rd Ave & Broad St                                      10737
PATH / 9th Ave & Westlake Ave                           10632
Occidental Park / Occidental Ave S & S Washington St     9584
Republican St & Westlake Ave N                           9305
Pine St & 9th Ave                                        9114
Seattle Aquarium / Alaskan Way S & Elliott Bay Trail     8931
1st Ave & Marion St                                      8713
'''


#%%

# Visualization

# trip durations for Annual members and short-term pass holders:

trip['minutes'] = trip.tripduration / 60

trip.groupby('usertype')['minutes'].hist(bins=np.arange(63), alpha=0.4, normed=True);



plt.title('Trip Durations ( in minutes )',fontsize=20)

plt.xlabel('Duration',fontsize=15)

plt.ylabel('relative frequency',fontsize=15)

plt.legend(['Annual Members', 'Short-term Pass'],fontsize=15)



plt.axvline(30, linestyle='--', color='red', alpha=0.3)

plt.text(28, 0.09, "Free Trips", ha='right', size=15, alpha=0.5, color='red');


# percentage of 
len(trip[trip.minutes>30]) / len(trip)
trip[trip.user

m = trip[trip.usertype=='Member']
len(m.minutes>30)/len(m)

trip[(trip.usertype=='Member') & (trip.minutes > 30)]
len(trip[[trip.usertype=='Member'] & [trip.minutes > 30]]) /len(trip[trip.usertype=='Member'])
len(len(trip[trip.usertype=='Short-Term Pass Holder'][trip.minutes>30]) /len(trip[trip.usertype=='Short-Term Pass Holder']))
#%%
# how holidays affect the number of trips

from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays('2014-10', '2016-7', return_name=True)
holidays_all = pd.concat([holidays,
                          "2 Days Before " + holidays.shift(-2, 'D'),
                          "Day Before " + holidays.shift(-1, 'D'),
                          "Day After " + holidays.shift(1, 'D')])
holidays_all = holidays_all.sort_index()
holidays_all.head()

'''
2014-10-11    2 Days Before Columbus Day
2014-10-12       Day Before Columbus Day
2014-10-13                  Columbus Day
2014-10-14        Day After Columbus Day
2014-11-09    2 Days Before Veterans Day
'''
# merge with trip data set
holidays_all = pd.DataFrame(holidays_all)
holidays_all['date']= holidays_all.index

trip = trip.merge(holidays_all, on = 'date',how = 'outer')

trip.head()



from ggplot import *

ggplot(trip, aes(x='date',y='tripduration')) +\
    geom_point(color='blue') +\
    stat_smooth(method='lm', se='False')


#%%
ind = pd.DatetimeIndex(trip.date)
ind.dayofweek

trip['weekend'] = (ind.dayofweek > 4)
hourly = trip.pivot_table('trip_id', aggfunc='count',
                           index=['date'], columns=['usertype', 'weekend', 'hour'])
fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
fig.subplots_adjust(wspace=0.1)
fmt = plt.FuncFormatter(lambda x, *args: '{0}:00'.format(int(x)))

color_cycle = plt.rcParams['axes.color_cycle']

for weekend in (False, True):
    axi = ax[int(weekend)]
    for i, col in enumerate(['Member', 'Short-Term Pass Holder']):
        vals = hourly[col][weekend].values
        vals = np.hstack([vals, vals[:, :1]])
        axi.plot(range(25), vals.T,
                 color=color_cycle[i], lw=1, alpha=0.05)
        axi.plot(range(25), np.nanmean(vals, 0),
                 color=color_cycle[i], lw=3, label=col)
    axi.xaxis.set_major_locator(plt.MultipleLocator(4))
    axi.xaxis.set_major_formatter(fmt)
    axi.set_ylim(0, 60)
    axi.set_title('Saturday - Sunday' if weekend else 'Monday - Friday')
    axi.legend(loc='upper left')
    axi.set_xlabel('Time of Day')
ax[0].set_ylabel('Number of Trips')
fig.suptitle('Hourly Trends: Weekdays and Weekends', size=14);


#%%
#  Visualization
#  This visualization shows the daily trend, separated by Annual members (top) and Day-Pass users (bottom). 
#  A couple observations:
#  Day pass users seem to show a steady increase and decrease in bike trip with the seasons 
#  The useage of annnual users didn't change significantly onver time.
#  Both annual members and day-pass users seem to show a distinct weekly trend, which can be observed from the left-top window
#  Anunal users tend to have higher usage during weekdays than weekend
by_date = trip.pivot_table('trip_id', aggfunc='count',
                            index='date',
                            columns='usertype', )

# Count trips by weekday
weekly = by_date.pivot_table(['Member', 'Short-Term Pass Holder'],
                             index=by_date.index.weekofyear,
                             columns=by_date.index.dayofweek)

color_cycle = plt.rcParams['axes.color_cycle']
fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.1)


color_cycle = plt.rcParams['axes.color_cycle']
for i, col in enumerate(['Member', 'Short-Term Pass Holder']):
    by_date[col].plot(ax=ax[i], title=col, color=color_cycle[i])
    ax[i].set_title(col + 's')

             
def add_inset(ax, rect, *args, **kwargs):
    box = ax.get_position()
    inax_position = ax.transAxes.transform(rect[0:2])
    infig_position = ax.figure.transFigure.inverted().transform(inax_position)
    new_rect = list(infig_position) + [box.width * rect[2], box.height * rect[3]]
    return fig.add_axes(new_rect, *args, **kwargs)
with sns.axes_style('whitegrid'):
    inset = [add_inset(ax[0], [0.07, 0.6, 0.2, 0.32]),
             add_inset(ax[1], [0.07, 0.6, 0.2, 0.32])]
             
for i, col in enumerate(['Member', 'Short-Term Pass Holder']):
    inset[i].plot(range(7), weekly[col].values.T, color=color_cycle[i], lw=2, alpha=0.05);
    inset[i].plot(range(7), weekly[col].mean(0), color=color_cycle[i], lw=3)
    inset[i].set_xticklabels(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
    inset[i].yaxis.set_major_locator(plt.MaxNLocator(5))
    inset[i].set_ylim(0, 500)
    inset[i].set_title('Average by Day of Week')
