# Cycle Share Dataset
###  Bicycle Trip Data from Seattle's Cycle Share System
"""
The project subject will not be assigned. Instead, your team will collaborate to determine a project.  Features of a project should include:  
1. A clear goal (or goals) and/or question(s) to answer. The more interesting and challenging, the better.

2. Identifiable data that can be obtained in support of the project.

3. A substantial data cleaning component, required prior to analysis or for the actual analysis (or both).

4. Analytic and visualization methods developed in this course our an introductory statistics course (the prerequisite) should be employed.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
from itertools import chain
from functools import wraps
import datetime
import sklearn
from scipy import stats
import statsmodels
import requests
import sympy
import bokeh
import scrapy
import os
import networkx
import re
import bs4


# The data to load
STATION_DATA = pd.read_csv('station.csv')
TRIP_DATA = pd.read_csv('trip.csv')
WEATHER_DATA = pd.read_csv('weather.csv')

STATION_DATA.info()
TRIP_DATA.info()
WEATHER_DATA.info()

# Count the lines
num_lines_trip = sum(1 for line in open(TRIP_DATA))

# Sample size - in this case 5%
trip_sample_n = int(num_lines_trip * 0.05)

# The row indices to skip - make sure 0 is not included to keep the header!
skip_idx = random.sample(range(1, num_lines_trip), num_lines_trip - trip_sample_n)

# Read the data
trip_df_raw = pd.read_csv(TRIP_DATA, skiprows=skip_idx)
print("Variables: {1}\nObservations: {0}\nFirst five rows:\n{2}\n".format(
    trip_df_raw.shape[0], trip_df_raw.shape[1], trip_df_raw.head(5)))
print(trip_df_raw.describe())


# Cleaning for station name
def station_name_to_alias(station_name):
    alias_name = station_name.split(" / ")
    return alias_name[0] if len(alias_name) > 1 else None


def simplify_station_name(station_name):
    alias_name = station_name.split(" / ")
    return alias_name[1] if len(alias_name) > 1 else alias_name[0]


trip_df = trip_df_raw.copy()
trip_df['from_station_alias'] = trip_df['from_station_name'].apply(station_name_to_alias)
trip_df['to_station_alias'] = trip_df['to_station_name'].apply(station_name_to_alias)
trip_df['from_station_name'] = trip_df['from_station_name'].apply(simplify_station_name)
trip_df['to_station_name'] = trip_df['to_station_name'].apply(simplify_station_name)

print(trip_df.sample(10))

# cleaning for datetime
print(trip_df['starttime'][0])
print(type(trip_df['starttime'][0]))


def to_pd_datetime(dt_string):
    pd_date = pd.to_datetime(dt_string).date()
    pd_time = pd.to_datetime(dt_string).time()
    return pd_date, pd_time


def get_date(dt_string):
    return to_pd_datetime(dt_string)[0]


def get_time(dt_string):
    return to_pd_datetime(dt_string)[1]

trip_df['start_date'] = trip_df['starttime'].apply(get_date)
trip_df['start_time'] = trip_df['starttime'].apply(get_time)
trip_df['stop_date'] = trip_df['stoptime'].apply(get_date)
trip_df['stop_time'] = trip_df['stoptime'].apply(get_time)
del trip_df['starttime'], trip_df['stoptime']
print(trip_df.sample(10))
