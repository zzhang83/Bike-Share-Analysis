# Bike-Share-Analysis
Introduction:
Our team worked on bike sharing data from a program in Seattle, Washington. Kaggle provided the 3 relevant data sets. The data are segmented in several ways, and include facets such as weather on a given day, trip identification numbers, and originations/destinations of specific trips. Moreover, we supplement the data with outside information on distances between stations and outside research on this particular bike sharing program. 

Guiding questions for our analysis:
1. What distinguishes the short term pass holders from members? In terms of distance travelled, speed, behavior around holidays, etc.
2. How does weather affect bike sharing? How reliable is this data for use in future analysis?

Our overall objectives are:
1. To understand the nuances of this bike sharing program from the given data
2. form a recommendation regarding how the program might exploit the different characteristics of riders in order to operate more efficiently.

Nature of the data:
- The data came in three different Excel files: trip.csv, stations.csv, and weather.csv. Trip.csv had approximately 300,000 observations on specific trips, including the duration and origin and destination. Stations.csv contains the latitude and longitude of each of the 50 stations, as well as information on the commission status of the station. Weather.csv shows information on general weather patterns-- temperature, visibility, and events, in particular. It links these data to a particular date. 
- We supplemented sharing data with a calculation of distance between origin and destination for each trip. We used Google API and Geopy for this part of the project.

Findings and recommendations:
- We found that pass holders exhibit significantly different behavior from annual members. Short-term pass holders tend to take trips that are longer in terms of time and shorter in terms of distance. They heavily prefer biking on the weekends: pass holders on the weekend almost double. Annual members most likely use the bikes for more specific purposes.
- After analyzing the weather data, we decided that the set was not very useful for our analysis given its limited scope. The vast majority of observations are either rain or sun. However, we are able to observe usage trends across monthly data.
- Bikes tend to be picked up in the east and dropped off in the west. This trend has important implications for the business, given that it must reliably have bikes at stations when bikers need them.
- We imagine that Pronto may be able to use our analysis on demographics to price discriminate effectively by altering charging patterns by user type and by month: for example, charging more for pass holders on the weekends. 
