Air Quality Data Collection and Feature Engineering

This project collects, cleans, and processes air quality, weather, and location data for modeling and analysis.

Modules
Module 1: Data Collection

Air Quality: PM2.5, PM10, NO₂, CO, SO₂, O₃ via OpenAQ API

Weather: Temperature, humidity, wind speed/direction via OpenWeatherMap API

Location Features: Roads, industrial zones, dump sites, agricultural fields via OSMnx

Data Storage: Tagged with latitude, longitude, timestamp, source; saved as CSV/JSON

Module 2: Data Cleaning & Feature Engineering

Remove duplicates and invalid records

Handle missing values, standardize units, normalize data

Add spatial (distance to road/industry/dump site) and temporal (hour, day, season) features

Merge all data into a single structured dataset
