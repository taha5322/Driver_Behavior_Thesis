"""
UNIMPLEMENTED: POINT OF GAZE PRE-PROCESSING W/ KALMAN'S FILTER
This aims to remove noise in gaze data due to the impreciseness of the imaging tool 
"""


import os 
import csv
from filterpy.kalman import KalmanFilter
import numpy as np
import pandas as pd



pog_path = "/home/tsiddi5/projects/def-bauer/tsiddi5/driver14_data/pog.csv"

kf = KalmanFilter(dim_x=4, dim_z=2)
kf.x = np.array([0, 0, 0, 0])  # Initial state (x, y, dx, dy)
kf.F = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]])  # Transition matrix
kf.H = np.array([[1,0,0,0], [0,1,0,0]])  # Measurement function
kf.P *= 1000  # Covariance matrix
kf.R = np.eye(2)  # Measurement noise
kf.Q = np.eye(4)  # Process noise

gaze_data = pd.read_csv(pog_path)
smoothed = []
output_path = "/home/tsiddi5/projects/def-bauer/tsiddi5/code/object_gaze_matching/prep_gaze.csv"


with open(output_path, "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["frame", "x_smooth", "y_smooth"])  # Write header

    # Iterate through rows
    for index, row in gaze_data.iterrows():
        print(index, row)
        f_number = row['f_number']
        x = row['x_position']
        y = row['y_position']
        print("curr_x")
        print(x, y)
        kf.predict()
        kf.update([x, y])
        new_vals =   kf.x[:2].tolist()
        writer.writerow([ f_number, new_vals[0], new_vals[1] ])
