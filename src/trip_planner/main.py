#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from trip_planner.crew import TripPlannerCrew

def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'destination': 'Kashmir, India',
        'age': 31,
        'travel_date': "13-04-2024",
        'trip_duration': '5 days'
    }
    result = TripPlannerCrew().crew().kickoff(inputs=inputs)
    print(result)