#!/usr/bin/env python3

'''
Implement function call limiting for multiple scenario choices
    1. Burst rate limit on ROWS of data (# of train/test examples passed through the system)
    2. Generic function for releasing f(x) rows with a g(x) sleep interval
'''

from ratelimit import limits, sleep_and_retry
from time import sleep
import random
import numpy as np

ONE_SEC = 1

############################################################
# GENERIC FUNCTIONS - For utility use

## Requirements
# - row_generator_fx returns n rows when called with parameter n. Defaults to 1.
############################################################


def construct_generic_limited_rows(row_generator_fx, rows_function, sleep_function):
    '''
    Case 2: Generic function
    - row_generator_fx returns n rows when called with parameter n. Defaults to 1.
    - rows_function(i) returns the numbers of rows we should return in the i-th step
    - sleep_function(i) returns how long we should sleep in the i-th step
    Returns a function that will return data at a known rate
    '''
    i = 0
    def inner():
        nonlocal i
        # Generate f(i) rows
        rows_to_generate = rows_function(i)
        rows = row_generator_fx(rows_to_generate)
        # Sleep for g(i) seconds
        sleep_period = sleep_function(i)
        # print("Sleep: " + str(sleep_period))
        sleep(sleep_period)
        # Increment for next generator iteration
        i = i + 1
        # Return data
        yield rows
    return inner

def file_row_generator(filename, num_rows):
    '''
    Compatible file generator that returns a certain number of rows each time
    '''
    from itertools import islice
    fileobj = open(filename, "r")
    data =  islice(fileobj, 0, num_rows)
    def inner():
        nonlocal fileobj
        try:
           return next(data)
        except StopIteration:
            return None

        

############################################################
# END GENERIC FUNCTIONS
############################################################



############################################################
# Public interface

## Requirements
# - row_generator_fx returns n rows when called with parameter n. Defaults to 1.
############################################################


def construct_ratelimit_rows(row_generator_fx, max_rows_per_minute, blocking=True):
    '''
    Case 1:
    - set a rate limit (min 1 per second)
    - option to make function blocking

    row_generator_fx releases 1 row per call (return, not yield)
    '''
    row_generator_fx = limits(calls=max_rows_per_minute, period=ONE_SEC)(row_generator_fx)
    if blocking:
        row_generator_fx = sleep_and_retry(row_generator_fx)
    return row_generator_fx
    

def construct_randomsleep_rows(row_generator_fx, min_sleep, max_sleep, rows=1):
    '''
    Returns `rows` rows every uniformly random number of seconds
    '''
    rows_fx = lambda x: rows
    return construct_generic_limited_rows(row_generator_fx, rows_fx,
                                          lambda x: random.uniform(min_sleep, max_sleep))

def construct_randnormsleep_rows(row_generator_fx, min_sleep, max_sleep, rows=1):
    '''
    Returns `rows` rows every vaguely normally distributed random number of seconds
    '''
    rows_fx = lambda x: rows
    mid_sleep = (max_sleep - min_sleep) / 2
    # Magic number for sd as a percentage of high-low range
    sd = (max_sleep - min_sleep) / 4
    def vague_normal():
        return min(max(random.gauss(mid_sleep, sd), 0), max_sleep)
    return construct_generic_limited_rows(row_generator_fx, rows_fx,
                                          lambda x: vague_normal())

def construct_randbetasleep_rows(row_generator_fx, min_sleep, max_sleep, alpha=1, beta=3, rows=1):
    '''
    Returns `rows` rows every beta-distributed random number of seconds
    '''
    rows_fx = lambda x: rows
    def beta_fx():
        return random.betavariate(alpha, beta) * max_sleep + min_sleep
    return construct_generic_limited_rows(row_generator_fx, rows_fx,
                                          lambda x: beta_fx())

def construct_sinusoidial_rows(row_generator_fx, amplitude, frequency, timestep):
    '''
    Generates a positive sine valued response of peak = 2 * amplitude and delta_T = timestep
    '''
    sleep_fx    = lambda x: 0
    rows_fx     = lambda x: int(amplitude + amplitude * np.sin((x * timestep) * (2 * np.pi * frequency)))
    return construct_generic_limited_rows(row_generator_fx, rows_fx, sleep_fx)

def construct_sinusoidial_rows_integrated(row_generator_fx, amplitude, frequency, timestep=None):
    '''
    Generates a positive sine valued response of peak = 2 * amplitude and delta_T = timestep
    Integrate between timesteps.
    Timestep also defines how fast we call the function: smaller step = more calls
    '''

    # Sleep for timestep units
    sleep_fx    = lambda x: timestep

    def sin_integral(t):
        # Integral computed from: https://www.wolframalpha.com/input/?i=integrate+with+respect+to+t%2C+f%28t%29+%3D+A+%2B+A*sin%282+*+pi+*+f+*+t%29
        # Amplitude, frequency, time
        A = amplitude
        f = frequency
        return (A * t) - (A * np.cos(2 * np.pi * f * t)) / (2 * np.pi * f)
        
    def rows_fx(t):
        # Integrate between timesteps
        return int(sin_integral(t * timestep) - sin_integral((t-1) * timestep))
        
    return construct_generic_limited_rows(row_generator_fx, rows_fx, sleep_fx)

# Shorthand alias
construct_sin_stream = construct_sinusoidial_rows_integrated

############################################################
# End Public interface
############################################################



############################################################
# Testing
############################################################

# Constant 1 - function for one row once a second
one_fx = lambda x: 1

# Example generator function
def test_row_generator(num_rows=1):
    rows = []
    for i in range(num_rows):
        rows.append([random.randint(0, 20), random.randint(0, 20)])
    return rows

if __name__ == "__main__":
    # Testing rate-limited function
    limited_fx = construct_ratelimit_rows(test_row_generator, 2)

    # Testing generically limited function (one row per second)
    #limited_fx_generic = construct_generic_limited_rows(test_row_generator, one_fx, one_fx)
    # limited_fx_generic = construct_randomsleep_rows(test_row_generator, 0, 1)
    # limited_fx_generic = construct_randnormsleep_rows(test_row_generator, min_sleep=0, max_sleep=1, rows=1)
    # limited_fx_generic =  construct_randbetasleep_rows(test_row_generator, 0, 5, alpha=1, beta=10)

    # Sinusoidal test - amplitude 50 frequency 1 ==> 50 samples per second. Timestep 0.5 = 
    limited_fx_generic = construct_sin_stream(test_row_generator, amplitude=10, frequency=1, timestep=0.25)

    while True:
        print(next(limited_fx_generic()))



