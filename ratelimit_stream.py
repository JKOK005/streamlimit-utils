#!/usr/bin/env python3

'''
Implement function call limiting for multiple scenario choices
    1. Burst rate limit on ROWS of data (# of train/test examples passed through the system)
    2. Generic function for releasing f(x) rows with a g(x) sleep interval
'''

from ratelimit import limits, sleep_and_retry
import random
from time import sleep

ONE_SEC = 1

############################################################
# GENERIC FUNCTIONS - For utility use
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


def construct_generic_limited_rows(row_generator_fx, rows_function, sleep_function):
    '''
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
        print("Sleep: " + str(sleep_period))
        sleep(sleep_period)
        # Increment for next generator iteration
        i = i + 1
        # Return data
        return rows
    return inner

############################################################
# END GENERIC FUNCTIONS
############################################################



############################################################
# Public interface

## Requirements
# - row_generator_fx returns n rows when called with parameter n. Defaults to 1.
############################################################

def construct_randomsleep_rows(row_generator_fx, min_sleep, max_sleep, rows=1):
    '''
    Returns `rows` rows every uniformly random number of seconds
    '''
    rows_fx = lambda x: rows
    return construct_generic_limited_rows(test_row_generator, rows_fx,
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
    return construct_generic_limited_rows(test_row_generator, rows_fx,
                                          lambda x: vague_normal())

def construct_randbetasleep_rows(row_generator_fx, min_sleep, max_sleep, alpha=1, beta=3, rows=1):
    '''
    Returns `rows` rows every beta-distributed random number of seconds
    '''
    rows_fx = lambda x: rows
    def beta_fx():
        return random.betavariate(alpha, beta) * max_sleep + min_sleep
    return construct_generic_limited_rows(test_row_generator, rows_fx,
                                          lambda x: beta_fx())


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
    limited_fx_generic =  construct_randnormsleep_rows(test_row_generator, 0, 0.1)
    # limited_fx_generic =  construct_randbetasleep_rows(test_row_generator, 0, 5, alpha=1, beta=10)
    while True:
        print(limited_fx_generic())



