# Stream Limiting Utils

Quick-and-dirty utils for limiting a row-generating function to simulate a stream of data.

Two types of stream limiting:

- Off-the-shelf limiting library
``` python
def construct_ratelimit_rows(row_generator_fx, max_rows_per_minute, blocking=True):
```

- Generic rate limiter that calls functions to find amount of rows to return and sleep time

``` python
def construct_generic_limited_rows(row_generator_fx, rows_function, sleep_function):
```

For the latter, there are util functions to make life easier. E.g., for a random interval:

``` python
def construct_randomsleep_rows(row_generator_fx, min_sleep, max_sleep, rows=1):
```

