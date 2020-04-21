import time
import math

class TimeKeeper(object):
    def __init__(self):
        self.creation_time = time.time()
    def __repr__(self):
        now = time.time()
        s = now - self.creation_time
        m = math.floor(s / 60)
        s -= m * 60
        return "%dm %ds" % (m, s)
