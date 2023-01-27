class BaseUnit:
    """ 
    Base unit of all hardware components

    latency_count: how many times the time of one operation/access
                    of the unit is metatime
    latency_counter: when latency_counter count to latency_count, a new operation can start
    """

    def __init__(self, latency_count):
        self.latency_count = latency_count
        self.latency_counter = 0

    def reset(self):
        self.latency_counter = 0