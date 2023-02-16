class Statistics:
    """ 
    Recording statistics

    util_counter: calculate core's utilization during whole template's calculation
                  we assume that only reading data from core SRAM to calculator and doing calculation means that a core is utilized, data transferred from other storage into the core 
                  and data removement from core's array to other storage do not count as utilized
    """

    def __init__(self):
        self.util_counter = 0