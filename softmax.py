from base_unit import BaseUnit
import utils

import numpy as np

class Softmax(BaseUnit):
    """ 
    Softmax Unit

    blocknum_col: number of mac_lane blocks in a row of A
    busy: if buffer is full of data for calculation/calculating/hasn't remove all result data to GB, busy is True,
          meaning that the data transfer from GB to Softmax is forbidden
    done: indicates whether this row of data finishes softmax calculation
    """

    def __init__(self, latency_count, blocknum_col):
        super(Softmax, self).__init__(latency_count)

        self.state_matrix = np.zeros(blocknum_col, dtype=int)

        self.blocknum_col = blocknum_col

        self.busy = False
        self.done = False

    def dump_configs(self):
        print("----------------------------------------------")
        print("| Softmax Configuration")
        print("| + access latency: " + str(self.latency_count * utils.METATIME) + "ns")
        print("| + buffer size: " + str(self.blocknum_col))
        print("----------------------------------------------")

    def dump_cal_status(self):
        print("---------------------------")
        print(" Softmax calculation status: ")
        print(" + state maxtrix:")
        print(self.state_matrix)
        print(" + is softmax busy: " + str(self.busy))
        print(" + is softmax done: " + str(self.done))
        print("---------------------------")

    def update_to_a(self, start, end):
        for i in range(start, end + 1):
            self.state_matrix[i] = utils.A
        if (end + 1) == self.blocknum_col:
            self.busy = True

    def update_to_null(self, start, end):
        for i in range(start, end + 1):
            self.state_matrix[i] = utils.NULL
        if (end + 1) == self.blocknum_col:
            self.busy = False
            self.done = False

    def update_to_asoftmax(self):
        for i in range(self.blocknum_col):
            self.state_matrix[i] = utils.A_SOFTMAX
        self.done = True
    
    def calculation(self):
        """ Return whether the whole row of data is ready for calculation """

        res = True
        for i in range(self.blocknum_col):
            if (self.state_matrix[i] != utils.A):
                res = False
                break
        return res
    