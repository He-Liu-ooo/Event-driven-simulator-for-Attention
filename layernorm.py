from base_unit import BaseUnit
import utils

import numpy as np

class LayerNorm(BaseUnit):
    """ 
    Layer Normalization Unit

    blocknum_col: LN is row-wise operation, we need to store a row of data for calculation
                  this variable indicates the number of mac_lane*mac_lane blocks in a row
    busy: True from a whole row completes transferring into LN and removed for FC calculation
    row_idx: record which row of X is now processing, only when all data transferred into next core SRAM will this variable increment
    sram_latency_counter: counter to count the latency of transferring data from LN to next core's SRAM
    """

    def __init__(self, latency_count, blocknum_col):
        super(LayerNorm, self).__init__(latency_count)

        self.state_matrix = np.zeros(blocknum_col, dtype=int)

        self.blocknum_col = blocknum_col

        self.busy = False

        self.row_idx = 0

        self.sram_latency_counter = 0

    def dump_configs(self):
        print("----------------------------------------------")
        print("| Layer Norm Configuration")
        print("| + access latency: " + str(self.latency_count * utils.METATIME) + "ns")
        print("| + buffer size: " + str(self.blocknum_col))
        print("----------------------------------------------")

    def dump_cal_status(self):
        print("---------------------------")
        print(" Layer Norm calculation status: ")
        print(" + state maxtrix:")
        print(self.state_matrix)
        print(" + is LN busy: " + str(self.busy))
        print(" + row idx: " + str(self.row_idx))
        print("---------------------------")

    def update_to_x(self, idx):
        self.state_matrix[idx] = utils.X
        # self.busy = True

    def update_to_xcal(self):
        for i in range(self.state_matrix.shape[0]):
            self.state_matrix[i] = utils.X_CAL
        self.busy = True

    def update_to_null(self):
        for i in range(self.state_matrix.shape[0]):
            self.state_matrix[i] = utils.NULL
        self.row_idx += 1
        self.busy = False

    def update_to_removing(self):
        for i in range(self.state_matrix.shape[0]):
            self.state_matrix[i] = utils.REMOVING

    def calculation(self):
        """ Return whether the whole row of data is ready for calculation """

        res = True
        for i in range(self.blocknum_col):
            if (self.state_matrix[i] != utils.X):
                res = False
                break
        return res

    def ln_complete(self):
        """ Return whether LN operation is complete and could be transferred to next core's SRAM """

        res = True
        for i in range(self.blocknum_col):
            if (self.state_matrix[i] != utils.X_CAL):
                res = False
                break
        return res

    def removing(self):
        """ Return whether row data in LN is now transferring to next core's SRAM """
        
        res = True
        for i in range(self.blocknum_col):
            if (self.state_matrix[i] != utils.REMOVING):
                res = False
                break
        return res

    
