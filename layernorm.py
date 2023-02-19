from base_unit import BaseUnit
import utils

import numpy as np

class LayerNorm(BaseUnit):
    """ 
    Layer Normalization Unit

    blocknum_col: LN is row-wise operation, we need to store a row of data for calculation
                  this variable indicates the number of mac_lane*mac_lane blocks in a row
    to_sram_bandwidth: number of mac_Lane*mac_lane blocks can be removing to next core's SRAM at a time
    busy: True from the start of LN calculation to the end of transferring the whole row of data for FC calculation
          during which the data transfer between LN and GB is prohibited
    partial_removing_to_core_busy: True if some data is removing from LN to next core's SRAM
    removing_to_core_busy: True if a row of data not complete transferring to next core's SRAM
    row_idx: record which row of X is now processing, only when all data transferred into next core SRAM will this variable increment
    sram_latency_counter: counter to count the latency of transferring data from LN to next core's SRAM
    """

    def __init__(self, latency_count, blocknum_col, to_sram_bandwidth):
        super(LayerNorm, self).__init__(latency_count)

        self.state_matrix = np.zeros(blocknum_col, dtype=int)

        self.blocknum_col = blocknum_col
        self.to_sram_bandwidth = to_sram_bandwidth
        self.remove_start = 0
        self.remove_end = to_sram_bandwidth - 1

        self.busy = False
        self.partial_removing_to_core_busy = False 
        self.removing_to_core_busy = False 

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

    def update_to_ready(self, start, end):
        for i in range(start, end + 1):
            self.state_matrix[i] = utils.A

    def update_to_xlayernorm(self):
        for i in range(self.state_matrix.shape[0]):
            self.state_matrix[i] = utils.A_SOFTMAX

    def update_to_null(self, start, end):
        for i in range(start, end + 1):
            self.state_matrix[i] = utils.NULL

        if (end + 1) == self.state_matrix.shape[0]:
            # if this is the last portion of data of a row
            self.row_idx += 1
            self.removing_to_core_busy = False
            self.busy = False
        
        self.partial_removing_to_core_busy = False

    def calculation(self):
        """ Return whether the whole row of data is ready for calculation """

        res = True
        for i in range(self.blocknum_col):
            if (self.state_matrix[i] != utils.A):
                res = False
                break
        return res

    def ln_complete(self):
        """ Return whether LN operation is complete and could be transferred to next core's SRAM """

        res = True
        for i in range(self.blocknum_col):
            if (self.state_matrix[i] != utils.A_SOFTMAX):
                res = False
                break
        return res

    def find_removing_target(self):
        """ Find layernorm result target and transfer it to next core's SRAM """

        start = 0
        end = 0
        if self.remove_end < (self.blocknum_col - 1):
            for i in range(self.remove_start, self.remove_end + 1): 
                self.state_matrix[i] = utils.REMOVING
            start = self.remove_start
            end = self.remove_end
            self.remove_start = self.remove_end + 1
            self.remove_end = self.remove_start + self.to_sram_bandwidth - 1
        else:
            for i in range(self.remove_start, self.state_matrix.shape[0]):
                self.state_matrix[i] = utils.REMOVING
            start = self.remove_start
            end = self.state_matrix.shape[0] - 1
            self.remove_start = 0
            self.remove_end = self.to_sram_bandwidth - 1
        
        self.removing_to_core_busy = True
        self.partial_removing_to_core_busy = True

        return (start, end)
