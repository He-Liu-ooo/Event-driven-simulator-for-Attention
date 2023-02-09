from base_unit import BaseUnit
import utils

import numpy as np

class CalculatorAndArray(BaseUnit):
    """ 
    The calculating and accumulating component of a core

    mac_lane: number of MAC lanes
    mac_num: number of MAC within a MAC lane

    complete: if all the calculation and write into array of a matrix is complete(True when the last data becomes complete sum in array)
    array_state_matrix: indicating data status of array

    array_idx_cal: array position that is going to hold the result calculated data 
    array_idx_rm: array position that is transferring data to next core's SRAM
    subsum_counter: counter to count how many subsums have been accumulated in the array
                    if first subsums, subsum_counter = 0

    block_cnt: number of mac_lane * mac_lane blocks in the result matrix
    block_counter_cal: number of mac_lane * mac_lane blocks in the result matrix that finish calculation
    block_counter_rm: number of mac_lane * mac_lane blocks in the result matrix that finish removing

    array_sram_busy: if this core is busy transferring data from array to next core's SRAM
    sram_latency_counter: 
    """

    def __init__(self, mac_lane, mac_num, block_cnt, latency_count=1):
        super(CalculatorAndArray, self).__init__(latency_count)

        self.mac_lane = mac_lane
        self.mac_num = mac_num

        self.complete = False
        self.array_state_matrix = np.zeros(mac_lane, dtype=int)

        self.array_idx_cal = 0
        self.array_idx_rm = 0
        self.subsum_counter = 0
        self.block_cnt = block_cnt
        self.block_counter_cal = 0
        self.block_counter_rm = 0

        self.array_sram_busy = False
        self.sram_latency_counter = 0
    
    def dump_configs(self): 
        print("| + MAC lane: " + str(self.mac_lane))
        print("| + MAC number in a MAC lane: " + str(self.mac_num))
        print("| + Array size: " + str(self.mac_lane) + "*" + str(self.mac_lane))
        print("| + Block number: " + str(self.block_cnt))
        print("| + operating latency: " + str(self.latency_count * utils.METATIME) + "ns")

    def dump_cal_status(self):
        print("array: [" + str(self.array_idx_cal)  + "(id), " + str(self.subsum_counter) + "(subsum_cnt/" + str(self.subsum_cnt - 1) + ")], block number: " + str(self.block_counter_cal))
        print("block_rm number: " + str(self.block_counter_rm))

    def dump_state_matrix(self):
        print(self.array_state_matrix)

    def dump_mappings(self):
        print("| + number of subsums acculmulated for a complete mac_lane*mac_lane block: " + str(self.subsum_cnt))

    def update_to_removing(self, array_idx):
        self.array_state_matrix[array_idx] = utils.REMOVING

    def update_to_null(self, array_idx):
        self.array_state_matrix[array_idx] = utils.NULL

    def update_to_subsum(self):
        self.array_state_matrix[self.array_idx_cal] = utils.SUBSUM

    def update_to_completesum(self):
        self.array_state_matrix[self.array_idx_cal] = utils.COMPLETESUM

    def update_array(self):
        if (self.array_idx_cal + 1) < self.mac_lane:
            if self.subsum_counter == 0:
                self.update_to_subsum()
            if self.subsum_counter == (self.subsum_cnt - 1):
                # if this is the last round of subsum 
                self.update_to_completesum()
            self.array_idx_cal += 1
        else:
            # for the last data in the array
            if self.subsum_counter == 0:
                self.update_to_subsum()
            # next subsum
            self.subsum_counter += 1

            if self.subsum_counter == self.subsum_cnt:
                self.subsum_counter = 0
                self.block_counter_cal += 1
                # for the last data in the array
                self.update_to_completesum()

            self.array_idx_cal = 0

            if self.block_counter_cal == self.block_cnt:
                self.complete = True

    def array_idx_rm_advance(self):
        if self.array_idx_rm + 1 < self.mac_lane:
            self.array_idx_rm += 1
        else:
            self.array_idx_rm = 0
            self.block_counter_rm += 1

    def find_array_target(self):
        """ Find the target data in array that will be transferred """
        idx = 0
        # print("array_idx_rm in array_idx_advance(): " + str(self.array_idx_rm))
        # print("self.array_state_matrix[self.array_idx_rm]: " + str(self.array_state_matrix[self.array_idx_rm]))
        # print("array state matrix")
        # print(self.array_state_matrix)
        if self.array_state_matrix[self.array_idx_rm] == utils.COMPLETESUM:
            self.array_sram_busy = True
            idx = self.array_idx_rm
            self.array_idx_rm_advance()
        return idx

    def add_mapping(self, subsum_cnt):
        """
        subsum_cnt: number of sub subsums to be accumulated to get the complete sum
        """
        self.subsum_cnt = subsum_cnt

    def ready(self):
        """ 
        Check whether next round of calculation can start 
        Sometimes subsum state is enough for next calculation, sometimes we need NULL state
        """

        if self.subsum_counter == 0:
            return (self.array_state_matrix[self.array_idx_cal] == utils.NULL)
        else: 
            return (self.array_state_matrix[self.array_idx_cal] == utils.SUBSUM)

    def reset(self):
        """ 
        Reset all variables except array_state_matrix
        """
        self.complete = False
        self.array_idx_cal = 0
        self.subsum_counter = 0
        self.block_counter_cal = 0
        super().reset()

    def reconfigure(self, block_cnt):
        self.block_cnt = block_cnt

        