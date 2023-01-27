import numpy as np

from base_unit import BaseUnit
import utils

class SRAM(BaseUnit):
    """ 
    Core SRAM

    num: how many sub-SRAMs in the SRAM
    height: height of every sub-SRAM
    width: width of every sub-SRAM

    sram_state_matrix: record states of data in the SRAM
                        three states: READY/REMOVE/REMOVING
    complete: True if the last data is read for calculation, same for SRAM1/2, SRAM1.complete is always False, ignore it
    """
    def __init__(self, latency_count, num, height, width):
        super(SRAM, self).__init__(latency_count)

        self.num = num
        self.height = height
        self.width = width

        self.sram_state_matrix = np.zeros((height, width), dtype=int)
        self.complete = False

    def dump_configs(self):
        print("| + sub-SRAM number: " + str(self.num))
        print("| + sub-SRAM height: " + str(self.height))
        print("| + sub-SRAM width: " + str(self.width))
        print("| + access latency: " + str(self.latency_count * utils.METATIME) + "ns")
    
    def dump_state_matrix(self, sram):
        print(str(sram) + ":")
        if sram == "SRAM1":
            print(self.sram_state_matrix)
        else:
            print(self.sram_state_matrix[30:])
    
    def update_to_ready(self, row, col):
        self.sram_state_matrix[row][col] = utils.READY

    def update_to_removing(self, row, col):
        # print("[row, col]: " + "[" + str(row) + ", " + str(col) + "]")
        self.sram_state_matrix[row][col] = utils.REMOVING

class SRAM1(SRAM):
    """
    Core SRAM1

    blocknum_row_sram_idx_cal: indicating which row in the sub-SRAM is now calculating
    subsum_cnt_idx_cal: indicating which subsum in the sub-SRAM is now calculating

    blocknum_row_sram_idx_rm: indicating which row in the sub-SRAM is now removing
    subsum_cnt_idx_rm: indicating which subsum in the sub-SRAM is now removing

    """

    def __init__(self, latency_count, num, height, width):
        super(SRAM1, self).__init__(latency_count, num, height, width)
    
        self.blocknum_row_sram_idx_cal = 0
        self.subsum_cnt_idx_cal = 0

        # self.blocknum_row_sram_idx_rm = 0
        # self.subsum_cnt_idx_rm = 0


    def dump_cal_status(self):
        print("SRAM1: [" + str(self.blocknum_row_sram_idx_cal) + ", " + str(self.subsum_cnt_idx_cal) + "]")

    def add_mapping(self, blocknum_row, blocknum_col, subsum_cnt, blocknum_row_sram):
        """ 
        Formulate mapping strategy

        blocknum_row_std: number of mac_lane * mac_lane blocks in the rowumn of result matrix
        blocknum_row_std: number of mac_lane * mac_lane blocks in the row of result matrix
        subsum_cnt_std: number of subsums accumulated to complete the calculation of a mac_lane * mac_lane block
        blocknum_row_sram_std: number of rows of mac_lane * mac_lane blocks simultaneously stores in sub-SRAM

        """

        self.blocknum_row_std = blocknum_row
        self.blocknum_col_std = blocknum_col
        self.subsum_cnt_std = subsum_cnt
        self.blocknum_row_sram_std = blocknum_row_sram

        # initialized all to READY
        self.sram_state_matrix = np.zeros((self.blocknum_row_sram_std, self.subsum_cnt_std), dtype=int)
        print("SRAM1 state matrix size: [" + str(self.sram_state_matrix.shape[0]) + ", " + str(self.sram_state_matrix.shape[1]) + "]")

    def ready(self):
        return (self.sram_state_matrix[self.blocknum_row_sram_idx_cal][self.subsum_cnt_idx_cal] == utils.READY)

    def update_to_remove(self, blocknum_row_sram_idx_cal):
        for i in range(self.subsum_cnt_std):
            self.sram_state_matrix[blocknum_row_sram_idx_cal][i] = utils.REMOVE
    
                                        # in order to get the last line of sram1 data to be replaced
    def cal_advance(self, blocknum_cal, sram2_complete):
        if self.complete == False:  # FIXME self.complete is always false
            # calculation of a block is not completed
            if (self.subsum_cnt_idx_cal + 1) < self.subsum_cnt_std:
                self.subsum_cnt_idx_cal += 1
            # calculation of a block completes, but the row not
            elif (blocknum_cal[1] < self.blocknum_col_std) & (blocknum_cal[1] != 0):
                self.subsum_cnt_idx_cal = 0
            # calculation of a row completes, switch to next row
            elif (self.blocknum_row_sram_idx_cal + 1) < self.blocknum_row_sram_std:
                self.subsum_cnt_idx_cal = 0
                self.update_to_remove(self.blocknum_row_sram_idx_cal)
                self.blocknum_row_sram_idx_cal += 1
            # calculation of a SRAM completes, but the whole calculation is not completed
            elif (blocknum_cal[0] + 1) < self.blocknum_row_std:
                self.subsum_cnt_idx_cal = 0
                self.update_to_remove(self.blocknum_row_sram_idx_cal)
                self.blocknum_row_sram_idx_cal = 0
            # calculation completes
            else: 
                self.update_to_remove(self.blocknum_row_sram_idx_cal)
                self.complete = True
        if sram2_complete:
        # sram2 completes, flush the last line in sram1
            self.update_to_remove(self.blocknum_row_sram_idx_cal)

    def reset(self):
        self.complete = False
        self.blocknum_row_sram_idx_cal = 0
        self.subsum_cnt_idx_cal = 0
        super().reset()


class SRAM2(SRAM):
    """ 
    Core SRAM2

    block_col_idx_cal: indicating which col in the block is now calculating, < mac_lane
    subsum_cnt_idx_cal: indicating which subsum in the SRAM is now calculating

    block_col_idx_rm: indicating which col in the block is now removing, < mac_lane
    subsum_cnt_idx_rm: indicating which subsum in the SRAM is now removing
    blocknum_col_idx_rm: indicating which col in the result matrix is now removing
    """

    def __init__(self, latency_count, num, height, width):
        super(SRAM2, self).__init__(latency_count, num, height, width)
    
        self.block_col_idx_cal = 0
        self.subsum_cnt_idx_cal = 0

        # self.block_col_idx_rm = 0
        # self.subsum_cnt_idx_rm = 0
        # self.blocknum_col_idx_rm = 0
     
    def dump_cal_status(self, blocknum_col_cal):
        print("SRAM2: [" + str(self.subsum_cnt_idx_cal) + ", " + str(blocknum_col_cal * self.block_col_std + self.block_col_idx_cal) + \
                "/" + str(self.block_col_idx_cal) + "]") 

    def add_mapping(self, blocknum_row, blocknum_col, block_col, subsum_cnt):
        """ 
        Formulate mapping strategy

        block_col_std: mac_lane
        blocknum_col_std: number of mac_lane * mac_lane blocks in the col of result matrix
        blocknum_row_std: number of mac_lane * mac_lane blocks in the row of result matrix
        subsum_cnt_std: number of subsums accumulated to complete the calculation of a mac_lane * mac_lane block

        sram_state_matrix: record states of data in the SRAM
                           three states: READY/REMOVE/REMOVING
        """

        self.block_col_std = block_col
        self.blocknum_col_std = blocknum_col
        self.blocknum_row_std = blocknum_row
        self.subsum_cnt_std = subsum_cnt
        self.num_working_std = block_col

        # initialized all to READY
        self.sram_state_matrix = np.zeros((self.subsum_cnt_std, self.block_col_std * self.blocknum_col_std), dtype=int)
        print("SRAM2 state matrix size: [" + str(self.sram_state_matrix.shape[0]) + ", " + str(self.sram_state_matrix.shape[1]) + "]")

    def ready(self, blocknum_col_cal):
        return (self.sram_state_matrix[self.subsum_cnt_idx_cal][blocknum_col_cal * self.block_col_std + self.block_col_idx_cal] == utils.READY)

    def update_to_remove(self, blocknum_col, block_col_idx_cal):
        self.sram_state_matrix[self.subsum_cnt_idx_cal][blocknum_col * self.block_col_std + block_col_idx_cal] = utils.REMOVE

    def cal_advance(self, blocknum_cal):
        is_sram1_advance = False
        if self.complete == False:
            # calculation of mac_lane width is not completed
            if (self.block_col_idx_cal + 1) < self.block_col_std:
                if blocknum_cal[0] == self.blocknum_row_std - 1:
                    # if it is the last round of calculation, we can set the data state as ROMOVE
                    self.update_to_remove(blocknum_cal[1], self.block_col_idx_cal)
                self.block_col_idx_cal += 1
            # calculation of mac_lane width completes, but the calculation of a block doesn't
            elif (self.subsum_cnt_idx_cal + 1) < self.subsum_cnt_std:
                if blocknum_cal[0] == self.blocknum_row_std - 1:
                    self.update_to_remove(blocknum_cal[1], self.block_col_idx_cal)
                self.block_col_idx_cal = 0
                self.subsum_cnt_idx_cal += 1
                is_sram1_advance = True
            # calculation of a block completes, but the calculation of a mac_lane row doesn't
            elif (blocknum_cal[1] + 1) < self.blocknum_col_std:
                if blocknum_cal[0] == self.blocknum_row_std - 1:
                    self.update_to_remove(blocknum_cal[1], self.block_col_idx_cal)
                blocknum_cal[1] += 1
                self.block_col_idx_cal = 0
                self.subsum_cnt_idx_cal = 0
                is_sram1_advance = True
            # calculation of a mac_lane row completes, but the whole calculation doesn't
            elif (blocknum_cal[0] + 1) < self.blocknum_row_std:
                # if blocknum_cal[0] == self.blocknum_row_std - 1:
                #     self.update_to_remove(blocknum_cal[1], self.block_col_idx_cal + 1)
                blocknum_cal[1] = 0
                blocknum_cal[0] += 1
                self.block_col_idx_cal = 0
                self.subsum_cnt_idx_cal = 0
                is_sram1_advance = True
            else:
                # update the data of last column and last row to REMOVE
                self.update_to_remove(blocknum_cal[1], self.block_col_idx_cal)
                self.complete = True
                is_sram1_advance = True

        return is_sram1_advance

    def reset(self):
        self.complete = False
        self.block_col_idx_cal = 0
        self.subsum_cnt_idx_cal = 0
        super().reset()