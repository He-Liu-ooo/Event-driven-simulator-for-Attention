import numpy as np

from base_unit import BaseUnit
import utils

class SRAM(BaseUnit):
    """ 
    Core SRAM

    num: how many sub-SRAMs in the SRAM
    height: height of every sub-SRAM
    width: width of every sub-SRAM

    cal_complete: True if the last data is read for calculation, same for SRAM1/2, SRAM1.complete is always False, ignore it
    write_complete: when SRAM's data is directly written from previous SRAM's array, we use this variable to indicate if all data is ready

    array_block_counter: number of mac_lane * mac_lane block that finishes written into this SRAM
                         this variable helps to update sram_state_matrix
    sram_state_matrix: record states of data in the SRAM
                        three states: READY/REMOVE/REMOVING
    """
    def __init__(self, latency_count, num, height, width):
        super(SRAM, self).__init__(latency_count)

        self.num = num
        self.height = height
        self.width = width

        self.cal_complete = False
        self.write_complete = False

        self.array_block_counter = 0

        self.sram_state_matrix = np.ones(self.height, dtype=int)
        print("SRAM2 state matrix size: " + str(self.height))

    def dump_configs(self):
        print("| + sub-SRAM number: " + str(self.num))
        print("| + sub-SRAM height: " + str(self.height))
        print("| + sub-SRAM width: " + str(self.width))
        print("| + access latency: " + str(self.latency_count * utils.METATIME) + "ns")
    
    def dump_state_matrix(self, sram, mode):
        print(str(sram) + ":")
        if sram == "SRAM1":
            if (mode == "Q") or (mode == "K") or (mode == "V"): 
                print(self.sram_state_matrix[:32])
                print(self.sram_state_matrix[32:64])
                print(self.sram_state_matrix[64:96])
                print(self.sram_state_matrix[96:])
            elif mode == "Q*K":
                print(self.sram_state_matrix.reshape(-1,2)[:24])
        else:
            if (mode == "Q") or (mode == "K") or (mode == "V"): 
                print("0")
                print(self.sram_state_matrix[:64])
                print("1")
                print(self.sram_state_matrix[64:128])
                print("30")
                print(self.sram_state_matrix[1920:1984])
                print("31")
                print(self.sram_state_matrix[1984:])
            elif mode == "Q*K":
                print("0")
                print(self.sram_state_matrix[:384])
                print("1")
                print(self.sram_state_matrix[384:768])

    def update_to_ready(self, idx):
        self.sram_state_matrix[idx] = utils.READY

    def update_to_removing(self, idx):
        self.sram_state_matrix[idx] = utils.REMOVING

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
        print("SRAM1: number of blocks from previous core: " + str(self.array_block_counter))

    def dump_cal_status1(self):
        print("Q*K SRAM1: ")
        print("subsum_cnt_idx_cal: " + str(self.subsum_cnt_idx_cal))
        print("blocknum_row_sram_idx_cal: " + str(self.blocknum_row_sram_idx_cal))
    
    def dump_mappings(self):
        print("| + number of mac_lane rows in the result matrix: " + str(self.blocknum_row_std))
        print("| + number of mac_lane columns in the result matrix: " + str(self.blocknum_col_std))
        print("| + logic state matrix size: [" + str(self.blocknum_row_std) + "/" + str(self.blocknum_row_sram_std) + ", " + str(self.subsum_cnt_std) + "]")

    def dump_ready(self):
        print("SRAM1 ready status, check state matrix idx: " + str(self.blocknum_row_sram_idx_cal * self.subsum_cnt_std + self.subsum_cnt_idx_cal))        

    def add_mapping(self, blocknum_row, blocknum_col, subsum_cnt, blocknum_row_sram):
        """ 
        Formulate mapping strategy

        blocknum_row_std: number of mac_lane * mac_lane blocks in the row of result matrix
        blocknum_col_std: number of mac_lane * mac_lane blocks in the column of result matrix
        subsum_cnt_std: number of subsums accumulated to complete the calculation of a mac_lane * mac_lane block
        blocknum_row_sram_std: number of rows of mac_lane * mac_lane blocks simultaneously stores in sub-SRAM

        flag: determine whether the state matrix should be initialized or kept

        """

        self.blocknum_row_std = blocknum_row
        self.blocknum_col_std = blocknum_col
        self.subsum_cnt_std = subsum_cnt #32
        self.blocknum_row_sram_std = blocknum_row_sram #4

    def ready(self):
        return (self.sram_state_matrix[self.blocknum_row_sram_idx_cal * self.subsum_cnt_std + self.subsum_cnt_idx_cal] == utils.READY)

    def update_to_remove(self, blocknum_row_sram_idx_cal):
        for i in range(self.subsum_cnt_std):
            self.sram_state_matrix[blocknum_row_sram_idx_cal * self.subsum_cnt_std + i] = utils.REMOVE
    
                                        # in order to get the last line of sram1 data to be replaced
    def cal_advance(self, blocknum_cal, sram2_cal_complete):
        if self.cal_complete == False:  # FIXME self.complete is always false
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
                self.cal_complete = True
        if sram2_cal_complete:
        # sram2 completes, flush the last line in sram1
            self.update_to_remove(self.blocknum_row_sram_idx_cal)

    def cal_advance_qk(self, blocknum_cal, sram2_cal_complete):
        if self.cal_complete == False:
            # calculation of a block is not completed
            if (self.subsum_cnt_idx_cal + 1) < self.subsum_cnt_std:
                self.subsum_cnt_idx_cal += 1
            # calculation of a block completes, but the rest not
            else:
                # if blocknum_cal[1] == (self.blocknum_col_std - 1):
                #     self.update_to_remove(self.blocknum_row_sram_idx_cal)
                self.subsum_cnt_idx_cal = 0
                self.blocknum_row_sram_idx_cal = blocknum_cal[0]

    def reset(self):
        self.cal_complete = False
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
        print("SRAM2: number of blocks from previous core: " + str(self.array_block_counter))
    
    def dump_cal_status1(self):
        print("Q*K SRAM2: ")
        print("block_col_idx_cal:" + str(self.block_col_idx_cal))
        print("subsum_cnt_idx_cal:" + str(self.subsum_cnt_idx_cal))
    
    def dump_mappings(self):
        print("| + number of mac_lane rows in the result matrix: " + str(self.blocknum_row_std))
        print("| + number of mac_lane columns in the result matrix: " + str(self.blocknum_col_std))
        print("| + logic state matrix size: [" + str(self.subsum_cnt_std) + ", " + str(self.logic_sram_col_cnt_std) + "]")
        print("| + mac lane: " + str(self.block_col_std))

    def dump_ready(self, blocknum_col_cal):
        print("SRAM2 ready status, check state matrix idx: " + str(self.subsum_cnt_idx_cal * self.logic_sram_col_cnt_std + blocknum_col_cal * self.block_col_std + self.block_col_idx_cal)) 

    def add_mapping(self, blocknum_row, blocknum_col, block_col, subsum_cnt):
        """ 
        Formulate mapping strategy

        block_col_std: mac_lane
        blocknum_col_std: number of mac_lane * mac_lane blocks in the col of result matrix
        blocknum_row_std: number of mac_lane * mac_lane blocks in the row of result matrix
        subsum_cnt_std: number of subsums accumulated to complete the calculation of a mac_lane * mac_lane block

        sram_state_matrix: record states of data in the SRAM
                           three states: READY/REMOVE/REMOVING

        flag: determine whether the state matrix should be initialized or kept
        """

        self.block_col_std = block_col
        self.blocknum_col_std = blocknum_col
        self.blocknum_row_std = blocknum_row
        self.subsum_cnt_std = subsum_cnt  #32
        self.logic_sram_col_cnt_std = block_col * blocknum_col #64

    def ready(self, blocknum_col_cal):
        return (self.sram_state_matrix[self.subsum_cnt_idx_cal * self.logic_sram_col_cnt_std + blocknum_col_cal * self.block_col_std + self.block_col_idx_cal] == utils.READY)

    def update_to_remove(self, blocknum_col, block_col_idx_cal):
        self.sram_state_matrix[self.subsum_cnt_idx_cal * self.logic_sram_col_cnt_std + blocknum_col * self.block_col_std + block_col_idx_cal] = utils.REMOVE

    def update_to_ready_from_array(self, blocknum_col, array_block_cnt_std):
        """ 
        Update SRAM2's state matrix according to next core's array data transfer 

        blocknum_col: number of blocks in col in the operand matrix
                      4 for head embedding dim = 64 and mac_lane = 16 case
        """

        # row idx (starts from 0) of sram state matrix 
        row = int((self.array_block_counter // 2 + 1) % 2)
        # col has mac_lane sub-cols
        col = (self.array_block_counter + 2) // blocknum_col - 1
        for i in range(self.block_col_std):
            self.sram_state_matrix[row * self.logic_sram_col_cnt_std + col * self.block_col_std + i] = utils.READY
        
        if self.array_block_counter == array_block_cnt_std:
            self.write_complete = True

    def cal_advance(self, blocknum_cal):
        is_sram1_advance = False
        if self.cal_complete == False:
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
                self.cal_complete = True
                is_sram1_advance = True

        return is_sram1_advance

    def cal_advance_qk(self, blocknum_cal):
        is_sram1_advance = False
        if self.cal_complete == False:
            # calculation of mac_lane width is not completed
            if (self.block_col_idx_cal + 1) < self.block_col_std:
                if blocknum_cal[0] == self.blocknum_col_std:
                    self.remove(blocknum_cal[1], self.block_col_idx_cal)
                self.block_col_idx_cal += 1
            # calculation of mac_lane width completes, but the calculation of a block doesn't
            elif (self.subsum_cnt_idx_cal + 1) < self.subsum_cnt_std:
                if blocknum_cal[0] == self.blocknum_col_std:
                    self.remove(blocknum_cal[1], self.block_col_idx_cal)
                self.block_col_idx_cal = 0
                self.subsum_cnt_idx_cal += 1
                is_sram1_advance = True
            # calculation of a block completes, but the calculation of column doesn't
            elif (blocknum_cal[0] + 1) < blocknum_cal[1]:
                self.block_col_idx_cal = 0
                self.subsum_cnt_idx_cal = 0
                is_sram1_advance = True
                blocknum_cal[0] += 1
            # calculation of the column completes, switch to row calculation
            elif (blocknum_cal[0] + 1) == blocknum_cal[1]:
                self.block_col_idx_cal = 0
                self.subsum_cnt_idx_cal = 0
                is_sram1_advance = True
                blocknum_cal[0] = blocknum_cal[1]
                blocknum_cal[1] = 0
            # calculation of the column completes, but the calculation of row doesn't
            elif (blocknum_cal[1] + 1) <= blocknum_cal[0]: 
                if blocknum_cal[0] == self.blocknum_col_std:
                    self.remove(blocknum_cal[1], self.block_col_idx_cal)
                self.block_col_idx_cal = 0
                self.subsum_cnt_idx_cal = 0
                is_sram1_advance = True
                blocknum_cal[1] += 1
            # calculation of a ring completes, reset to the next beginning block
            elif (blocknum_cal[0] == blocknum_cal[1]) and ((blocknum_cal[0] - 1)!= self.blocknum_col_std):
                self.block_col_idx_cal = 0
                self.subsum_cnt_idx_cal = 0
                is_sram1_advance = True
                blocknum_cal[0] = 0
                blocknum_cal[1] += 1
            # calculation complete
            else:
                self.remove(blocknum_cal[1], self.block_col_idx_cal)
                self.cal_complete = True
                is_sram1_advance = True 

        return is_sram1_advance

    def reset(self):
        self.cal_complete = False
        self.block_col_idx_cal = 0
        self.subsum_cnt_idx_cal = 0
        super().reset()