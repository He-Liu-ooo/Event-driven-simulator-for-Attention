from base_unit import BaseUnit
import utils

import numpy as np

""" 
TODO
1. combine the 3 "find_sram_target" function
2. combine the 2 SRAM2 case: whether weight matrix can be hold in SRAM2 at once
"""

class GlobalBuffer(BaseUnit):
    """ 
    Global buffer of a cluster

    Suppose it can support the data transfer of all sub-SRAMs in one access

    sram1_busy: if the GB is transferring data from core sram1 now
    sram2_busy: if the GB is transferring data from core sram2 now
    array_busy: if the GB is transferring data from core array now
    softmax_busy: if the GB is transferring data to softmax unit now
    layernorm_busy: if the GB is transferring data to layernorm unit now

    row, col: which data in core's sram is the global buffer transfer now
    colnum2: record which mac_lane col of logic sram2 is now transferring
    colnum2_sram: record which mac_lane col of physical sram2 is now transferring 
    rownum2: sram2 record which row of the result matrix is now calculating
    rownum1: record which time of the whole sram1 is now transferring(in seq-len=384 case, sram1 will be updated 384/mac_lane/(sram1_height/(embedding_dim/mac_num))-1=5 times)
    array_idx_rm: which data in core's array is the global buffer transfer now
               difference between array_idx_cal in calculator_and_array, which indicates position id that needs to accept the next data
    a_row: record which row of A is executing softmax
    layernorm_row: record which row of X is executing layernorm
    softmax_start: record the transfer to/from softmax starts from which block
    softmax_end: record the transfer to/from softmax ends at which block
    layernorm_start: record the transfer to layernorm starts from which block
    layernorm_end: record the transfer to layernorm ends at which block

    blocknum_row_cnt: number of mac_lane rows need to be replaced
    array_data_cnt: number of mac_lane data need to be replaced
    blocknum_counter_from_last_core: record how many blocks from last core has been written into GB

    sram1_complete1: indicates whether data update of SRAM1 is complete(True when the last data starts transferring)
    sram1_complete2: indicates whether data update of SRAM1 is complete(True when the last data finishes transferring)
    sram2_complete1: indicates whether data update of SRAM2 is complete(True when the last data starts transferring)
    sram1_complete2: indicates whether data update of SRAM1 is complete(True when the last data finishes transferring)
    array_complete1: indicates whether the calculated data is all transferred into gb(True when the last data starts transferring)
    array_complete2: indicates whether the calculated data is all transferred into gb(True when the last data finishes transferring)

    sram2_latency_counter: latency counter for sram2, width equal to chosen bandwidth
    array_latency_counter: latency counter for the data transfer of array
    softmax_latency_counter: latency counter for the data transfer between GB and Softmax
    layernorm_latency_counter: latency counter for the data transfer between GB and Layernorm

    array_data_counter: record how many data has been moved into gb

    gb_sram_bandwidth: number of mac_lane*mac_num BYTE of data that can be transferred from GB to core SRAM during one access time               
    softmax_bandwidth: number of mac_lane*mac_lane blocks can be transferred from GB to Softmax Unit at a time
    layermorm_bandwidth: number of mac_lane*mac_lane blocks can be transferred from GB to Layernorm Unit at a time
    """

    def __init__(self, latency_count, gb_sram_bandwidth, softmax_bandwidth=0, layernorm_bandwidth=0):
        super(GlobalBuffer, self).__init__(latency_count)

        self.sram1_busy = False
        self.sram2_busy = False
        self.array_busy = False
        self.softmax_busy = False
        self.layernorm_busy = False

        self.row = [0, 0]
        self.col = [0, 0]
        self.colnum2 = 1
        self.colnum2_sram = 1
        self.rownum2 = 0
        self.rownum1 = 1
        self.array_idx_rm = 0
        self.a_row = 0
        self.layernorm_row = 0
        self.softmax_start = 0
        self.softmax_end = softmax_bandwidth - 1
        self.layernorm_start = 0
        self.layernorm_end = layernorm_bandwidth - 1

        self.blocknum_row_cnt = 0
        self.array_data_cnt = 0
        self.blocknum_counter_from_last_core = 0
        
        self.sram1_complete1 = False
        self.sram1_complete2 = False
        self.sram2_complete1 = False
        self.sram2_complete2 = False
        self.array_complete1 = False
        self.array_complete2 = False

        self.sram2_latency_counter = 0
        self.array_latency_counter = 0
        self.softmax_latency_counter = 0
        self.layernorm_latency_counter = 0

        self.array_data_counter = 0

        self.gb_sram_bandwidth = gb_sram_bandwidth
        self.softmax_bandwidth = softmax_bandwidth
        self.layernorm_bandwidth = layernorm_bandwidth

        
    def dump_configs(self):
        print("----------------------------------------------")
        print("| Global Buffer Configuration")
        print("| + access latency: " + str(self.latency_count * utils.METATIME) + "ns")
        print("| + softmax bandwidth: " + str(self.softmax_bandwidth))
        print("| + SRAM bandwidth: " + str(self.gb_sram_bandwidth))
        print("----------------------------------------------")
    
    def dump_mappings(self, id):
        print("----------------------------------------------")
        print("| " + id + " Global buffer Mappings")
        print("| + number of mac_lane rows in result matrix: " + str(self.blocknum_row_cnt))
        print("| + number of mac_lane*mac_lane blocks in result matrix: " + str(self.array_data_cnt))
        print("| + logic SRAM1 state matrix: [" + str(self.blocknum_row_cnt) + "/" + str(self.sram1_rownum_cnt) + ", " + str(self.sram_subsum_cnt) + "]")
        print("| + logic SRAM2 state matrix: [" + str(self.sram_subsum_cnt) + ", " + str(self.sram2_colnum_cnt) + "/" + str(self.sram2_sram_colnum_cnt) + "]")
        print("----------------------------------------------")

    def dump_rm_status(self, idx):
        print("---------------------------")
        print(" Global buffer " + str(idx) + " status:")
        print(" + Next transferring data from sram1: [" + str(self.row[0]) + ", " + str(self.col[0]) + "]")
        print(" + Next transferring data from sram2: [" + str(self.row[1]) + ", " + str(self.col[1]) + "]")
        print(" + Next transferring data from array: " + str(self.array_idx_rm))
        if idx == "FC2":
            print(" Received block number from FC1 core: " + str(self.blocknum_counter_from_last_core))
        print("---------------------------")

    def dump_a_state_matrix(self):
        print("A state matrix in GB:")
        print(self.a_state_matrix)
        print("Row of softmax: " + str(self.a_row))
        print("[start, end] = [" + str(self.softmax_start) + ", " + str(self.softmax_end) + "]")
        print("softmax_busy = " + str(self.softmax_busy))
        
    def add_mapping(self, blocknum_row_cnt, array_data_cnt, sram_subsum_cnt, sram1_rownum_cnt, sram2_colnum_cnt, sram2_sram_colnum_cnt, flag=False):
        """   
        blocknum_row_cnt: number of mac_lane rows in result matrix
        sram_subsum_cnt: number of subsums acculmulated to complete the calculation of a  mac_lane * mac_lane block
                         1024/32=32 for Q/K/V calculation, 64/32=2 for Q*K calculation, 384/32=12 for A'*V calculation, under seq-len = 384
                         column number for sram1 logic state matrix and row number for sram2 logic state matrix
        sram1_rownum_cnt: number of mac_lane rows in a sub-SRAM
                          128/32=4 for Q/K/V calculation, 128/2=64 for Q*K calculation, 128/12=10 for A'*V calculation, under seq-len = 384  
                          row number for sram1 logic state matrix
        sram2_colnum_cnt: number of columns of valid data in sram2
                          64 for Q/K/V calculation, 384 for Q*K calculation, 64 for A'*V calculation, under seq-len = 384
                          column number for sram2 logic state matrix
        sram2_sram_colnum_cnt: number of columns of data a SRAM2 can store at the same time
                               64 for row=1024 case and 16 for row=4096 case
        """

        self.blocknum_row_cnt = blocknum_row_cnt
        self.array_data_cnt = array_data_cnt
        self.sram_subsum_cnt = sram_subsum_cnt
        self.sram1_rownum_cnt = sram1_rownum_cnt
        self.sram2_colnum_cnt = sram2_colnum_cnt
        self.sram2_sram_colnum_cnt = sram2_sram_colnum_cnt
        

        if flag:
            self.a_state_matrix = np.zeros((self.blocknum_row_cnt, int(self.array_data_cnt // self.blocknum_row_cnt)), dtype=int)
            print("A state matrix size: [" + str(self.a_state_matrix.shape[0]) + ", " + str(self.a_state_matrix.shape[1]) + "]")

    def update_to_a2(self, row, col):
        self.a_state_matrix[row][col] = utils.A

    def update_to_cal(self, start, end, mode):
        row_idx = self.layernorm_row if mode == "ln" else self.a_row

        for i in range(start, end + 1):
            self.a_state_matrix[row_idx][i] = utils.A_CAL

        if (mode == "ln") and (end == (self.a_state_matrix.shape[1] - 1)):
            self.layernorm_row += 1

    def update_to_asoftmax(self, start, end):
        for i in range(start, end + 1):
            self.a_state_matrix[self.a_row][i] = utils.A_SOFTMAX
        if end == (self.a_state_matrix.shape[1] - 1):
            self.a_row += 1
    
    def update_blocknum_counter_from_last_core(self, block_counter_rm, blocknum_col_std):
        """ 
        When transferring block from last core to GB and then will be transferred to next core's SRAM, for LP case,  
        we need to update the variable blocknum_counter_from_last_core
        """

        if (block_counter_rm % blocknum_col_std) == 0:
            # a whole row finishes calculation
            self.blocknum_counter_from_last_core = (block_counter_rm // blocknum_col_std) * self.sram_subsum_cnt * 2
        else:
            self.blocknum_counter_from_last_core = (block_counter_rm // blocknum_col_std) * self.sram_subsum_cnt * 2 + block_counter_rm - \
                                                        (block_counter_rm // blocknum_col_std) * blocknum_col_std

    def rowcol_advance1(self):
        """ For SRAM1 """
        if (self.col[0] + 1) < self.sram_subsum_cnt:
            self.col[0] += 1
        elif ((self.row[0] + 1) < self.sram1_rownum_cnt) and ((self.row[0] + (self.rownum1 - 1) * self.sram1_rownum_cnt + 1) < self.blocknum_row_cnt):
            self.row[0] += 1
            self.col[0] = 0
        elif (self.row[0] + (self.rownum1 - 1) * self.sram1_rownum_cnt + 1) < self.blocknum_row_cnt:
            self.row[0] = 0
            self.col[0] = 0
            self.rownum1 += 1
        else:
            self.sram1_complete1 = True

    def rowcol_advance2(self, mac_lane, flag):
        """ For SRAM2 """
        if flag:
            # if all matrix data can be stored in physical sram2 at the same time
            if (self.col[1] + 1 - self.colnum2 * mac_lane) < 0:
                self.col[1] += 1
            elif (self.row[1] + 1) < self.sram_subsum_cnt:
                self.row[1] += 1
                self.col[1] = (self.colnum2 - 1) * mac_lane
            elif (self.col[1] + 1) < self.sram2_colnum_cnt:
                self.col[1] += 1
                self.row[1] = 0
                self.colnum2 += 1
            else: 
                self.sram2_complete1 = True
        else:
            # if not all matrix data can be stored in physical sram2 at the same time
            # if subsum of a mac_lane*mac_lane block not ready
            if (self.col[1] + 1 - self.colnum2_sram * mac_lane) < 0:
                self.col[1] += 1
            # if subsum of a mac_lane*mac_lane block ready, but the complete sum not
            elif (self.row[1] + 1) < self.sram_subsum_cnt:
                self.row[1] += 1
                self.col[1] = (self.colnum2_sram - 1) * mac_lane
            # if a mac_lane*mac_lane block is complete, but a "sram row" not
            elif (self.col[1] + 1) < self.sram2_sram_colnum_cnt:
                self.col[1] += 1
                self.row[1] = 0
                self.colnum2 += 1
                self.colnum2_sram += 1
            # if a "sram row" completes, but a "real row" not
            elif (self.col[1] + self.colnum2  * mac_lane - self.sram2_sram_colnum_cnt + 1) < self.sram2_colnum_cnt:
                # here means a brand-new SRAM update begins
                self.col[1] = 0
                self.colnum2_sram = 1
                self.colnum2 += 1
                self.row[1] = 0
            # a "real row" completes, but whole not
            # this means a new row's calculation begins
            elif (self.rownum2 + 1) < self.blocknum_row_cnt:
                self.col[1] = 0
                self.colnum2_sram = 1
                self.colnum2 = 1
                self.row[1] = 0
                self.rownum2 += 1
            else:
                self.sram2_complete1 = True
                
    def array_idx_advance(self, num_data):
        if self.array_idx_rm + 1 < num_data:
            self.array_idx_rm += 1
        else:
            self.array_idx_rm = 0
            self.array_data_counter += 1
            if self.array_data_counter == self.array_data_cnt:
                self.array_complete1 = True

    def find_sram_target(self, sram_state_matrix, mac_lane, sram):
        """ Find the target data in sram1 that will be transferred """

        # the returning flattened idx
        idx_start = 0
        idx_end = 0

        # recording the original value of the class, incase we regret modification
        row_raw = 0
        col_raw = 0
        rownum1_raw = 0
        sram1_complete1_raw = 0

        colnum2_sram_raw = 0
        colnum2_raw = 0
        rownum2_raw = 0
        sram2_complete1_raw = 0

        # last target that satisfy the conditions
        row_end = 0
        col_end = 0
        colnum2_sram_end = 0

        sram2_colnum_cnt_tmp = 0

        # indicate whether SRAM2 can hold the whole matrix of data
        flag = False

        if sram == 1:
            # if for sram1 
            # we record the values at the begining, in case of restoration
            row_raw = self.row[0]
            col_raw = self.col[0] 
            rownum1_raw = self.rownum1
            sram1_complete1_raw = self.sram1_complete1

            # find the band of data
            for i in range(self.gb_sram_bandwidth):
                if sram_state_matrix[self.row[0] * self.sram_subsum_cnt + self.col[0]] == utils.REMOVE:
                    # hit = True
                    if i == (self.gb_sram_bandwidth - 1):
                        # last data still satisfies, which means we successfully find a removable band of data
                        self.sram1_busy = True
                        row_end = self.row[0]
                        col_end = self.col[0]
                        self.rowcol_advance1()
                    elif ((self.col[0] + 1) >= self.sram_subsum_cnt) and \
                            ((self.row[0] + 1) >= self.sram1_rownum_cnt) or ((self.row[0] + (self.rownum1 - 1) * self.sram1_rownum_cnt + 1) >= self.blocknum_row_cnt) and \
                                ((self.col[1] + 1) < self.sram2_colnum_cnt):
                        # if row idx will advance to 0, stop here
                        self.sram1_busy = True
                        row_end = self.row[0]
                        col_end = self.col[0]
                        self.rowcol_advance1()
                        break
                    else:
                        # go and find whether the next data is true
                        self.rowcol_advance1()
                    
                    if self.sram1_complete1:
                        # completes
                        self.sram1_busy = True
                        row_end = self.row[0]
                        col_end = self.col[0]
                        break
                else:
                    break

            if self.sram1_busy:
                # if we successfully found a removable band of data
                idx_start = row_raw * self.sram_subsum_cnt + col_raw
                idx_end = row_end * self.sram_subsum_cnt + col_end
            else:
                # if not, we should restore the state
                self.row[0] = row_raw
                self.col[0] = col_raw
                self.rownum1 = rownum1_raw 
                self.sram1_complete1 = sram1_complete1_raw
                
            return (idx_start, idx_end)
        
        elif sram == 2:
            # if for sram2 
            # we record the values at the begining, in case of restoration
            row_raw = self.row[1]
            col_raw = self.col[1]
            colnum2_sram_raw = self.colnum2_sram
            colnum2_raw = self.colnum2
            rownum2_raw = self.rownum2
            sram2_complete1_raw = self.sram2_complete1

            if self.sram2_colnum_cnt <= self.sram2_sram_colnum_cnt:
                sram2_colnum_cnt_tmp = self.sram2_colnum_cnt
                flag = True
            else:
                sram2_colnum_cnt_tmp = self.sram2_sram_colnum_cnt

            # find the band of data
            for i in range(self.gb_sram_bandwidth * mac_lane):
                if sram_state_matrix[self.row[1] * sram2_colnum_cnt_tmp + self.col[1]] == utils.REMOVE:
                    # hit = True
                    if i == (self.gb_sram_bandwidth * mac_lane - 1):
                        # last data still satisfies, which means we successfully find a removable band of data
                        self.sram2_busy = True
                        row_end = self.row[1]
                        colnum2_sram_end = self.colnum2 if flag else self.colnum2_sram
                        self.rowcol_advance2(mac_lane, flag)
                    elif (flag == False) and ((self.col[1] + 1 - self.colnum2_sram * mac_lane) >= 0) and ((self.row[1] + 1) >= self.sram_subsum_cnt) and \
                        ((self.col[1] + 1) >= self.sram2_sram_colnum_cnt) and ((self.col[1] + self.colnum2  * mac_lane - self.sram2_sram_colnum_cnt + 1) < self.sram2_colnum_cnt):
                        # if a brand new SRAM2 is going to be updated
                        self.sram2_busy = True
                        row_end = self.row[1]
                        colnum2_sram_end = self.colnum2 if flag else self.colnum2_sram
                        self.rowcol_advance2(mac_lane, flag)
                        break
                    else:
                        # go and find whether the next data is true
                        self.rowcol_advance2(mac_lane, flag)
                    
                    if self.sram2_complete1:
                        # completes
                        self.sram2_busy = True
                        row_end = self.row[1]
                        # col_end = self.col[1]
                        colnum2_sram_end = self.colnum2 if flag else self.colnum2_sram
                        break
                else:
                    break

            if self.sram2_busy == False:
                # if we cannot successfully found a removable band of data, we should restore the state
                self.row[1] = row_raw
                self.col[1] = col_raw
                self.colnum2_sram = colnum2_sram_raw
                self.colnum2 = colnum2_raw
                self.rownum2 = rownum2_raw
                self.sram2_complete1 = sram2_complete1_raw
            else:
               pass

            return (row_raw, row_end, colnum2_raw if flag else colnum2_sram_raw, colnum2_sram_end)
        else:
            assert(0)

    def find_sram1_target_with_gb_check(self, sram_state_matrix, mac_lane, mode):
        """ 
        Check if the data of FC1 result matrix that is going to update FC2's core SRAM1 is ready 

        mode: if this checks for LP or FC2
        """

        # the returning flattened idx
        idx_start = 0
        idx_end = 0
        # recording the original value of the class, incase we regret modification
        row_raw = 0
        col_raw = 0
        rownum1_raw = 0
        sram1_complete1_raw = 0
        # last target that satisfy the conditions
        row_end = 0
        col_end = 0

        # if for sram1 
        # we record the values at the begining, in case of restoration
        row_raw = self.row[0]
        col_raw = self.col[0] 
        rownum1_raw = self.rownum1
        sram1_complete1_raw = self.sram1_complete1

        # find the band of data
        for i in range(self.gb_sram_bandwidth):
            if (sram_state_matrix[self.row[0] * self.sram_subsum_cnt + self.col[0]] == utils.REMOVE) and \
                self.check_gb(self.row[0], self.col[0], mac_lane, mode):
                # hit = True
                if i == (self.gb_sram_bandwidth - 1):
                    # last data still satisfies, which means we successfully find a removable band of data
                    row_end = self.row[0]
                    col_end = self.col[0]
                    self.sram1_busy = True
                    self.rowcol_advance1()
                elif ((self.col[0] + 1) >= self.sram_subsum_cnt) and \
                        ((self.row[0] + 1) >= self.sram1_rownum_cnt) or ((self.row[0] + (self.rownum1 - 1) * self.sram1_rownum_cnt + 1) >= self.blocknum_row_cnt) and \
                            ((self.col[1] + 1) < self.sram2_colnum_cnt):
                    # this data is not the band's end, and if row idx will advance to 0, stop here
                    row_end = self.row[0]
                    col_end = self.col[0]
                    self.sram1_busy = True
                    self.rowcol_advance1()
                    break
                else:
                    # go and find whether the next data is true
                    self.rowcol_advance1()


                if self.sram1_complete1:
                    # completes
                    self.sram1_busy = True
                    row_end = self.row[0]
                    col_end = self.col[0]
                    break
            else:
                break

        if self.sram1_busy:
            # if we successfully found a removable band of data
            idx_start = row_raw * self.sram_subsum_cnt + col_raw
            idx_end = row_end * self.sram_subsum_cnt + col_end
        else:
            # if not, we should restore the state
            self.row[0] = row_raw
            self.col[0] = col_raw
            self.rownum1 = rownum1_raw 
            self.sram1_complete1 = sram1_complete1_raw
        
        return (idx_start, idx_end)

    def check_gb(self, row, col, mac_lane, mode):
        if mode == "fc2":
            if (((self.sram2_colnum_cnt * 4) // mac_lane) * self.sram1_rownum_cnt * (self.rownum1 - 1) + (row * self.sram_subsum_cnt + col) * 2) <= self.blocknum_counter_from_last_core:
                return True
        elif mode == "lp":
            if ((self.sram2_colnum_cnt // mac_lane) * self.sram1_rownum_cnt * (self.rownum1 - 1) + (row * self.sram_subsum_cnt + col) * 2) <= self.blocknum_counter_from_last_core:
                return True
        else:
            assert(0)
        return False
    
    def find_sram_target_a(self, sram_state_matrix, a_state_matrix, sram1_rownum_cnt):
        """ 
        Find the target data in sram1 that will be transferred and check if GB has the corresponding data 

        sram1_rownum_cnt: number of mac_lane rows a sub-SRAM can hold at the same time
        """

        # the returning flattened idx
        idx_start = 0
        idx_end = 0
        # recording the original value of the class, incase we regret modification
        row_raw = 0
        col_raw = 0
        rownum1_raw = 0
        sram1_complete1_raw = 0
        # last target that satisfy the conditions
        row_end = 0
        col_end = 0

        # if for sram1 
        # we record the values at the begining, in case of restoration
        row_raw = self.row[0]
        col_raw = self.col[0] 
        rownum1_raw = self.rownum1
        sram1_complete1_raw = self.sram1_complete1

        # find the band of data
        for i in range(self.gb_sram_bandwidth):
            if (sram_state_matrix[self.row[0] * self.sram_subsum_cnt + self.col[0]] == utils.REMOVE) and \
                self.check_a(self.row[0], self.col[0], a_state_matrix, sram1_rownum_cnt):
                # hit = True
                if i == (self.gb_sram_bandwidth - 1):
                    # last data still satisfies, which means we successfully find a removable band of data
                    row_end = self.row[0]
                    col_end = self.col[0]
                    self.sram1_busy = True
                    self.rowcol_advance1()
                elif ((self.col[0] + 1) >= self.sram_subsum_cnt) and \
                        ((self.row[0] + 1) >= self.sram1_rownum_cnt) or ((self.row[0] + (self.rownum1 - 1) * self.sram1_rownum_cnt + 1) >= self.blocknum_row_cnt) and \
                            ((self.col[1] + 1) < self.sram2_colnum_cnt):
                    # this data is not the band's end, and if row idx will advance to 0, stop here
                    row_end = self.row[0]
                    col_end = self.col[0]
                    self.sram1_busy = True
                    self.rowcol_advance1()
                    break
                else:
                    # go and find whether the next data is true
                    self.rowcol_advance1()


                if self.sram1_complete1:
                    # completes
                    self.sram1_busy = True
                    row_end = self.row[0]
                    col_end = self.col[0]
                    break
            else:
                break

        if self.sram1_busy:
            # if we successfully found a removable band of data
            idx_start = row_raw * self.sram_subsum_cnt + col_raw
            idx_end = row_end * self.sram_subsum_cnt + col_end
        else:
            # if not, we should restore the state
            self.row[0] = row_raw
            self.col[0] = col_raw
            self.rownum1 = rownum1_raw 
            self.sram1_complete1 = sram1_complete1_raw
        
        return (idx_start, idx_end)

    def check_a(self, row, col, a_state_matrix, sram1_rownum_cnt):
        a_row = row + (self.rownum1 - 1) * sram1_rownum_cnt
        if (a_state_matrix[a_row][col * 2] == utils.A_SOFTMAX) and (a_state_matrix[a_row][col * 2 + 1] == utils.A_SOFTMAX):
            return True
        return False
                
    def prev_core_result_matrix_write_complete(self, sram1_idx, mac_lane, mode):
        """ Check whether all A'*V/FC1 result matrix is written into LP/FC2's core SRAM1 """ 
        if mode == "fc2":
            if (((self.sram2_colnum_cnt * 4) // mac_lane) * self.sram1_rownum_cnt * (self.rownum1 - 1) + sram1_idx * 2) == self.blocknum_row_cnt * (self.sram2_colnum_cnt * 4 // mac_lane):
                return True
        elif mode == "lp":
            if ((self.sram2_colnum_cnt // mac_lane) * self.sram1_rownum_cnt * (self.rownum1 - 1) + sram1_idx * 2) == self.blocknum_row_cnt * (self.sram2_colnum_cnt // mac_lane):
                return True
        else:
            assert(0)
        
        return False

    def find_array_target(self, array_state_matrix):
        """ Find the target data in array that will be transferred """

        idx = 0
        if array_state_matrix[self.array_idx_rm] == utils.COMPLETESUM:
            self.array_busy = True
            idx = self.array_idx_rm
            self.array_idx_advance(array_state_matrix.shape[0])

        return idx

    def find_softmax_null_target(self):
        """ Find the target data in GB that will be transferred to Softmax """

        start = 0
        end = 0
        if self.a_row < self.blocknum_row_cnt:
            if self.softmax_end < (self.a_state_matrix.shape[1] - 1):
                if (self.a_state_matrix[self.a_row][self.softmax_end] == utils.A):
                    for i in range(self.softmax_start, self.softmax_end + 1):
                        self.a_state_matrix[self.a_row][i] = utils.REMOVING
                    start = self.softmax_start
                    end = self.softmax_end
                    self.softmax_start = self.softmax_end + 1
                    self.softmax_end = self.softmax_start + self.softmax_bandwidth - 1
                    self.softmax_busy = True
            else:
                if (self.a_state_matrix[self.a_row][-1] == utils.A):
                    for i in range(self.softmax_start, self.a_state_matrix.shape[1]):
                        self.a_state_matrix[self.a_row][i] = utils.REMOVING
                    start = self.softmax_start
                    end = self.a_state_matrix.shape[1] - 1
                    self.softmax_start = 0
                    self.softmax_end = self.softmax_bandwidth - 1
                    self.softmax_busy = True

        return (start, end)

    def find_layernorm_null_target(self):
        """ Find the target data in GB that will be transferred to Layernorm """

        start = 0
        end = 0
        if self.layernorm_row < self.blocknum_row_cnt:
            if self.layernorm_end < (self.a_state_matrix.shape[1] - 1):
                if (self.a_state_matrix[self.layernorm_row][self.layernorm_end] == utils.A):
                    for i in range(self.layernorm_start, self.layernorm_end + 1):
                        self.a_state_matrix[self.layernorm_row][i] = utils.REMOVING
                    start = self.layernorm_start
                    end = self.layernorm_end
                    self.layernorm_start = self.layernorm_end + 1
                    self.layernorm_end = self.layernorm_start + self.layernorm_bandwidth - 1
                    self.layernorm_busy = True
            else:
                if (self.a_state_matrix[self.layernorm_row][-1] == utils.A):
                    for i in range(self.layernorm_start, self.a_state_matrix.shape[1]):
                        self.a_state_matrix[self.layernorm_row][i] = utils.REMOVING
                    start = self.layernorm_start
                    end = self.a_state_matrix.shape[1] - 1
                    self.layernorm_start = 0
                    self.layernorm_end = self.layernorm_bandwidth - 1
                    self.layernorm_busy = True

        return (start, end)

    def find_softmax_res_target(self):
        """ Find softmax result target and transfer it back to GB """
        start = 0
        end = 0
        if self.softmax_end < (self.a_state_matrix.shape[1] - 1):
            for i in range(self.softmax_start, self.softmax_end + 1):
                self.a_state_matrix[self.a_row][i] = utils.REMOVING
            start = self.softmax_start
            end = self.softmax_end
            self.softmax_start = self.softmax_end + 1
            self.softmax_end = self.softmax_start + self.softmax_bandwidth - 1
        else:
            for i in range(self.softmax_start, self.a_state_matrix.shape[1]):
                self.a_state_matrix[self.a_row][i] = utils.REMOVING
            start = self.softmax_start
            end = self.a_state_matrix.shape[1] - 1
            self.softmax_start = 0
            self.softmax_end = self.softmax_bandwidth - 1
        
        self.softmax_busy = True

        return (start, end)
    
    def softmax_complete(self):
        return (self.a_state_matrix[-1][-1] == utils.A_SOFTMAX)

    def transfer_to_softmax_complete(self):
        return (self.a_state_matrix[-1][-1] == utils.A_CAL)