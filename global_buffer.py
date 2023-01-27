from base_unit import BaseUnit
import utils

class GlobalBuffer(BaseUnit):
    """ 
    Global buffer of a cluster

    Suppose it can support the data transfer of all sub-SRAMs in one access

    sram1_busy: if the GB is transferring data from core sram1 now
    array_busy: if the GB is transferring data from core array now
    row, col: which data in core's sram is the global buffer transfer now
    colnum2: record which mac_lane * mac_lane col of sram2 is now transferring
    rownum1: record which time of the  whole sram1 is now transferring(in seq-len=384 case, sram1 will be updated 384/mac_lane/(sram1_height/(embedding_dim/mac_num))-1=5 times)
    array_idx_rm: which data in core's array is the global buffer transfer now
               difference between array_idx_cal in calculator_and_array, which indicates position id that needs to accept the next data

    blocknum_row_cnt: number of mac_lane rows need to be replaced
    array_data_cnt: number of mac_lane data need to be replaced

    sram1_complete1: indicates whether data update of SRAM1 is complete(True when the last data starts transferring)
    sram1_complete2: indicates whether data update of SRAM1 is complete(True when the last data finishes transferring)
    sram2_complete1: indicates whether data update of SRAM2 is complete(True when the last data starts transferring)
    sram1_complete2: indicates whether data update of SRAM1 is complete(True when the last data finishes transferring)
    array_complete1: indicates whether the calculated data is all transferred into gb(True when the last data starts transferring)
    array_complete2: indicates whether the calculated data is all transferred into gb(True when the last data finishes transferring)

    array_latency_counter: latency counter for the data transfer of array
    array_data_counter: record how many data has been moved into gb

    sram2_latency_counter: latency counter for sram2, width equal to chosen bandwidth

    num_working: number of data that are transferred at the same time, < self.num_working_std
                 this is for sram2 only
    """

    def __init__(self, latency_count):
        super(GlobalBuffer, self).__init__(latency_count)

        self.sram1_busy = False
        self.sram2_busy = False
        self.array_busy = False

        self.row = [0, 0]
        self.col = [0, 0]
        self.colnum2 = 1
        self.rownum1 = 1
        self.array_idx_rm = 0

        self.blocknum_row_cnt = 0
        self.array_data_cnt = 0
        self.sram1_complete1 = False
        self.sram1_complete2 = False
        self.sram2_complete1 = False
        self.sram2_complete2 = False
        self.array_complete1 = False
        self.array_complete2 = False

        self.array_latency_counter = 0
        self.array_data_counter = 0

        self.sram2_latency_counter = 0

        self.num_working = 0
        
    def dump_configs(self):
        print("----------------------------------------------")
        print("| Global Buffer Configuration")
        print("| + access latency: " + str(self.latency_count * utils.METATIME) + "ns")
        print("----------------------------------------------")

    def dump_rm_status(self, idx):
        print("---------------------------")
        print(" Global buffer " + str(idx) + " status:")
        print(" + is busy for sram1: " + str(self.sram1_busy))
        print(" + is busy for array: " + str(self.array_busy))
        print(" + Next transferring data from sram1: [" + str(self.row[0]) + ", " + str(self.col[0]) + "]")
        print(" + Next transferring data from sram2: [" + str(self.row[1]) + ", " + str(self.col[1]) + "]")
        print(" + Next transferring data from array: " + str(self.array_idx_rm))
        print("---------------------------")

    def dump_mappings(self, idx):
        print("----------------------------------------------")
        print("| Global buffer " + str(idx) + " Mappings")
        print("| + number of mac_lane rows in result matrix: " + str(self.blocknum_row_cnt))
        print("| + number of mac_lane*mac_lane blocks in result matrix: " + str(self.array_data_cnt))
        print("| + logic SRAM1 state matrix: [" + str(self.blocknum_row_cnt) + "/" + str(self.sram1_rownum_cnt) + ", " + str(self.sram_subsum_cnt) + "]")
        print("| + logic SRAM2 state matrix: [" + str(self.sram_subsum_cnt) + ", " + str(self.sram2_colnum_cnt) + "]")
        print("----------------------------------------------")

    def add_mapping(self, blocknum_row_cnt, array_data_cnt, sram_subsum_cnt, sram1_rownum_cnt, sram2_colnum_cnt):
        """  
        # TODO not implemented yet
        num_working_std: maximum number of data can be transferred at the same time, which is equal to mac_lane, 
                         with the purpose of match the bandwidth of SRAM1
                         # NOTE this could be modified if the bandwidth of GB is far different from general standards 
       
        blocknum_row_cnt: number of mac_lane rows in result matrix
        sram_subsum_cnt: number of subsums acculmulated to complete the calculation of a  mac_lane * mac_lane block
                         1024/32=32 for Q/K/V calculation, 64/32=2 for Q*K calculation, 384/32=12 for A'*V calculation, under seq-len = 384
                         column number for sram1 logic state matrix and row number for sram2 logic state matrix
        sram1_rownum_cnt: number of mac_lane rows in a sub-SRAM
                          128/32=4 for Q/K/V calculation, 128/2=64 for Q*K calculation, 128/12=10 for A'*V calculation, under seq-len = 384  
                          row number for sram1 logic state matrix
        sram2_colnum_cnt: number of rows of valid data in sram2
                          64 for Q/K/V calculation, 384 for Q*K calculation, 64 for A'*V calculation, under seq-len = 384
                          column number for sram2 logic state matrix
        """

        self.blocknum_row_cnt = blocknum_row_cnt
        self.array_data_cnt = array_data_cnt
        self.sram_subsum_cnt = sram_subsum_cnt
        self.sram1_rownum_cnt = sram1_rownum_cnt
        self.sram2_colnum_cnt = sram2_colnum_cnt

        # self.num_working_std = num_working

    def rowcol_advance1(self):
        """ For SRAM1 """
        if (self.col[0] + 1) < self.sram_subsum_cnt:
            self.col[0] += 1
        elif ((self.row[0] + 1) < self.sram1_rownum_cnt) and ((self.row[0] + 1) < self.blocknum_row_cnt):
            self.row[0] += 1
            self.col[0] = 0
        elif (self.row[0] + (self.rownum1 - 1) * self.sram1_rownum_cnt + 1) < self.blocknum_row_cnt:
            self.row[0] = 0
            self.col[0] = 0
            self.rownum1 += 1
        else:
            self.sram1_complete1 = True

    def rowcol_advance2(self, mac_lane):
        """ For SRAM2 """
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

        row = 0
        col = 0
        idx = 0
        if sram == 1:
            # if for sram1 
            if sram_state_matrix[self.row[0] * self.sram_subsum_cnt + self.col[0]] == utils.REMOVE:
                # hit = True
                row = self.row[0]
                col = self.col[0]
                self.sram1_busy = True
                self.rowcol_advance1()
            idx = row * self.sram_subsum_cnt + col
        elif sram == 2:
            # if for sram2 
            if sram_state_matrix[self.row[1] * self.sram2_colnum_cnt + self.col[1]] == utils.REMOVE:
                # hit = True
                # self.num_working += 1
                row = self.row[1]
                col = self.col[1]
                # if self.num_working == self.num_working_std:
                self.sram2_busy = True
                self.rowcol_advance2(mac_lane)
            idx = row * self.sram2_colnum_cnt + col
        else:
            assert(0)

        return idx

    def find_array_target(self, array_state_matrix):
        """ Find the target data in array that will be transferred """

        idx = 0
        if array_state_matrix[self.array_idx_rm] == utils.COMPLETESUM:
            self.array_busy = True
            idx = self.array_idx_rm
            self.array_idx_advance(array_state_matrix.shape[0])

        return idx