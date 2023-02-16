from calculator_and_array import CalculatorAndArray
from sram import SRAM1, SRAM2
from statistics import Statistics
import utils

import numpy as np

class Core:
    """ 
    Core of the cluster
    A core composes of 2 SRAMs, 1 calculator

    block_cal: 2 elements, in the shape of result matrix, record which block is now under calculation
               [row, col]
    block_rownum_softxmax: when the core is calculating Q*K, this variable indicates which row is going to execute softmax next
    """

    def __init__(self, sram1_num, sram1_height, sram1_width,
                sram2_height, sram2_width,
                mac_lane, mac_num, block_cnt, 
                sram_latency_count, array_and_calculator_latency_count, sram2_num=1):

        self.sram1 = SRAM1(sram_latency_count, sram1_num, sram1_height, sram1_width)
        self.sram2 = SRAM2(sram_latency_count, sram2_num, sram2_height, sram2_width)

        self.calculator_and_array = CalculatorAndArray(mac_lane, mac_num, block_cnt, array_and_calculator_latency_count)

        self.blocknum_cal = [0, 0]

        self.statistics = Statistics()

    def dump_configs(self, id):
        print("----------------------------------------------")
        print("| " + id + " Core Configuration")
        print("|")
        print("| SRAM1")
        self.sram1.dump_configs()
        print("|")
        print("| SRAM2")
        self.sram2.dump_configs()
        print("|")
        print("| Calculator")
        self.calculator_and_array.dump_configs()
        print("----------------------------------------------")
    
    def dump_mappings(self, id):
        print("----------------------------------------------")
        print("| " + id + " Core Mappings")
        print("|")
        print("| SRAM1")
        self.sram1.dump_mappings()
        print("|")
        print("| SRAM2")
        self.sram2.dump_mappings()
        print("|")
        print("| Calculator")
        self.calculator_and_array.dump_mappings()
        print("----------------------------------------------")

    def dump_cal_status(self, id):
        print("-------------------")
        print(id + " Calculation status")
        print("block: [" + str(self.blocknum_cal[0]) + ", " + str(self.blocknum_cal[1]) + "]")
        self.sram1.dump_cal_status()
        self.sram2.dump_cal_status(self.blocknum_cal[1])
        self.calculator_and_array.dump_cal_status()
        print("-------------------")

    def dump_state_matrix(self, id):
        # if (self.blocknum_cal[0] == 3) & (self.blocknum_cal[1] ==3):
        print(id + " Core State Matrix")
        self.sram1.dump_state_matrix("SRAM1", id)
        self.sram2.dump_state_matrix("SRAM2", id)
        self.calculator_and_array.dump_state_matrix()

    def sram_ready(self):
        """ Check whether SRAMs are ready to calculate """
        # print("sram1 ready: " + str(self.sram1.ready()))
        # print("sram2 ready: " + str(self.sram2.ready()))
        return (self.sram1.ready() & self.sram2.ready(self.blocknum_cal[1]))
    
    def sram_cal_advance(self):
        """ Advance SRAMs' calculation index """

        # sram2 has a finer granularity
        if self.sram2.cal_advance(self.blocknum_cal):
            self.sram1.cal_advance(self.blocknum_cal, self.sram2.cal_complete)

    def sram_cal_advance_qk(self):
        """ Advance SRAM's calculation index in case of Q*K calculation, which means the index should be advanced in a ring format """
        
        if self.sram2.cal_advance_qk(self.blocknum_cal):
            self.sram1.cal_advance_qk(self.blocknum_cal, self.sram2.cal_complete)

    def is_complete(self):
        return (self.sram1.cal_complete | self.sram2.cal_complete)     

    def reset(self):
        self.sram1.reset()
        self.sram2.reset()
        self.calculator_and_array.reset()
        self.blocknum_cal = [0, 0]

    def reconfigure(self, block_cnt):
        """ 
        When switch to a different calculation stage, some configurations may change

        block_cnt: number of mac_lane * mac_lane blocks in the result matrix
        """
        
        self.calculator_and_array.reconfigure(block_cnt)
    
