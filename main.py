from core import Core
from global_buffer import GlobalBuffer
from softmax import Softmax
from layernorm import LayerNorm
import utils

import argparse
import sys
import math
import numpy as np

"""
TODO
1. combine function "coresram1_gb_data_transfer" and "coresram1_gb_data_transfer_a" 
"""

def argparser():
    """ Argument parser. """

    ap = argparse.ArgumentParser()

    """ HW configs """
    ap.add_argument('--core-num', type = int, default = 1, \
                    help = 'number of matrix computation core')
    ap.add_argument('--SRAM-capacity', type = int, default = 65536, \
                    help = 'capacity of SRAM in matrix computation core, in term of BYTE')
    ap.add_argument('--MAC-lane', type = int, default = 16, \
                    help = 'number of MAC lane in matrix computation core(how many vector dot production can calculate in parallel)')
    ap.add_argument('--MAC-num', type = int, default = 32, \
                    help = 'number of MAC within a lane(dimension of a dot production)')
    ap.add_argument('--SRAM-access-latency', type = int, default = 1, \
                    help = 'how many times the time of SRAM access is metatime')
    ap.add_argument('--GB-access-latency', type = int, default = 50, \
                    help = 'how many times the time of global buffer access is metatime') 
    ap.add_argument('--GB-SRAM-bandwidth', type = int, default = 32, \
                    help = 'number of mac_lane*mac_num BYTE of data can be transferred from GB to core SRAM during a GB-access-latency')
    ap.add_argument('--array-access-and-calculation-latency', type = int, default = 1, \
                    help = 'how many times the time of array access and calculation is metatime') 
    ap.add_argument('--softmax-cal-latency', type = int, default = 60, \
                    help = 'how many times the time of softmax calculation is metatime')
    ap.add_argument('--softmax-throughput', type = int, default = 6, \
                    help = 'number of mac_lane*mac_lane blocks can be transferred from GB to Softmax Unit at a time')     
    ap.add_argument('--layernorm-cal-latency', type = int, default = 10, \
                    help = 'how many times the time of layernorm operation is metatime') 
    ap.add_argument('--GB-LN-bandwidth', type = int, default = 4, \
                    help = 'number of mac_lane*mac_lane BYTE can be transferred from GB to Layer Normalization')
    ap.add_argument('--LN-SRAM-bandwidth', type = int, default = 4, \
                    help = 'number of mac_lane*mac_lane BYTE can be transferred from Layer Normalization to core SRAM')
    ap.add_argument('--head-id', type = int, default = 0, \
                    help = 'which split head is this template simulating, < head-num')

    """ SW configs """
    ap.add_argument('--seq-length', type = int, default = 384, \
                    help = 'sequence length of the workload')
    ap.add_argument('--embedding-dim', type = int, default = 1024, \
                    help = 'embedding dimension of a token')
    ap.add_argument('--head-num', type = int, default = 16, \
                    help = 'number of attention heads')
    
    """ Others """
    ap.add_argument('--debug-flag', type = bool, default = False, \
                    help = 'whether to print intermediate results')
    return ap

def dump_configs(args):
    print("----------------------------------------------")
    print("| Configuration")
    print("|")
    print("| HW configs")
    print("| + core number: " + str(args.core_num))
    print("| + core SRAM capacity: " + str(args.SRAM_capacity))
    print("| + mac lane number: " + str(args.MAC_lane))
    print("| + mac number within a lane: " + str(args.MAC_num))
    print("| + SRAM access latency: " + str(args.SRAM_access_latency * utils.METATIME) + "ns")
    print("| + Global buffer access latency: " + str(args.GB_access_latency * utils.METATIME) + "ns")
    print("|")
    print("| SW configs")
    print("| + sequence length: " + str(args.seq_length))
    print("| + embedding dimension: " + str(args.embedding_dim))
    print("| + head number: " + str(args.head_num))
    print("----------------------------------------------")

def dump_latency(latency):
    print("Latency: " + str(latency) + "ns")

def read_from_core_sram(cores, stage, idx, flag=False):
    stage = stage
    if cores[idx].sram2.cal_complete == False:
        # if we can read SRAM and accumulating buffer is ready for result data
        if cores[idx].sram_ready() & cores[idx].calculator_and_array.ready():
            cores[idx].sram2.latency_counter += 1
            cores[idx].statistics.util_counter += 1
        # if data is ready for calculation
        if cores[idx].sram2.latency_counter == cores[idx].sram2.latency_count:
            cores[idx].sram2.latency_counter = 0
            if flag:
                cores[idx].sram_cal_advance_qk()
            else:
                cores[idx].sram_cal_advance()
            stage = stage + 1

    return stage

def dot_production(cores, stage, count, idx, core_num=1, a_row_idx=[0], a_col_idx=[0], a_idx_idx=0):
    stage = stage
    if cores[idx].calculator_and_array.complete == False:
        cores[idx].statistics.util_counter += 1
        cores[idx].calculator_and_array.latency_counter += 1
        if cores[idx].calculator_and_array.latency_counter == cores[idx].calculator_and_array.latency_count:
            cores[idx].calculator_and_array.latency_counter = 0
            cores[idx].calculator_and_array.update_array()
            if cores[idx].calculator_and_array.array_state_matrix[0] == utils.COMPLETESUM:
                a_row_idx[a_idx_idx] = cores[idx].blocknum_cal[0]
                a_col_idx[a_idx_idx] = cores[idx].blocknum_cal[1]
            if cores[idx].sram2.cal_complete == False:
                stage = stage - 1 
    if cores[idx].calculator_and_array.complete:
    # if the calculation of all data in Q completes, switch to K calculation
        # print("count: " + str(count))
        if (count[0] == 1):
            # print("################### calculation stage switch #################")
            if core_num == 1:
                cores[0].reset()
            stage = stage + 1
            count[0] = 0
        else: 
            count[0] += 1
    return stage

def coresram1_gb_data_transfer(cores, global_buffers, core_idx, gb_idx, sram1_idx_gb_start, sram1_idx_gb_end, mac_lane=0, core_num=0):
    # if global buffer can update SRAM data now
    if global_buffers[gb_idx].sram1_busy == False:
        if (core_num == 8) and ((gb_idx == 5) or (gb_idx == 7)):
            # if this is the data transfer from global_buffer6 into FC2's core SRAM1
            # besides checking whether FC2's core SRAM1 has a vacancy for holding data, we also need to check whether GB6 already has FC1's result matrix data
            # NOTE: [sram1_idx_gb_start, sram1_idx_gb_end]
            mode = "lp" if gb_idx == 5 else "fc2"
            (sram1_idx_gb_start[gb_idx], sram1_idx_gb_end[gb_idx]) = global_buffers[gb_idx].find_sram1_target_with_gb_check(cores[core_idx].sram1.sram_state_matrix, cores[core_idx].calculator_and_array.mac_lane, mode)
        else:
            (sram1_idx_gb_start[gb_idx], sram1_idx_gb_end[gb_idx]) = global_buffers[gb_idx].find_sram_target(cores[core_idx].sram1.sram_state_matrix, cores[core_idx].calculator_and_array.mac_lane, 1)
            
        # if global buffer actually removes a data
        if global_buffers[gb_idx].sram1_busy:
            cores[core_idx].sram1.update_to_removing(sram1_idx_gb_start[gb_idx], sram1_idx_gb_end[gb_idx])
    # if global buffer is transferring data
    else: 
        global_buffers[gb_idx].latency_counter += 1
        # if global buffer finishes 
        if global_buffers[gb_idx].latency_counter == global_buffers[gb_idx].latency_count:
            global_buffers[gb_idx].latency_counter = 0
            global_buffers[gb_idx].sram1_busy = False
            global_buffers[gb_idx].sram1_complete2 = global_buffers[gb_idx].sram1_complete1
            cores[core_idx].sram1.update_to_ready(sram1_idx_gb_start[gb_idx], sram1_idx_gb_end[gb_idx])
            if (gb_idx == 5) or (gb_idx == 7):
                mode = "lp" if gb_idx == 5 else "fc2"
                if global_buffers[6].prev_core_result_matrix_write_complete(sram1_idx_gb_end[gb_idx], mac_lane, mode):   # TODO check this 
                    cores[core_idx].sram1.write_complete = True

def coresram1_gb_data_transfer_a(cores, global_buffers, core_idx, gb_idx, sram1_idx_gb_start, sram1_idx_gb_end):
    """ A' matrix data transfer from GB3 to core SRAM1 """
    # if global buffer can update SRAM data now
    if global_buffers[gb_idx].sram1_busy == False:
        (sram1_idx_gb_start[gb_idx], sram1_idx_gb_end[gb_idx]) = global_buffers[gb_idx].find_sram_target_a(cores[core_idx].sram1.sram_state_matrix, global_buffers[gb_idx-1].a_state_matrix,
                                                                    global_buffers[4].sram1_rownum_cnt)
        if global_buffers[gb_idx].sram1_busy:
            # if global buffer actually has the corresponding data, we can transfer this data to core sram1
            cores[core_idx].sram1.update_to_removing(sram1_idx_gb_start[gb_idx], sram1_idx_gb_end[gb_idx])
    # if global buffer is transferring data
    else: 
        global_buffers[gb_idx].latency_counter += 1
        # if global buffer finishes 
        if global_buffers[gb_idx].latency_counter == global_buffers[gb_idx].latency_count:
            global_buffers[gb_idx].latency_counter = 0
            global_buffers[gb_idx].sram1_busy = False
            global_buffers[gb_idx].sram1_complete2 = global_buffers[gb_idx].sram1_complete1
            cores[core_idx].sram1.update_to_ready(sram1_idx_gb_start[gb_idx], sram1_idx_gb_end[gb_idx])

def coresram2_gb_data_transfer(cores, global_buffers, core_idx, gb_idx, rownum_sram2_idx_gb_start, rownum_sram2_idx_gb_end, colnum_sram2_idx_gb_start, colnum_sram2_idx_gb_end):
    if global_buffers[gb_idx].sram2_busy == False:
        (rownum_sram2_idx_gb_start[gb_idx], rownum_sram2_idx_gb_end[gb_idx], colnum_sram2_idx_gb_start[gb_idx], colnum_sram2_idx_gb_end[gb_idx]) = \
            global_buffers[gb_idx].find_sram_target(cores[core_idx].sram2.sram_state_matrix, cores[core_idx].calculator_and_array.mac_lane, 2)
        if global_buffers[gb_idx].sram2_busy:
            cores[core_idx].sram2.update_to_removing(rownum_sram2_idx_gb_start[gb_idx], rownum_sram2_idx_gb_end[gb_idx], colnum_sram2_idx_gb_start[gb_idx], colnum_sram2_idx_gb_end[gb_idx])
    else:
        global_buffers[gb_idx].sram2_latency_counter += 1
        if global_buffers[gb_idx].sram2_latency_counter == global_buffers[gb_idx].latency_count:
            global_buffers[gb_idx].sram2_latency_counter = 0
            global_buffers[gb_idx].sram2_busy = False
            global_buffers[gb_idx].sram2_complete2 = global_buffers[gb_idx].sram2_complete1
            cores[core_idx].sram2.update_to_ready(rownum_sram2_idx_gb_start[gb_idx], rownum_sram2_idx_gb_end[gb_idx], colnum_sram2_idx_gb_start[gb_idx], colnum_sram2_idx_gb_end[gb_idx])

def corearray_gb_data_transfer(cores, global_buffers, core_idx, gb_idx, array_idx_gb, stage, mac_lane, core_num=1, a_row_idx=[0], a_col_idx=[0], a_idx_idx=0):
    if global_buffers[gb_idx].array_busy == False:
        # choose the data in array that is ready to remove
        array_idx_gb[gb_idx] = global_buffers[gb_idx].find_array_target(cores[core_idx].calculator_and_array.array_state_matrix)
        # if there is data in array satisfies the condition to remove
        if global_buffers[gb_idx].array_busy:
            cores[core_idx].calculator_and_array.update_to_removing(array_idx_gb[gb_idx])
            if (gb_idx == 5) or (gb_idx == 7):
                # if this is the case that transferring remaining X/FC1 results into LP/FC2's core SRAM1, we need to keep up X/FC1 core's block_counter_rm
                cores[core_idx].calculator_and_array.array_idx_rm_advance_keep(array_idx_gb[gb_idx])
    else: 
        global_buffers[gb_idx].array_latency_counter += 1
        if global_buffers[gb_idx].array_latency_counter == global_buffers[gb_idx].latency_count:
            global_buffers[gb_idx].array_latency_counter = 0
            global_buffers[gb_idx].array_busy = False
            global_buffers[gb_idx].array_complete2 = global_buffers[gb_idx].array_complete1
            # all data finish transferring
            cores[core_idx].calculator_and_array.update_to_null(array_idx_gb[gb_idx])
            if core_num == 1:
                if (stage > 6) and (array_idx_gb[gb_idx] == mac_lane - 1) and (gb_idx == 3):
                    # part of a block completes, we may need to update state matrix of A
                    # written when the second last data of array(out of mac_lane data) finishes transferring into GB
                    global_buffers[gb_idx].update_to_a1(cores[core_idx].calculator_and_array.block_counter_cal)
            elif core_num == 8:
                if (gb_idx == 3) or (gb_idx == 6):
                    if array_idx_gb[gb_idx] == (mac_lane - 1):
                        global_buffers[gb_idx].update_to_a2(a_row_idx[a_idx_idx], a_col_idx[a_idx_idx])
                if gb_idx == 5:
                    # case that X core is transferring remaining X result matrix to LP's GB
                    global_buffers[gb_idx].update_blocknum_counter_from_last_core(cores[core_idx].calculator_and_array.block_counter_rm, cores[core_idx].sram1.blocknum_col_std)
                if gb_idx == 7:
                    # case that FC1 core is transferring remaining FC1 result matrix to FC2's GB
                    if array_idx_gb[gb_idx] == (mac_lane - 1):
                        # result block written into GB5/GB7 increments
                        global_buffers[gb_idx].blocknum_counter_from_last_core = cores[core_idx].calculator_and_array.block_counter_rm

def gb_layernorm_data_transfer(global_buffers, layernorm, gb_idx, gb_idx_layernorm_start, gb_idx_layernorm_end):
    if layernorm[0].busy == False:
        if (global_buffers[gb_idx].layernorm_busy == False):
            (gb_idx_layernorm_start[0], gb_idx_layernorm_end[0]) = global_buffers[gb_idx].find_layernorm_null_target()
        elif (global_buffers[gb_idx].layernorm_busy == True):
            global_buffers[gb_idx].layernorm_latency_counter += 1
            if global_buffers[gb_idx].layernorm_latency_counter == global_buffers[gb_idx].latency_count:
                global_buffers[gb_idx].layernorm_latency_counter = 0
                global_buffers[gb_idx].layernorm_busy = False
                layernorm[0].update_to_ready(gb_idx_layernorm_start[0], gb_idx_layernorm_end[0])
                global_buffers[gb_idx].update_to_cal(gb_idx_layernorm_start[0], gb_idx_layernorm_end[0], "ln")

def layernorm_coresram1_data_transfer(cores, layernorm, core_idx, prev_core_idx, gb_idx_layernorm_start, gb_idx_layernorm_end):
    if (layernorm[0].ln_complete() or layernorm[0].removing_to_core_busy) and (layernorm[0].partial_removing_to_core_busy == False):
        # only if LN calculation of a row is complete and next core's SRAM has vacancy can we transfer LN's data into next core's SRAM
        
        start = layernorm[0].remove_start
        end = layernorm[0].remove_end if layernorm[0].remove_end < layernorm[0].state_matrix.shape[0] else layernorm[0].state_matrix.shape[0] - 1
        if cores[core_idx].sram1.check_remove_state(layernorm[0].row_idx, start, end):
            (gb_idx_layernorm_start[0], gb_idx_layernorm_end[0]) = layernorm[0].find_removing_target()
    elif layernorm[0].partial_removing_to_core_busy:
        layernorm[0].sram_latency_counter += 1
        if layernorm[0].sram_latency_counter == cores[core_idx].sram1.latency_count:
            layernorm[0].sram_latency_counter = 0  
            layernorm[0].partial_removing_to_core_busy = False
            cores[core_idx].sram1.update_to_ready_from_ln(layernorm[0].row_idx, cores[prev_core_idx].sram1.blocknum_row_std, gb_idx_layernorm_start[0], gb_idx_layernorm_end[0])
            layernorm[0].update_to_null(gb_idx_layernorm_start[0], gb_idx_layernorm_end[0])   # row_idx increment

def corearray_coresram_data_transfer(cores, prev_core_idx, nxt_core_idx, array_idx_gb, mac_lane, sram, matrix):
    """ 
    Data transfer from previous core array to next core SRAM
    NOTE: here we assume Q/K/V won't exceed SRAM1/2's capacity 
    """
    assert((sram == 1) or (sram == 2))
    if cores[prev_core_idx].calculator_and_array.array_sram_busy == False:   
        array_idx_gb[prev_core_idx] = cores[prev_core_idx].calculator_and_array.find_array_target("sram")
        if cores[prev_core_idx].calculator_and_array.array_sram_busy:
            cores[prev_core_idx].calculator_and_array.update_to_removing(array_idx_gb[prev_core_idx])
    else:
        cores[prev_core_idx].calculator_and_array.sram_latency_counter += 1
        if cores[prev_core_idx].calculator_and_array.sram_latency_counter == cores[nxt_core_idx].sram1.latency_count:
            cores[prev_core_idx].calculator_and_array.sram_latency_counter = 0
            cores[prev_core_idx].calculator_and_array.array_sram_busy = False
            cores[prev_core_idx].calculator_and_array.update_to_null(array_idx_gb[prev_core_idx])
            if array_idx_gb[prev_core_idx] == (mac_lane - 1):
                if sram == 1:
                    cores[nxt_core_idx].sram1.array_block_counter += 1
                    
                    if matrix == "A'*V":
                        if (cores[nxt_core_idx].sram1.array_block_counter % 2) == 0:
                            cores[nxt_core_idx].sram1.update_to_ready_from_array_av(cores[prev_core_idx].sram1.blocknum_col_std)
                        if (cores[prev_core_idx].calculator_and_array.block_counter_rm % cores[prev_core_idx].sram1.blocknum_col_std) == 0:
                            # if a mac_lane row of data finishes calcualtion, we need to update all row in next core's sram
                            cores[nxt_core_idx].sram1.update_to_ready_from_array_abrupt(cores[prev_core_idx].sram1.blocknum_col_std)
                    else:
                        if (cores[nxt_core_idx].sram1.array_block_counter % 2) == 0:
                            cores[nxt_core_idx].sram1.update_to_ready_from_array((int(cores[nxt_core_idx].sram1.array_block_counter // 2) - 1))
                    
                    if cores[nxt_core_idx].sram1.array_block_counter == cores[prev_core_idx].calculator_and_array.block_cnt:
                        cores[nxt_core_idx].sram1.write_complete = True

                else:
                    # print("in corearray_coresram_data_transfer sram2.array_block_counter: " + str(cores[nxt_core_idx].sram2.array_block_counter))
                    cores[nxt_core_idx].sram2.array_block_counter += 1
                    if matrix == "K":
                        if (cores[nxt_core_idx].sram2.array_block_counter % 2) == 0:
                            cores[nxt_core_idx].sram2.update_to_ready_from_array(cores[prev_core_idx].sram2.blocknum_col_std, cores[prev_core_idx].calculator_and_array.block_cnt, matrix)
                    elif matrix == "V":
                        if (((cores[nxt_core_idx].sram2.array_block_counter - 1) // int(cores[nxt_core_idx].sram2.logic_sram_col_cnt_std // cores[nxt_core_idx].calculator_and_array.mac_lane)) % 2 == 1):
                            cores[nxt_core_idx].sram2.update_to_ready_from_array(cores[prev_core_idx].sram2.blocknum_col_std, cores[prev_core_idx].calculator_and_array.block_cnt, matrix)
                    else:
                        assert(0)

def gb_softmax_data_transfer(global_buffers, softmax, idx, gb_idx_softmax_start, gb_idx_softmax_end):
    if (global_buffers[idx].softmax_busy == False) and (softmax[0].busy == False):
        (gb_idx_softmax_start[0], gb_idx_softmax_end[0]) = global_buffers[idx].find_softmax_null_target()
    elif (global_buffers[idx].softmax_busy == True) and (softmax[0].busy == False):
        global_buffers[idx].softmax_latency_counter += 1
        if global_buffers[idx].softmax_latency_counter == global_buffers[idx].latency_count:
            global_buffers[idx].softmax_latency_counter = 0
            global_buffers[idx].softmax_busy = False
            # softmax[0].busy = True
            softmax[0].update_to_a(gb_idx_softmax_start[0], gb_idx_softmax_end[0])
            global_buffers[idx].update_to_cal(gb_idx_softmax_start[0], gb_idx_softmax_end[0], "softmax")

def softmax_gb_data_transfer(global_buffers, softmax, idx, gb_idx_softmax_start, gb_idx_softmax_end):
    if (global_buffers[idx].softmax_busy == False) and softmax[0].busy and softmax[0].done:
        (gb_idx_softmax_start[0], gb_idx_softmax_end[0]) = global_buffers[idx].find_softmax_res_target()
    elif (global_buffers[idx].softmax_busy == True) and softmax[0].busy and softmax[0].done:
        global_buffers[idx].softmax_latency_counter += 1
        if global_buffers[idx].softmax_latency_counter == global_buffers[idx].latency_count:
            global_buffers[idx].softmax_latency_counter = 0
            global_buffers[idx].softmax_busy = False
            # softmax[0].busy possibly updates to True
            softmax[0].update_to_null(gb_idx_softmax_start[0], gb_idx_softmax_end[0])
            global_buffers[idx].update_to_asoftmax(gb_idx_softmax_start[0], gb_idx_softmax_end[0])

def softmax_coresram1_data_transfer(global_buffers, softmax, cores, gb_idx, core_idx, gb_idx_softmax_start, gb_idx_softmax_end):
    if (global_buffers[gb_idx].softmax_busy == False) and softmax[0].busy and softmax[0].done:
        (gb_idx_softmax_start[0], gb_idx_softmax_end[0]) = global_buffers[gb_idx].find_softmax_res_target()
    elif (global_buffers[gb_idx].softmax_busy == True) and softmax[0].busy and softmax[0].done:
        global_buffers[gb_idx].softmax_latency_counter += 1
        if global_buffers[gb_idx].softmax_latency_counter == cores[core_idx].sram1.latency_count:
            global_buffers[gb_idx].softmax_latency_counter = 0
            global_buffers[gb_idx].softmax_busy = False
            # softmax[0].busy possibly updates to True
            softmax[0].update_to_null(gb_idx_softmax_start[0], gb_idx_softmax_end[0])
            cores[core_idx].sram1.update_to_ready_from_softmax(global_buffers[gb_idx].a_row, gb_idx_softmax_start[0], gb_idx_softmax_end[0])
            global_buffers[gb_idx].update_to_asoftmax(gb_idx_softmax_start[0], gb_idx_softmax_end[0])


def softmax_cal(softmax):
    """ Execution of Softmax """

    if softmax[0].calculation():
        softmax[0].latency_counter += 1
        if softmax[0].latency_counter == softmax[0].latency_count:
            softmax[0].latency_counter = 0
            softmax[0].update_to_asoftmax()


def layernorm_cal(layernorm):
    """ Execution of Layernorm """

    if layernorm[0].calculation():
        layernorm[0].latency_counter += 1
        layernorm[0].busy = True
        if layernorm[0].latency_counter == layernorm[0].latency_count:
            layernorm[0].latency_counter = 0
            layernorm[0].update_to_xlayernorm()

def dump_all(cores, global_buffers, softmax, layernorm, stage, latency, core_num):
    if core_num == 1:
        cores[0].dump_state_matrix("#", "Q*k")
        cores[0].dump_cal_status("#")
        for i in range(5):
            global_buffers[i].dump_rm_status(i)
        global_buffers[3].dump_a_state_matrix()
        softmax[0].dump_cal_status()
        # print("stage: " + str(stage))
        if (stage == 0) or (stage == 1):
            print("gb0-array complete1: " + str(global_buffers[0].array_complete1))
            print("gb0-array complete2: " + str(global_buffers[0].array_complete2))
            print("gb0-sram1 complete1: " + str(global_buffers[0].sram1_complete1))
            print("gb0-sram1 complete2: " + str(global_buffers[0].sram1_complete2))
            print("gb0-sram2 complete1: " + str(global_buffers[0].sram2_complete1))
            print("gb0-sram2 complete2: " + str(global_buffers[0].sram2_complete2))
            print("gb1-array complete1: " + str(global_buffers[1].array_complete1))
            print("gb1-array complete2: " + str(global_buffers[1].array_complete2))
            print("gb1-sram1 complete1: " + str(global_buffers[1].sram1_complete1))
            print("gb1-sram1 complete2: " + str(global_buffers[1].sram1_complete2))
            print("gb1-sram2 complete1: " + str(global_buffers[1].sram2_complete1))
            print("gb1-sram2 complete2: " + str(global_buffers[1].sram2_complete2))
        elif (stage == 2) or (stage == 3):
            print("gb1-array complete1: " + str(global_buffers[1].array_complete1))
            print("gb1-array complete2: " + str(global_buffers[1].array_complete2))
            print("gb1-sram1 complete1: " + str(global_buffers[1].sram1_complete1))
            print("gb1-sram1 complete2: " + str(global_buffers[1].sram1_complete2))
            print("gb1-sram2 complete1: " + str(global_buffers[1].sram2_complete1))
            print("gb1-sram2 complete2: " + str(global_buffers[1].sram2_complete2))
            print("gb2-array complete1: " + str(global_buffers[2].array_complete1))
            print("gb2-array complete2: " + str(global_buffers[2].array_complete2))
            print("gb2-sram1 complete1: " + str(global_buffers[2].sram1_complete1))
            print("gb2-sram1 complete2: " + str(global_buffers[2].sram1_complete2))
            print("gb2-sram2 complete1: " + str(global_buffers[2].sram2_complete1))
            print("gb2-sram2 complete2: " + str(global_buffers[2].sram2_complete2))
        elif (stage == 4) or (stage == 5):
            print("gb2-array complete1: " + str(global_buffers[2].array_complete1))
            print("gb2-array complete2: " + str(global_buffers[2].array_complete2))
            print("gb2-sram1 complete1: " + str(global_buffers[2].sram1_complete1))
            print("gb2-sram1 complete2: " + str(global_buffers[2].sram1_complete2))
            print("gb2-sram2 complete1: " + str(global_buffers[2].sram2_complete1))
            print("gb2-sram2 complete2: " + str(global_buffers[2].sram2_complete2))
            print("gb3-array complete1: " + str(global_buffers[3].array_complete1))
            print("gb3-array complete2: " + str(global_buffers[3].array_complete2))
            print("gb3-sram1 complete1: " + str(global_buffers[3].sram1_complete1))
            print("gb3-sram1 complete2: " + str(global_buffers[3].sram1_complete2))
            print("gb3-sram2 complete1: " + str(global_buffers[3].sram2_complete1))
            print("gb3-sram2 complete2: " + str(global_buffers[3].sram2_complete2))
        print("gb3-array complete1: " + str(global_buffers[3].array_complete1))
        print("gb3-array complete2: " + str(global_buffers[3].array_complete2))
        print("gb3-sram1 complete1: " + str(global_buffers[3].sram1_complete1))
        print("gb3-sram1 complete2: " + str(global_buffers[3].sram1_complete2))
        print("gb3-sram2 complete1: " + str(global_buffers[3].sram2_complete1))
        print("gb3-sram2 complete2: " + str(global_buffers[3].sram2_complete2))
        print("gb4-array complete1: " + str(global_buffers[4].array_complete1))
        print("gb4-array complete2: " + str(global_buffers[4].array_complete2))
        print("gb4-sram1 complete1: " + str(global_buffers[4].sram1_complete1))
        print("gb4-sram1 complete2: " + str(global_buffers[4].sram1_complete2))
        print("gb4-sram2 complete1: " + str(global_buffers[4].sram2_complete1))
        print("gb4-sram2 complete2: " + str(global_buffers[4].sram2_complete2))
        print("core-sram1 complete: " + str(cores[0].sram1.cal_complete))
        print("core-sram2 complete: " + str(cores[0].sram2.cal_complete))
        print("core-calculator complete: " + str(cores[0].calculator_and_array.complete))
    elif core_num == 8:
        # if global_buffers[0].array_complete2:
        #     cores[0].dump_state_matrix("FC1")
        # else:
        #     cores[0].dump_state_matrix("Q")
        # if global_buffers[0].sram1_complete1 == False:
        cores[0].dump_cal_status("Q")
        # cores[1].dump_state_matrix("K")
        cores[1].dump_cal_status("K")
        # cores[2].dump_state_matrix("V")
        cores[2].dump_cal_status("V")
        # cores[3].dump_state_matrix("Q*K")
        # if cores[3].blocknum_cal[1] < 12:
        cores[3].dump_cal_status("Q*K")
        # for i in range(5):
        #     global_buffers[i].dump_rm_status(i)
        print("GB3 a state matrix")
        print(global_buffers[3].a_state_matrix)
        softmax[0].dump_cal_status()
        cores[4].dump_state_matrix("A'*V")
        cores[4].dump_cal_status("A'*V")
        cores[5].dump_state_matrix("LP")
        cores[5].dump_cal_status("LP")
        print("GB6 x state matrix")
        print(global_buffers[6].a_state_matrix)
        layernorm[0].dump_cal_status()
        cores[6].dump_state_matrix("FC1")
        cores[6].dump_cal_status("FC1")
        cores[7].dump_state_matrix("FC2")
        cores[7].dump_cal_status("FC2")
        # global_buffers[5].dump_rm_status("FC1")
        # global_buffers[6].dump_rm_status("FC2")
        # print("cores[0].sram1.cal_complete: " + str(cores[0].sram1.cal_complete))
        # print("cores[0].sram2.cal_complete: " + str(cores[0].sram2.cal_complete))
        # print("cores[5].sram1.cal_complete: " + str(cores[5].sram1.cal_complete))
        # print("cores[6].sram1.cal_complete: " + str(cores[6].sram1.cal_complete))
        # print("globalbuffer[7].array_complete2: " + str(global_buffers[7].array_complete2))
        # print("GB3.array_complete2: " + str(global_buffers[3].array_complete2))
    else:
        raise NotImplementedError("Core number of " + str(core_num) + " is not supported yet!")

    print(latency)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

def simulating(args):
    """ 
    Remaining problems: 
    2. It is not ready simply for sram1 write data into gb, but gb needs to provides new data for sram1, which is not implemented

    global_buffer.sram1_complete -> global_buffer.sram2_complete -> core.sram.complete -> core.calculator_and_array.complete -> global_buffer.array_complete 
    """
    
    latency = 0.0
    stop = False
    # when core = 5, this variable indicates whether matrix A can be put into the core SRAM
    use_sram = False
    # global utilization counter, for calculating each core's utilization
    util_counter = 0

    """ HW initialization """

    head_embedding_dim = int(args.embedding_dim // args.head_num) 
    sram1_height = int(args.SRAM_capacity // args.MAC_lane // args.MAC_num)
    sram2_height = int(args.SRAM_capacity // args.MAC_num)
    blocknum_row = int(args.seq_length // args.MAC_lane)
    blocknum_col_qkv = int(head_embedding_dim // args.MAC_lane)
    blocknum_col_a = blocknum_row
    blocknum_col_subx = int(head_embedding_dim // args.MAC_lane)
    blocknum_col_lp = int(args.embedding_dim // args.MAC_lane)
    blocknum_col_fc1 = 4 * int(args.embedding_dim // args.MAC_lane)
    blocknum_col_fc2 = int(args.embedding_dim // args.MAC_lane)
    subsum_cnt_qkv = int(args.embedding_dim // args.MAC_num)
    subsum_cnt_a = int(head_embedding_dim // args.MAC_num)
    subsum_cnt_subx = int(args.seq_length // args.MAC_num)
    subsum_cnt_lp = subsum_cnt_qkv
    subsum_cnt_fc1 = subsum_cnt_qkv
    subsum_cnt_fc2 = subsum_cnt_fc1 * 4
    blocknum_row_sram1_qkv = int(sram1_height // subsum_cnt_qkv)
    blocknum_row_sram1_a = int(sram1_height // subsum_cnt_a)
    blocknum_row_sram1_subx = int(sram1_height // subsum_cnt_subx)
    blocknum_row_sram1_lp = int(sram1_height // subsum_cnt_lp)
    blocknum_row_sram1_fc1 = blocknum_row_sram1_qkv
    blocknum_row_sram1_fc2 = int(sram1_height // subsum_cnt_fc2)
    blocknum_col_sram2_qkv = int(sram2_height // subsum_cnt_qkv // args.MAC_lane)
    blocknum_col_sram2_a = int(sram2_height // subsum_cnt_a // args.MAC_lane)
    blocknum_col_sram2_subx = int(sram2_height // subsum_cnt_subx // args.MAC_lane)
    blocknum_col_sram2_lp = int(sram2_height // subsum_cnt_lp // args.MAC_lane)
    blocknum_col_sram2_fc1 = int(sram2_height // subsum_cnt_fc1 // args.MAC_lane)
    blocknum_col_sram2_fc2 = int(sram2_height // subsum_cnt_fc2 // args.MAC_lane)

    ## cores
    cores = []
    if args.core_num == 1:
        cores.append(Core(sram1_num=args.MAC_lane, sram1_height=int(args.SRAM_capacity // args.MAC_lane // args.MAC_num),
                            sram1_width=args.MAC_num, sram2_height=int(args.SRAM_capacity // args.MAC_num),
                            sram2_width=args.MAC_num, mac_lane=args.MAC_lane, mac_num=args.MAC_num, block_cnt=blocknum_row * blocknum_col_qkv,
                            sram_latency_count=args.SRAM_access_latency, array_and_calculator_latency_count=args.array_access_and_calculation_latency))

        cores[0].dump_configs("Q/K/V")
    elif args.core_num == 8:
        # Q/K/V
        for i in range(3):
            cores.append(Core(sram1_num=args.MAC_lane, sram1_height=int(args.SRAM_capacity // args.MAC_lane // args.MAC_num),
                            sram1_width=args.MAC_num, sram2_height=int(args.SRAM_capacity // args.MAC_num),
                            sram2_width=args.MAC_num, mac_lane=args.MAC_lane, mac_num=args.MAC_num, block_cnt=blocknum_row * blocknum_col_qkv,
                            sram_latency_count=args.SRAM_access_latency, array_and_calculator_latency_count=args.array_access_and_calculation_latency))
        
        # Q*K
        cores.append(Core(sram1_num=args.MAC_lane, sram1_height=int(args.SRAM_capacity // args.MAC_lane // args.MAC_num), 
                        sram1_width=args.MAC_num, sram2_height=int(args.SRAM_capacity // args.MAC_num),
                        sram2_width=args.MAC_num, mac_lane=args.MAC_lane, mac_num=args.MAC_num, block_cnt=blocknum_row * blocknum_col_a,
                        sram_latency_count=args.SRAM_access_latency, array_and_calculator_latency_count=args.array_access_and_calculation_latency))
 
        # A'*V
        cores.append(Core(sram1_num=args.MAC_lane, sram1_height=int(args.SRAM_capacity // args.MAC_lane // args.MAC_num), 
                        sram1_width=args.MAC_num, sram2_height=int(args.SRAM_capacity // args.MAC_num),
                        sram2_width=args.MAC_num, mac_lane=args.MAC_lane, mac_num=args.MAC_num, block_cnt=blocknum_row * blocknum_col_subx,
                        sram_latency_count=args.SRAM_access_latency, array_and_calculator_latency_count=args.array_access_and_calculation_latency))

        # Linear Projection after MH
        cores.append(Core(sram1_num=args.MAC_lane, sram1_height=int(args.SRAM_capacity // args.MAC_lane // args.MAC_num), 
                        sram1_width=args.MAC_num, sram2_height=int(args.SRAM_capacity // args.MAC_num),
                        sram2_width=args.MAC_num, mac_lane=args.MAC_lane, mac_num=args.MAC_num, block_cnt=blocknum_row * blocknum_col_lp,
                        sram_latency_count=args.SRAM_access_latency, array_and_calculator_latency_count=args.array_access_and_calculation_latency))

        # FC1
        cores.append(Core(sram1_num=args.MAC_lane, sram1_height=int(args.SRAM_capacity // args.MAC_lane // args.MAC_num),
                        sram1_width=args.MAC_num, sram2_height=int(args.SRAM_capacity // args.MAC_num),
                        sram2_width=args.MAC_num, mac_lane=args.MAC_lane, mac_num=args.MAC_num, block_cnt=blocknum_row * blocknum_col_fc1,
                        sram_latency_count=args.SRAM_access_latency, array_and_calculator_latency_count=args.array_access_and_calculation_latency))

        # FC2
        cores.append(Core(sram1_num=args.MAC_lane, sram1_height=int(args.SRAM_capacity // args.MAC_lane // args.MAC_num),
                        sram1_width=args.MAC_num, sram2_height=int(args.SRAM_capacity // args.MAC_num),
                        sram2_width=args.MAC_num, mac_lane=args.MAC_lane, mac_num=args.MAC_num, block_cnt=blocknum_row * blocknum_col_fc2,
                        sram_latency_count=args.SRAM_access_latency, array_and_calculator_latency_count=args.array_access_and_calculation_latency))

        cores[0].dump_configs("Q/K/V")
        cores[3].dump_configs("Q*K")
        cores[4].dump_configs("A'*V")
        cores[5].dump_configs("FC1")
        cores[6].dump_configs("FC2")
    else:
        raise NotImplementedError("Core number of " + str(args.core_num) + " is not supported yet!")

    ## global_buffers
    global_buffers = []
    if args.core_num == 1:
        for i in range(3):
            global_buffers.append(GlobalBuffer(latency_count=args.GB_access_latency, gb_sram_bandwidth=args.GB_SRAM_bandwidth))
        global_buffers.append(GlobalBuffer(latency_count=args.GB_access_latency, gb_sram_bandwidth=args.GB_SRAM_bandwidth, softmax_bandwidth=args.softmax_throughput))
        
        global_buffers.append(GlobalBuffer(latency_count=args.GB_access_latency, gb_sram_bandwidth=args.GB_SRAM_bandwidth))
        global_buffers[3].dump_configs()        
    elif args.core_num == 8:
        for i in range(3):
            global_buffers.append(GlobalBuffer(latency_count=args.GB_access_latency, gb_sram_bandwidth=args.GB_SRAM_bandwidth))
        
        global_buffers.append(GlobalBuffer(latency_count=args.GB_access_latency, gb_sram_bandwidth=args.GB_SRAM_bandwidth, softmax_bandwidth=args.softmax_throughput))
        global_buffers[3].dump_configs()        
        
        global_buffers.append(GlobalBuffer(latency_count=args.GB_access_latency, gb_sram_bandwidth=args.GB_SRAM_bandwidth))
        global_buffers[4].rownum1 = 2
        
        global_buffers.append(GlobalBuffer(latency_count=args.GB_access_latency, gb_sram_bandwidth=args.GB_SRAM_bandwidth))
        global_buffers.append(GlobalBuffer(latency_count=args.GB_access_latency, gb_sram_bandwidth=args.GB_SRAM_bandwidth, layernorm_bandwidth=args.GB_LN_bandwidth))
        global_buffers[6].dump_configs()        


        for i in range(2):
            global_buffers.append(GlobalBuffer(latency_count=args.GB_access_latency, gb_sram_bandwidth=args.GB_SRAM_bandwidth))


        if args.seq_length <= int(math.sqrt(args.SRAM_capacity)):
            # whether A is stored in GB or core SRAM
            use_sram = True
        print("If all A can be stored in cores' SRAM: " + str(use_sram))

        if use_sram:
            global_buffers[3].latency_count = args.SRAM_access_latency
    else:
        raise NotImplementedError("Core number of " + str(args.core_num) + " is not supported yet!")

    ## softmax
    softmax = []
    softmax.append(Softmax(latency_count=args.softmax_cal_latency, blocknum_col=blocknum_row))
    softmax[0].dump_configs()

    ## layernorm
    layernorm = []
    layernorm.append(LayerNorm(latency_count=args.layernorm_cal_latency, blocknum_col=blocknum_col_lp, to_sram_bandwidth=args.LN_SRAM_bandwidth))
    layernorm[0].dump_configs()


    """ Add Mappings """
    
    if args.core_num == 1:
        cores[0].sram1.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_qkv, 
                                    subsum_cnt=subsum_cnt_qkv, blocknum_row_sram=blocknum_row_sram1_qkv)
        cores[0].sram2.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_qkv,
                                    block_col=args.MAC_lane, subsum_cnt=subsum_cnt_qkv, blocknum_col_sram=blocknum_col_sram2_qkv)
        cores[0].calculator_and_array.add_mapping(subsum_cnt=subsum_cnt_qkv)    

        cores[0].dump_mappings("Q/K/V")   

    elif args.core_num == 8:
        for i in range(3):
            cores[i].sram1.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_qkv, 
                                    subsum_cnt=subsum_cnt_qkv, blocknum_row_sram=blocknum_row_sram1_qkv)
            cores[i].sram2.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_qkv,
                                        block_col=args.MAC_lane, subsum_cnt=subsum_cnt_qkv, blocknum_col_sram=blocknum_col_sram2_qkv)
            cores[i].calculator_and_array.add_mapping(subsum_cnt=subsum_cnt_qkv)  
        cores[0].dump_mappings("Q/K/V")   
        
        cores[3].sram1.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_a, 
                                    subsum_cnt=subsum_cnt_a, blocknum_row_sram=blocknum_row_sram1_a)
        cores[3].sram2.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_a,
                                    block_col=args.MAC_lane, subsum_cnt=subsum_cnt_a, blocknum_col_sram=blocknum_col_sram2_a)
        cores[3].calculator_and_array.add_mapping(subsum_cnt=subsum_cnt_a)  
        cores[3].dump_mappings("Q*K")

        cores[4].sram1.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_subx, 
                                    subsum_cnt=subsum_cnt_subx, blocknum_row_sram=blocknum_row_sram1_subx)
        cores[4].sram2.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_subx,
                                    block_col=args.MAC_lane, subsum_cnt=subsum_cnt_subx, blocknum_col_sram=blocknum_col_sram2_subx)
        cores[4].calculator_and_array.add_mapping(subsum_cnt=subsum_cnt_subx)  
        cores[4].dump_mappings("A'*V")

        cores[5].sram1.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_lp,
                                    subsum_cnt=subsum_cnt_lp, blocknum_row_sram=blocknum_row_sram1_lp)
        cores[5].sram2.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_lp,
                                    block_col=args.MAC_lane, subsum_cnt=subsum_cnt_lp, blocknum_col_sram=blocknum_col_sram2_lp)
        cores[5].calculator_and_array.add_mapping(subsum_cnt=subsum_cnt_lp)
        cores[5].dump_mappings("Linear Projection after MH")

        cores[6].sram1.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_fc1,
                                    subsum_cnt=subsum_cnt_fc1, blocknum_row_sram=blocknum_row_sram1_fc1)
        cores[6].sram2.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_fc1,
                                    block_col=args.MAC_lane, subsum_cnt=subsum_cnt_fc1, blocknum_col_sram=blocknum_col_sram2_fc1)
        cores[6].calculator_and_array.add_mapping(subsum_cnt=subsum_cnt_fc1)
        cores[6].dump_mappings("FC1")
        
        cores[7].sram1.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_fc2,
                                    subsum_cnt=subsum_cnt_fc2, blocknum_row_sram=blocknum_row_sram1_fc2)
        cores[7].sram2.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_fc2,
                                    block_col=args.MAC_lane, subsum_cnt=subsum_cnt_fc2, blocknum_col_sram=blocknum_col_sram2_fc2)
        cores[7].calculator_and_array.add_mapping(subsum_cnt=subsum_cnt_fc2)
        cores[7].dump_mappings("FC2")

    else:
        raise NotImplementedError("Core number of " + str(args.core_num) + " is not supported yet!")
    
    """ 
    1 core case:
    GB0 provides operand matrices for Q calculation and takes Q matrix
    GB1 provides operand matrices for K calculation and takes K matrix
    GB2 provides operand matrices for V calculation and takes V matrix
    GB3 provides operand matrices for Q * K calculation and takes A matrix
        provides A matrix for Softmax and takes A' matrix
    GB4 provides operand matrices for A' * V calculation and takes subX matrix
        asks GB3 for the state matrix of A'

    8 core case:
    GB0-Q     provides X and W_Q for Q calculation
    GB1-K     provides X and W_K for K calculation
    GB2-V     provides X and W_V for K calculation
    GB3-Q*K   takes A and provides A for Softmax calculation and takes A' from Softmax under some circumstance
    GB4-A'*V  provides A' for A'*V calculation and 
    GB5-LP    takes remaining subX provides remaining subX and Weight for LP calculation and 
    GB6-FC1   takes X from LP and provides X for LN and provides Weight for FC1 calculation
    GB7-FC2   takes remaining expanded_X and provides remaining expended_X and Weight for FC2 calculation
    GB8       takes the output matrix of FC2
    """
    for i in range(3):
        global_buffers[i].add_mapping(blocknum_row_cnt=blocknum_row, array_data_cnt=blocknum_row * blocknum_col_qkv,
                                        sram_subsum_cnt=subsum_cnt_qkv, sram1_rownum_cnt=blocknum_row_sram1_qkv, 
                                        sram2_colnum_cnt=head_embedding_dim, sram2_sram_colnum_cnt=blocknum_col_sram2_qkv * args.MAC_lane)
    global_buffers[0].dump_mappings("Q/K/V")

    global_buffers[3].add_mapping(blocknum_row_cnt=blocknum_row, array_data_cnt=blocknum_row * blocknum_col_a,
                                    sram_subsum_cnt=subsum_cnt_a, sram1_rownum_cnt=blocknum_row_sram1_a, 
                                    sram2_colnum_cnt=args.seq_length, sram2_sram_colnum_cnt=blocknum_col_sram2_a * args.MAC_lane, flag=True)
    global_buffers[3].dump_mappings("Q*K")

    global_buffers[4].add_mapping(blocknum_row_cnt=blocknum_row, array_data_cnt=0,
                                    sram_subsum_cnt=subsum_cnt_subx, sram1_rownum_cnt=blocknum_row_sram1_subx, 
                                    sram2_colnum_cnt=head_embedding_dim, sram2_sram_colnum_cnt=blocknum_col_sram2_subx * args.MAC_lane)
    global_buffers[4].dump_mappings("A'*V")

    global_buffers[5].add_mapping(blocknum_row_cnt=blocknum_row, array_data_cnt=blocknum_row * blocknum_col_subx,
                                    sram_subsum_cnt=subsum_cnt_lp, sram1_rownum_cnt=blocknum_row_sram1_lp, 
                                    sram2_colnum_cnt=args.embedding_dim, sram2_sram_colnum_cnt=blocknum_col_sram2_lp * args.MAC_lane)
    global_buffers[5].dump_mappings("Linear Projection after MH")
    
    global_buffers[6].add_mapping(blocknum_row_cnt=blocknum_row, array_data_cnt=blocknum_row * blocknum_col_lp,
                                    sram_subsum_cnt=subsum_cnt_fc1, sram1_rownum_cnt=blocknum_row_sram1_fc1, 
                                    sram2_colnum_cnt=4 * args.embedding_dim, sram2_sram_colnum_cnt=blocknum_col_sram2_fc1 * args.MAC_lane, flag=True)
    global_buffers[6].dump_mappings("FC1")
    
    
    global_buffers[7].add_mapping(blocknum_row_cnt=blocknum_row, array_data_cnt=blocknum_row * blocknum_col_fc1, 
                                    sram_subsum_cnt=subsum_cnt_fc2, sram1_rownum_cnt=blocknum_row_sram1_fc2, 
                                    sram2_colnum_cnt=args.embedding_dim, sram2_sram_colnum_cnt=blocknum_col_sram2_fc2 * args.MAC_lane)

    global_buffers[8].add_mapping(blocknum_row_cnt=0, array_data_cnt=blocknum_row * blocknum_col_fc2, 
                                    sram_subsum_cnt=0, sram1_rownum_cnt=0, 
                                    sram2_colnum_cnt=0, sram2_sram_colnum_cnt=0)
    
    # since LP/FC2's SRAM1 is already hold a sub-SRAM of data from A'*V/FC1's core directly, this gb-sram transfer is the second time
    global_buffers[5].rownum1 = 2
    global_buffers[7].rownum1 = 2


    """ 
    NotImplementedError 
    
    Configurations which are not supported yet
    """
    # Capacity of SRAM2 in the core cannot be exceeded
    if args.seq_length * (args.embedding_dim // args.head_num) > args.SRAM_capacity:
        raise NotImplementedError("Q/K/V size CAN'T exceed blue SRAM capacity!")
    elif args.embedding_dim * (args.embedding_dim // args.head_num) > args.SRAM_capacity:
        raise NotImplementedError("Weight matrix size CAN'T exceed core SRAM capacity!")
    else:
        pass 
    
    # capacity of SRAM2 cannot be exceeded by a mac_lane column of FC2 weight matrix
    # eg. 2048 >= 1024*4*16/32  
    if (args.SRAM_capacity // args.MAC_num) < (args.embedding_dim * 4 * args.MAC_lane // args.MAC_num):
        raise NotImplementedError("A mac_lane column of FC2 weight matrix size CAN'T exceed SRAM capacity!")

    # layernorm-core bandwidth must be even number
    if (args.LN_SRAM_bandwidth % 2) != 0:
        raise ValueError("Layernorm to Core bandwidth must be the even times of number of mac_lane*mac_lane BYTE")

    """ Simulating """
    # stall for one cycle between different calculation stages
    count = [0]
    # 1 if last block of A should be written into GB
    flag = [0] * 2
    # helper counters for marking the end calculation of a core 
    end_counter = [0] * 8

    # for core_num = 1
    stage = 0
    # for core_num = 5    
    qkv_stage = 0
    a_stage = 0
    x_stage = 0
    lp_stage = 0
    fc1_stage = 0
    fc2_stage = 0

    sram1_idx_gb_start = [0] * 8
    sram1_idx_gb_end = [0] * 8

    rownum_sram2_idx_gb_start = [0] * 8
    rownum_sram2_idx_gb_end = [0] * 8
    colnum_sram2_idx_gb_start = [0] * 8
    colnum_sram2_idx_gb_end = [0] * 8

    array_idx_gb = [0] * 9
    # for solving conflict: two function use 6 to index array_idx_gb
    array_idx_gb_copy = [0] * 7
    # layernorm
    gb_idx_layernorm_start = [0]
    gb_idx_layernorm_end = [0]
    # softmax
    gb_idx_softmax_start = [0]
    gb_idx_softmax_end = [0]

    # used only when core_num is 5, for recording which block of a matrix is transferring from array to GB/core SRAM
    a_row_idx = [0] * 2
    a_col_idx = [0] * 2

    counter = 0

    while stop == False:

        if args.core_num == 1:
            """ 
            stage 0/1: Q calculation
            stage 2/3: K calculation
            stage 4/5: V calculation
            stage 6: core reconfiguration
            stage 7/8: Q*K calculation
            stage 9: core reconfiguration
            stage 10/11: A'*V calculation 
            """

            """ Data transfer between GB and core SRAM/Array """
            if (stage == 0) or (stage == 1):
                if global_buffers[0].sram1_complete2 == False:
                    # Read X data from GB to core sram1 for X * W_Q calculation
                    coresram1_gb_data_transfer(cores, global_buffers, 0, 0, sram1_idx_gb_start, sram1_idx_gb_end)
                else:
                    # Read X data from GB to core sram1 for X * W_K calculation
                    coresram1_gb_data_transfer(cores, global_buffers, 0, 1, sram1_idx_gb_start, sram1_idx_gb_end)

                if global_buffers[0].sram2_complete2 == False:
                    # Read W_Q data from GB to core sram1 for X * W_Q calculation
                    coresram2_gb_data_transfer(cores, global_buffers, 0, 0, rownum_sram2_idx_gb_start, rownum_sram2_idx_gb_end, colnum_sram2_idx_gb_start, colnum_sram2_idx_gb_end)
                else:
                    # Read W_K data from GB to core sram1 for X * W_K calculation
                    coresram2_gb_data_transfer(cores, global_buffers, 0, 1, rownum_sram2_idx_gb_start, rownum_sram2_idx_gb_end, colnum_sram2_idx_gb_start, colnum_sram2_idx_gb_end)

                if global_buffers[0].array_complete2 == False:
                    # Transfer Q data to GB
                    corearray_gb_data_transfer(cores, global_buffers, 0, 0, array_idx_gb, stage, args.MAC_lane, flag, a_row_idx, a_col_idx)
            elif (stage == 2) or (stage == 3):
                if global_buffers[1].sram1_complete2 == False:
                    # Read X data from GB to core sram1 for X * W_K calculation
                    coresram1_gb_data_transfer(cores, global_buffers, 0, 1, sram1_idx_gb_start, sram1_idx_gb_end)
                else:
                    # Read X data from GB to core sram1 for X * W_V calculation
                    coresram1_gb_data_transfer(cores, global_buffers, 0, 2, sram1_idx_gb_start, sram1_idx_gb_end)

                if global_buffers[1].sram2_complete2 == False:
                    # Read W_K data from GB to core sram1 for X * W_K calculation
                    coresram2_gb_data_transfer(cores, global_buffers, 0, 1, rownum_sram2_idx_gb_start, rownum_sram2_idx_gb_end, colnum_sram2_idx_gb_start, colnum_sram2_idx_gb_end)
                else:
                    # Read W_V data from GB to core sram1 for X * W_V calculation
                    coresram2_gb_data_transfer(cores, global_buffers, 0, 2, rownum_sram2_idx_gb_start, rownum_sram2_idx_gb_end, colnum_sram2_idx_gb_start, colnum_sram2_idx_gb_end)

                if global_buffers[0].array_complete2 == False:
                    # Complete transferring Q data to GB
                    corearray_gb_data_transfer(cores, global_buffers, 0, 0, array_idx_gb, stage, args.MAC_lane, flag, a_row_idx, a_col_idx)
                if global_buffers[1].array_complete2 == False:
                    # Transfer K data to GB
                    corearray_gb_data_transfer(cores, global_buffers, 0, 1, array_idx_gb, stage, args.MAC_lane, flag, a_row_idx, a_col_idx)
            elif (stage == 4) or (stage == 5):
                if global_buffers[2].sram1_complete2 == False:
                    # Read X data from GB to core sram1 for X * W_V calculation
                    coresram1_gb_data_transfer(cores, global_buffers, 0, 2, sram1_idx_gb_start, sram1_idx_gb_end)
                else:
                    # Read Q data from GB to core sram1 for Q * K calculation
                    coresram1_gb_data_transfer(cores, global_buffers, 0, 3, sram1_idx_gb_start, sram1_idx_gb_end)

                if global_buffers[2].sram2_complete2 == False:
                    # Read W_V data from GB to core sram1 for X * W_V calculation
                    coresram2_gb_data_transfer(cores, global_buffers, 0, 2, rownum_sram2_idx_gb_start, rownum_sram2_idx_gb_end, colnum_sram2_idx_gb_start, colnum_sram2_idx_gb_end)
                else:
                    # Read K data from GB to core sram1 for Q * K calculation
                    coresram2_gb_data_transfer(cores, global_buffers, 0, 3, rownum_sram2_idx_gb_start, rownum_sram2_idx_gb_end, colnum_sram2_idx_gb_start, colnum_sram2_idx_gb_end)

                if global_buffers[1].array_complete2 == False:
                    # Complete transferring K data to GB
                    corearray_gb_data_transfer(cores, global_buffers, 0, 1, array_idx_gb, stage, args.MAC_lane, flag, a_row_idx, a_col_idx)
                if global_buffers[2].array_complete2 == False:
                    # Transfer V data to GB
                    corearray_gb_data_transfer(cores, global_buffers, 0, 2, array_idx_gb, stage, args.MAC_lane, flag, a_row_idx, a_col_idx)
            elif (stage == 6) or (stage == 7) or (stage == 8):
                if global_buffers[3].sram1_complete2 == False:
                    # Read Q data from GB to core sram1 for Q * K calculation
                    coresram1_gb_data_transfer(cores, global_buffers, 0, 3, sram1_idx_gb_start, sram1_idx_gb_end)
                else:
                    # Read A' data from GB to core sram1 for A' * V calculation
                    coresram1_gb_data_transfer_a(cores, global_buffers, 0, 4, sram1_idx_gb_start, sram1_idx_gb_end)

                if global_buffers[3].sram2_complete2 == False:
                    # Read K data from GB to core sram1 for Q * K calculation
                    coresram2_gb_data_transfer(cores, global_buffers, 0, 3, rownum_sram2_idx_gb_start, rownum_sram2_idx_gb_end, colnum_sram2_idx_gb_start, colnum_sram2_idx_gb_end)
                else: 
                    # Read V data from GB to core sram1 for A' * V calculation
                    coresram2_gb_data_transfer(cores, global_buffers, 0, 4, rownum_sram2_idx_gb_start, rownum_sram2_idx_gb_end, colnum_sram2_idx_gb_start, colnum_sram2_idx_gb_end)


                if global_buffers[2].array_complete2 == False:
                    # Complete transferring V data to GB
                    corearray_gb_data_transfer(cores, global_buffers, 0, 2, array_idx_gb, stage, args.MAC_lane, flag, a_row_idx, a_col_idx)    
                if global_buffers[3].array_complete2 == False:
                    # Transfer A data to GB
                    corearray_gb_data_transfer(cores, global_buffers, 0, 3, array_idx_gb, stage, args.MAC_lane, flag, a_row_idx, a_col_idx)

                
                if global_buffers[3].softmax_complete() == False:
                    # Executing softmax for A
                    gb_softmax_data_transfer(global_buffers, softmax, 3, gb_idx_softmax_start, gb_idx_softmax_end)
                    softmax_gb_data_transfer(global_buffers, softmax, 3, gb_idx_softmax_start, gb_idx_softmax_end)
            elif (stage == 9) or (stage == 10) or (stage == 11):
                if global_buffers[4].sram1_complete2 == False:
                    # Read A' data from GB to core sram1 for A' * V calculation
                    coresram1_gb_data_transfer_a(cores, global_buffers, 0, 4, sram1_idx_gb_start, sram1_idx_gb_end)
                
                if global_buffers[4].sram2_complete2 == False:
                    # Read V data from GB to core sram1 for A' * V calculation
                    coresram2_gb_data_transfer(cores, global_buffers, 0, 4, rownum_sram2_idx_gb_start, rownum_sram2_idx_gb_end, colnum_sram2_idx_gb_start, colnum_sram2_idx_gb_end)


                if global_buffers[3].array_complete2 == False:
                    # Complete transferring V data to GB
                    corearray_gb_data_transfer(cores, global_buffers, 0, 3, array_idx_gb, stage, args.MAC_lane, flag, a_row_idx, a_col_idx)    
                if global_buffers[4].array_complete2 == False:
                    # Transfer A data to GB
                    corearray_gb_data_transfer(cores, global_buffers, 0, 4, array_idx_gb, stage, args.MAC_lane, flag, a_row_idx, a_col_idx)

                if global_buffers[3].softmax_complete() == False:
                    # Executing softmax for A
                    gb_softmax_data_transfer(global_buffers, softmax, 3, gb_idx_softmax_start, gb_idx_softmax_end)
                    softmax_gb_data_transfer(global_buffers, softmax, 3, gb_idx_softmax_start, gb_idx_softmax_end)
            elif (stage == 12):
                if global_buffers[4].array_complete2 == False:
                    # Transfer A data to GB
                    corearray_gb_data_transfer(cores, global_buffers, 0, 4, array_idx_gb, stage, args.MAC_lane, flag, a_row_idx, a_col_idx)

            """ Calculation """
            if (stage == 0) or (stage == 2) or (stage == 4) or (stage == 7) or (stage == 10):
                """ Reading data from core SRAM """
                stage = read_from_core_sram(cores, stage, 0)
            elif (stage == 1) or (stage == 3) or (stage == 5) or (stage == 8) or (stage == 11):
                """ Dot production """ 
                stage = dot_production(cores, stage, count, 0)
            elif stage == 6:
                """ Q * K Reconfiguration """
                # print()
                # Here core should be reconfigured
                cores[0].reconfigure(block_cnt=blocknum_row * blocknum_col_a)
                cores[0].dump_configs("Q*K")
                # Here we assume sram1/2 can hold all Q/K
                cores[0].sram1.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_a, 
                                            subsum_cnt=subsum_cnt_a, blocknum_row_sram=blocknum_row_sram1_a)
                cores[0].sram2.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_a,
                                            block_col=args.MAC_lane, subsum_cnt=subsum_cnt_a, blocknum_col_sram=blocknum_col_sram2_a)
                cores[0].calculator_and_array.add_mapping(subsum_cnt=subsum_cnt_a) 
                cores[0].dump_mappings("Q*K")
                stage = 7     
            elif stage == 9:
                """ A' * V Reconfiguration """
                cores[0].reconfigure(block_cnt=blocknum_row * blocknum_col_subx)
                cores[0].dump_configs("A'*V")
                cores[0].sram1.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_subx,
                                            subsum_cnt=subsum_cnt_subx, blocknum_row_sram=blocknum_row_sram1_subx)
                cores[0].sram2.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_subx,
                                            block_col=args.MAC_lane, subsum_cnt=subsum_cnt_subx, blocknum_col_sram=blocknum_col_sram2_subx)
                cores[0].calculator_and_array.add_mapping(subsum_cnt=subsum_cnt_subx)
                cores[0].dump_mappings("A'*V")
                stage = 10  

            if stage > 6:
                """ Softmax execution """
                softmax_cal(softmax)

            """ For debug """
            # if stage > 10:
            #     dump_all(cores, global_buffers, softmax, layernorm, stage, latency, args.core_num)

            if global_buffers[4].array_complete2:
                stop = True
                dump_all(cores, global_buffers, softmax, layernorm, stage, latency, args.core_num)
        
        elif args.core_num == 8:
            """ 
            qkv_stage 0/1: Q/K/V calculation
            a_stage 0/1: Q*K calculation
            x_stage 0/1: A'*V calculation
            fc1_stage 0/1: FC1 calculation
            fc2_stage 0/1: FC2 calculation
            """

            """ Data transfer between GB and core SRAM/Array """
            if (qkv_stage == 0) or (qkv_stage == 1):
                for i in range(3):
                    if global_buffers[i].sram1_complete2 == False:
                        # Read X data from GB to core sram1 for X * W_Q/W_K/W_V calculation
                        coresram1_gb_data_transfer(cores, global_buffers, i, i, sram1_idx_gb_start, sram1_idx_gb_end)
            
                    if global_buffers[i].sram2_complete2 == False:
                        # Read W_Q/W_K/W_V data from GB to core sram1 for X * W_Q/W_K/W_V calculation
                        coresram2_gb_data_transfer(cores, global_buffers, i, i, rownum_sram2_idx_gb_start, rownum_sram2_idx_gb_end, colnum_sram2_idx_gb_start, colnum_sram2_idx_gb_end)
            
            if (a_stage == 0) or (a_stage == 1) or (a_stage == 2):
                if global_buffers[3].array_complete2 == False: 
                    # Transfer A data to GB/core SRAM                                   FIXME fix the stage judgement in this function
                    corearray_gb_data_transfer(cores, global_buffers, 3, 3, array_idx_gb, 7, args.MAC_lane, args.core_num, a_row_idx, a_col_idx, 0)
            
            if (x_stage == 0) or (x_stage == 1) or (x_stage == 2):
                if use_sram == False:
                    # if not all A' data can be stored in core SRAM, we need to update core SRAM data
                    if global_buffers[4].sram1_complete2 == False:
                        # Read A' data from GB to core sram1 for A' * V calculation
                        coresram1_gb_data_transfer_a(cores, global_buffers, 4, 4, sram1_idx_gb_start, sram1_idx_gb_end)   

            if (lp_stage == 0) or (lp_stage == 1):
                if global_buffers[5].sram1_complete2 == False:
                    # Transfer remaining X from GB to core5 for LP calulation
                    coresram1_gb_data_transfer(cores, global_buffers, 5, 5, sram1_idx_gb_start, sram1_idx_gb_end, args.MAC_lane, args.core_num)
                if global_buffers[5].sram2_complete2 == False:
                    # Transfer Weight matrix for LP calculation
                    coresram2_gb_data_transfer(cores, global_buffers, 5, 5, rownum_sram2_idx_gb_start, rownum_sram2_idx_gb_end, colnum_sram2_idx_gb_start, colnum_sram2_idx_gb_end)

            if (fc1_stage == 0) or (fc1_stage == 1):
                if global_buffers[6].sram2_complete2 == False:
                    # Transfer Weight Matrix from GB to core6 for FC1 calculation
                    coresram2_gb_data_transfer(cores, global_buffers, 6, 6, rownum_sram2_idx_gb_start, rownum_sram2_idx_gb_end, colnum_sram2_idx_gb_start, colnum_sram2_idx_gb_end)
                    
            if (fc2_stage == 0) or (fc2_stage == 1):
                if global_buffers[7].sram1_complete2 == False:
                    # Transfer remaining X_FC2 from GB to core7 for FC2 calculation
                    coresram1_gb_data_transfer(cores, global_buffers, 7, 7, sram1_idx_gb_start, sram1_idx_gb_end, args.MAC_lane, args.core_num)
                if global_buffers[7].sram2_complete2 == False:
                    # Transfer Weight Matrix for FC2 calculation
                    coresram2_gb_data_transfer(cores, global_buffers, 7, 7, rownum_sram2_idx_gb_start, rownum_sram2_idx_gb_end, colnum_sram2_idx_gb_start, colnum_sram2_idx_gb_end)
            
            if (lp_stage == 0) or (lp_stage == 1) or (lp_stage == 2):
                # Transfer LP's result matrix from core array into GB
                if global_buffers[6].array_complete2 == False:
                    corearray_gb_data_transfer(cores, global_buffers, 5, 6, array_idx_gb, 7, args.MAC_lane, args.core_num, a_row_idx, a_col_idx, 1)

            if (fc2_stage == 0) or (fc2_stage == 1) or (fc2_stage == 2):
                # Transfer FC2's result matrix from core array into GB
                if global_buffers[8].array_complete2 == False:
                    corearray_gb_data_transfer(cores, global_buffers, 7, 8, array_idx_gb, 7, args.MAC_lane, args.core_num)
                

            """ Data transfer to softmax and transfer back to GB/core SRAM """
            if global_buffers[3].transfer_to_softmax_complete() == False: 
                # Executing softmax for A
                gb_softmax_data_transfer(global_buffers, softmax, 3, gb_idx_softmax_start, gb_idx_softmax_end)

            if global_buffers[3].softmax_complete() == False:
                if use_sram:
                    softmax_coresram1_data_transfer(global_buffers, softmax, cores, 3, 4, gb_idx_softmax_start, gb_idx_softmax_end)
                else:
                    if global_buffers[3].a_row < (cores[4].sram1.height // cores[4].sram1.subsum_cnt_std):
                        # if core SRAM still has vacancy for A', transfer A' data from softmax to core SRAM1
                        softmax_coresram1_data_transfer(global_buffers, softmax, cores, 3, 4, gb_idx_softmax_start, gb_idx_softmax_end)
                    else:
                        # if core SRAM is full, transfer the reset of A' data to GB
                        softmax_gb_data_transfer(global_buffers, softmax, 3, gb_idx_softmax_start, gb_idx_softmax_end)


            """ Data transfer from GB to LN and transfer back to core SRAM """
            if layernorm[0].row_idx != blocknum_row:
                # Transfer X(LP result) data in GB to LN
                gb_layernorm_data_transfer(global_buffers, layernorm, 6, gb_idx_layernorm_start, gb_idx_layernorm_end)
                # only if LN.busy is True will this function be called

            if cores[6].sram1.write_complete == False: 
                # Transfer X data for FC1 calculation, the transfer process may be blocked by the FC1 calculation
                # NOTE: this means when FC1 SRAM is not empty, it won't receive data from LN, therefore there's no need to design a backup GB for FC1 SRAM1
                layernorm_coresram1_data_transfer(cores, layernorm, 6, 5, gb_idx_layernorm_start, gb_idx_layernorm_end)


            """ Data transfer between previous core's array and next cores SRAM """
            if (qkv_stage == 0) or (qkv_stage == 1) or (a_stage == 0) or (a_stage == 1):
                if cores[3].sram1.write_complete == False:
                    # Read Q from core0 to core3 sram1 for Q * K calculation
                    corearray_coresram_data_transfer(cores, 0, 3, array_idx_gb, args.MAC_lane, 1, "Q")
                
                if cores[3].sram2.write_complete == False:
                    # Read K from core1 to core3 sram2 for Q * K calculation
                    corearray_coresram_data_transfer(cores, 1, 3, array_idx_gb, args.MAC_lane, 2, "K")
 
                if cores[4].sram2.write_complete == False:
                    # Read V from core1 to core4 sram2 for A' * V calculation
                    corearray_coresram_data_transfer(cores, 2, 4, array_idx_gb, args.MAC_lane, 2, "V")

            if (x_stage == 0) or (x_stage == 1) or (lp_stage == 0) or (lp_stage == 1):
                if cores[5].sram1.write_complete == False:
                    if cores[4].calculator_and_array.is_next_core_sram_full(blocknum_row_sram1_lp, blocknum_col_subx) == False:
                        # Read result matrix from core4(A'*V) to core5(LP) for LP calculation
                        corearray_coresram_data_transfer(cores, 4, 5, array_idx_gb, args.MAC_lane, 1, "A'*V")
                    elif (cores[4].calculator_and_array.block_counter_rm == blocknum_row_sram1_lp * blocknum_col_subx) and (cores[4].calculator_and_array.array_state_matrix[-1] == utils.REMOVING): 
                        corearray_coresram_data_transfer(cores, 4, 5, array_idx_gb, args.MAC_lane, 1, "A'*V")
                    else:
                        # Read result matrix from core4(A'*V) to core5(LP)'s global buffer, since core5's SRAM is running out of capacity
                        corearray_gb_data_transfer(cores, global_buffers, 4, 5, array_idx_gb, 7, args.MAC_lane, args.core_num)

            if (fc1_stage == 0) or (fc1_stage == 1) or (fc2_stage == 0) or (fc2_stage == 1):
                if cores[7].sram1.write_complete == False:
                    if cores[6].calculator_and_array.block_counter_rm < blocknum_row_sram1_fc2 * blocknum_col_fc1:
                        # Read result matrix from core6(FC1) to core7(FC2) for FC2 calculation
                        corearray_coresram_data_transfer(cores, 6, 7, array_idx_gb_copy, args.MAC_lane, 1, "FC1")
                    elif (cores[6].calculator_and_array.block_counter_rm == blocknum_row_sram1_fc2 * blocknum_col_fc1) and (cores[6].calculator_and_array.array_state_matrix[-1] == utils.REMOVING): 
                        # make sure the last data of last block(that can be written directly into core's SRAM1) is successfully transferred
                        corearray_coresram_data_transfer(cores, 6, 7, array_idx_gb_copy, args.MAC_lane, 1, "FC1")
                    else:
                        # Read result matrix from core6(FC1) to core7(FC2)'s global buffer, since core6's SRAM is running out of capacity
                        corearray_gb_data_transfer(cores, global_buffers, 6, 7, array_idx_gb, 7, args.MAC_lane, args.core_num)
                                         

            """ Q/K/V Calculation """
            if (qkv_stage == 0):
                """ Reading data from core SRAM """
                read_from_core_sram(cores, qkv_stage, 0)
                read_from_core_sram(cores, qkv_stage, 1)
                qkv_stage = read_from_core_sram(cores, qkv_stage, 2)
            elif (qkv_stage == 1):
                """ Dot production """ 
                dot_production(cores, qkv_stage, count, 0, args.core_num)
                dot_production(cores, qkv_stage, count, 1, args.core_num)
                qkv_stage = dot_production(cores, qkv_stage, count, 2, args.core_num)


            """ Q * K Calculation """
            if (a_stage == 0):
                """ Reading data from core SRAM """
                a_stage = read_from_core_sram(cores, a_stage, 3, True)
            elif (a_stage == 1):
                """ Dot production """ 
                a_stage = dot_production(cores, a_stage, count, 3, args.core_num, a_row_idx, a_col_idx, 0)


            """ Softmax execution """
            softmax_cal(softmax)


            """ A' * V Calculation """
            if (x_stage == 0):
                """ Reading data from core SRAM """
                x_stage = read_from_core_sram(cores, x_stage, 4)
            elif (x_stage == 1):
                """ Dot production """ 
                x_stage = dot_production(cores, x_stage, count, 4, args.core_num)


            """ LayerNorm execution """
            layernorm_cal(layernorm)


            """ Linear Projection """
            if (lp_stage == 0): 
                """ Read data from core SRAM """
                lp_stage = read_from_core_sram(cores, lp_stage, 5)
            elif (lp_stage == 1):
                """ Dot production """
                lp_stage = dot_production(cores, lp_stage, count, 5, args.core_num, a_row_idx, a_col_idx, 1)


            """ FC1 calculation """
            if (fc1_stage == 0):
                """ Read data from core SRAM """
                fc1_stage = read_from_core_sram(cores, fc1_stage, 6)
            elif (fc1_stage == 1):
                """ Dot production """
                fc1_stage = dot_production(cores, fc1_stage, count, 6, args.core_num)


            """ FC2 calculation """
            if (fc2_stage == 0):
                """ Read data from core SRAM """
                fc2_stage = read_from_core_sram(cores, fc2_stage, 7)
            elif (fc2_stage == 1):
                """ Dot production """
                fc2_stage = dot_production(cores, fc2_stage, count, 7, args.core_num)


            """ For debug """
            if args.seq_length < 100:
                if counter == 100:
                    dump_all(cores, global_buffers, softmax, layernorm, stage, latency, args.core_num)
                    counter = 0
            elif args.seq_length == 192:
                if counter == 500:
                    dump_all(cores, global_buffers, softmax, layernorm, stage, latency, args.core_num)
                    counter = 0
            elif args.seq_length == 384:
                if counter == 1000:
                    dump_all(cores, global_buffers, softmax, layernorm, stage, latency, args.core_num)
                    counter = 0
            else:
                if counter == 3000:
                    dump_all(cores, global_buffers, softmax, layernorm, stage, latency, args.core_num)
                    counter = 0
            # print("in end of while: a_row_idx[a_idx_idx], a_col_idx[a_idx_idx]: [" + str(a_row_idx[1]) + ", " + str(a_col_idx[1]) + "]")
            # if latency > 28149:
            # if counter == 500:
            # if (cores[5].blocknum_cal[0] >= 2) and (cores[5].blocknum_cal[0] <= 4): 
                # dump_all(cores, global_buffers, softmax, layernorm, stage, latency, args.core_num)
                # counter = 0

            if global_buffers[8].array_complete2:
            # if cores[5].blocknum_cal[1] == 4:
            # if latency > 5255:
                stop = True
                dump_all(cores, global_buffers, softmax, layernorm, stage, latency, args.core_num)
 
            ii = 0
            for core in cores:
                if end_counter[ii] == 0:
                    if core.calculator_and_array.complete:
                        print()
                        print("###################### core" + str(ii) + " computation completes! #################")
                        print("latency: " + str(latency))
                        print()
                        end_counter[ii] += 1
                ii += 1

        else:
            raise NotImplementedError("Core number of " + str(args.core_num) + " is not supported yet!")
        
        latency += utils.METATIME
        util_counter += 1
        counter += 1

    """ Calculate utilization """
    ii = 0
    print("Utilization of each core: ")
    for core in cores:
        print("core" + str(ii) + ": " + str(round(100 * (core.statistics.util_counter / util_counter), 2)) + " %")
        ii += 1

    return latency

def main():
    """ Main function """

    args = argparser().parse_args()
    dump_configs(args)
    latency = simulating(args)
    dump_latency(latency)

    return 0

if __name__ == '__main__':
    main()
    # sys.exit(main())