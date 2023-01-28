from core import Core
from global_buffer import GlobalBuffer
from softmax import Softmax

import argparse
import sys
import utils
import numpy as np

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
    ap.add_argument('--SRAM-access-latency', type = int, default = 100, \
                    help = 'how many times the time of SRAM access is metatime')
    ap.add_argument('--GB-access-latency', type = int, default = 800, \
                    help = 'how many times the time of global buffer access is metatime') 
    ap.add_argument('--array-access-and-calculation-latency', type = int, default = 2, \
                    help = 'how many times the time of array access and calculation is metatime') 
    ap.add_argument('--softmax-cal-latency', type = int, default = 10, \
                    help = 'how many times the time of softmax calculation is metatime')
    ap.add_argument('--softmax-throughput', type = int, default = 6, \
                    help= 'number of mac_lane*mac_lane blocks can be transferred from GB to Softmax Unit at a time')      

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

def read_from_core_sram(cores, stage):
    stage = stage
    if cores[0].sram2.complete == False:
        # if we can read SRAM and accumulating buffer is ready for result data
        if cores[0].sram_ready() & cores[0].calculator_and_array.ready():
            cores[0].sram2.latency_counter += 1
        # if data is ready for calculation
        if cores[0].sram2.latency_counter == cores[0].sram2.latency_count:
            cores[0].sram2.latency_counter = 0
            cores[0].sram_cal_advance()
            #FIXME use a directory instead
            if stage == 0:
                stage = 1 
            elif stage == 2:
                stage = 3
            elif stage == 4:
                stage = 5
            elif stage == 7:
                stage = 8
            else:
                pass
                #TODO
    return stage

def dot_production(cores, stage):
    stage = stage
    if cores[0].calculator_and_array.complete == False:
        cores[0].calculator_and_array.latency_counter += 1
        if cores[0].calculator_and_array.latency_counter == cores[0].calculator_and_array.latency_count:
            cores[0].calculator_and_array.latency_counter = 0
            cores[0].calculator_and_array.update_array()
            if cores[0].sram2.complete == False:
                #FIXME use a directory instead
                if stage == 1:
                    stage = 0   
                elif stage == 3:
                    stage = 2
                elif stage == 5:
                    stage = 4
                elif stage == 8:
                    stage = 7
                else:
                    pass
                    #TODO   
            if cores[0].calculator_and_array.complete:
            # if the calculation of all data in Q completes, switch to K calculation
                print("################### calculation stage switch #################")
                cores[0].reset()
                #FIXME use a directory instead
                if stage == 1:
                    stage = 2
                elif stage == 3:
                    stage = 4
                elif stage == 5:
                    stage = 6
                elif stage == 8:
                    stage = 9
                else:
                    assert(0)
                    #TODO
    return stage

def coresram1_gb_data_transfer(cores, global_buffers, idx, sram1_idx_gb):
    # if global buffer can update SRAM data now
    if global_buffers[idx].sram1_busy == False:
        sram1_idx_gb[idx] = global_buffers[idx].find_sram_target(cores[0].sram1.sram_state_matrix, cores[0].calculator_and_array.mac_lane, 1)
        # if global buffer actually removes a data
        if global_buffers[idx].sram1_busy:
            cores[0].sram1.update_to_removing(sram1_idx_gb[idx])
    # if global buffer is transferring data
    else: 
        global_buffers[idx].latency_counter += 1
        # if global buffer finishes 
        if global_buffers[idx].latency_counter == global_buffers[idx].latency_count:
            global_buffers[idx].latency_counter = 0
            global_buffers[idx].sram1_busy = False
            global_buffers[idx].sram1_complete2 = global_buffers[idx].sram1_complete1
            cores[0].sram1.update_to_ready(sram1_idx_gb[idx])

def coresram2_gb_data_transfer(cores, global_buffers, idx, sram2_idx_gb):
    if global_buffers[idx].sram2_busy == False:
        sram2_idx_gb[idx] = global_buffers[idx].find_sram_target(cores[0].sram2.sram_state_matrix, cores[0].calculator_and_array.mac_lane, 2)
        if global_buffers[idx].sram2_busy:
            cores[0].sram2.update_to_removing(sram2_idx_gb[idx])
    else:
        global_buffers[idx].sram2_latency_counter += 1
        if global_buffers[idx].sram2_latency_counter == global_buffers[idx].latency_count:
            global_buffers[idx].sram2_latency_counter = 0
            global_buffers[idx].sram2_busy = False
            global_buffers[idx].sram2_complete2 = global_buffers[idx].sram2_complete1
            cores[0].sram2.update_to_ready(sram2_idx_gb[idx])

def corearray_gb_data_transfer(cores, global_buffers, idx, array_idx_gb, stage, mac_lane):
    if global_buffers[idx].array_busy == False:
        # choose the  data in array that is ready to remove
        array_idx_gb[idx] = global_buffers[idx].find_array_target(cores[0].calculator_and_array.array_state_matrix)
        # if there is data in array satisfies the condition to remove
        if global_buffers[idx].array_busy:
            cores[0].calculator_and_array.update_to_removing(array_idx_gb[idx])
    else: 
        global_buffers[idx].array_latency_counter += 1
        if global_buffers[idx].array_latency_counter == global_buffers[idx].latency_count:
            global_buffers[idx].array_latency_counter = 0
            global_buffers[idx].array_busy = False
            global_buffers[idx].array_complete2 = global_buffers[idx].array_complete1
            # all data finish transferring
            cores[0].calculator_and_array.update_to_null(array_idx_gb[idx])
            if (stage > 6) and (array_idx_gb[idx] == mac_lane - 1):
                # part of a block completes, we may need to update state matrix of A
                # written when the second last data of array(out of mac_lane data) finishes transferring into GB
                global_buffers[idx].update_to_a(cores[0].calculator_and_array.block_counter)

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
            global_buffers[idx].update_to_cal(gb_idx_softmax_start[0], gb_idx_softmax_end[0])

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

def softmax_cal(softmax):
    """ Execution of Softmax """

    if softmax[0].calculation():
        softmax[0].latency_counter += 1
        if softmax[0].latency_counter == softmax[0].latency_count:
            softmax[0].latency_counter = 0
            softmax[0].update_to_asoftmax()


def simulating(args):
    """ 
    Remaining problems: 
    2. It is not ready simply for sram1 write data into gb, but gb needs to provides new data for sram1, which is not implemented

    global_buffer.sram1_complete -> global_buffer.sram2_complete -> core.sram.complete -> core.calculator_and_array.complete -> global_buffer.array_complete 
    """
    
    latency = 0.0
    stop = False

    """ HW initialization """
    head_embedding_dim = int(args.embedding_dim // args.head_num) 
    sram1_height = int(args.SRAM_capacity // args.MAC_lane // args.MAC_num)
    blocknum_row = int(args.seq_length // args.MAC_lane)
    blocknum_col_qkv = int(head_embedding_dim // args.MAC_lane)
    blocknum_col_a = blocknum_row
    subsum_cnt_qkv = int(args.embedding_dim // args.MAC_num)
    subsum_cnt_a = int(head_embedding_dim // args.MAC_num)
    blocknum_row_sram1_qkv = int(sram1_height // subsum_cnt_qkv)
    blocknum_row_sram1_a = int(sram1_height // subsum_cnt_a)

    ## cores
    cores = []
    for i in range(args.core_num):
        cores.append(Core(sram1_num=args.MAC_lane, sram1_height=int(args.SRAM_capacity // args.MAC_lane // args.MAC_num),
                            sram1_width=args.MAC_num, sram2_height=int(args.SRAM_capacity // args.MAC_num),
                            sram2_width=args.MAC_num, mac_lane=args.MAC_lane, mac_num=args.MAC_num, block_cnt=blocknum_row * blocknum_col_qkv,
                            sram_latency_count=args.SRAM_access_latency, array_and_calculator_latency_count=args.array_access_and_calculation_latency))

    # dump core configs, suppose all cores' configuration are the same
    cores[0].dump_configs()

    ## global_buffers
    global_buffers = []
    if args.core_num == 1:
        for i in range(3):
            global_buffers.append(GlobalBuffer(args.GB_access_latency))
        global_buffers.append(GlobalBuffer(latency_count=args.GB_access_latency, softmax_bandwidth=args.softmax_throughput))
    global_buffers[3].dump_configs()        

    ## softmax
    softmax = []
    softmax.append(Softmax(latency_count=args.softmax_cal_latency, blocknum_col=blocknum_row))
    softmax[0].dump_configs()

    """ Add Mappings """
    
    cores[0].sram1.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_qkv, 
                                subsum_cnt=subsum_cnt_qkv, blocknum_row_sram=blocknum_row_sram1_qkv)
    cores[0].sram2.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_qkv,
                                block_col=args.MAC_lane, subsum_cnt=subsum_cnt_qkv)
    cores[0].calculator_and_array.add_mapping(subsum_cnt=subsum_cnt_qkv)    

    cores[0].dump_mappings()   

    if args.core_num == 1:
        for i in range(3):
            global_buffers[i].add_mapping(blocknum_row_cnt=blocknum_row, array_data_cnt=blocknum_row * blocknum_col_qkv,
                                            sram_subsum_cnt=subsum_cnt_qkv, sram1_rownum_cnt=blocknum_row_sram1_qkv, sram2_colnum_cnt=head_embedding_dim)
        global_buffers[0].dump_mappings(0)

        global_buffers[3].add_mapping(blocknum_row_cnt=blocknum_row, array_data_cnt=blocknum_row * blocknum_col_a,
                                        sram_subsum_cnt=subsum_cnt_a, sram1_rownum_cnt=blocknum_row_sram1_a, sram2_colnum_cnt=args.seq_length, flag=True)
        global_buffers[3].dump_mappings(3)


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

    # Array access and calculation latency must exceed 1, or A written into GB goes WRONG
    # if args.array_access_and_calculation_latency < 2:
    #     raise NotImplementedError("rray access and calculation latency must exceed 1!")


    """ Simulating """
    counter = 0
    stage = 0
    # stage 0/1: Q calculation
    # stage 2/3: K calculation
    # stage 4/5: V calculation
    # stage 6: core reconfiguration
    # stage 7/8: Q*K calculation

    sram1_idx_gb = [0, 0, 0, 0]
    sram2_idx_gb = [0, 0, 0, 0]
    array_idx_gb = [0, 0, 0, 0]
    # softmax is now processing which row
    gb_idx_softmax_start = [0]
    gb_idx_softmax_end = [0]
    while stop == False:

        if args.core_num == 1:

            """ Data transfer between GB and core SRAM/Array """
            if (stage == 0) or (stage == 1):
                if global_buffers[0].sram1_complete2 == False:
                    ## Data transfer between GB and core SRAM1
                    coresram1_gb_data_transfer(cores, global_buffers, 0, sram1_idx_gb)
                else:
                    coresram1_gb_data_transfer(cores, global_buffers, 1, sram1_idx_gb)

                if global_buffers[0].sram2_complete2 == False:
                    ## Data transfer between GB and core SRAM2
                    coresram2_gb_data_transfer(cores, global_buffers, 0, sram2_idx_gb)
                else:
                    coresram2_gb_data_transfer(cores, global_buffers, 1, sram2_idx_gb)

                if global_buffers[0].array_complete2 == False:
                    ## Data transfer from core array to GB
                    corearray_gb_data_transfer(cores, global_buffers, 0, array_idx_gb, stage, args.MAC_lane)
            elif (stage == 2) or (stage == 3):
                if global_buffers[1].sram1_complete2 == False:
                    coresram1_gb_data_transfer(cores, global_buffers, 1, sram1_idx_gb)
                else:
                    coresram1_gb_data_transfer(cores, global_buffers, 2, sram1_idx_gb)

                if global_buffers[1].sram2_complete2 == False:
                    coresram2_gb_data_transfer(cores, global_buffers, 1, sram2_idx_gb)
                else:
                    coresram2_gb_data_transfer(cores, global_buffers, 2, sram2_idx_gb)

                if global_buffers[0].array_complete2 == False:
                    corearray_gb_data_transfer(cores, global_buffers, 0, array_idx_gb, stage, args.MAC_lane)
                if global_buffers[1].array_complete2 == False:
                    corearray_gb_data_transfer(cores, global_buffers, 1, array_idx_gb, stage, args.MAC_lane)
            elif (stage == 4) or (stage == 5):
                if global_buffers[2].sram1_complete2 == False:
                    coresram1_gb_data_transfer(cores, global_buffers, 2, sram1_idx_gb)
                else:
                    coresram1_gb_data_transfer(cores, global_buffers, 3, sram1_idx_gb)

                if global_buffers[2].sram2_complete2 == False:
                    coresram2_gb_data_transfer(cores, global_buffers, 2, sram2_idx_gb)
                else:
                    coresram2_gb_data_transfer(cores, global_buffers, 3, sram2_idx_gb)

                if global_buffers[1].array_complete2 == False:
                    corearray_gb_data_transfer(cores, global_buffers, 1, array_idx_gb, stage, args.MAC_lane)
                if global_buffers[2].array_complete2 == False:
                    corearray_gb_data_transfer(cores, global_buffers, 2, array_idx_gb, stage, args.MAC_lane)
            elif (stage == 6) or (stage == 7) or (stage == 8):
                # Read data from GB to core for Q * K calculation
                if global_buffers[3].sram1_complete2 == False:
                    print("!!!!!!!!!!!!!!!!!!!!")
                    coresram1_gb_data_transfer(cores, global_buffers, 3, sram1_idx_gb)
                else:
                    pass
                    #TODO GB3 provides A' for sram
                if global_buffers[3].sram2_complete2 == False:
                    coresram2_gb_data_transfer(cores, global_buffers, 3, sram2_idx_gb)
                else: 
                    pass
                    #TODO GB3 provides V for sram

                # Complete transferring V data to GB
                if global_buffers[2].array_complete2 == False:
                    corearray_gb_data_transfer(cores, global_buffers, 2, array_idx_gb, stage, args.MAC_lane)    
                if global_buffers[3].array_complete2 == False:
                    corearray_gb_data_transfer(cores, global_buffers, 3, array_idx_gb, stage, args.MAC_lane)
                #TODO
                
                if global_buffers[3].softmax_complete() == False:
                    gb_softmax_data_transfer(global_buffers, softmax, 3, gb_idx_softmax_start, gb_idx_softmax_end)
                    softmax_gb_data_transfer(global_buffers, softmax, 3, gb_idx_softmax_start, gb_idx_softmax_end)
               
            # Reading data from core SRAM
            if (stage == 0) or (stage == 2) or (stage == 4) or (stage == 7):
                stage = read_from_core_sram(cores, stage)
            ## Dot production 
            elif (stage == 1) or (stage == 3) or (stage == 5) or (stage == 8):
                stage = dot_production(cores, stage)
            elif stage == 6:
                """ Q * K Reconfiguration """
                # print()
                # Here core should be reconfigured
                cores[0].reconfigure(block_cnt=blocknum_row * blocknum_col_a)
                cores[0].dump_configs()
                # Here we assume sram1/2 can hold all Q/K
                cores[0].sram1.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_a, 
                                subsum_cnt=subsum_cnt_a, blocknum_row_sram=blocknum_row_sram1_a)
                cores[0].sram2.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_a,
                                            block_col=args.MAC_lane, subsum_cnt=subsum_cnt_a)
                cores[0].calculator_and_array.add_mapping(subsum_cnt=subsum_cnt_a) 
                cores[0].dump_mappings()
                stage = 7     
            else:
                """ A' * V Reconfiguration """
                pass
                #TODO   

            if stage > 6:
                softmax_cal(softmax)
         
        latency += utils.METATIME
        if counter == 0:
            counter += 1
        if stage > 15:
            cores[0].dump_state_matrix()
            cores[0].dump_cal_status()
            for i in range(4):
                global_buffers[i].dump_rm_status(i)
            global_buffers[3].dump_a_state_matrix()
            softmax[0].dump_cal_status()
            print("stage: " + str(stage))
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
            print("core-sram1 complete: " + str(cores[0].sram1.complete))
            print("core-sram2 complete: " + str(cores[0].sram2.complete))
            print("core-calculator complete: " + str(cores[0].calculator_and_array.complete))
            print(latency)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        if stage == 9:
            stop = True
            # if args.debug_flag:
            cores[0].dump_state_matrix()
            cores[0].dump_cal_status()
            for i in range(4):
                global_buffers[i].dump_rm_status(i)
            global_buffers[3].dump_a_state_matrix()
            softmax[0].dump_cal_status()
            print("stage: " + str(stage))
            print("gb2-array complete1: " + str(global_buffers[2].array_complete1))
            print("gb2-array complete2: " + str(global_buffers[2].array_complete2))
            print("gb2-sram1 complete1: " + str(global_buffers[2].sram1_complete1))
            print("gb2-sram1 complete2: " + str(global_buffers[2].sram1_complete2))
            print("gb2-sram2 complete1: " + str(global_buffers[2].sram2_complete1))
            print("gb2-sram2 complete2: " + str(global_buffers[2].sram2_complete2))
            print("core-sram1 complete: " + str(cores[0].sram1.complete))
            print("core-sram2 complete: " + str(cores[0].sram2.complete))
            print("core-calculator complete: " + str(cores[0].calculator_and_array.complete))

    return latency

def main():
    """ Main function """

    """ Sys setup """
    # np.set_printoptions(threshold=np.inf)
    sys.stdout = open('./results.txt', mode = 'w', encoding='utf-8')

    args = argparser().parse_args()
    dump_configs(args)
    latency = simulating(args)
    dump_latency(latency)
    return 0

if __name__ == '__main__':
    main()
    # sys.exit(main())