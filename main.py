from core import Core
from global_buffer import GlobalBuffer

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
            stage = 1 
    return stage

def dot_production(cores, stage):
    stage = stage
    if cores[0].calculator_and_array.complete == False:
        cores[0].calculator_and_array.latency_counter += 1
        if cores[0].calculator_and_array.latency_counter == cores[0].calculator_and_array.latency_count:
            cores[0].calculator_and_array.latency_counter = 0
            cores[0].calculator_and_array.update_array()
            if cores[0].sram2.complete == False:
                stage = 0
            if cores[0].calculator_and_array.complete:
            # if the calculation of all data in Q completes, switch to K calculation
                stage = 2
    return stage


def simulating(args):
    """ 
    Remaining problems: 
    1. How to mark the ending of every stage
    2. It is not ready simply for sram1 write data into gb, but gb needs to provides new data for sram1, which is not implemented
    """
    
    latency = 0.0
    stop = False

    """ HW initialization """
    head_embedding_dim = int(args.embedding_dim // args.head_num) 
    sram1_height = int(args.SRAM_capacity // args.MAC_lane // args.MAC_num)
    blocknum_row = int(args.seq_length // args.MAC_lane)
    blocknum_col_qkv = int(head_embedding_dim // args.MAC_lane)
    subsum_cnt_qkv = int(args.embedding_dim // args.MAC_num)
    blocknum_row_sram1 = int(sram1_height // subsum_cnt_qkv)

    cores = []
    for i in range(args.core_num):
        cores.append(Core(sram1_num=args.MAC_lane, sram1_height=int(args.SRAM_capacity // args.MAC_lane // args.MAC_num),
                            sram1_width=args.MAC_num, sram2_height=int(args.SRAM_capacity // args.MAC_num),
                            sram2_width=args.MAC_num, mac_lane=args.MAC_lane, mac_num=args.MAC_num, block_cnt=blocknum_row * blocknum_col_qkv,
                            sram_latency_count=args.SRAM_access_latency, array_and_calculator_latency_count=args.array_access_and_calculation_latency))

    # dump core configs, suppose all cores' configuration are the same
    cores[0].dump_configs()

    global_buffers = []
    # for i in range(args.core_num):
    #     global_buffers.append(GlobalBuffer(args.GB_access_latency))

    # global_buffers[0].dump_configs()

    """ Preprocessing """
    
    cores[0].sram1.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_qkv, 
                                subsum_cnt=subsum_cnt_qkv, blocknum_row_sram=blocknum_row_sram1)
    cores[0].sram2.add_mapping(blocknum_row=blocknum_row, blocknum_col=blocknum_col_qkv,
                                block_col=args.MAC_lane, subsum_cnt=subsum_cnt_qkv)
    cores[0].calculator_and_array.add_mapping(subsum_cnt=subsum_cnt_qkv)                                    

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


    """ Simulating """
    counter = 0
    stage = 0
    # row1 = []
    # col1 = []
    # row2 = []
    # col2 = []
    while stop == False:

        if args.core_num == 1:
            for i in range(3):
                global_buffers.append(GlobalBuffer(args.GB_access_latency))
                global_buffers[i].add_mapping(blocknum_row_cnt=blocknum_row, array_data_cnt=blocknum_row * blocknum_col_qkv)
            ## Data transfer between GB and core SRAM1
            # if global buffer can update SRAM data now
            if global_buffers[0].sram1_busy == False:
                if args.debug_flag:
                    print("gb for sram1 not busy, try to find whether data in sram1 needs to be transfered.")
                (row1, col1) = global_buffers[0].find_sram_target(cores[0].sram1.sram_state_matrix, cores[0].calculator_and_array.mac_lane, 1)
                # if global buffer actually removes a data
                if global_buffers[0].sram1_busy:
                    if args.debug_flag:
                        print("find data in sram1 needs to be transferred")
                    cores[0].sram1.update_to_removing(row1, col1)
            # if global buffer is transferring data
            else: 
                if args.debug_flag:
                    print("gb for sram1 is busy, data is transferring")
                global_buffers[0].latency_counter += 1
                # if global buffer finishes 
                if global_buffers[0].latency_counter == global_buffers[0].latency_count:
                    if args.debug_flag:
                        print("data transfer from sram1 to gb is done")
                    global_buffers[0].latency_counter = 0
                    global_buffers[0].sram1_busy = False
                    cores[0].sram1.update_to_ready(row1, col1)

            ## Data transfer between GB and core SRAM2
            if global_buffers[0].sram2_busy == False:
                (row2, col2) = global_buffers[0].find_sram_target(cores[0].sram2.sram_state_matrix, cores[0].calculator_and_array.mac_lane, 2)
                if global_buffers[0].sram2_busy:
                    cores[0].sram2.update_to_removing(row2, col2)
            else:
                global_buffers[0].sram2_latency_counter += 1
                if global_buffers[0].sram2_latency_counter == global_buffers[0].latency_count:
                    global_buffers[0].sram2_latency_counter = 0
                    global_buffers[0].sram2_busy = False
                    cores[0].sram2.update_to_ready(row2, col2)
            # global_buffers[0].is_sram2_update_done(cores[0].sram2.sram_state_matrix)

            ## Data transfer from core array to GB
            if global_buffers[0].array_busy == False:
                # choose the  data in array that is ready to remove
                if args.debug_flag:
                    print("gb for array not busy, try to find whether data in array needs to be transfered.")
                array_idx_gb = global_buffers[0].find_array_target(cores[0].calculator_and_array.array_state_matrix)
                # if there is data in array satisfies the condition to remove
                if global_buffers[0].array_busy:
                    if args.debug_flag:
                        print("find data in array needs to be transferred")
                    cores[0].calculator_and_array.update_to_removing(array_idx_gb)
            else: 
                if args.debug_flag:
                    print("gb for array is busy, data is transferring")
                global_buffers[0].array_latency_counter += 1
                if global_buffers[0].array_latency_counter == global_buffers[0].latency_count:
                    if args.debug_flag:
                        print("data transfer from array to gb is done")
                    global_buffers[0].array_latency_counter = 0
                    global_buffers[0].array_busy = False
                    # all data finish transferring
                    # global_buffers[0].array_complete2 = global_buffers[0].array_complete1
                    cores[0].calculator_and_array.update_to_null(array_idx_gb)
                    

            """ Q calculation """
            ## Reading data from core SRAM
            if stage == 0:
                stage = read_from_core_sram(cores, stage)
            ## Dot production
            elif stage == 1:
                stage = dot_production(cores, stage)
            """ K calculation """
            elif stage == 2:
                pass
            """ V calculation """
         
        latency += utils.METATIME
        counter += 1
        # if counter == 10:
        # if latency > 81600:
        # if args.debug_flag:
        cores[0].dump_state_matrix()
        cores[0].dump_cal_status()
        global_buffers[0].dump_rm_status()
        counter = 0
        print("array complete: " + str(global_buffers[0].array_complete))
        print("sram1 complete: " + str(global_buffers[0].sram1_complete))
        print("sram2 complete: " + str(global_buffers[0].sram2_complete))
        print("core sram1 complete: " + str(cores[0].sram1.complete))
        print("core sram2 complete: " + str(cores[0].sram2.complete))
        print("calculator complete: " + str(cores[0].calculator_and_array.complete))
        print("calculator process: " + str(cores[0].calculator_and_array.block_counter))
        print(latency)

        if (global_buffers[0].array_complete == True) and (global_buffers[0].sram2_complete == True):
            stop = True
            # if args.debug_flag:
            cores[0].dump_state_matrix()
            cores[0].dump_cal_status()
            global_buffers[0].dump_rm_status()
            print("stage: " + str(stage))
            print("array complete: " + str(global_buffers[0].array_complete))
            print("sram1 complete: " + str(global_buffers[0].sram1_complete))
            print("sram2 complete: " + str(global_buffers[0].sram2_complete))
            print("core sram1 complete: " + str(cores[0].sram1.complete))
            print("core sram2 complete: " + str(cores[0].sram2.complete))
            print("calculator complete: " + str(cores[0].calculator_and_array.complete))
            print("calculator process: " + str(cores[0].calculator_and_array.block_counter))
            print("row[0,1]: " + str(global_buffers[0].row))
            print("col[0,1]: " + str(global_buffers[0].col))

    return latency

def main():
    """ Main function. """

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