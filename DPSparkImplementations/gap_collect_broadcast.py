__author__ = "Zafar Ahmad, Mohammad Mahdi Javanmard"
__copyright__ = "Copyright (c) 2019 Tealab@SBU"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zafar Ahmad"
__email__ = "zafahmad@cs.stonybrook.edu"
__status__ = "Development"

import numpy as np
import glob 

'''
 The algorithm is originally based on the following r-way algorithm:
 A_Gap(X, r, r_shared):
   if input matrix 'X' is small, then: call A_loop_based_Gap(X)
   else:
     for k in range(2*r-1):
       parallel_for i from (k < r ? 0 : k-r+1) to (k < r ? k : r-1) do:
         j = (k-i)
         A_Gap(X_{ij}, r_shared)
         parallel_for l from (j+1) to (r-1): do:
           B_Gap(X_{il}, X_{ij}, r_shared)
         parallel_for l from (i+1) to (r-1) do:
           C_Gap(X_{lj}, X_{ij}, r_shared)
'''


def funcA_iter(block_info, x_seq_bcast, y_seq_bcast):
    ((I_, J_), x_block) = block_info
    for i in range(x_block.shape[0]):
        for j in range(x_block.shape[0]):
            XI = I_*x_block.shape[0]+i
            XJ = J_*x_block.shape[0]+j
            #TODO: This can't happen in distributed setting!!!
            if XI != 0 and XJ != 0 :
                x_block[i, j] = min(x_block[i, j], x_block[i-1, j-1] + (0 if x_seq_bcast[XI] == y_seq_bcast[XJ] else 1))
            for k in range(x_block.shape[0]):
                if k < j:
                    K = J_*x_block.shape[0]+k
                    x_block[i, j] = min(x_block[i, j], x_block[i, k] + K + XJ) #, which is w1(K, XJ)
                if k < i:
                    K = I_*x_block.shape[0]+k
                    x_block[i, j] = min(x_block[i, j], x_block[k, j] + K + XI) #, which is w2(K, XI)
    return x_block


def funcB_iter(block_info, u_block_info):
    ((I_, J_), x_block) = block_info
    ((UI_, UJ_), u_block) = u_block_info
    for i in range(x_block.shape[0]):
        for j in range(x_block.shape[0]):
            XI = I_*x_block.shape[0]+i
            XJ = J_*x_block.shape[0]+j
            UI = UI_*u_block.shape[0]+j
            for k in range(x_block.shape[0]):
                K = UJ_*u_block.shape[0]+k
                x_block[i, j] = min(x_block[i, j], u_block[i, k] + K + XJ) #, which is w1(K, XJ)
    return x_block

def funcC_iter(block_info, v_block_info):
    ((I_, J_), x_block) = block_info
    ((VI_, VJ_), v_block) = v_block_info
    for i in range(x_block.shape[0]):
        for j in range(x_block.shape[0]):
            XI = I_*x_block.shape[0]+i
            XJ = J_*x_block.shape[0]+j
            VJ = VJ_*x_block.shape[0]+j
            for k in range(x_block.shape[0]):
                K = VI_*x_block.shape[0]+k
                x_block[i, j] = min(x_block[i, j], v_block[k, j] + K + XI) #, which is w2(K, XI)
    return x_block


def solve(k, dp_table, r, is_rec, r_shared, partitions, base_size, rdd_partitioner, blocks_dir, x_seq_broadcast, y_seq_broadcast, sc):
    def funcA(block_info, k):
        ((I_, J_), x_block) = block_info
        if is_rec:
            x_block = funcA_iter(block_info, x_seq_broadcast, y_seq_broadcast)#funcA_rec(block_info, x_seq_broadcast, y_seq_broadcast, r_shared, base_size)
        else:
            x_block = funcA_iter(block_info, x_seq_broadcast, y_seq_broadcast)
        return ((I_, J_), x_block)

    def funcB(block_info, k):
        ((I_, J_), x_block) = block_info
        blkfname = blocks_dir+'block_q'+str(k)+'_'+str(I_)+'_'+str(k-I_)+'_*.csv'
        blkfname = glob.glob(blkfname)
        blkfname = blkfname[0]
        bsize = tuple(map(int, blkfname.split('/')[-1].split('.')[0].split('_')[-2:]))
        u_block = np.memmap(blkfname, shape=bsize, dtype='int', mode='r')
        if is_rec:
            x_block = funcB_iter(block_info, ((I_, k-I_), u_block))#funcB_rec(block_info, ((I_, k-I_), u_block), r_shared, base_size)
        else:
            x_block = funcB_iter(block_info, ((I_, k-I_), u_block))
        return ((I_, J_), x_block)

    def funcC(block_info, k):
        ((I_, J_), x_block) = block_info
        blkfname = blocks_dir+'block_q'+str(k)+'_'+str(k-J_)+'_'+str(J_)+'_*.csv'
        blkfname = glob.glob(blkfname)
        blkfname = blkfname[0]
        bsize = tuple(map(int, blkfname.split('/')[-1].split('.')[0].split('_')[-2:]))
        v_block = np.memmap(blkfname, shape=bsize, dtype='int', mode='r')
        if is_rec:
            x_block = funcC_iter(x_block, ((k-J_, J_), v_block))#funcC_rec(x_block, ((k-J_, J_), v_block), r_shared, base_size)
        else:
            x_block = funcC_iter(x_block, ((k-J_, J_), v_block))
        return ((I_, J_), x_block)

    def funcBC(block_info, k):
        ((I_, J_), x_block) = block_info
        u_blkfname = blocks_dir+'block_q'+str(k)+'_'+str(I_)+'_'+str(k-I_)+'_*.csv'
        u_blkfname = glob.glob(u_blkfname)
        u_blkfname = u_blkfname[0]
        u_bsize = tuple(map(int, u_blkfname.split('/')[-1].split('.')[0].split('_')[-2:]))
        u_block = np.memmap(u_blkfname, shape=u_bsize, dtype='int', mode='r')

        if is_rec:
            x_block = funcB_iter(block_info, ((I_, k-I_), u_block))#funcB_rec(block_info, ((I_, k-I_), u_block), r_shared, base_size)
        else:
            x_block = funcB_iter(block_info, ((I_, k-I_), u_block))

        v_blkfname = blocks_dir+'block_q'+str(k)+'_'+str(k-J_)+'_'+str(J_)+'_*.csv'
        v_blkfname = glob.glob(v_blkfname)
        v_blkfname = v_blkfname[0]
        v_bsize = tuple(map(int, v_blkfname.split('/')[-1].split('.')[0].split('_')[-2:]))
        v_block = np.memmap(blkfname, shape=v_bsize, dtype='int', mode='r')

        if is_rec:
            x_block = funcC_iter(x_block, ((k-J_, J_), v_block))#funcC_rec(x_block, ((k-J_, J_), v_block), r_shared, base_size)
        else:
            x_block = funcC_iter(x_block, ((k-J_, J_), v_block))
        return ((I_, J_), x_block)


    a_blocksRDD = dp_table.filter(lambda x : (x[0][0] + x[0][1] == k)).map(lambda x : funcA(x, k), preservesPartitioning=False)

    a_blocks = a_blocksRDD.collectAsMap()
    for i,j in a_blocks:
        (shape1, shape2) = map(str, a_blocks[(i,j)].shape)
        blkfname = blocks_dir+'block_q'+str(k)+'_'+str(i)+'_'+str(j)+'_'+shape1+'_'+shape2+'.csv'
        a_blocks[(i,j)].tofile(blkfname)
    
    if k < r:
        only_b_blocks = dp_table.filter(lambda x : (((x[0][0] >= 0) and (x[0][0] <= k) and (x[0][1] >= k-x[0][0]+1) and (x[0][1] < r)) and\
                                                    not((x[0][0] >= k - x[0][1]+1) and (x[0][0] < r) and (x[0][1] >= 0) and (x[0][1] <= k))))\
                                .map(lambda x : funcB(x, k), preservesPartitioning=False)
        only_c_blocks = dp_table.filter(lambda x : (not((x[0][0] >= 0) and (x[0][0] <= k) and (x[0][1] >= k-x[0][0]+1) and (x[0][1] < r)) and\
                                                    ((x[0][0] >= k - x[0][1]+1) and (x[0][0] < r) and (x[0][1] >= 0) and (x[0][1] <= k))))\
                                .map(lambda x : funcC(x, k), preservesPartitioning=False)
        b_and_c_blocks = dp_table.filter(lambda x : (((x[0][0] >= 0) and (x[0][0] <= k) and (x[0][1] >= k-x[0][0]+1) and (x[0][1] < r)) and\
                                                     ((x[0][0] >= k - x[0][1]+1) and (x[0][0] < r) and (x[0][1] >= 0) and (x[0][1] <= k))))\
                                 .map(lambda x : funcBC(x, k), preservesPartitioning=False)
    else:
        only_b_blocks = dp_table.filter(lambda x : (((k-r+1 <= x[0][0]) and (x[0][0] < r) and (k-x[0][0]+1 <= x[0][1]) and (x[0][1] < r)) and\
                                                    not((k-r < x[0][1]) and (x[0][1] < r) and (k-x[0][1]+1 <= x[0][0]) and (x[0][0] < r))))\
                                .map(lambda x : funcB(x, k), preservesPartitioning=False)
        only_c_blocks = dp_table.filter(lambda x : (not((k-r+1 <= x[0][0]) and (x[0][0] < r) and (k-x[0][0]+1 <= x[0][1]) and (x[0][1] < r)) and\
                                                    ((k-r < x[0][1]) and (x[0][1] < r) and (k-x[0][1]+1 <= x[0][0]) and (x[0][0] < r))))\
                                .map(lambda x : funcC(x, k), preservesPartitioning=False)
        b_and_c_blocks = dp_table.filter(lambda x : ((k-r+1 <= x[0][0]) and (x[0][0] < r) and (k-x[0][0]+1 <= x[0][1]) and (x[0][1] < r)) and\
                                                    ((k-r < x[0][1]) and (x[0][1] < r) and (k-x[0][1]+1 <= x[0][0]) and (x[0][0] < r)))\
                                 .map(lambda x : funcBC(x, k), preservesPartitioning=False)

    # No need to write back the only_b_blocks, only_c_blocks and b_and_c_blocks to the auxiliary drive as we do not need them for the current iteration 
    previous_blocks = dp_table.filter(lambda x : (x[0][0] + x[0][1] < k))
    if k < r:
        not_yet_touched_blocks = dp_table.filter(lambda x : (not((x[0][0] >= 0) and (x[0][0] <= k) and (x[0][1] >= k-x[0][0]+1) and (x[0][1] < r)) and\
                                                             not((x[0][0] >= k - x[0][1]+1) and (x[0][0] < r) and (x[0][1] >= 0) and (x[0][1] <= k)) and\
                                                             (x[0][0] + x[0][1] > k)))
    else:
        not_yet_touched_blocks = dp_table.filter(lambda x : (not((k-r+1 <= x[0][0]) and (x[0][0] < r) and (k-x[0][0]+1 <= x[0][1]) and (x[0][1] < r)) and\
                                                             not((k-r < x[0][1]) and (x[0][1] < r) and (k-x[0][1]+1 <= x[0][0]) and (x[0][0] < r)) and\
                                                             (x[0][0] + x[0][1] > k)))
    #TODO: Getting runtime error for having empty RDDs to be unioned!
    dp_table = sc.union([previous_blocks, a_blocksRDD, only_b_blocks, only_c_blocks, b_and_c_blocks, not_yet_touched_blocks])\
                 .partitionBy(partitions, rdd_partitioner)

    return dp_table
