__author__ = "Zafar Ahmad, Mohammad Mahdi Javanmard"
__copyright__ = "Copyright (c) 2019 Tealab@SBU"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zafar Ahmad"
__email__ = "zafahmad@cs.stonybrook.edu"
__status__ = "Development"

import numpy as np
import glob
import paf_kernels
import protien as pf #TODO: Fix the name, it should be protein

def solve(k, dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, blocks_dir, n, b, sc):
    def funcA(block_info, k):
        ((I_, J_), x_block) = block_info
        if is_rec:
            x_block = pf.pafA(x_block, x_block.shape[0], n, I_, J_, I_, r_shared)
        else:
            x_block = paf_kernels.funcA_iter(block_info, n)
        return ((I_, J_), x_block)

    def funcB(block_info, k):
        ((I_, J_), x_block) = block_info
        blkfname = blocks_dir+'block_q'+str(k)+'_'+str(k)+'_'+str(k)+'_*.csv'
        blkfname = glob.glob(blkfname)
        blkfname = blkfname[0]
        bsize = tuple(map(int, blkfname.split('/')[-1].split('.')[0].split('_')[-2:]))
        u_block = np.memmap(blkfname, shape=bsize, dtype='float', mode='r')
        if is_rec:
            x_block = pf.pafB(x_block, u_block, x_block.shape[0], n, I_, J_, k, r_shared)
        else:
            x_block = paf_kernels.funcX_iter(block_info, ((k, k), u_block), n)
        return ((I_, J_), x_block)

    def funcCD(block_info, k):
        ((I_, J_), x_block) = block_info
        blkfname = blocks_dir+'block_q'+str(k)+'_'+str(J_)+'_'+str(k)+'_*.csv'
        blkfname = glob.glob(blkfname)
        blkfname = blkfname[0]
        bsize = tuple(map(int, blkfname.split('/')[-1].split('.')[0].split('_')[-2:]))
        u_block = np.memmap(blkfname, shape=bsize, dtype='float', mode='r')
        if is_rec:
            x_block = pf.pafB(x_block, u_block, x_block.shape[0], n, I_, J_, k, r_shared)
        else:
            x_block = paf_kernels.funcX_iter(block_info, ((J_, k), u_block), n)
        return ((I_, J_), x_block)
            
    if is_cached:
        a_blockRDD = dp_table.filter(lambda x : x[0][0] == k and x[0][1] == k)\
                             .map(lambda x : funcA(x, k), preservesPartitioning=False).cache()
    else:
        a_blockRDD = dp_table.filter(lambda x : x[0][0] == k and x[0][1] == k)\
                             .map(lambda x : funcA(x, k), preservesPartitioning=False)
    a_block = a_blockRDD.collectAsMap()
    a_block = a_block[(k,k)]

    (shape1, shape2) = map(str, a_block.shape)
    blkfname = blocks_dir+'block_q'+str(k)+'_'+str(k)+'_'+str(k)+'_'+shape1+'_'+shape2+'.csv'
    a_block.tofile(blkfname)

    if is_cached:
        b_blocksRDD = dp_table.filter(lambda x : x[0][1] == k and x[0][0] < k)\
                              .map(lambda x : funcB(x, k), preservesPartitioning=False).cache()
    else:
        b_blocksRDD = dp_table.filter(lambda x : x[0][1] == k and x[0][0] < k)\
                              .map(lambda x : funcB(x, k), preservesPartitioning=False)

    b_blocks = b_blocksRDD.collectAsMap()
    for i, j in b_blocks:
        (shape1, shape2) = map(str, b_blocks[(i,j)].shape)
        blkfname = blocks_dir+'block_q'+str(k)+'_'+str(i)+'_'+str(j)+'_'+shape1+'_'+shape2+'.csv'
        b_blocks[(i,j)].tofile(blkfname)

    cd_blocks = dp_table.filter(lambda x : x[0][1] < k)\
                        .map(lambda x : funcCD(x, k), preservesPartitioning=False)

    previous_blocks = dp_table.filter(lambda x : x[0][1] > k)
    dp_table = sc.union([previous_blocks, a_blockRDD, b_blocksRDD, cd_blocks])\
                 .partitionBy(partitions, rdd_partitioner)

    return dp_table

def solve_dp(n, dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, blocks_dir, sc):
    b = int(n//r)
    for k in range(r-1, -1, -1):
        dp_table = solve(k, dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, blocks_dir, n, b, sc)
    return dp_table
