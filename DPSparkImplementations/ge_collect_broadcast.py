__author__ = "Zafar Ahmad, Mohammad Mahdi Javanmard"
__copyright__ = "Copyright (c) 2019 Tealab@SBU"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zafar Ahmad"
__email__ = "zafahmad@cs.stonybrook.edu"
__status__ = "Development"

import numpy as np
import glob
import ge_kernels
import gaussian as ge

def solve(k, n, dp_table, r, is_cached, is_rec, r_shared, partitions, base_size, rdd_partitioner, blocks_dir, sc):
    def funcA(block_info, k):
        ((I_,J_), x_block) = block_info
        if is_rec:
            x_block = ge.geA(x_block, x_block.shape[0], n, I_, J_, k, r_shared)
        else:
            x_block = ge_kernels.funcA_iter(x_block, n, I_, J_, k)
        return ((I_, J_), x_block)

    def funcBC(block_info, k):
        ((I_, J_), x_block) = block_info
        blkfname = blocks_dir+'block_q'+str(k)+'_'+str(k)+'_'+str(k)+'_*.csv'
        blkfname = glob.glob(blkfname)
        blkfname = blkfname[0]
        bsize = tuple(map(int, blkfname.split('/')[-1].split('.')[0].split('_')[-2:]))
        uv_block = np.memmap(blkfname, shape=bsize, dtype='float', mode='r')
        if I_ == k: # doing function B
            if is_rec:
                x_block = ge.geB(x_block, uv_block, x_block.shape[0], n, I_, J_, k, r_shared)
            else:
                x_block = ge_kernels.funcB_iter(x_block, uv_block, n, I_, J_, k)
        else: # doing function C
            if is_rec:
                x_block = ge.geC(x_block, uv_block, x_block.shape[0], n, I_, J_, k, r_shared)
            else:
                x_block = ge_kernels.funcC_iter(x_block, uv_block, n, I_, J_, k)
        return ((I_, J_), x_block)

    def funcD(block_info, k):
        ((I_, J_), x_block) = block_info
        u_blkfname = blocks_dir+'block_q'+str(k)+'_'+str(I_)+'_'+str(k)+'_*.csv'
        u_blkfname = glob.glob(u_blkfname)
        u_blkfname = u_blkfname[0]
        u_bsize = tuple(map(int, u_blkfname.split('/')[-1].split('.')[0].split('_')[-2:]))
        u_block = np.memmap(u_blkfname, shape=u_bsize, dtype='float', mode='r')

        v_blkfname = blocks_dir+'block_q'+str(k)+'_'+str(k)+'_'+str(J_)+'_*.csv'
        v_blkfname = glob.glob(v_blkfname)
        v_blkfname = v_blkfname[0]
        v_bsize = tuple(map(int, v_blkfname.split('/')[-1].split('.')[0].split('_')[-2:]))
        v_block = np.memmap(v_blkfname, shape=v_bsize, dtype='float', mode='r')

        w_blkfname = blocks_dir+'block_q'+str(k)+'_'+str(k)+'_'+str(k)+'_*.csv'
        w_blkfname = glob.glob(w_blkfname)
        w_blkfname = w_blkfname[0]
        w_bsize = tuple(map(int, v_blkfname.split('/')[-1].split('.')[0].split('_')[-2:]))
        w_block = np.memmap(w_blkfname, shape=w_bsize, dtype='float', mode='r')

        if is_rec:
            x_block = ge.geD(x_block, u_block, v_block, w_block, x_block.shape[0], n, I_, J_, k, r_shared)
        else:
            x_block = ge_kernels.funcD_iter(x_block, u_block, v_block, w_block, n, I_, J_, k)
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
        bc_blocksRDD = dp_table.filter(lambda x : (x[0][0] == k and x[0][1] > k) or (x[0][0] > k and x[0][1] == k))\
                               .map(lambda x : funcBC(x, k), preservesPartitioning=False).cache()
    else:
        bc_blocksRDD = dp_table.filter(lambda x : (x[0][0] == k and x[0][1] > k) or (x[0][0] > k and x[0][1] == k))\
                               .map(lambda x : funcBC(x, k), preservesPartitioning=False)

    bc_blocks = bc_blocksRDD.collectAsMap()
    for i, j in bc_blocks:
        (shape1, shape2) = map(str, bc_blocks[(i,j)].shape)
        blkfname = blocks_dir+'block_q'+str(k)+'_'+str(i)+'_'+str(j)+'_'+shape1+'_'+shape2+'.csv'
        bc_blocks[(i,j)].tofile(blkfname)

    d_blocks = dp_table.filter(lambda x : x[0][0] > k and x[0][1] > k)\
                       .map(lambda x : funcD(x, k), preservesPartitioning=False)

    previous_blocks = dp_table.filter(lambda x : x[0][0] < k or x[0][1] < k)
    dp_table = sc.union([previous_blocks, a_blockRDD, bc_blocksRDD, d_blocks]).partitionBy(partitions, rdd_partitioner)
    return dp_table

def solve_dp(n, dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, blocks_dir, sc):
    for k in range(0, r, 1):
        dp_table = solve(k, n, dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, blocks_dir, sc)
    return dp_table
