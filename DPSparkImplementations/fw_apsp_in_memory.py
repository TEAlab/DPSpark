__author__ = "Zafar Ahmad, Mohammad Mahdi Javanmard"
__copyright__ = "Copyright (c) 2019 Tealab@SBU"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zafar Ahmad"
__email__ = "zafahmad@cs.stonybrook.edu"
__status__ = "Development"

import numpy as np
import fw_apsp_kernels
import floyd as fw

'''
    Copyright Note: (for comparison purposes) Iterative implementation of fw_apsp algorithm is taken 
                    from (Schoeneman and Zola)'s ICPP 2019 paper.
'''

def solve(k, dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, sc):
    def funcA(block_info, k):
        ((I_, J_), x_block) = block_info
        if is_rec:
            x_block = fw.fwA(x_block, x_block.shape[0], r_shared)
        else:
            x_block = fw_apsp_kernels.funA_iter(x_block)
        yield((I_, J_), (-2, x_block, 'N'))
        # making required copies for functions B
        for j in range(0, r, 1):
            if j != J_:
                yield((I_, j), (-1, x_block, 'B'))
        #making required copies for functions C
        for i in range(0, r, 1):
            if i != I_:
                yield((i, J_), (-1, x_block, 'C'))


    def funcBC(blocks_info, k):
        ((I_, J_), blocks) = blocks_info
        if isinstance(blocks[0], np.ndarray):
            (x_block, (typeFlag, uv_block, whichFunc)) = blocks
        else:
            ((typeFlag, uv_block, whichFunc), x_block) = blocks
        if typeFlag != -2:
            if whichFunc == 'B':
                if is_rec:
                    x_block = fw.fwB(x_block, uv_block, x_block.shape[0], r_shared)
                else:
                    x_block = fw_apsp_kernels.funcB_iter(x_block, uv_block)
            else:
                if is_rec:
                    x_block = fw.fwC(x_block, uv_block, x_block.shape[0], r_shared)
                else:
                    x_block = fw_apsp_kernels.funcC_iter(x_block, uv_block)
            yield((I_, J_), x_block) # returning the result of function call B/C
        else:
            yield((I_, J_), uv_block) # to keep diagonal block [k,k]
        # making required copies for functions D
        if (I_ == k and J_ != k): # it was function call B
            for i in range(0, r, 1):
                if i != I_:
                    yield((i, J_), (x_block, 'V'))
        elif (I_ != k and J_ == k): # it was function call C
            for j in range(0, r, 1):
                if j != J_:
                    yield((I_, j), (x_block, 'U'))

    def funcD(blocks_info, k):
        for((I_, J_), blocks) in blocks_info:
            if (I_ == k or J_ == k):
                yield((I_,J_), blocks[0]) # to keep blocks [i,k], [k,k], [k,j] (results of previous stages)
            else: # function D computation needed
                if isinstance(blocks[0], np.ndarray):
                    x_block, (uv_block1, whichBlock1), (uv_block2, whichBlock2) = blocks
                elif isinstance(blocks[1], np.ndarray):
                    (uv_block1, whichBlock1), x_block, (uv_block2, whichBlock2) = blocks
                else:
                    (uv_block1, whichBlock1), (uv_block2, whichBlock2), x_block = blocks
                if whichBlock1 == 'U': # uv_block1 is u_block and uv_block2 is v_block
                    if is_rec:
                        x_block = fw.fwD(x_block, uv_block1, uv_block2, x_block.shape[0], r_shared)
                    else:
                        x_block = fw_apsp_kernels.funcD_iter(x_block, uv_block1, uv_block2)
                else: # uv_block1 is v_block and uv_block2 is u_block
                    if is_rec:
                        x_block = fw.fwD(x_block, uv_block2, uv_block1, x_block.shape[0], r_shared)
                    else:
                        x_block = fw_apsp_kernels.funcD_iter(x_block, uv_block2, uv_block1)
                yield((I_, J_), x_block)
                    

    # Calling function A on diagonal block [k,k] and make (2r-1) copies of it for the next stage (func calls B/C)
    if is_cached:
        a_blocks = dp_table.filter(lambda x : x[0][0] == k and x[0][1] == k)\
                           .flatMap(lambda x : funcA(x, k), preservesPartitioning=False)\
                           .partitionBy(partitions, rdd_partitioner).cache()
    else:
        a_blocks = dp_table.filter(lambda x : x[0][0] == k and x[0][1] == k)\
                           .flatMap(lambda x : funcA(x, k), preservesPartitioning=False)\
                           .partitionBy(partitions, rdd_partitioner)
    # Calling functions B/C on blocks [i,k] and [k,j] for all i,j in {0, ..., r-1} - {k}
    # and making copies of them for the next stage (func calls D)
    # Note: it contains a_block (block [k,k]) as well
    if is_cached:
        a_bc_blocks = dp_table.filter(lambda x : (x[0][0] == k) or (x[0][1] == k))\
                            .union(a_blocks)\
                            .combineByKey((lambda x : [x]), (lambda x, y : x + [y]),\
                                          (lambda x, y : x + y), numPartitions=partitions,\
                                          partitionFunc=rdd_partitioner)\
                            .flatMap(lambda x : funcBC(x, k), preservesPartitioning=False)\
                            .partitionBy(partitions, rdd_partitioner).cache()
    else:
        a_bc_blocks = dp_table.filter(lambda x : (x[0][0] == k) or (x[0][1] == k))\
                            .union(a_blocks)\
                            .combineByKey((lambda x : [x]), (lambda x, y : x + [y]),\
                                          (lambda x, y : x + y), numPartitions=partitions,\
                                          partitionFunc=rdd_partitioner)\
                            .flatMap(lambda x : funcBC(x, k), preservesPartitioning=False)\
                            .partitionBy(partitions, rdd_partitioner)
    # Calling functions D on blocks [i,j] for all i, j in {0, 1, ..., r-1} - {k}
    if is_cached:
        a_bc_d_blocks = dp_table.filter(lambda x : x[0][0] != k and x[0][1] != k)\
                                .union(a_bc_blocks)\
                                .combineByKey((lambda x : [x]), (lambda x, y : x + [y]),\
                                              (lambda x, y : x + y), numPartitions=partitions,\
                                              partitionFunc=rdd_partitioner)\
                                .mapPartitions(lambda x : funcD(x, k),preservesPartitioning=False)\
                                .partitionBy(partitions, rdd_partitioner).cache()
    else:
        a_bc_d_blocks = dp_table.filter(lambda x : x[0][0] != k and x[0][1] != k)\
                                .union(a_bc_blocks)\
                                .combineByKey((lambda x : [x]), (lambda x, y : x + [y]),\
                                              (lambda x, y : x + y), numPartitions=partitions,\
                                              partitionFunc=rdd_partitioner)\
                                .mapPartitions(lambda x : funcD(x, k),preservesPartitioning=False)\
                                .partitionBy(partitions, rdd_partitioner)
    dp_table = a_bc_d_blocks
    return dp_table


def solve_dp(n, dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, sc):
    for k in range(0, r, 1):
        dp_table = solve(k, dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, sc)
    return dp_table
