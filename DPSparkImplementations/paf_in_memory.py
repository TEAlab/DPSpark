__author__ = "Zafar Ahmad, Mohammad Mahdi Javanmard"
__copyright__ = "Copyright (c) 2019 Tealab@SBU"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zafar Ahmad"
__email__ = "zafahmad@cs.stonybrook.edu"
__status__ = "Development"

import numpy as np
import paf_kernels
import protien as pf #TODO: Fix the name, it should be protein

def solve(k, dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, n, b, sc):
    def funcA(block_info, k):
        ((I_, J_), x_block) = block_info
        if is_rec:
            x_block = pf.pafA(x_block, x_block.shape[0], n, I_, J_, I_, r_shared)
        else:
            x_block = paf_kernels.funcA_iter(block_info, n)
        yield((I_,J_), (-2, x_block))
        # making required copies for functions B
        for i in range(0, I_, 1):
            yield((i, J_), (-1, x_block))

    def funcB(blocks_info, k):
        ((I_, J_), blocks) = blocks_info
        if isinstance(blocks[0], np.ndarray):
            (x_block, (typeFlag, u_block)) = blocks
        else:
            ((typeFlag, u_block), x_block) = blocks
        
        if typeFlag == -2:
            yield((I_, J_), u_block) # to keep the digonal block [k,k]
        else: # function B computation needed
            if is_rec:
                x_block = pf.pafB(x_block, u_block, x_block.shape[0], n, I_, J_, k, r_shared)
            else:
                block_info = ((I_, J_), x_block)
                x_block = paf_kernels.funcX_iter(block_info, ((k, k), u_block), n)
            yield((I_, J_), x_block) # returning the result of function call B
            # Making required copies for function D
            for i in range(0, I_+1, 1):
                yield((i, I_), (-2, x_block))

    def funcCD(blocks_info, k):
        for ((I_, J_), blocks) in blocks_info:
            if (J_ == k):
                yield((I_, J_), blocks[0]) # to keep the results of previous stages
            else: # function C/D computation needed
                if isinstance(blocks[0], np.ndarray):
                    x_block, (typeFlag, u_block) = blocks
                else:
                    (typeFlag, u_block), x_block = blocks
                if is_rec:
                    x_block = pf.pafB(x_block, u_block, x_block.shape[0], n, I_, J_, k, r_shared)
                else:
                    block_info = ((I_, J_), x_block)
                    x_block = paf_kernels.funcX_iter(block_info, ((J_, k), u_block), n)
                yield((I_, J_), x_block)

    if is_cached:
        a_blocks = dp_table.filter(lambda x : x[0][0] == k and x[0][1] == k)\
                          .flatMap(lambda x : funcA(x, k), preservesPartitioning=False)\
                          .partitionBy(partitions, rdd_partitioner).cache()
    else:
        a_blocks = dp_table.filter(lambda x : x[0][0] == k and x[0][1] == k)\
                           .flatMap(lambda x : funcA(x, k), preservesPartitioning=False)\
                           .partitionBy(partitions, rdd_partitioner)

    if is_cached:
        a_b_blocks = dp_table.filter(lambda x : x[0][1] == k and x[0][0] <= k)\
                             .union(a_blocks)\
                             .combineByKey((lambda x : [x]), (lambda x, y : x + [y]),\
                                           (lambda x, y : x + y), numPartitions=partitions,\
                                           partitionFunc=rdd_partitioner)\
                             .flatMap(lambda x : funcB(x, k), preservesPartitioning=False)\
                             .partitionBy(partitions, rdd_partitioner).cache()
    else:
        a_b_blocks = dp_table.filter(lambda x : x[0][1] == k and x[0][0] <= k)\
                             .union(a_blocks)\
                             .combineByKey((lambda x : [x]), (lambda x, y : x + [y]),\
                                           (lambda x, y : x + y), numPartitions=partitions,\
                                           partitionFunc=rdd_partitioner)\
                             .flatMap(lambda x : funcB(x, k), preservesPartitioning=False)\
                             .partitionBy(partitions, rdd_partitioner)


    if is_cached:
        a_b_cd_blocks = dp_table.filter(lambda x : x[0][1] < k)\
                                .union(a_b_blocks)\
                                .combineByKey((lambda x : [x]), (lambda x, y : x + [y]),\
                                              (lambda x, y : x + y), numPartitions=partitions,\
                                              partitionFunc=rdd_partitioner)\
                                .mapPartitions(lambda x : funcCD(x, k),preservesPartitioning=False)\
                                .partitionBy(partitions, rdd_partitioner).cache()
    else:
        a_b_cd_blocks = dp_table.filter(lambda x : x[0][1] < k)\
                                .union(a_b_blocks)\
                                .combineByKey((lambda x : [x]), (lambda x, y : x + [y]),\
                                              (lambda x, y : x + y), numPartitions=partitions,\
                                              partitionFunc=rdd_partitioner)\
                                .mapPartitions(lambda x : funcCD(x, k),preservesPartitioning=False)\
                                .partitionBy(partitions, rdd_partitioner)

    previous_blocks = dp_table.filter(lambda x : x[0][1] > k)
    dp_table = sc.union([previous_blocks, a_b_cd_blocks]).partitionBy(partitions, rdd_partitioner)

    return dp_table

def solve_dp(n, dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, sc):
    b = int(n//r)
    for k in range(r-1, -1, -1):
        dp_table = solve(k, dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, n, b, sc)
    return dp_table 
