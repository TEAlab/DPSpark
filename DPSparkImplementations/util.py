__author__ = "Zafar Ahmad, Mohammad Mahdi Javanmard"
__copyright__ = "Copyright (c) 2019 Tealab@SBU"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zafar Ahmad"
__email__ = "zafahmad@cs.stonybrook.edu"
__status__ = "Development"


from pyspark.rdd import portable_hash as pyspark_portable_hash

import numpy as np
import random
import sys
import math

'''
   The method get_partitioner (as well as sendBlkToBlk and doBlock) has been taken from
   Schoeneman's ICPP 2019 implementation (for the sake of experimental comparision):
'''

def get_partitioner(partitioner, r, compute_nodes):
    assert partitioner == 'md' or partitioner == 'ph' or partitioner == 'cus', \
        'Error: Unrecognized partitioner \'' + F + '\'. Use \'md\' or \'ph\' or \'cus\'.'
    if partitioner == 'md':
        def multi_diag(x):
            (k1, k2) = x
            if k1 <= k2:
                return int(k1 - (0.5) * (k1 - k2) * (k1 - k2 + 2 * r  + 1))
            else:
                return int(k2 - (0.5) * (k2 - k1) * (k2 - k1 + 2 * r  + 1))
        return multi_diag
    if partitioner == 'cus': # building 2D grid of executors
       def multi_diag2(x):
           (k1, k2) = x
           compute_nodes_sqrt = int(math.sqrt(compute_nodes))
           batch = int(r/compute_nodes_sqrt)
           k11 = int(k1/batch)
           k22 = int(k2/batch)
           return int(k11*compute_nodes_sqrt + k22)
       return multi_diag2
    return pyspark_portable_hash

def get_dp_table(n, r, partitions, rdd_partitioner, problem_type, sc):
    def sendBlkToBlk(x):
        (blkId, blkCnt) = x
        if problem_type == "paf": #paf works on upper_triangular DP table
            for i in range(0, r, 1):
                yield(tuple(sorted((blkId, i))), (blkCnt, blkId))
        else: # other algorithms (fw-apsp and ge) work on square DP table
            for i in range(0, r, 1):
                yield(tuple((blkId, i)), (blkCnt, blkId))

    def doBlock(iter_):
        random.seed(30)#NOTE: for test/debug purposes. To be removed in production code
        for ((I_, J_), _LIST_) in iter_:
            if problem_type == "fw-apsp":
                block = np.ones(shape = (b, b))
                for i in range(b):
                    for j in range(b):
                        block[i, j] = random.uniform(1, 10)
                if (I_ == J_):
                    np.fill_diagonal(block, 0)
            elif problem_type == "ge":
                block = np.ones(shape = (b, b)) # by default it is floating point
                for i in range(b):
                    for j in range(b):
                        block[i, j] = random.uniform(1, 100)
            elif problem_type == "paf":
                block = np.ones(shape = (b,b))
                for i in range(b):
                    for j in range(b):
                        block[i, j] = 0.
            yield((I_, J_), block)
                
    b = int(n//r)
    blocks = sc.parallelize([i for i in range(n)], numSlices=partitions)\
               .map(lambda x : (x // b, x))\
               .combineByKey((lambda x : 1),(lambda x, y : x + 1),(lambda x, y : x + y))\
               .flatMap(lambda x : sendBlkToBlk(x))\
               .combineByKey((lambda x : [x]),(lambda x, y : x + [y]),(lambda x, y : x + y),numPartitions=partitions,partitionFunc=rdd_partitioner)\
               .mapPartitions(lambda x : doBlock(x),preservesPartitioning=False)
    return blocks


#def get_dp_table_info_gap(n, r, partitions, rdd_partitioner, sc):
#    def sendBlkToBlk(x):
#        (blkId, blkCnt) = x
#        for i in range(0, r, 1):
#            yield(tuple((blkId, i)), (blkCnt, blkId))
#    def doBlock(iter_):
#        for ((I_, J_), _LIST_) in iter_:
#            block = np.ones(shape = (b,b), dtype = int)
#            for i in range(b):
#                for j in range(b):
#                    block[i, j] = sys.maxint#np.inf
#            if I_ == 0:
#                for idx in range(b):
#                    block[0][idx] = (i * b) + idx #, which is w2(0, i*b+idx)
#            elif J_ == 0:
#                for idx in range(b):
#                    block[idx][0] = (j * b) + idx #, which is w1(j*b+idx, 0)
#            yield block
#    x_seq = np.chararray(n+1)
#    for i in range(n+1):
#        x_seq[i] = random.choice('ABCD')
#    y_seq = np.chararray(n+1)
#    for i in range(n+1):
#        y_seq[i] = random.choice('ABCD')
#    x_seq_broadcast = sc.broadcast(x_seq)
#    y_seq_broadcast = sc.broadcast(y_seq)
#    b  = int(n//r)
#    blocks = sc.parallelize([i for i in range(n)], numSlices=partitions)\
#               .map(lambda x : (x // b, x))\
#               .combineByKey((lambda x : 1),(lambda x, y : x + 1),(lambda x, y : x + y))\
#               .flatMap(lambda x : sendBlkToBlk(x))\
#               .combineByKey((lambda x : [x]),(lambda x, y : x + [y]),(lambda x, y : x + y),numPartitions=partitions,partitionFunc=rdd_partitioner)\
#               .mapPartitions(lambda x : doBlock(x),preservesPartitioning=False)
#    return (x_seq_broadcast, y_seq_broadcast, blocks)

#def get_dp_table_info_gap(n, r, partitions, rdd_partitioner, sc):
#    def get_block(x, b):
#        (i,j) = x
#        block = np.ones(shape = (b,b), dtype = int)
#        #block = np.full(shape = (b, b), np.inf)
#        for cell in block.flat:
#            cell = np.inf
#        if i == 0:
#            for idx in range(b):
#                block[0][idx] = (i * b) + idx #, which is w2(0, i*b+idx)
#        elif j == 0:
#            for idx in range(b):
#                block[idx][0] = (j * b) + idx #, which is w1(j*b+idx, 0)
#        yield block
#        
#    x_seq = np.chararray(n+1)
#    for cell in x_seq.flat:
#        cell = random.choice('ABCD')
#    y_seq = np.chararray(n+1)
#    for cell in y_seq.flat:
#        cell = random.choice('ABCD')
#    x_seq_broadcast = sc.broadcast(x_seq)
#    y_seq_broadcast = sc.broadcast(y_seq)
#    b  = int(n//r)
#    block_coordinates = list((i,j) for i in range(0, b) for j in range(0, b))
#    blocks = sc.parallelize(block_coordinates, numSlices=partitions)\
#                .map(lambda x: get_block(x, b, problem_type)).partitionBy(rdd_partitioner, partitions)
#    return (x_seq_broadcast, y_seq_broadcast, blocks)
