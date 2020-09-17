__author__ = "Zafar Ahmad, Mohammad Mahdi Javanmard"
__copyright__ = "Copyright (c) 2019 Tealab@SBU"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zafar Ahmad"
__email__ = "zafahmad@cs.stonybrook.edu"
__status__ = "Development"


import fw_apsp_collect_broadcast
import ge_collect_broadcast
import paf_collect_broadcast
#import gap_collect_broadcast

import fw_apsp_in_memory
import ge_in_memory
import paf_in_memory
#import gap_in_memory



def solve(problem_type, is_in_memory, n, dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, blocks_dir, sc):
    if problem_type == "fw-apsp":
        if is_in_memory:
            dp_table = fw_apsp_in_memory.solve_dp(n, dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, sc)
        else:
            dp_table = fw_apsp_collect_broadcast.solve_dp(n, dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, blocks_dir, sc)
    elif problem_type == "ge":
        if is_in_memory:
            dp_table = ge_in_memory.solve_dp(n, dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, sc)
        else:
            dp_table = ge_collect_broadcast.solve_dp(n, dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, blocks_dir, sc)
    elif problem_type == "paf":
        if is_in_memory:
            dp_table = paf_in_memory.solve_dp(n, dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, sc)
        else:
            dp_table = paf_collect_broadcast.solve_dp(n, dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, blocks_dir, sc)
    return dp_table

#def solveGap(dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, blocks_dir, x_seq_broadcast, y_seq_broadcast, sc):
#    # TODO: The for loop is different for Gap Problem!
#    for k in range(0, 2*r-1, 1):
#        dp_table = gap.solve(k, dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, blocks_dir, x_seq_broadcast, y_seq_broadcast, sc)
#    return dp_table
