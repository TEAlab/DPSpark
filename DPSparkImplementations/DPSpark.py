__author__ = "Zafar Ahmad, Mohammad Mahdi Javanmard"
__copyright__ = "Copyright (c) 2019 Tealab@SBU"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zafar Ahmad"
__email__ = "zafahmad@cs.stonybrook.edu"
__status__ = "Development"

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, 'SparkFiles.getRootDirectory()')

import util
import solver 

from pyspark import SparkContext, SparkConf
from pyspark import StorageLevel as stglev


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-isInMem", "--is_in_memory", help="Should run in memory solution? (Y/y/N/n)", type=str, required=True)
    parser.add_argument("-problemSize", "--input_dp_size", help="Size of input DP table.", type=int, required=True)
    parser.add_argument("-blockingFactor", "--decomposition_factor", help="Spark-level decomposition parameter [Should divide n for now]", type=int, required=True)
    parser.add_argument("-isCached", "--is_cached", help="Caching the blocks not to recompute them (Y/y/N/n)", type=str, required=True)
    parser.add_argument("-isRec", "--is_recursive", help="Should run recursive solution? (Y/y/N/n)", type=str, required=True)
    parser.add_argument("-rShared", "--r_factor_recursive", help="Parameter r for r-way functions in executors.", type=int, required=False)
    parser.add_argument("-baseSize", "--base_size", help="base size for kernels.", type=int, required=True)
    parser.add_argument("-numPartitions", "--partitions", help="Number of RDD partitions.", type=int, required=True)
    parser.add_argument("-partitioner", "--partitioner", help="Partitioning function. [ph (default one) or custom md].", type=str, required=True)
    parser.add_argument("-problemType", "--problem_type", help="Problem type [fw-apsp, ge, paf].", type=str, required=True)
    # new argument to creat the auxiliary directory in the scratch folder    
    parser.add_argument("-d", "--blocks_dir", help="SCRATCH directory", type=str, required=True)
    parser.add_argument("-n", "--compute_nodes", help="number of executors", type=int, required=True)
    parser.add_argument("-o", "--output_dir", help="Folder name for storing the results.", type=str, required=True)
    parser.add_argument("-i", "--input_dir", help="Folder name for the randomly generated input dp table.", type=str, required=True)
    parser.add_argument("-e", "--log_dir", help="Spark event log dir.", type=str, required=False)

    args = parser.parse_args()
    is_in_memory = (args.is_in_memory == 'y')
    n = args.input_dp_size
    r = args.decomposition_factor
    is_cached = (args.is_cached.lower() == 'y')
    is_rec = (args.is_recursive.lower() == 'y')
    r_shared = 0
    if is_rec:
        r_shared = args.r_factor_recursive
    base_size = args.base_size
    partitions = args.partitions
    partitioner = args.partitioner.lower()
    problem_type = args.problem_type.lower()
    compute_nodes = args.compute_nodes
    inp = args.input_dir
    out = args.output_dir
    # blocks_dir = args.blocks_dir + "/" + args.job_number + "/"
    blocks_dir = args.blocks_dir + "/"
    
    rdd_partitioner = util.get_partitioner(partitioner, r, compute_nodes)

    conf = SparkConf()

    # optional log for history server
    save_history = args.log_dir is not None
    if save_history:
        conf.set("spark.eventLog.enabled", "true")\
            .set("spark.eventLog.dir", args.log_dir)\
            .set("spark.history.fs.logDirectory", args.log_dir)

    sc = SparkContext(conf=conf)
    log4jLogger = sc._jvm.org.apache.log4j
    logger = log4jLogger.LogManager.getLogger("DPSPark")
    logger.setLevel(sc._jvm.org.apache.log4j.Level.ALL)
    logger.info('is_in_memory: {}, n: {}, r: {}, is_cached: {}, is_rec: {}, r_shared: {}, base_size: {}, partitions: {}, rdd_partitioner: {}, problem_type: {}'.format(is_in_memory, n, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, problem_type))
 
    if problem_type != "gap":
        dp_table = util.get_dp_table(n, r, partitions, rdd_partitioner, problem_type, sc) 
        dp_table.persist(stglev.MEMORY_AND_DISK)
        dp_table.count()
        #dp_table.saveAsTextFile(inp) # for verification, uncomment and check the inp folder
        t0 = time.time()
        #os.system("rm -r " + blocks_dir)
        #os.system("mkdir " + blocks_dir)
        # solving the specific DP problem P
        dp_table = solver.solve(problem_type, is_in_memory, n, dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, blocks_dir, sc) 
        dp_table.persist(stglev.MEMORY_AND_DISK)
        dp_table.count()
        os.system("rm -r " + blocks_dir)
        t1 = time.time()
        #dp_table.saveAsTextFile(out) # for verification, uncomment and check the out folder
        sc.stop()
        logger.info("time to solution: " + str(t1-t0) + " s")

    # NOTE: if the problem_type is "gap", we need to make two broadcast variables x_seq and y_seq and use them
    #elif problem_type == "gap":   
        #dp_table_info = util.get_dp_table_info_gap(n, r, partitions, rdd_partitioner, sc)
        #x_seq_broadcast = dp_table_info[0]
        #y_seq_broadcast = dp_table_info[1]
        #dp_table = dp_table_info[2]
        #dp_table.persist(stglev.MEMORY_AND_DISK)
        #dp_table.count()
        #dp_table.saveAsTextFile(inp)
        #blocks_dir = os.getcwd() + '/_auxdir_/'
        #os.system("mkdir " + blocks_dir)
        #t0 = time.time()
        ## solving the gap problem
        #dp_table = solver.solveGap(dp_table, r, is_cached, is_rec, r_shared, base_size, partitions, rdd_partitioner, blocks_dir, x_seq_broadcast, y_seq_broadcast, sc)
        #dp_table.persist(stglev.MEMORY_AND_DISK)
        #dp_table.count()
        #os.system("rm -r " + blocks_dir)
        #t1 = time.time()
        #dp_table.saveAsTextFile(out)
        #sc.stop()
        #logger.info("time to solution: " + str(t1-t0) + " s")

    logger.info('is_in_memory: {}, n: {}, r: {}, is_rec: {}, r_shared: {}, base_size {}, partitions: {}, rdd_partitioner: {}, problem_type: {}'.format(is_in_memory, n, r, is_rec, r_shared, base_size, partitions, rdd_partitioner, problem_type))
    logger.info("using Python: " + sys.version)
    logger.info('done!')
