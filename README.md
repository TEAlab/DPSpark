# DPSPark: Efficient Execution of Dynamic Programming Algorithms on Apache Spark

**Authors:** 
Mohammad Javanmard <mjavanmard@cs.stonybrook.edu>,
Zafar Ahmad <zafahmad@cs.stonybrook.edu>,
Jaroslaw Zola <jzola@buffalo.edu>,
Louisnoel Pouchet <Louis-Noel.Pouchet@colostate.edu>,
Rezaul Chowdhury <rezaul@cs.stonybrook.edu>,
Robert Harrison <robert.harrison@stonybrook.edu>

This project is the implementation of the paper [Efficient Execution of Dynamic Programming Algorithms on Apache Spark](https://ieeexplore.ieee.org/abstract/document/9229617), which has been accepted in the [IEEE-CLUSTER 2020](https://clustercomp.org/2020/) conference

To cite `DPSPark`, please refer to this repository and our paper:

Mohammad Mahdi Javanmard, Zafar Ahmad, Jaroslaw Zola, Louis-Noël Pouchet, Rezaul Chowdhury, Robert Harrison, "Efficient Execution of Dynamic Programming Algorithms on Apache Spark", 2020 IEEE International Conference on Cluster Computing (CLUSTER). https://ieeexplore.ieee.org/abstract/document/9229617

## Environment Setup

Use Anaconda3 for virtual environment and install the following packages in your environment: Anaconda3, Python 2.7, Spark 2.2.0, NumPy, Numba, Cython


## Installation

Run the following commands to generate required shared libraries:

`cd floyd_r-way`

`make`

`cp *.so ../DPSparkImplementations/`

`cd ..`

Repeat the above steps for `gaussian_r-way` and `paf_r-way`.

## Running the sources

`DPSPark` supports the following command line options:

* `-isInMem` solver type (use `y` for In memory and `n` for Collect Broadcast)
* `-problemSize` number of vertices in the input graph
* `-blockingFactor` block size for adjacency matrix decomposition
* `-isCached` caching type (`y` for caching is on and `n` for turning it off)
* `-numPartitions` number of Spark RDD partitions to store adjacency matrix
* `-partitioner` type of partitioner
* `-problemType` type of the problem (use `fw-apsp` for Floyd-Warshall All Pair Shortest Paths, `ge` for Gaussian elimination, and `paf` for Protein Accordion Folding). Please note that based of the type of the problem the shared obect need to be passed according in `--file` argument (`floyd.so` for `fw-apsp`, `gaussian.so` for `ge`, and `protien.so` for `paf`)
* `-isRec` type of the kernel (use `y` for recursive kernels and `n` for iterative implementation)
* `-rShared` parameter `R` for shared memory implementation of R-way recursive devide and conquer approch (e.g: 2 means, at each recursion depth, the computation grid will be devided in 2 parts at each dimension)
* `-baseSize` base case of R-way shared memory implementation where the R-way recursive solver switches to iterative computation
* `-o` output folder

##### Example invocation

`spark-submit --master spark://$SPARK_MASTER_IP:$SPARK_MASTER_PORT --num-executors 16 --executor-cores 31 --executor-memory 160g --driver-memory 160g --conf spark.driver.maxResultSize=2g --py-files DPSpark-master.zip --files floyd.so DPSpark.py -isInMem n -problemSize 32768 -blockingFactor 128 -isCached n -numPartitions 992 -partitioner ph -problemType fw-apsp -isRec n -rShared 2 -baseSize 64 -d $BLOCKS_DIR -n 16 -i inp_dp_table -o out_dp_table &> output.out`
