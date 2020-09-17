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

Sample Command:

`spark-submit --master spark://$SPARK_MASTER_IP:$SPARK_MASTER_PORT --num-executors 16 --executor-cores 31 --executor-memory 160g --driver-memory 160g --conf spark.driver.maxResultSize=2g --py-files DPSpark-master.zip --files floyd.so DPSpark.py -isInMem n -problemSize 32768 -blockingFactor 128 -isCached n -numPartitions 992 -partitioner ph -problemType fw-apsp -isRec n -rShared 2 -baseSize 64 -d $BLOCKS_DIR -n 16 -i inp_dp_table -o out_dp_table &> output.out`
