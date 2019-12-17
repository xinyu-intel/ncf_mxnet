#!bin/bash
source ~/.bashrc
export PYTHONPATH=~/workspace/MXNet/python
export MXNET_EXEC_BULK_EXEC_INFERENCE=0
export MXNET_EXEC_BULK_EXEC_TRAIN=0
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN=0
export MXNET_PROFILER_MODE=1

CORES=$1
INS=$2
BS=$3

NUM_SOCKET=`lscpu | grep 'Socket(s)' | awk '{print $NF}'`
NUM_NUMA_NODE=`lscpu | grep 'NUMA node(s)' | awk '{print $NF}'`
CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $NF}'`
NUM_CORES=$((CORES_PER_SOCKET * NUM_SOCKET))
CORES_PER_NUMA=$((NUM_CORES / NUM_NUMA_NODE))
echo "target machine has $NUM_CORES physical core(s) on $NUM_NUMA_NODE numa nodes of $NUM_SOCKET socket(s)."

if [ -z $INS ]; then
  echo "Default: launch one instance per core."
  INS=$NUM_CORES
fi
if [ -z $CORES ]; then
  echo "Default: divide full physical cores."
  CORES=1
fi
if [ -z $BS ]; then
  echo "Default: set batch size to 1000."
  BS=1000
fi

echo "  cores/instance: $CORES"
echo "  total instances: $INS"
echo "  batch size: $BS"
echo ""

rm INT8_NCF_*.log

for((i=0;i<$INS;i++));
do
  ((a=$i*$CORES))
  ((b=$a+$CORES-1))
  memid=$((b/CORES_PER_NUMA))
  LOG=INT8_NCF_$i.log
  echo "  $i instance use $a-$b cores and $memid mem with $LOG"
  KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0 \
  OMP_NUM_THREADS=$CORES \
  numactl --physcpubind=$a-$b --membind=$memid python ncf.py --batch-size=$BS --dataset='ml-20m' --epoch=7  --benchmark --prefix=./model/ml-20m/neumf-quantized 2>&1 | tee $LOG &
done
wait

grep speed INT8_NCF_*.log | awk '{ sum += $(NF-1) }; END { print sum }'
