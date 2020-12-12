#!/bin/bash  -l

#SBATCH -A m844
#SBATCH -t 00:05:00
#SBATCH --nodes=4
#SBATCH --tasks-per-node=32
#SBATCH --constraint=haswell
#SBATCH -L SCRATCH
#SBATCH -o qout.128%j
#SBATCH -e qout.128%j
#SBATCH --qos=debug

cd $PWD

NP=$(($SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE))

echo "------------------------------------------------------"
echo "---- Running on $NP MPI processes on Cori Haswell nodes ----"
echo "---- SLURM_JOB_QOS         = $SLURM_JOB_QOS"
echo "---- SLURM_CLUSTER_NAME    = $SLURM_CLUSTER_NAME"
echo "---- SLURM_JOB_PARTITION   = $SLURM_JOB_PARTITION"
echo "---- SLURM_JOB_NUM_NODES   = $SLURM_JOB_NUM_NODES"
echo "---- SLURM_NTASKS_PER_NODE = $SLURM_NTASKS_PER_NODE"
echo "---- SBATCH_CONSTRAINT     = $SBATCH_CONSTRAINT"
echo "------------------------------------------------------"
echo ""

export LD_LIBRARY_PATH=$HOME/HDF5/1.10.6/lib:$LD_LIBRARY_PATH

echo "==========================================================="
IN_DIR=/global/cscratch1/sd/wkliao/FS_1M_32
echo "==== Lustre stripe count = 32"
LIST_FILE=dset.txt
echo "==== LIST_FILE = $LIST_FILE"
echo "==== INPUT $IN_DIR/NOvA_ND_1951.h5caf_*.h5"
echo "==========================================================="
srun -n $NP ./pandana_read -p 0 -s 0 -m 0 -l $LIST_FILE -i $IN_DIR/NOvA_ND_1951.h5caf_1.h5
srun -n $NP ./pandana_read -p 0 -s 0 -m 1 -l $LIST_FILE -i $IN_DIR/NOvA_ND_1951.h5caf_2.h5
srun -n $NP ./pandana_read -p 0 -s 0 -m 2 -l $LIST_FILE -i $IN_DIR/NOvA_ND_1951.h5caf_3.h5
srun -n $NP ./pandana_read -p 0 -s 2 -m 0 -l $LIST_FILE -i $IN_DIR/NOvA_ND_1951.h5caf_4.h5
srun -n $NP ./pandana_read -p 0 -s 2 -m 1 -l $LIST_FILE -i $IN_DIR/NOvA_ND_1951.h5caf_1.h5
srun -n $NP ./pandana_read -p 0 -s 2 -m 2 -l $LIST_FILE -i $IN_DIR/NOvA_ND_1951.h5caf_2.h5
srun -n $NP ./pandana_read -p 0 -s 3 -m 0 -l $LIST_FILE -i $IN_DIR/NOvA_ND_1951.h5caf_3.h5
srun -n $NP ./pandana_read -p 0 -s 3 -m 1 -l $LIST_FILE -i $IN_DIR/NOvA_ND_1951.h5caf_4.h5
srun -n $NP ./pandana_read -p 0 -s 3 -m 2 -l $LIST_FILE -i $IN_DIR/NOvA_ND_1951.h5caf_1.h5
srun -n $NP ./pandana_read -p 1 -s 4 -m 0 -l $LIST_FILE -i $IN_DIR/NOvA_ND_1951.h5caf_2.h5
srun -n $NP ./pandana_read -p 0 -s 4 -m 1 -l $LIST_FILE -i $IN_DIR/NOvA_ND_1951.h5caf_3.h5
srun -n $NP ./pandana_read -p 0 -s 4 -m 2 -l $LIST_FILE -i $IN_DIR/NOvA_ND_1951.h5caf_4.h5
exit

