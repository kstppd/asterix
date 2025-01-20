#!/bin/bash
#SBATCH -t 01:30:00        # Run time (hh:mm:ss)
#SBATCH --job-name=TP_gpu
#SBATCH -M ukko
#SBATCH -p gpu
##SBATCH -p gpu-oversub
#SBATCH --constraint=a100
#SBATCH --cpus-per-gpu=32
#SBATCH --hint=nomultithread
##SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH -c 32                 # CPU cores per task
#SBATCH -n 1                  # number of tasks
##SBATCH --mem=0 # do not request all node memory or it's equal to exclusive
#SBATCH --mem=60G

#If 1, the reference vlsv files are generated
# if 0 then we check the v1
create_verification_files=0

# folder for all reference data
reference_dir="/proj/group/spacephysics/vlasiator_testpackage/"
cd $SLURM_SUBMIT_DIR
#cd $reference_dir # don't run on /proj

#compare agains which revision
reference_revision="CI_reference"
#reference_revision="current"

bin="$SLURM_SUBMIT_DIR/vlasiator"
diffbin="$SLURM_SUBMIT_DIR/vlsvdiff_DP"

module purge
ml GCC/11.2.0
ml OpenMPI/4.1.1-GCC-11.2.0
ml PMIx/4.1.0-GCCcore-11.2.0
ml PAPI/6.0.0.1-GCCcore-11.2.0
ml CUDA
ml Boost/1.55.0-GCC-11.2.0

#--------------------------------------------------------------------
#---------------------DO NOT TOUCH-----------------------------------
nodes=$SLURM_NNODES
#Carrington has 2 x 16 cores
cores_per_node=128
# Hyperthreading
# ht=1
#Change PBS parameters above + the ones here
# total_units=$(echo $nodes $cores_per_node $ht | gawk '{print $1*$2*$3}')
# units_per_node=$(echo $cores_per_node $ht | gawk '{print $1*$2}')
# tasks=$(echo $total_units $t  | gawk '{print $1/$2}')
# tasks_per_node=$(echo $units_per_node $t  | gawk '{print $1/$2}')
export t=$SLURM_CPUS_PER_TASK # used by TP script
export OMP_NUM_THREADS=$t
export tasks=$SLURM_NTASKS

#command for running stuff
run_command="mpirun --mca btl self -mca pml ^vader,tcp,openib,uct,yalla -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc,sm -x UCX_IB_ADDR_TYPE=ib_global -np $tasks"
small_run_command="mpirun --mca btl self -mca pml ^vader,tcp,openib,uct,yalla -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc,sm -x UCX_IB_ADDR_TYPE=ib_global -n 1 -N 1"
run_command_tools="mpirun --mca btl self -mca pml ^vader,tcp,openib,uct,yalla -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc,sm -x UCX_IB_ADDR_TYPE=ib_global -n 1 -N 1"
#run_command="mpirun -np $tasks"
#small_run_command="mpirun -n 1 -N 1"
#run_command_tools="srun -n 1 "

umask 007
# Launch the OpenMP job to the allocated compute node
echo "Running $exec on $tasks mpi tasks, with $t threads per task on $nodes nodes ($ht threads per physical core)"

# Define test
source test_definitions_small.sh
wait
# Run tests
source run_tests.sh
wait 

