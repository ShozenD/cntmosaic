#!/bin/bash
REPO_PATH="/rds/general/user/sd121/home/high_res_brc"
OUTPUT_DIR="/rds/general/user/sd121/home/high_res_brc_outputs"

# Main script
cat > "$OUTPUT_DIR/run_sim_omp.pbs" <<EOF
#!/bin/bash
#PBS -l select=1:ncpus=4:ompthreads=1:mem=16gb
#PBS -l walltime=24:00:00

module load tools/prod
module load Python/3.10.8-GCCcore-12.2.0

cd $REPO_PATH
source .venv/bin/activate

python scripts/run_sim.py
EOF

cd $OUTPUT_DIR
qsub "run_sim_omp.pbs"