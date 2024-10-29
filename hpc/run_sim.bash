#!/bin/bash
REPO_PATH="/rds/general/user/sd121/home/high_res_brc"
OUTPUT_DIR="/rds/general/user/sd121/home/high_res_brc_outputs"

# Main script
cat > "$OUTPUT_DIR/run_sim.pbs" <<EOF
#!/bin/bash
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=4:ompthreads=1:mem=50gb

module load anaconda3/personal
source activate brc

cd $REPO_PATH
python scripts/run_sim.py
EOF

cd $OUTPUT_DIR
qsub "run_sim.pbs"