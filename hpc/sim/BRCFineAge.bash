#!/bin/bash
REPO_PATH="/rds/general/user/sd121/home/high_res_brc"
OUTPUT_DIR="/rds/general/user/sd121/home/high_res_brc_outputs"

# Main script
cat > "$OUTPUT_DIR/BRCFineAge.pbs" <<EOF
#!/bin/bash
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=8:gpu_type=RTX6000

module load anaconda3/personal
source activate brc

cd $REPO_PATH
python scripts/sim/BRCFineAge.py
EOF

cd $OUTPUT_DIR
qsub "BRCFineAge.pbs"