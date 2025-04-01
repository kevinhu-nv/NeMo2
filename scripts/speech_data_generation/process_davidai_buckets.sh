#!/bin/bash
#SBATCH -A llmservice_nemo_mlops
#SBATCH -N 1 # number of nodes
#SBATCH -t 04:00:00
#SBATCH --time-min 04:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

set -x

# Parse command line arguments
while getopts ":b:n:c:s:o:r:h:" opt; do
  case $opt in
    b) BUCKET_DIR="$OPTARG";;
    n) NUM_BUCKETS="$OPTARG";;
    c) CONTAINER="$OPTARG";;
    s) SCRIPT_PATH="$OPTARG";;
    o) OUTPUT_DIR="$OPTARG";;
    r) RESULTS_DIR="$OPTARG";;
    h) NUM_SHARDS="$OPTARG";;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1;;
    \?) echo "Invalid option: -$OPTARG" >&2
        echo "Usage: $0 -b bucket_dir -n num_buckets -c container -s script_path -o output_dir -r results_dir -h num_shards" >&2
        exit 1;;
  esac
done

# Check if required arguments are provided
if [ -z "$BUCKET_DIR" ] || [ -z "$NUM_BUCKETS" ] || [ -z "$CONTAINER" ] || [ -z "$SCRIPT_PATH" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$RESULTS_DIR" ] || [ -z "$NUM_SHARDS" ]; then
    echo "Required arguments missing"
    echo "Usage: $0 -b bucket_dir -n num_buckets -c container -s script_path -o output_dir -r results_dir -h num_shards"
    exit 1
fi

# Define fixed mounts
MOUNTS="--container-mounts=/lustre/:/lustre/,$RESULTS_DIR:/results,/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data:/data,/lustre/fsw:/lustre/fsw,/lustre/fs12:/lustre/fs12"

mkdir -p ${RESULTS_DIR}
mkdir -p ${OUTPUT_DIR}

# Create a temporary script for each bucket
TEMP_SCRIPT_DIR="${RESULTS_DIR}/temp_scripts"
mkdir -p ${TEMP_SCRIPT_DIR}

# Process each bucket
for i in $(seq 0 $((NUM_BUCKETS-1))); do
    DIRS_FILE="${BUCKET_DIR}/davidai_dirs_bucket_${i}.txt"
    OUTPUT_SUBDIR="${OUTPUT_DIR}/bucket_${i}"
    
    # Create output directory for this bucket
    mkdir -p ${OUTPUT_SUBDIR}
    
    # Create a temporary script for this bucket
    TEMP_SCRIPT="${TEMP_SCRIPT_DIR}/process_bucket_${i}.sh"
    cat << EOF > ${TEMP_SCRIPT}
#!/bin/bash
#SBATCH -A llmservice_nemo_mlops
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH --time-min 04:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH -J "llmservice_nemo_mlops:process_davidai_${i}"
#SBATCH --output=${RESULTS_DIR}/slurm-${i}-%j.out
#SBATCH --error=${RESULTS_DIR}/error-${i}-%j.out

set -x

# Run the python command inside the container
srun --container-image="${CONTAINER}" ${MOUNTS} python ${SCRIPT_PATH} \\
    --dirs_file ${DIRS_FILE} \\
    --out_dir ${OUTPUT_SUBDIR} \\
    --num_shards ${NUM_SHARDS}

# Add echo statement to indicate job completion
echo "====== Finished processing bucket ${i} ======"
EOF

    # Make the temporary script executable
    chmod +x ${TEMP_SCRIPT}
    
    # Submit the job using sbatch
    sbatch ${TEMP_SCRIPT}
done 