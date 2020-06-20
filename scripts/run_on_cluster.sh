#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets

if [ ! -d "${DATASET_DIR}/${2}/" ]; then
  echo "Dataset set not in node, getting it..."
  rsync -ua /home/${STUDENT_ID}/RecommendationSystem/dataset/${2}.tar.gz "${DATASET_DIR}"
  tar -xzf "${DATASET_DIR}/${2}.tar.gz" -C "${DATASET_DIR}"
fi

source /home/${STUDENT_ID}/miniconda3/bin/activate tezi

python ${1} --json_configs_string "${3}" --dataset_location "${DATASET_DIR}/${2}/"