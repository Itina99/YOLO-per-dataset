#!/bin/bash

# --- CONFIGURAZIONE PERCORSI ---
WORK_DIR="/seidenas/users/mtinacci/YOLO-per-dataset"
RAW_DATA_DIR="/seidenas/datasets/SimAdapt"
CONDA_ENV_NAME="vmr_yolo"

# --- INIT DEBUG MODE ---
# set -e  <-- COMMENTATO PER EVITARE CHE SI FERMI SUBITO
set -x  # <-- ATTIVA LA MODALITÀ VERBOSA (Stampa tutto)

echo "========================================================"
echo "🚀 VMR YOLO PIPELINE - DEBUG MODE"
echo "========================================================"
echo "📅 Date: $(date)"
echo "📍 Work Dir: $WORK_DIR"
echo "💾 Raw Data: $RAW_DATA_DIR"

# 1. Spostiamoci nella cartella di lavoro
cd "$WORK_DIR" || { echo "❌ Cannot cd to WORK_DIR"; exit 1; }

# 2. Attivazione Conda
echo "🐍 Activating Conda Environment: $CONDA_ENV_NAME..."
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$CONDA_ENV_NAME"

# Check veloce della GPU
python -c "import torch; print(f'🔥 GPU Detected: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"

echo "--------------------------------------------------------"

# 3. Creazione Symlink
echo "🔗 Linking datasets from $RAW_DATA_DIR..."

# Pulizia
echo "   Cleaning old links..."
rm -rf output_batch_*

# DIAGNOSTICA: Vediamo cosa vede effettivamente lo script
echo "🔍 LISTA CARTELLE TROVATE (DEBUG):"
ls -d "$RAW_DATA_DIR"/output_*

num_batches=0
for batch_path in "$RAW_DATA_DIR"/output_*; do
    echo "   ➡️ Processing: $batch_path"
    
    if [ -d "$batch_path" ]; then
        dirname=$(basename "$batch_path")
        link_name="${dirname/output_/output_batch_}"
        
        echo "      Tentativo link: $link_name -> $batch_path"
        
        # Forza il link
        ln -sf "$batch_path" "$link_name"
        
        # Verifica se il link è stato creato
        if [ -L "$link_name" ]; then
             echo "      ✅ Link creato correttamente: $link_name"
             ((num_batches++))
        else
             echo "      ❌ ERRORE CREAZIONE LINK: $link_name"
        fi
    else
        echo "      ⚠️ Non è una directory: $batch_path"
    fi
done

echo "✅ Linked $num_batches batches."

echo "--------------------------------------------------------"

# 4. Step 1: Analisi Luminosità
echo "💡 STEP 1: Recovering Brightness Levels..."
# Controllo preventivo librerie
python -c "import cv2; import numpy; print('📚 Libraries OK')" || echo "❌ LIBRERIE PYTHON MANCANTI O ROTTE"

python recover_brightness.py

# 5. Step 2: Preparazione Dataset YOLO
echo "--------------------------------------------------------"
echo "🏗️ STEP 2: Preparing YOLO Dataset Structure..."
python prepare_yolo.py

# 6. Step 3: Training
echo "--------------------------------------------------------"
echo "🏋️ STEP 3: Starting YOLO Training..."
python train.py

echo "--------------------------------------------------------"
echo "📊 PIPELINE FINISHED (Check errors above)"