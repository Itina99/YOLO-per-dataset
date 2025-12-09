#!/bin/bash

# --- CONFIGURAZIONE PERCORSI ---
# Directory dove risiedono i tuoi script Python
WORK_DIR="/seidenas/users/mtinacci/YOLO-per-dataset"

# Directory dove sono salvati i dati grezzi (le cartelle output_1, output_2...)
RAW_DATA_DIR="/seidenas/datasets/SimAdapt"

# Nome del tuo environment conda
CONDA_ENV_NAME="vmr_yolo"

# --- INIT ---
# Interrompi lo script se un comando fallisce
set -e 

echo "========================================================"
echo "🚀 VMR YOLO PIPELINE - STARTING"
echo "========================================================"
echo "📅 Date: $(date)"
echo "📍 Work Dir: $WORK_DIR"
echo "💾 Raw Data: $RAW_DATA_DIR"

# 1. Spostiamoci nella cartella di lavoro
cd "$WORK_DIR"

# 2. Attivazione Conda
echo "🐍 Activating Conda Environment: $CONDA_ENV_NAME..."
# Trick per attivare conda negli script shell
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$CONDA_ENV_NAME"

# Check veloce della GPU
python -c "import torch; print(f'🔥 GPU Detected: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
echo "--------------------------------------------------------"

# 3. Creazione Symlink ai dati
# TRUCCO FONDAMENTALE: Rinomina output_N -> output_batch_N
echo "🔗 Linking datasets from $RAW_DATA_DIR..."

# Rimuove vecchi link per pulizia
find . -maxdepth 1 -name "output_*" -type l -delete

num_batches=0
# Cerca cartelle che iniziano con output_
for batch_path in "$RAW_DATA_DIR"/output_*; do
    if [ -d "$batch_path" ]; then
        # Prende il nome originale (es. output_1)
        dirname=$(basename "$batch_path")
        
        # Sostituisce la stringa 'output_' con 'output_batch_'
        # Esempio: output_1 diventa output_batch_1
        link_name="${dirname/output_/output_batch_}"
        
        # Crea il link col NUOVO nome
        ln -s "$batch_path" "$link_name"
        
        # Feedback visuale
        echo "   Attached: $dirname -> $link_name"
        ((num_batches++))
    fi
done

if [ "$num_batches" -eq 0 ]; then
    echo "❌ ERRORE: Nessuna cartella 'output_*' trovata in $RAW_DATA_DIR"
    exit 1
fi

echo "✅ Linked $num_batches batches."

echo "--------------------------------------------------------"

# 4. Step 1: Analisi Luminosità
echo "💡 STEP 1: Recovering Brightness Levels..."
python recover_brightness.py

if [ -f "scene_brightness.json" ]; then
    echo "✅ Brightness map generated."
else
    echo "❌ Error generating brightness map."
    exit 1
fi

echo "--------------------------------------------------------"

# 5. Step 2: Preparazione Dataset YOLO
echo "🏗️ STEP 2: Preparing YOLO Dataset Structure..."
python prepare_yolo.py

if [ -f "yolo_dataset/data.yaml" ]; then
    echo "✅ YOLO dataset ready at $WORK_DIR/yolo_dataset"
else
    echo "❌ Error preparing YOLO dataset."
    exit 1
fi

echo "--------------------------------------------------------"

# 6. Step 3: Training
echo "🏋️ STEP 3: Starting YOLO Training..."
python train.py

echo "--------------------------------------------------------"

# 7. Risultati
echo "📊 PIPELINE COMPLETED SUCCESSFULLY!"
echo "   Results are stored in: $WORK_DIR/vmr_project"

# Opzionale: Pulizia symlink alla fine
# echo "🧹 Cleaning up symlinks..."
# find . -maxdepth 1 -name "output_batch_*" -type l -delete

echo "========================================================"