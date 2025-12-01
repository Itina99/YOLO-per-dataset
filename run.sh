#!/bin/bash

# --- CONFIGURAZIONE PERCORSI ---
# Directory dove risiedono i tuoi script Python
WORK_DIR="/seidenas/users/mtinacci/YOLO-per-dataset"

# Directory dove sono salvati i dati grezzi (le cartelle output_batch_X)
RAW_DATA_DIR="/seidenas/datasets"

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
# Nota: su alcuni server serve inizializzare conda nello script bash
echo "🐍 Activating Conda Environment: $CONDA_ENV_NAME..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# Check veloce della GPU
python -c "import torch; print(f'🔥 GPU Detected: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}')"

echo "--------------------------------------------------------"

# 3. Creazione Symlink ai dati
# Questo step evita di copiare GB di dati. Crea dei "puntatori" nella cartella corrente
# che puntano ai batch originali. Gli script Python leggeranno questi link.
echo "🔗 Linking datasets from $RAW_DATA_DIR..."

# Rimuove vecchi link per pulizia (non cancella i dati veri, solo i link)
find . -maxdepth 1 -name "output_batch_*" -type l -delete

# Crea i nuovi link
num_batches=0
for batch_path in "$RAW_DATA_DIR"/output_batch_*; do
    if [ -d "$batch_path" ]; then
        ln -s "$batch_path" .
        ((num_batches++))
    fi
done

if [ "$num_batches" -eq 0 ]; then
    echo "❌ ERRORE: Nessuna cartella 'output_batch_*' trovata in $RAW_DATA_DIR"
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
# Questo script userà i symlink creati al punto 3
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
# Il train.py salverà i risultati nella cartella 'vmr_project'
python train.py

echo "--------------------------------------------------------"

# 7. Risultati
echo "📊 PIPELINE COMPLETED SUCCESSFULLY!"
echo "   Results are stored in: $WORK_DIR/vmr_project"
echo "   Model weights: $WORK_DIR/vmr_project/exp_yolo_nano/weights/best.pt"

# Opzionale: Pulizia symlink alla fine (puoi commentarlo se vuoi tenerli)
# echo "🧹 Cleaning up symlinks..."
# find . -maxdepth 1 -name "output_batch_*" -type l -delete

echo "========================================================"