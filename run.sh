#!/bin/bash

# --- CONFIGURAZIONE PERCORSI ---
WORK_DIR="/seidenas/users/mtinacci/YOLO-per-dataset"
RAW_DATA_DIR="/seidenas/datasets/SimAdapt"
CONDA_ENV_NAME="vmr_yolo"

# --- INIT ---
set -e 

echo "========================================================"
echo "🚀 VMR YOLO PIPELINE - FINAL VERSION"
echo "========================================================"
echo "📅 Date: $(date)"
echo "📍 Work Dir: $WORK_DIR"
echo "💾 Raw Data: $RAW_DATA_DIR"

# 1. Spostiamoci nella cartella di lavoro
cd "$WORK_DIR"

# 2. Attivazione Conda
echo "🐍 Activating Conda Environment: $CONDA_ENV_NAME..."
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$CONDA_ENV_NAME"

python -c "import torch; print(f'🔥 GPU Detected: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"

echo "--------------------------------------------------------"

# 3. Creazione Symlink (Modalità FORZATA)
echo "🔗 Linking datasets from $RAW_DATA_DIR..."

# Pulizia preventiva aggressiva dei vecchi link locali
rm -rf output_batch_*

num_batches=0
# Cerca cartelle che iniziano con output_
for batch_path in "$RAW_DATA_DIR"/output_*; do
    if [ -d "$batch_path" ]; then
        dirname=$(basename "$batch_path")
        
        # Sostituzione nome: output_1 -> output_batch_1
        link_name="${dirname/output_/output_batch_}"
        
        # USARE -sf (Symlink Force) per non crashare se il file esiste
        ln -sf "$batch_path" "$link_name"
        
        echo "   Attached: $dirname -> $link_name"
        ((num_batches++))
    fi
done

if [ "$num_batches" -eq 0 ]; then
    echo "❌ ERRORE: Nessuna cartella trovata in $RAW_DATA_DIR"
    exit 1
fi

echo "✅ Linked $num_batches batches."

echo "--------------------------------------------------------"

# 4. Step 1: Analisi Luminosità
echo "💡 STEP 1: Recovering Brightness Levels..."
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
echo "📊 PIPELINE COMPLETED SUCCESSFULLY!"
echo "   Results are stored in: $WORK_DIR/vmr_project"
echo "========================================================"