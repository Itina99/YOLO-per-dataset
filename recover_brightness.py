import sys
print("🐍 Python sta partendo...", file=sys.stderr)

try:
    import os
    import glob
    import cv2
    import numpy as np
    import json
    from pathlib import Path
    print("📚 Librerie importate correttamente.", file=sys.stderr)
except ImportError as e:
    print(f"❌ ERRORE IMPORT LIBRERIE: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"❌ ERRORE GENERICO ALL'AVVIO: {e}", file=sys.stderr)
    sys.exit(1)

DATASET_ROOT = "."  # Dove sono le cartelle output_batch_X

def get_kmeans_centroids(values, k=3):
    """
    Versione semplificata di K-Means con numpy per non installare scikit-learn.
    Trova 3 centroidi (scuro, medio, chiaro) nei dati.
    """
    # Inizializza centroidi usando percentile (es. 10%, 50%, 90%)
    centroids = np.percentile(values, [10, 50, 90])
    
    for _ in range(10): # 10 iterazioni bastano per convergere
        # Assegna ogni valore al centroide più vicino
        distances = np.abs(values[:, None] - centroids)
        labels = np.argmin(distances, axis=1)
        
        # Ricalcola centroidi
        new_centroids = np.array([values[labels == i].mean() for i in range(k)])
        
        # Gestione caso cluster vuoto (raro ma possibile)
        if np.any(np.isnan(new_centroids)):
            return centroids # Ritorna i vecchi se crasha
            
        centroids = new_centroids
        
    return np.sort(centroids) # Ritorna ordinati: [scuro, medio, chiaro]

def main():
    print("🕵️  Analisi luminosità in corso...")
    
    sequences_data = [] # Lista di tuple (batch, seq_name, mean_brightness)
    
    # 1. Raccogli i dati di luminosità
    batch_folders = sorted(glob.glob(os.path.join(DATASET_ROOT, "output_batch_*")))
    
    for batch in batch_folders:
        batch_name = os.path.basename(batch)
        # Cerca le cartelle delle sequenze dentro rgb
        seq_paths = glob.glob(os.path.join(batch, "rgb", "seq_*"))
        
        for seq_path in seq_paths:
            seq_name = os.path.basename(seq_path)
            
            # Prendi il primo frame (frame_00000.png o frame_00001.png)
            frames = sorted(glob.glob(os.path.join(seq_path, "*.png")))
            if not frames:
                continue
            
            first_frame = frames[0]
            
            # Leggi immagine e calcola media in scala di grigi
            img = cv2.imread(first_frame)
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray) # Valore da 0 a 255
            
            sequences_data.append({
                "batch": batch_name,
                "seq": seq_name,
                "value": mean_brightness
            })

    if not sequences_data:
        print("❌ Nessuna sequenza trovata.")
        return

    # 2. Clustering per trovare le soglie
    all_values = np.array([x['value'] for x in sequences_data])
    centroids = get_kmeans_centroids(all_values, k=3)
    
    print(f"\n📊 Centroidi rilevati (0-255): {centroids}")
    print(f"   Corrispondono circa a: 0.1 (Scuro), 0.5 (Medio), 1.0 (Chiaro)")

    # 3. Assegna le etichette finali
    # Mappiamo i centroidi ordinati ai tuoi valori noti
    known_labels = [0.1, 0.5, 1.0]
    
    final_mapping = {} # Chiave: batch/seq, Valore: label
    
    counts = {0.1: 0, 0.5: 0, 1.0: 0}
    
    for item in sequences_data:
        val = item['value']
        # Trova a quale centroide è più vicino
        idx = (np.abs(centroids - val)).argmin()
        label = known_labels[idx]
        
        # Chiave univoca per il json finale
        key = f"{item['batch']}/{item['seq']}"
        final_mapping[key] = label
        
        counts[label] += 1

    # 4. Salva il risultato
    with open("scene_brightness.json", "w") as f:
        json.dump(final_mapping, f, indent=4)
        
    print("\n✅ Analisi completata!")
    print(f"💾 Mappa salvata in: scene_brightness.json")
    print("\n--- Distribuzione Recuperata ---")
    print(f"🌑 Scene Scure (0.1): {counts[0.1]}")
    print(f"🌗 Scene Medie (0.5): {counts[0.5]}")
    print(f"🌕 Scene Chiare (1.0): {counts[1.0]}")

if __name__ == "__main__":
    main()