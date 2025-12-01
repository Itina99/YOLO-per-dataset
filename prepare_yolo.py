import os
import json
import glob
import random
import shutil
from pathlib import Path
from tqdm import tqdm  # Se non ce l'hai: pip install tqdm

# --- CONFIGURAZIONE ---
DATASET_ROOT_DIR = "."  # Cartella corrente dove sono gli output_batch_X
OUTPUT_DIR = "yolo_dataset" # Cartella che verrà creata per YOLO
TRAIN_RATIO = 0.8  # 80% train, 20% val
FORCE_RECREATE = True # Se True, cancella la cartella output prima di iniziare

def convert_bbox_coco_to_yolo(bbox, img_width, img_height):
    """
    Converte bbox da COCO [x_min, y_min, width, height] 
    a YOLO [x_center, y_center, width, height] normalizzato (0-1).
    """
    x_min, y_min, w, h = bbox
    
    x_center = (x_min + w / 2) / img_width
    y_center = (y_min + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    return x_center, y_center, w_norm, h_norm

def main():
    # 1. Preparazione cartelle
    output_path = Path(OUTPUT_DIR)
    if output_path.exists() and FORCE_RECREATE:
        shutil.rmtree(output_path)
    
    # Creiamo la struttura: images/train, images/val, labels/train, labels/val
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    print(f"🚀 Inizio creazione dataset YOLO in: {OUTPUT_DIR}")

    # Trova tutte le cartelle dei batch
    batch_folders = sorted(glob.glob(os.path.join(DATASET_ROOT_DIR, "output_batch_*")))
    if not batch_folders:
        print("❌ Nessuna cartella 'output_batch_' trovata!")
        return

    categories_info = {} # Per salvare i nomi delle classi

    # Iteriamo su ogni batch
    for batch_folder in batch_folders:
        batch_name = os.path.basename(batch_folder)
        print(f"📂 Processando {batch_name}...")

        # Carica annotazioni COCO
        anno_file = os.path.join(batch_folder, "annotations.json")
        if not os.path.exists(anno_file):
            print(f"⚠️ {anno_file} non trovato, salto il batch.")
            continue
            
        with open(anno_file, 'r') as f:
            coco_data = json.load(f)

        # Mappatura ID categoria -> Nome (per il file yaml)
        for cat in coco_data['categories']:
            categories_info[cat['id']] = cat['name']
        
        # Creiamo un dizionario rapido per le annotazioni: image_id -> list of anns
        img_to_anns = {img['id']: [] for img in coco_data['images']}
        for ann in coco_data['annotations']:
            img_to_anns[ann['image_id']].append(ann)

        # Processiamo le immagini
        for img_info in tqdm(coco_data['images'], desc="Converting"):
            img_id = img_info['id']
            file_name = img_info['file_name'] # es: "seq_0/frame_00001.png" o solo "frame_00001.png"
            width = img_info['width']
            height = img_info['height']

            # Costruiamo il percorso sorgente dell'immagine
            # Kubric di solito salva in: output_batch_X/rgb/seq_Y/frame_Z.png
            # Ma il json potrebbe dire solo "frame_Z.png". Controlliamo.
            
            # Tentativo 1: Percorso diretto come nel JSON (se include sottocartelle)
            src_img_path = Path(batch_folder) / "rgb" / file_name
            
            # Se il path nel json non ha la cartella della sequenza, dobbiamo indovinarla o sperare sia inclusa
            if not src_img_path.exists():
                # Fallback: cerca ricorsivamente nella cartella rgb del batch
                found = list(Path(batch_folder).glob(f"rgb/**/{os.path.basename(file_name)}"))
                if found:
                    src_img_path = found[0]
                else:
                    # print(f"⚠️ Immagine mancante: {file_name}")
                    continue

            # Decidiamo se è TRAIN o VAL
            split = 'train' if random.random() < TRAIN_RATIO else 'val'

            # Nome univoco per destinazione: batch_seq_frame.png
            # Sostituiamo gli slash con underscore per appiattire la struttura
            safe_filename = f"{batch_name}_{file_name.replace('/', '_')}"
            dst_img_path = output_path / 'images' / split / safe_filename
            dst_label_path = output_path / 'labels' / split / safe_filename.replace('.png', '.txt')

            # 1. CREAZIONE SYMLINK
            try:
                # Path assoluti sono più sicuri per i symlink
                os.symlink(src_img_path.resolve(), dst_img_path.resolve())
            except FileExistsError:
                pass
            except OSError:
                # Fallback per Windows se non si hanno permessi, copia fisica
                shutil.copy(src_img_path, dst_img_path)

            # 2. CREAZIONE LABEL FILE (YOLO TXT)
            anns = img_to_anns.get(img_id, [])
            with open(dst_label_path, 'w') as label_f:
                for ann in anns:
                    # COCO category id parte spesso da 1 o numeri random. 
                    # YOLO vuole indici da 0 a N-1.
                    # Qui assumiamo che i tuoi ID siano mappati correttamente o li usiamo raw.
                    # IMPORTANTE: Se i tuoi ID sono tipo [1, 2, 5], YOLO fallirà se non rimappi.
                    # Per ora usiamo cat_id - 1 assumendo partano da 1.
                    # Se ShapeNet usa ID strani, serve un rimappatore.
                    
                    # *FIX RAPIDO*: Assumiamo che Kubric dia ID sequenziali o usiamo l'ID diretto se parte da 0.
                    # Verifichiamo il minimo ID. Se è 1, sottraiamo 1.
                    cat_id = ann['category_id'] 
                    # Se necessario: cat_id -= 1 
                    
                    bbox = convert_bbox_coco_to_yolo(ann['bbox'], width, height)
                    label_line = f"{cat_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                    label_f.write(label_line)

    # Creazione file data.yaml
    print("📝 Creazione data.yaml...")
    
    # Assicuriamoci che le chiavi siano ordinate per l'indice YOLO
    names_list = [categories_info[k] for k in sorted(categories_info.keys())]
    
    yaml_content = f"""
path: {output_path.absolute()} # dataset root dir
train: images/train
val: images/val

# Classes
nc: {len(names_list)}
names: {names_list}
    """
    
    with open(output_path / "data.yaml", 'w') as f:
        f.write(yaml_content)

    print("✅ Finito! Dataset pronto per YOLO.")
    print(f"   YAML file: {output_path / 'data.yaml'}")

if __name__ == "__main__":
    main()