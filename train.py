from ultralytics import YOLO

def main():
    # 1. Carica il modello
    # 'yolov8n.pt' verrà scaricato automaticamente la prima volta
    model = YOLO('yolov8n.pt')  

    # 2. Avvia il training
    # data: percorso al file yaml creato dallo script precedente
    # epochs: numero di giri completi sul dataset (50-100 è un buon numero standard, 5 per test)
    # imgsz: risoluzione immagini (deve matchare la tua generazione Kubric, es. 256)
    # batch: quanti frame caricare insieme (dipende dalla VRAM della tua 1050, prova 16 o 32)
    results = model.train(
        data='yolo_dataset/data.yaml',
        epochs=10,          # Aumenta a 50 o 100 per il report finale
        imgsz=256,          # Risoluzione del tuo dataset VMR
        batch=16,           # Se la GPU va "Out of Memory", abbassa a 8 o 4
        device=0,           # Usa la GPU 0
        project='vmr_project', # Nome della cartella dei risultati
        name='exp_yolo_nano',  # Nome dell'esperimento specifico
        plots=True          # Salva grafici di loss e metriche automaticamente
    )

    # 3. Validazione finale (opzionale, la fa già il train alla fine)
    metrics = model.val()
    print(f"Mappa mAP50-95: {metrics.box.map}")

if __name__ == '__main__':
    main()