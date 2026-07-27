import shutil
import os
from ultralytics import YOLO


model = YOLO("yolo11n.pt")

model.train(
    data = "dataset/data.yaml",
    epochs = 30,
    batch = 12,
    device = "cpu",
    imgsz = 640,
    workers = 4,
    mosaic = 1.0,
    fliplr = 0.5,

    name = "train",
    exist_ok = True
)

best_weights = os.path.join("runs", "detect", "train", "weights", "best.pt")
final_model = "model.pt"

if os.path.exists(best_weights):
    shutil.copy(best_weights, final_model)
    print(f"Modelo salvo com sucesso como '{final_model}'")
else:
    print(f"Treinamento falhou. Não foi possível encontrar '{best_weights}'")


