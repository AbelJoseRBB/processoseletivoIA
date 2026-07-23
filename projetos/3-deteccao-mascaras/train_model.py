import shutil
import os
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data = "dataset/data.yaml",
    epochs = 20,
    batch = 8,
    device = "cpu",
    project = "runs/detect",
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



# ---------------------------------------------------------------------------
# Projeto 3 — Detecção de Máscaras Faciais (Fine-tuning do YOLO11n)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo pré-treinado YOLO11n: YOLO("yolo11n.pt")
#      (única exceção à regra de "sem modelos pré-treinados" do processo seletivo)
#   2. Fazer fine-tuning em dataset/data.yaml, em CPU (device="cpu"),
#      com um número de épocas modesto (ex: 15-30)
#   3. Copiar os pesos resultantes (results.save_dir / "weights" / "best.pt")
#      para "model.pt", na raiz desta pasta
# ---------------------------------------------------------------------------

# insira seu código aqui

# Dica de estrutura (não é obrigatório seguir exatamente assim):
#
# model = YOLO("yolo11n.pt")
# results = model.train(
#     data="dataset/data.yaml",
#     epochs=...,
#     imgsz=...,
#     batch=...,
#     device="cpu",
# )
# shutil.copy(results.save_dir / "weights" / "best.pt", "model.pt")
