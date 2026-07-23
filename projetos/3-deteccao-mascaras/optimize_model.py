from ultralytics import YOLO
import shutil
import os
# ---------------------------------------------------------------------------
# Projeto 3 — Otimização do Modelo (Exportação para Edge)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.pt"
#   2. Exportar para TensorFlow Lite via model.export(format="tflite")
#      (a Ultralytics gera automaticamente "model.tflite" na mesma pasta)
# ---------------------------------------------------------------------------
if not os.path.exists("model.pt"):
    print(f"Erro, arquivo model.pt nao encontrado")
    exit()

model = YOLO("model.pt")

try:
    model.export(format= "tflite")
    tflite_16 = "model_saved_model/model_float16.tflite"
    tflite_32 = "model_saved_model/model_float32.tflite"
    model_path = "model.tflite"

    if os.path.exists(tflite_16):
        os.rename(tflite_path, model_path)
        shutil.rmtree("model_saved_model")
    elif os.path.exists(tflite_32):
        os.rename(tflite_32, model_path)
        shutil.rmtree("model_saved_model")
    elif not os.path.exists(model_path):
        print("Atenção: A exportação terminou, mas o arquivo 'model.tflite' não foi encontrado na raiz.")

    if os.path.exists(model_path):
        print("O modelo otimizado foi salvo como 'model.tflite' na raiz do projeto.")

except Exception as e:
    print(f"Ocorreu um erro durante a exportação: {e}")


# Dica de estrutura (não é obrigatório seguir exatamente assim):
#
# model = YOLO("model.pt")
# model.export(format="tflite", imgsz=...)
