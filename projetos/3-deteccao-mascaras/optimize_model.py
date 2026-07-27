from ultralytics import YOLO
import shutil
import os

if not os.path.exists("model.pt"):
    print(f"Erro, arquivo model.pt nao encontrado")
    exit()

model = YOLO("model.pt")

try:
    model.export(format= "tflite", int8 = True, data = "dataset/data.yaml")
    tflite_int8 = "model_int8.tflite"
    model_path = "model.tflite"

    if os.path.exists(tflite_int8):
        
        if os.path.exists(model_path):
            os.remove(model_path)
            
        os.rename(tflite_int8, model_path)
        print("O modelo otimizado foi salvo como 'model.tflite' na raiz do projeto.")
    elif not os.path.exists(model_path):
        print("Atenção: A exportação terminou, mas o arquivo 'model.tflite' não foi encontrado.")

    if os.path.exists(model_path):
        print("O modelo otimizado foi salvo como 'model.tflite' na raiz do projeto.")

except Exception as e:
    print(f"Ocorreu um erro durante a exportação: {e}")

