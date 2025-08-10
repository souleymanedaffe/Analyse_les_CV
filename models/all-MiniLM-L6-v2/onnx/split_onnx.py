import os

def split_file(file_path, part_size_mb=25):
    part_size = part_size_mb * 1024 * 1024  # en octets
    file_size = os.path.getsize(file_path)
    
    with open(file_path, "rb") as f:
        part_num = 1
        while chunk := f.read(part_size):
            part_filename = f"{file_path}.part{part_num}"
            with open(part_filename, "wb") as part_file:
                part_file.write(chunk)
            print(f"Créé : {part_filename} ({os.path.getsize(part_filename)/1024/1024:.2f} Mo)")
            part_num += 1

    print(f"Découpage terminé : {file_path} -> {part_num - 1} parties")

# Liste des fichiers à découper
files = [
    "model_O2.onnx",
    "model_O2.onnx.bak",
    "model_O3.onnx",
    "model_O3.onnx.bak"
]

for file in files:
    if os.path.exists(file):
        split_file(file, part_size_mb=25)
    else:
        print(f"⚠️ Fichier introuvable : {file}")
