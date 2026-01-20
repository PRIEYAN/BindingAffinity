import kagglehub

# Download latest version
path = kagglehub.dataset_download("ankerhuang/protein-data")

print("Path to dataset files:", path)