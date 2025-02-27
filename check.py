import os

folder_path = r"D:\semester 5\CO542-2024 Neural Networks and Fuzzy Systems\project\data\by_merge"

print("Extracted files and directories:")
for root, dirs, files in os.walk(folder_path):
    for name in files:
        print(os.path.join(root, name))
