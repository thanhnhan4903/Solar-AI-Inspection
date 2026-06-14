import os
from pathlib import Path

def find_files(name):
    print(f"Searching for '{name}'...")
    for root, dirs, files in os.walk("."):
        for file in files:
            if name in file:
                print(os.path.join(root, file))

find_files("DJI_0959")
