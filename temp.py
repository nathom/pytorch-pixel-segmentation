import os

for file in os.listdir(os.getcwd() + "/models"):
    if file.endswith(".pkl"):
        print(file)
