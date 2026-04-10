import os
import urllib.request

url = "https://raw.githubusercontent.com/tarekjundi10/mental-health-predictor/main/data/raw/Mental_Health_Lifestyle_Dataset.csv"
filepath = "data/raw/Mental_Health_Lifestyle_Dataset.csv"

os.makedirs("data/raw", exist_ok=True)

if not os.path.exists(filepath):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, filepath)
    print("Download complete.")
else:
    print("Dataset already exists.")