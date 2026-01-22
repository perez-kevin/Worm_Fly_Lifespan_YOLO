from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# 1. Download the weights from Hugging Face
weights_path = hf_hub_download(
    repo_id="kprz/Worms_Fly_Lifespan_YOLO", 
    # filename="Worm_YOLO_Weights.pt"
    filename="Fly_YOLO_Weights.pt"
)

# 2. Load the model using the local path returned by the download
model = YOLO(weights_path)

path = 'Images/'

# 2. Run detection on an image
# results = model(path + "Worms/2.jpg")
results = model(path + "Flies/1.jpg")

# 3. Process results
for result in results:
    result.show()        # Displays the image with boxes
    result.save(filename="result.jpg")  # Saves the image
    
    # Optional: Print raw data (coordinates, classes, confidence)
    print(result.boxes.data)