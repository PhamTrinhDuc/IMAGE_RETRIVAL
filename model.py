from lib import *

def get_model(mode):
    model = None
    processor = None
    try: 
        if mode == "ViT":
            checkpoint = "google/vit-base-patch16-224"
            processor = ViTImageProcessor.from_pretrained(checkpoint)
            model = ViTForImageClassification.from_pretrained(checkpoint)
        elif mode == "CLIP":
            checkpoint = "openai/clip-vit-base-patch32"
            model = CLIPModel.from_pretrained(checkpoint)
            processor = CLIPProcessor.from_pretrained(checkpoint)
    except: 
        print("Model Invalid")
    
    return model, processor
