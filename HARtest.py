import argparse
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from HARmain import HARResNet50  

def load_model(model_path, num_classes):
    model = HARResNet50(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(model, image_path, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
        predicted_class = output.argmax(dim=1).item()
    
    return class_names[predicted_class]

def main():
    parser = argparse.ArgumentParser(description="Test HAR model on an image or all images in test folder")
    parser.add_argument("filename", type=str, help="Filename without extension (e.g., 'a' for test/a.jpg) or 'all' to test all images")
    parser.add_argument("--model", type=str, default="checkpoints/har_resnet50.pth", help="Path to trained model")
    parser.add_argument("--test_dir", type=str, default="test", help="Path to test images directory")
    parser.add_argument("--train_dir", type=str, default="train", help="Path to training directory (to extract class names)")
    args = parser.parse_args()

    # Get Class from train dir
    class_names = sorted(os.listdir(args.train_dir))
    num_classes = len(class_names)
    
    # Load model
    model = load_model(args.model, num_classes)
    
    if args.filename.lower() == "all":
        # Predict all in test dir
        for img_name in sorted(os.listdir(args.test_dir)):
            img_path = os.path.join(args.test_dir, img_name)
            if img_path.endswith(".jpg") or img_path.endswith(".png"):
                action = predict_image(model, img_path, class_names)
                print(f"{img_name} --> Predicted activity: {action}")
    else:
        # Predict a specific pic
        img_path = os.path.join(args.test_dir, args.filename + ".jpg")
        if not os.path.exists(img_path):
            print(f"Error: file not found {img_path}")
            return
        action = predict_image(model, img_path, class_names)
        print(f"{args.filename}.jpg --> Predicted activity: {action}")

if __name__ == "__main__":
    main()