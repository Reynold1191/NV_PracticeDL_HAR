import os
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from HARmain import HARResNet50  

object_detector = YOLO("yolov8n.pt") 
class_names = sorted(os.listdir("train"))  
har_model = HARResNet50(len(class_names))
har_model.load_state_dict(torch.load("checkpoints/har_resnet50.pth", map_location=torch.device('cpu')))
har_model.eval()

# Transform input 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_action(person_crop):
    image = transform(person_crop).unsqueeze(0)
    with torch.no_grad():
        output = har_model(image)
        predicted_class = output.argmax(dim=1).item()
    return class_names[predicted_class]

def find_best_text_position(x1, y1, x2, y2, used_positions, img_w, img_h, font, text, draw):
    # Get size of text by textbox
    text_bbox = draw.textbbox((x1, y1), text, font=font)
    text_width = text_bbox[2] - text_bbox[0] 
    text_height = text_bbox[3] - text_bbox[1] 
    
    candidates = [
        (x1 , y1 - text_height - 5),  # UP
        (x1, y2 + 5),  # DOWN
        (x1 - 5, y1),  # LEFT
        (x2 - text_width, y1)  # RIGHT
    ]
    
    for tx, ty in candidates:
        if 0 <= tx < img_w and 0 <= ty < img_h:
            # Check whether it overlap or not
            text_bbox = (tx, ty, tx + text_width, ty + text_height)
            overlap = False
            for pos in used_positions:
                # Check overlap
                if (pos[0] < text_bbox[2] and pos[2] > text_bbox[0] and
                    pos[1] < text_bbox[3] and pos[3] > text_bbox[1]):
                    overlap = True
                    break
            if not overlap:
                used_positions.add((tx, ty, tx + text_width, ty + text_height))
                return tx, ty
    
    # Return default position
    return x1, y1 - text_height - 5

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    results = object_detector(image)
    draw = ImageDraw.Draw(image)

    try:
        base_font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        base_font = ImageFont.load_default()

    objects = []  
    people_boxes = []  
    action_results = {}  

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            objects.append((cls, x1, y1, x2, y2))

    for cls, x1, y1, x2, y2 in objects:
        if cls == 0:  
            expand_ratio = 0.2
            img_w, img_h = image.size

            x1 = max(0, int(x1 - (x2 - x1) * expand_ratio))
            y1 = max(0, int(y1 - (y2 - y1) * expand_ratio))
            x2 = min(img_w, int(x2 + (x2 - x1) * expand_ratio))
            y2 = min(img_h, int(y2 + (y2 - y1) * expand_ratio))

            cropped_person = image.crop((x1, y1, x2, y2))
            action = predict_action(cropped_person)

            people_boxes.append((x1, y1, x2, y2))
            action_results[(x1, y1, x2, y2)] = action

    used_positions = set()
    for (x1, y1, x2, y2) in people_boxes:
        action = action_results[(x1, y1, x2, y2)]
        
        font_size = 30  
        font = ImageFont.truetype("arial.ttf", font_size) if isinstance(base_font, ImageFont.FreeTypeFont) else base_font

        text_x, text_y = find_best_text_position(x1, y1, x2, y2, used_positions, image.size[0], image.size[1], font, action, draw)
        
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        draw.text((text_x, text_y), action, font=font, fill="red")

    output_dir = "Output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "output_" + os.path.basename(image_path))
    image.save(output_path)
    print(f"Processed image saved as {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="File name or 'all' for running all the test set")
    args = parser.parse_args()

    test_dir = "test/"
    if args.image.lower() == "all":
        for img_file in os.listdir(test_dir):
            if img_file.endswith(".jpg"):
                print(f"Processing {img_file}...")
                process_image(os.path.join(test_dir, img_file))
    else:
        image_path = os.path.join(test_dir, args.image + ".jpg")
        if os.path.exists(image_path):
            process_image(image_path)
        else:
            print("File not found!")

if __name__ == "__main__":
    main()
