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

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def expand_bbox(person_box, object_boxes, threshold=10):
    """ 
    Expand BBox of people if there is a object nearby
    - person_box: Original bounding box of human
    - object_boxes: List bounding box of other objects
    """
    px1, py1, px2, py2 = person_box
    expanded = False

    for (ox1, oy1, ox2, oy2, obj_id) in object_boxes:
        if obj_id != 0:  # Not humans
            if (ox2 > px1 - threshold or ox1 < px2 + threshold or 
                oy2 > py1 - threshold or oy1 < py2 + threshold):  # Check distance
                px1 = min(px1, ox1)
                py1 = min(py1, oy1)
                px2 = max(px2, ox2)
                py2 = max(py2, oy2)
                expanded = True

    return (px1, py1, px2, py2) if expanded else person_box  # Only expend if has object nearby
def predict_action(person_crop):
    """ Predict activity form cropped image """
    image = transform(person_crop).unsqueeze(0)
    with torch.no_grad():
        output = har_model(image)
        predicted_class = output.argmax(dim=1).item()
    return class_names[predicted_class]

def process_image(image_path):
    """ Detect human, expand bbox (if there is an object nearby) and make prediction """
    image = Image.open(image_path).convert("RGB")
    results = object_detector(image)
    os.makedirs("Output1", exist_ok=True)
    os.makedirs("Output2", exist_ok=True)

    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()

    people_boxes = []  
    object_boxes = []  

    # Check object == people or not
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])  # Class ID of object
            if class_id == 0:  # ID = 0 is humans un YOLO
                people_boxes.append((x1, y1, x2, y2))
            else:
                object_boxes.append((x1, y1, x2, y2, class_id))

    action_results = []  

    image1 = image.copy()
    draw1 = ImageDraw.Draw(image1)

    for person_box in people_boxes:
        expanded_box = expand_bbox(person_box, object_boxes)
        x1, y1, x2, y2 = expanded_box

        cropped_person = image.crop((x1, y1, x2, y2))
        action = predict_action(cropped_person)

        # Save infor (ori bbox, expand edbbox, action)
        action_results.append((person_box, expanded_box, action))

        # BBox for humans (expanded) with red color (Output1)
        draw1.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        draw1.text((x1, y1 - 30), action, font=font, fill="red")

    # BBox for other objects that are not humans (No predict) (Output1)
    for (ox1, oy1, ox2, oy2, obj_id) in object_boxes:
        draw1.rectangle([(ox1, oy1), (ox2, oy2)], outline="blue", width=2)

    # Save expanded Bbox
    output_path1 = os.path.join("Output1", "output_" + os.path.basename(image_path))
    image1.save(output_path1)
    print(f"Processed image saved as {output_path1}")

    # =======================
    # DRAW IMAGE 2 - ONLY DRAW THE ORIGINAL BBOX BUT KEEP THE CORRECT ACTION FOR EACH PERSON
    # =======================
    image2 = image.copy()
    draw2 = ImageDraw.Draw(image2)

    for (original_box, expanded_box, action) in action_results:
        x1, y1, x2, y2 = original_box  # Giữ nguyên bbox ban đầu

        # BBox for humans (ori) with red color
        draw2.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        draw2.text((x1, y1 - 30), action, font=font, fill="red")

    # BBox for other objects that are not humans (No predict) (Output2)
    for (ox1, oy1, ox2, oy2, obj_id) in object_boxes:
        draw2.rectangle([(ox1, oy1), (ox2, oy2)], outline="blue", width=2)

    # Save
    output_path2 = os.path.join("Output2", "output_" + os.path.basename(image_path))
    image2.save(output_path2)
    print(f"Processed image saved as {output_path2}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="File name of 'all' for running all the test set")
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
