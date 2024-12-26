# grounding DINO

import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from PIL import Image, ImageOps, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


def object_detection(image_original, text):
    image_crop = image_original
    image_crop = image_original.crop((0, 0, 512, 320))
    image_crop.save("crop.png")
    
    inputs = processor(images=image_crop, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image_crop.size[::-1]]
    )
    # Filter results to keep only the highest score for each label
    filtered_results = {"boxes": [], "scores": [], "labels": []}
    label_score_map = {}

    for box, score, label in zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"]):
        if label not in label_score_map or score > label_score_map[label]:
            label_score_map[label] = score
            filtered_results["boxes"].append(box)
            filtered_results["scores"].append(score)
            filtered_results["labels"].append(label)

    results[0] = filtered_results


    # print(results)
    draw = ImageDraw.Draw(image_original)
    for box, score in zip(results[0]["boxes"], results[0]["scores"]):
        draw.rectangle(box.tolist(), outline="red", width=3)
        draw.text((box[0], box[1]-10), f"{score:.2f}", fill="black")
    
    return image_original, results

# image_detection, results = object_detection(Image.open("./0-0.png"), "a white robot arm. a blue object.")
# image_detection.save("./output_scene.png")