import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import re
import copy
from copy import deepcopy
import os
# This is for using the locally installed repo clone when using slurm
import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig, BlipProcessor, BlipForQuestionAnswering, AutoProcessor
import json

#cuda device
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set the CUDA device ID
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor_blip2 = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model_blip2 = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", quantization_config=quantization_config, device_map={"": 0}, torch_dtype=torch.float16)

model_vqa = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor_vqa = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

prompt_answer = []

img_path = "/tmp2/danzel/GR-MG/static_rgb_0_0.png"
raw_image = Image.open(img_path)
# Format: (left, upper, right, lower)
blue_part_bbox = (20, 30, 150, 130)  # Adjust as per the actual coordinates
crop_image = raw_image.crop(blue_part_bbox)
crop_image.save("/tmp2/danzel/GR-MG/cropped_image.png")

crop_path = "/tmp2/danzel/GR-MG/cropped_image.png"
raw_image = Image.open(crop_path).convert("RGB")



# question = lang_annotation + "? Answer:"
# question = "which direction is the white arm with blue block? Answer: " 

prompt_answer = []

question_formats = [
    ("Is white arm on the left of blue object?", "The arm is on the [direction] of the blue block."),
    ("Is white object on the blue object?", "Yes/No"),
]

for question, answer in question_formats:
    prompt = f"{question} Answer:"
    # Generate answer with blip2
    # inputs = processor_blip2(raw_image, prompt, return_tensors="pt").to(device, torch.float16)
    # out = model_blip2.generate(**inputs)
    # generated_text = processor_blip2.batch_decode(out, skip_special_tokens=True)[0].strip()

    inputs = processor_vqa(images=raw_image, text=question, return_tensors="pt")
    outputs = model_vqa.generate(**inputs)
    generated_text = processor_vqa.decode(outputs[0], skip_special_tokens=True)
    print(processor_vqa.decode(outputs[0], skip_special_tokens=True))

    print(generated_text)
    answer = generated_text.split("Answer:")[-1].strip()
    print(f"Question: {question}\nAnswer: {answer}")

    if "yes" in answer:
        prompt_temp = question.replace("Is ", "", 1)
        prompt_temp = prompt_temp.replace("?", "").strip()

        prompt_answer.append(prompt_temp)
    elif "no" in answer:
        prompt_temp = question.replace("Is ", "", 1)
        prompt_temp = prompt_temp.replace("?", "").strip()
        prompt_temp = prompt_temp.replace("on", "not on", 1)
        prompt_answer.append(prompt_temp)   

print(prompt_answer)