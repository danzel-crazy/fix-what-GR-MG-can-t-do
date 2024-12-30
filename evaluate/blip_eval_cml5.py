# MIT License

# Copyright (c) 2021 Oier Mees
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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

repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))
print(sys.path)

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition
)
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm
from utils.utils import print_and_save
from wrapper.model_wrapper import CustomModel
from goal_gen.evaluate import IP2PEvaluation
    
logger = logging.getLogger(__name__)
EP_LEN = 360
NUM_SEQUENCES = 1000
SAVE_DIR = None
FAIL_COUNTER=0

#cuda device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# processor_blip2 = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
# model_blip2 = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b", quantization_config=quantization_config, device_map={"": 0}, torch_dtype=torch.float16)  # doctest: +IGNORE_RESULT

#VQA
model_vqa = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor_vqa = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")


def make_env(dataset_path, observation_space, device_id):
    val_folder = Path(dataset_path) / "validation"
    # insert your own env wrapper
    from wrapper.calvin_env_wrapper_raw import CalvinEnvWrapperRaw
    device = torch.device('cuda', device_id)
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env


def evaluate_policy(model, env, eval_sr_path, eval_result_path, ip2p_model):
    """Run this function to evaluate a model on the CALVIN challenge."""
    conf_dir = Path("./calvin/calvin_models/conf")
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    eval_sequences = get_sequences(NUM_SEQUENCES)
    results = []
    sequence_i = 0

    check_result_path = "results.txt"
    file = open(check_result_path, 'a')
    with open(check_result_path, 'w') as f:
        f.write("Evaluation Results\n")

    for index,(initial_state, eval_sequence) in enumerate(eval_sequences):
        with open(check_result_path, 'a') as f:
            f.write(f"eval_sequences_index: {index}\n")
        result= evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, sequence_i,ip2p_model)
        with open(check_result_path, 'a') as f:
            f.write(f"Result: {result}\n")
        results.append(result)
        success_list = count_success(results)
        with open(eval_sr_path, 'a') as f:
            line =f"{sequence_i}/{NUM_SEQUENCES}: "
            for sr in success_list:
                line += f"{sr:.3f} | "
            sequence_i += 1
            line += "\n"
            f.write(line)
        file.write("\n")
        if index%100==0 and index!=0: #save every 100 sequences
            print_and_save(results, eval_sequences[:index+1], eval_result_path[:-5]+f"_{index+1}"+".json", None)
    print_and_save(results, eval_sequences, eval_result_path, None)
    
    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, sequence_i,ip2p_model):
    """Evaluates a sequence of language instructions."""
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    for subtask_i, subtask in enumerate(eval_sequence):
        if subtask == "push_blue_block_right" or subtask == "push_red_block_right" or subtask == "push_pink_block_right" or subtask == "push_blue_block_down":
            print(subtask)
            success = rollout(env, model, task_checker, subtask, val_annotations, subtask_i, sequence_i,ip2p_model)
            if success:
                success_counter += 1
            else:
                return success_counter
        else:
            return success_counter 
    return success_counter

def blip_prompt(img, lang_annotation, img_path, progess_state, subtask):
    prompt_answer = []

    print(lang_annotation)
    print(img_path)

    blue_part_bbox = (20, 30, 150, 130)  # Adjust as per the actual coordinates
    crop_image = img.crop(blue_part_bbox)

    # raw_image = Image.open(img_path).convert("RGB")
    raw_image = crop_image.convert("RGB")
    
    if subtask == "push_blue_block_right":
        question_formats = [
            ("Is white arm on the left of blue object?", "The arm is on the [direction] of the blue block."),
            ("Is white object on the blue object?", "Yes/No"),
        ]
    elif subtask == "push_red_block_right":
        question_formats = [
            ("Is white arm on the left of red object?", "The arm is on the [direction] of the red block."),
            ("Is white object on the red object?", "Yes/No"),
        ]
    elif subtask == "push_pink_block_right":
        question_formats = [
            ("Is white arm on the left of pink object?", "The arm is on the [direction] of the pink block."),
            ("Is white object on the pink object?", "Yes/No"),
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
    
    
    print(f"Prompt: {prompt_answer}")
    return prompt_answer

def rollout(env, model, task_oracle, subtask, val_annotations, subtask_i, sequence_i,ip2p_model):
    """Run the actual rollout on one subtask."""
    """Add blip2 to give more information about image movement"""
    
    obs = env.get_obs()
    # print(obs)

    print("Dictionary has been written to 'output.json'.")
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()
    debug_image=[]
    progress=0
    progess_state = True

    for i in range(EP_LEN):
        if i % 20 == 0:  # hardcode
            static_rgb = obs['rgb_obs']['rgb_static'] # (200, 200, 3)
            print(static_rgb.shape)
            #save static_rgb as image
            img = Image.fromarray(static_rgb)
            folder_path = f"static_images"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            path = f"static_rgb_{sequence_i}_{subtask_i}_{lang_annotation}.png"
            file_path = os.path.join(folder_path, path)
            img.save(file_path)
           
            hand_rgb = obs['rgb_obs']['rgb_gripper']
            image_patch=[static_rgb]
            text_patch=[lang_annotation + f".And {progress}% of the instruction has been finished."]
            print(text_patch)
            
            if progess_state: 
                text_prompt = blip_prompt(img, lang_annotation, file_path, progess_state, subtask)
                print(f'VQA_Prompt: {text_prompt}')
                for i in range(len(text_prompt)):
                    text_patch[0] = text_patch[0] + text_prompt[i] + ". "
            
            print(text_patch[0])
            goal_image=ip2p_model.inference(image_patch,text_patch)
            temp_image=[static_rgb,goal_image[0],hand_rgb]
            debug_image.append(temp_image)

        action,progress = model.step(obs,deepcopy(goal_image),[lang_annotation]) 
        obs, _, _, current_info = env.step(action)

        # if progress <= 40:
        #     progess_state = True
        # else:
        #     progess_state = False

        # # Save dictionary to a text file
        # output_file = "output.txt"
        # with open(output_file, "w") as f:
        #     for key, value in current_info.items():
        #         f.write(f"{key}: {value}\n")
        

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        # # Save current_task_info to a text file
        # with open(output_file, "a") as f:
        #     f.write(f"Step {i} - current_task_info: str{current_task_info}\n")
        if len(current_task_info) > 0:
            print("success!")
            return True
    print("fail!")
    
    global FAIL_COUNTER
    FAIL_COUNTER+=1
    if FAIL_COUNTER % 30 ==0:  # save every 30 failure cases
        length=len(debug_image) 
        fig, ax = plt.subplots(length, 2,figsize=(5.5, 46.58))
        for ax_ in ax.flat:
            # ax_.plot([1, 2, 3], [4, 5, 6])
            ax_.axis('off')  # 隐藏每个子图的刻度和边框
        for i in range(length):
            ax[i][0].imshow(debug_image[i][0])
            ax[i][1].imshow(debug_image[i][1])
            # ax[i][2].imshow(debug_image[i][2])
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(os.path.join(SAVE_DIR, f"{sequence_i}-{subtask_i}-{subtask}.png"),dpi=100)
        plt.close()
    return False


def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", default='/tmp2/danzel/GR-MG/calvin/dataset/calvin_debug_dataset',
                        type=str, help="Path to the dataset root directory.")  # modify it before opensource
    # evaluation
    parser.add_argument('--config_path', type=str, default="", help='path to the policy config file')
    parser.add_argument('--ckpt_dir', type=str, default="",help="path to the policy ckpt file")
    parser.add_argument('--epoch', type=int,default=41, help="epoch index for evaluating")
    parser.add_argument('--device_id', default=0, type=int, help="CUDA device")
    parser.add_argument('--ip2p_ckpt_path', default="", type=str, help="ip2p ckpt path")
    args = parser.parse_args()
    config_path = args.config_path
    ckpt_dir = args.ckpt_dir
    epoch = args.epoch
    device_id = args.device_id
    ip2p_ckpt_path=args.ip2p_ckpt_path
    assert config_path != None
    # Load config file
    with open(config_path, 'r') as f:
        configs = json.load(f)
                
    # Get checkpoint path
    ckpt_path = None
    ckpt_files = os.listdir(ckpt_dir)
    for ckpt_file in ckpt_files:
        match = re.search(r'epoch=(\d+)', ckpt_file)
        if match:
            temp_epoch = int(match.group(1))
            if temp_epoch == epoch:
                ckpt_path = os.path.join(ckpt_dir, ckpt_file)
                break

    device = torch.device('cuda', device_id)
    model = CustomModel(
        ckpt_path=ckpt_path,
        configs=configs,
        device=device)
    observation_space = {
        'rgb_obs': ['rgb_static', 'rgb_gripper'], 
        'depth_obs': [], 
        'state_obs': ['robot_obs'], 
        'actions': ['rel_actions'], 
        'language': ['language']} 
    env = make_env(args.dataset_path, observation_space, device_id) 
    print("Start evaluating policy...")
    # Success rate and result files
    # flag="opensourcesd"
    flag="blip2_debug_push_blue_block_right"
    sub_dir=f"{flag}_{epoch}_epoch"
    # set a global variable
    global SAVE_DIR
    SAVE_DIR=os.path.join(ckpt_dir,sub_dir)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    sr_path = os.path.join(SAVE_DIR, f"success_rate.txt")
    result_path = os.path.join(SAVE_DIR, f"results.json")
    ip2p_model=IP2PEvaluation(ip2p_ckpt_path)

    # model_qa, vis_processors_qa, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)
    print("Start evaluating policy...")

    evaluate_policy(
        model, 
        env,
        eval_sr_path=sr_path,
        eval_result_path=result_path,
        ip2p_model=ip2p_model)
if __name__ == "__main__":
    main()