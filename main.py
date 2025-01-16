import os
from os import path
import pandas as pd
from huggingface_hub import login
from transformers import pipeline
from sklearn.model_selection import train_test_split
import keras
from env import SkinCancerEnv 
import torch
from tensorflow.keras import utils
from tensorflow.keras import models
from dataset import build_training_ds
import numpy as np

def llm(prompt, pipe, stop=["\n"]):
    response = pipe(
            prompt,
            max_new_tokens=200,
            truncation=True,
            do_sample=True,
            top_p=1,
            return_full_text=False,
            pad_token_id=pipe.tokenizer.eos_token_id
            )
    output = response[0]["generated_text"]

    for stop_token in stop:
        text_output = output.split(stop_token)[0]

    return text_output

def think(env, idx, prompt, pat_id, pipe):
    question = env.reset()

    print(idx, question)

    prompt += question
    prompt += "\n"
    n_calls = 0
    n_badcalls = 0

    for i in range(1, 8):
        n_calls += 1
        thought_action = llm(prompt + f"Thought: {i}:", pipe, stop=[f"\nObservation {i}:"])


        try:
            thought, action = thought_action.strip().split("f\nAction{i}")
        except:
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", pipe, stop=[f"\n"]).strip()
            #action = llm(prompt + f"Action {i}: ", pipe, stop=[f"\n"]).strip()

        obs, done, info = env.step(pat_id, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')

        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str

        print(step_str)

        if done:
            break

    if not done:
        obs, done, info = env.step(pat_id, "finish[]")

    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return info

def main():
    login()

    pipe = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-1B",
            torch_dtype=torch.bfloat16,
            device_map="auto")

    pipe("The key to life is")

    env = SkinCancerEnv()

    env.nn = keras.saving.load_model("best_model.keras")

    env.loadPatientData("test-metadata.csv", "test-image.hdf5")


    instruction = """
    Categorize a skin lesion as benign or malignant utilizing interleaving Thought, Action, and Observation steps. Thought can reason about the current situation, and Action can be one of three types:

(1) retrieve[patient], which collects patient data

(2) process[data], which provides the patient data to your internal neural network and allows you to determine whether or not the patient is a progressor or a non-progressor

(3) finish[answer], which returns the answer and diagnoses the patient. The action "finish[benign]" will categorize a given skin lesion as benign; while the action "finish[malignant]" will diagnose a given skin lesion as malignant.

Here are some examples.
    """

    examples = "Question: Predict if a skin lesion located on patient IP_1234567 is malignant or benign.\nThought 1: Patient ID = IP_1234567. I will check the patient's images using my neural network.\nAction 1: process[IP_1234567]\nObservation 1: result is 0.87.\nThought 2: The result is 0.87, which is higher than 0.5. A result higher than 0.5 indicates that the identified skin lesion is likely to be malignant. Therefore, the skin lesion is likely malignant. I will submit this as my answer.\nAction 2: [malignant]\nQuestion: Predict if a skin lesion located on patient IP_7654321 is malignant or benign.\nPatient ID = IP_7654321. I will check the patient's images using my neural network.\nAction 1: process[IP_7654321]\nObservation 1: result is 0.43.\nThought 2: The result is 0.43, which is less than 0.5. A result less than than 0.5 indicates the skin lesion is likely to be benign. Therefore, the skin lesion is likely benign. I will submit this as my answer.\nAction 2[benign]\n"

    question = "Predict if a skin lesion located on patient IP_6074337 is malignant or benign"

    env.initial_obs = question    

    prompt = instruction + examples

    think(env, 1, prompt, 0, pipe)

    question = "Predict if a skin lesion located on patient IP_6074337 is malignant or benign"
    env.initial_obs = question
    think(env, 1, prompt, 0, pipe)

if __name__ == "__main__":
    main()
