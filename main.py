import sys
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
    if len(sys.argv) > 1:
        if "-l" in sys.argv:
            login()

    pipe = pipeline(
            "text-generation",
            #model="meta-llama/Llama-3.2-1B",
            model="deepseek-ai/DeepSeek-V3",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True)

    pipe("The key to life is")

    env = SkinCancerEnv()

    env.nn = keras.saving.load_model("best_model.keras")

    env.loadPatientData("test-metadata.csv", "test-image.hdf5")


    instruction = """
    You are an AI assistant in charge of help a medical practitioner diagnose skin lesions as either malignant or benign utilizing interleaving Thought, Action, and Observation steps. Thought can reason about the current situation, and Action can be one of ten types:

(1) retrieve[patient], which collects patient data.

(2) process[data], which provides the patient data to your internal neural network and allows you to determine whether or not the patient is a progressor or a non-progressor.

(3) diagnose[answer], which returns the answer and diagnoses the patient. The action "finish[benign]" will categorize a given skin lesion as benign; while the action "finish[malignant]" will diagnose a given skin lesion as malignant.

(4) false_pos_rate[], which returns the false positive rate of the model and provides it to the user.

(5) get_accuracy[], which returns the model's accuracy and provides it to the user.

(6) pred_features[], which returns the predictive features of the model and provides it to the user.

(7) visualize_decision_making[], which returns a visualization of the model and provides it to the user.

(8) about[], provide the user with information about the model

(9) think[thought], give yourself more time to think.

(10) finish[], which ends the current episode. 

Here are some examples.
    """

    # Diagnose examples
    examples = "Question: Predict if a skin lesion located on patient IP_1234567 is malignant or benign.\nThought 1: Patient ID = IP_1234567. I will check the patient's images using my neural network.\nAction 1: process[IP_1234567]\nObservation 1: result is 0.87.\nThought 2: The result is 0.87, which is higher than 0.5. A result higher than 0.5 indicates that the identified skin lesion is likely to be malignant. Therefore, the skin lesion is likely malignant. I will submit this as my answer.\nAction 2: diagnose[malignant]\nThought 3: The user asked for a diagnosis, which I provided in Action 2. I have answered the user's question and may finish.\nAction 3: finish[]\nQuestion: What is the probability that the skin lesion on patient IP_7654321 is cancerous?\nThought 1: cancerous means malignant.\nAction 1: think[]\nThought 2: Patient ID = IP_7654321. I will check the patient's images using my neural network.\nAction 2: process[IP_7654321]\nObservation 2: result is 0.43.\nThought 3: The result is 0.43, which is less than 0.5. A result less than than 0.5 indicates the skin lesion is likely to be benign. Therefore, the skin lesion is likely benign. I will submit this as my answer.\nAction 3[benign]\nObservation 3: The user has been provided with the diagnosis.\nThought 3: The user asked for a diagnosis, which I provided in action 3. I have answered the user's question and may finish the episode.\nAction 3: finish[]\n"

    # Get accuracy examples
    examples += "Question: How accurate is this model in predicting skin cancer?\nThought 1: the user is requesting information about the neural network. I need to provide them with the neural network's accuracy.\nAction 1: get_accuracy[]\nObservation 1: The user has been provided with the neural network's accuracy.\nThought 2: The user asked for the model accuracy, which I provided in Action 1. I answered the user's question and may finish the episode.\nAction 2: finish[]\nQuestion: What is your accuracy?\nThought 1: the user is requesting information about the neural network. I need to provide them with the neural network's accuracy.\nAction 1: get_accuracy[]\nObservation 1: The user has been provided with the neural network's accuracy.\nThought 2: The user asked for the model's accuracy, which I provided in Action 1. I answered the user's question and may finish the episode.\nAction 2: finish[]\n"

    # False positive rate examples
    examples += "Question: What is the model's false positive rate for benign lesions?\nThought 1: the user is requesting information about the neural network's false positive rate.\nAction 1: false_pos_rate[]\nObservation 1: The user has been provided with the neural network's false positive rate.\nThought 2: The user asked for the model's false positive rate, which I provided in Action 1. I answered the user's question and may finish the episode.\nAction 2: finish[]\nQuestion: what is the false positive rate?\nThought 1: the user is requesting information about the neural network's false positive rate.\nAction 1: false_positive_rate[]\nObservation 1: the user has been provided with the model's false positive rate.\nThought 2: The user asked for the model's false positive rate, which I provided in Action 1. I answered the user's question and may finish the episode.\nAction 2: finish[]\n"

    # Predictive features examples
    examples += "Question: What predictive features does the model use in its analysis?\nThought 1: the user is requesting information about the neural network's predictive features.\nAction 1: pred_features[]\nObservation 1: the user has been provided with the neural network's predictive features.\nThought 2: The user asked for information on predictive features, which I provided in Action 1. I have answered the user's question and may finish the episode.\nAction 2: finish[]\nQuestion: What do you look at to know if a lesion is malign?\nThought 1: the user is requesting information about the neural network's predictive features.\nAction 1: pred_features[]\nObservation 1: the user has been provided with the neural network's predictive features.\nThought 2: The user asked for information on predictive features, which I provided in Action 1. I have answered the user's question and may finish the episode.\nAction 2: finish[]\n"

    # About examples 
    examples += "Question: How does the model decide if a lesion is cancerous?\nThought 1: the user is requesting information about how the model functions.\nAction 1: about[]\nObservation 1: The user has been provided with information about how the model functions.\nThought 2: The user asked for information about the model, which I provided in Action 1. The user's question has been answered and I may finish the episode.\nAction 2: finish[]\nQuestion: how does the model work?\nThought 1: the user is requesting information about how the model functions.\nAction 1: about[]\nObservation 1: The user has been provided with information about how the model functions.\nThought 2: the user's question has been answered and I may finish the episode.\nAction 2: finish[]\nQuestion: how do you diagnose patients?\nThought 1: the user is requesting information about how the model functions.\nAction 1: about[]\nObservation 1: The user has been provided with information about how the model functions.\nThought 2: The user asked for information about the model, which I provided in Action 1. The user's question has been answered and I may finish the episode.\nAction 2: finish[]\nQuestion: how do you work?\nThought 1: the user is requesting information about how the model functions.\nAction 1: about[]\nObservation 1: The user has been provided with information about how the model functions.\nThought 2: The user asked for information about the model, which I provided in Action 1. The user's question has been answered and I may finish the episode.\nAction 2: finish[]\nQuestion: how do you know if a lesion is cancerous or not?\nThought 1: the user is requesting information about how the model functions.\nAction 1: about[]\nObservation 1: The user has been provided with information about how the model functions.\nThought 2: The user asked for information about the model, which I provided in Action 1. The user's question has been answered and I may finish the episode.\nAction 2: finish[]\n"

    # Visualize examples
    examples += "Question: Visualize which parts of the image influenced the model's decision.\nThought 1: The user is asking for a visualization of the neural network's image processing.\nAction 1: visualize_decision_making[]\nObservation 1: the user has been provided with a visualization of the neural network's decision making.\nThought 2: The user asked for a visualization of the model, which I provided in Action 1. I have answered the user's question and may finish the episode.\nAction 2: finish[]\nQuestion: can you show me which parts of the image look cancerous?\nThought 1: the user is asking for a visualization of the neural network's image processing.\nAction 1: visualize_decision_making[]\nObservation 1: the user has been provided with a visualization of the neural network's decision making.\nThought 2: The user asked for a visualization of the neural network's decision making, which I provided in Action 1. I have answered the user's question and may finish the episode.\nAction 2: finish[]\n"

    prompt = instruction + examples

    questions = []
    questions.append("How does the model decide if a lesion is cancerous?")
    questions.append("What features does the model use in its analysis?")
    questions.append("Predict if a skin lesion located on patient IP_6074337 is malignant or benign")
    questions.append("What is the model's false positive rate for benign skin lesions?")
    questions.append("Visualize which parts of the image influenced the model's decision.")
    
    for question in questions:
        env.initial_obs = "Question: " + question + "\n"
        think(env, 1, prompt, 0, pipe)

if __name__ == "__main__":
    main()
