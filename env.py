import pandas as pd
import gymnasium as gym
import os
from os import path
import h5py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from dataset import build_dataset, feature_space, FEAT_COLS
import keras
import PIL
import numpy as np

class SkinCancerEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.patient = None     # Patient ID
        self.obs = None         # Current observation
        self.steps = 0
        self.answer = None
        self.patientData = None
        self.patientIDs = None
        self.observation_space = self.action_space = None
        self.nn = None
        self.data_handler = None
        self.initial_obs = None
        self.batch_size = 128

    def getObs(self):
        return self.obs

    def getInfo(self):
        return {"steps": self.steps, "answer": self.answer}

    def reset(self, return_info=False):
        self.obs = self.initial_obs
        self.steps = 0
        self.answer = None
        observation = self.getObs()
        info = self.getInfo()
        if (return_info):
            return observation, info
        else:
            return observation

    def loadPatientData(self, metadata, images):
        images = h5py.File(f"{images}", 'r')
        
        df = pd.read_csv(f"{metadata}")
        features = dict(df[FEAT_COLS])
        df = df.ffill()
        self.patientIDs = df.isic_id.values
        #dataset = build_dataset(self.patientIDs, images, features, batch_size=self.batch_size, shuffle=False, augment=False, cache=False)

        #self.patientData = dataset.map(lambda x: {"images": x["images"], "features": feature_space(x["features"])}, num_parallel_calls=tf.data.AUTOTUNE)

    def process(self, entity):
        #if entity not in patientIDs:
        #    self.obs = f"Invalid ID: {entity} is not a known patient ID"
        #    return

        #index = self.patientIDs.index(entity)
        #item = self.patientData.skip(index).take(1)
        image = load_img("test.jpg", target_size=(128, 128))
        item = img_to_array(image) / 255.0
        item = np.expand_dims(item, axis=0)
        preds = self.nn.predict(item).squeeze()

        self.obs = f"the result is {preds}"

    def get_accuracy(self):
        self.obs = "The user has been provided with the neural network's accuracy."

    def get_false_pos_rate(self):
        self.obs = "The user has been provided with the neural network's false positive rate."

    def get_pred_features(self):
        print(f"Agent Output: The model predictive features are: {FEAT_COLS}")
        self.obs = "The user has been provided with the neural network's predictive features."

    def visualize_decision_making(self):
        self.obs = "The user has been provided with a visualization of the neural network's decision making."

    def about(self):
        print("Agent Output: The model decides if a lesion is cancerous by analyzing both image data (lesion patterns, textures, and colors) and tabular features (patient demographics or clinical metrics) through deep learnig layers. It outputs a probability score that can be thresholded (e.g., 0.5), to classify a lesion as cancerous or not.")

        self.nn.summary()

        self.obs = "The user has been provided with information about how the model functions"


    def step(self, patientID, action):
        done = False

        if self.answer is not None:
            done = True
            return self.obs, done, self.getInfo()

        if action.startswith("retrieve[") and action.endswith("]"):
            entity = action[len("retrieve["):-1]
            self.searchStep(patientID)

        elif action.startswith("process[") and action.endswith("]"):
            entity = action[len("process["):-1]
            self.process(entity)

        elif action.startswith("get_accuracy[") and action.endswith("]"):
            self.get_accuracy()
            self.obs = "The user has been provided with the neural network's accuracy."

        elif action.startswith("false_pos_rate[") and action.endswith("]"):
            self.get_false_pos_rate()
            self.obs = "The user has been provided with the neural network's false positive rate."

        elif action.startswith("pred_features[") and action.endswith("]"):
            self.get_pred_features()
            self.obs = "The user has been provided with the neural network's predictive features."

        elif action.startswith("visualize_decision_making[") and action.endswith("]"):
            self.visualize_decision_making()
            self.obs = "The user has been provided with a visualization of the neural network's decision making."

        elif action.startswith("diagnose[") and action.endswith("]"):
            diagnosis = action[len("diagnose["):-1]
            print(f"My diagnosis is {diagnosis}")

        elif action.startswith("about[") and action.endswith("]"):
            self.about()

        elif action.startswith("finish[") and action.endswith("]"):
            answer = action[len("finish["):-1]
            self.answer = answer
            done = True
            self.obs = f"Episode finished"

        elif action.startswith("think[") and action.endswith("]"):
            thought = action[len("think["):-1]
            self.obs = thought
            
        else:
            self.obs = f"invalid action: {action}"

        self.steps += 1

        return self.obs, done, self.getInfo()
