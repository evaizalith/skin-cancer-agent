import pandas as pd
import gymnasium as gym
import os
from os import path
import h5py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from dataset import build_dataset, feature_space
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

    def get_accuracy():
        return

    def get_false_pos_rate():
        return

    def get_pred_features():
        return

    def visualize_decision_making():
        return

    def step(self, patientID, action):
        done = False

        if self.answer is not None:
            done = True
            return self.obs, done, self.getInfo()

        if action.startswith("retrieve[") and action.endswith("]"):
            entity = action[len("retrieve["):-1]
            self.searchStep(patientID)

        elif action.startswith("analyze_image[") and action.endswith("]"):
            entity = action[len("process["):-1]
            self.process(entity)

        elif actions.startswith("get_accuracy[") and action.endswith("]"):
            get_accuracy()

        elif actions.startswith("false_pos_rate[") and action.endswith("]"):
            get_false_pos_rate()

        elif action.startswith("pred_features[") and action.endswith("]"):
            get_pred_features()

        elif action.startswith("visualize_decision_making[") and action.endswith("]"):
            visualize_decision_making()

        elif action.startswith("finish[") and action.endswith("]"):
            answer = action[len("finish["):-1]
            self.answer = answer
            done = True
            self.obs = f"Episode finished"

        elif action.startswith("think[") and action.endswith("]"):
            self.obs = "Thought"
            
        else:
            self.obs = f"invalid action: {action}"

        self.steps += 1

        return self.obs, done, self.getInfo()

