import pandas as pd
import gymnasium as gym
import os
from os import path
import h5py
import tensorflow as tf
import cfg

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
        df = df.ffill()
        self.patientIDs = isic_id.values
        dataset = build_dataset(self.patientIDs, images, features, batch_size=self.batch_size, shuffle=False, augment=False, cache=False)

        self.patientData = dataset.map(lambda x: {"images": x["images"], "features": feature_space(x["features"])}, num_parallel_calls=tf.data.AUTOTUNE)

    def process(self, entity):
        if entity not in patientIDs:
            self.obs = f"Invalid ID: {entity} is not a known patient ID"
            return

        index = self.patientIDs.index(entity)
        item = self.patientData.skip(index).take(1)
        preds = self.nn.predict(item).squeeze()

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


def build_dataset(
    isic_ids,
    hdf5,
    features,
    labels=None,
    batch_size=32,
    decode_fn=None,
    augment_fn=None,
    augment=False,
    shuffle=1024,
    cache=True,
    drop_remainder=False,
):
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter()

    AUTO = tf.data.experimental.AUTOTUNE

    images = [None]*len(isic_ids)
    for i, isic_id in enumerate(tqdm(isic_ids, desc="Loading Images ")):
        images[i] = hdf5[isic_id][()]
        
    inp = {"images": images, "features": features}
    slices = (inp, labels) if labels is not None else inp

    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.cache() if cache else ds
    ds = ds.map(decode_fn, num_parallel_calls=AUTO)
    if shuffle:
        ds = ds.shuffle(shuffle, seed=CFG.seed)
        opt = tf.data.Options()
        opt.deterministic = False
        ds = ds.with_options(opt)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.map(augment_fn, num_parallel_calls=AUTO) if augment else ds
    ds = ds.prefetch(AUTO)
    return ds
