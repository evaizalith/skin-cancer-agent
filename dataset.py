from cfg import CFG
import tensorflow as tf
import keras_cv
import h5py
import pandas as pd
from tqdm import tqdm
import keras
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import cv2

# Categorical features which will be one hot encoded
CATEGORICAL_COLUMNS = ["sex", "anatom_site_general",
            "tbp_tile_type","tbp_lv_location", ]

# Numeraical features which will be normalized
NUMERIC_COLUMNS = ["age_approx", "tbp_lv_nevi_confidence", "clin_size_long_diam_mm",
           "tbp_lv_areaMM2", "tbp_lv_area_perim_ratio", "tbp_lv_color_std_mean",
           "tbp_lv_deltaLBnorm", "tbp_lv_minorAxisMM", ]

# Tabular feature columns
FEAT_COLS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

def build_training_ds():
    # Train + Valid
    df = pd.read_csv(f'train-metadata.csv')
    df = df.ffill()
    print(df.head(2))

    # Testing
    testing_df = pd.read_csv(f'test-metadata.csv')
    testing_df = testing_df.ffill()
    print(testing_df.head(2))

    class_weights = compute_class_weight('balanced', classes=np.unique(df['target']), y=df['target'])
    class_weights = dict(enumerate(class_weights))
    print("Class Weights:", class_weights)

    training_validation_hdf5 = h5py.File(f"train-image.hdf5", 'r')
    testing_hdf5 = h5py.File(f"test-image.hdf5", 'r')

    isic_id = df.isic_id.iloc[0]

    # Image as Byte String
   # byte_string = training_validation_hdf5[isic_id][()]
   # print(f"Byte String: {byte_string[:20]}....")

    # Convert byte string to numpy array
    #nparr = np.frombuffer(byte_string, np.uint8)

    #print("Image:")
    #image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[...,::-1] # reverse last axis for bgr -> rgb

    df = df.reset_index(drop=True) # ensure continuous index
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CFG.seed)
    for i, (training_idx, validation_idx) in enumerate(sgkf.split(df, y=df.target, groups=df.patient_id)):
        df.loc[validation_idx, "fold"] = int(i)

    # Use first fold for training and validation
    training_df = df.query("fold!=0")
    validation_df = df.query("fold==0")
    print(f"# Num Train: {len(training_df)} | Num Valid: {len(validation_df)}")

    training_df.target.value_counts()

    validation_df.target.value_counts()

    ## Train
    print("# Training:")
    training_features = dict(training_df[FEAT_COLS])
    training_ids = training_df.isic_id.values
    training_labels = training_df.target.values
    training_ds = build_dataset(training_ids, training_validation_hdf5, training_features, 
                         training_labels, batch_size=CFG.batch_size,
                         shuffle=True, augment=True)

    # Valid
    print("# Validation:")
    validation_features = dict(validation_df[FEAT_COLS])
    validation_ids = validation_df.isic_id.values
    validation_labels = validation_df.target.values
    validation_ds = build_dataset(validation_ids, training_validation_hdf5, validation_features,
                         validation_labels, batch_size=CFG.batch_size,
                         shuffle=False, augment=False)

    training_ds_with_no_labels = training_ds.map(lambda x, _: x["features"])
    feature_space.adapt(training_ds_with_no_labels)

    return training_ds, validation_ds 
    
feature_space = keras.utils.FeatureSpace(
    features={
        # Categorical features encoded as integers
        "sex": "string_categorical",
        "anatom_site_general": "string_categorical",
        "tbp_tile_type": "string_categorical",
        "tbp_lv_location": "string_categorical",
        # Numerical features to discretize
        "age_approx": "float_discretized",
        # Numerical features to normalize
        "tbp_lv_nevi_confidence": "float_normalized",
        "clin_size_long_diam_mm": "float_normalized",
        "tbp_lv_areaMM2": "float_normalized",
        "tbp_lv_area_perim_ratio": "float_normalized",
        "tbp_lv_color_std_mean": "float_normalized",
        "tbp_lv_deltaLBnorm": "float_normalized",
        "tbp_lv_minorAxisMM": "float_normalized",
    },
    output_mode="concat",
)

def build_augmenter():
    # Define augmentations
    aug_layers = [
        keras_cv.layers.RandomCutout(height_factor=(0.02, 0.06), width_factor=(0.02, 0.06)),
        keras_cv.layers.RandomFlip(mode="horizontal"),
    ]
    
    # Apply augmentations to random samples
    aug_layers = [keras_cv.layers.RandomApply(x, rate=0.5) for x in aug_layers]
    
    # Build augmentation layer
    augmenter = keras_cv.layers.Augmenter(aug_layers)

    # Apply augmentations
    def augment(inp, label):
        images = inp["images"]
        aug_data = {"images": images}
        aug_data = augmenter(aug_data)
        inp["images"] = aug_data["images"]
        return inp, label
    return augment


def build_decoder(with_labels=True, target_size=CFG.image_size):
    def decode_image(inp):
        # Read jpeg image
        file_bytes = inp["images"]
        image = tf.io.decode_jpeg(file_bytes)
        
        # Resize
        image = tf.image.resize(image, size=target_size, method="area")
        
        # Rescale image
        image = tf.cast(image, tf.float32)
        image /= 255.0
        
        # Reshape
        image = tf.reshape(image, [*target_size, 3])
        
        inp["images"] = image
        return inp

    def decode_label(label, num_classes):
        label = tf.cast(label, tf.float32)
        label = tf.reshape(label, [num_classes])
        return label

    def decode_with_labels(inp, label=None):
        inp = decode_image(inp)
        label = decode_label(label, CFG.num_classes)
        return (inp, label)

    return decode_with_labels if with_labels else decode_image


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
