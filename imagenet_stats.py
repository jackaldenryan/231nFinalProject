import pickle
import numpy as np


# shape: activations[layer name][batch index] = nparray of [image idx, channel-mean activation]
with open("./outputs/imagenet-activations.pickle", "rb") as f:
    activations = pickle.load(f)

fc_acts_dict = activations["fc"]

acts = np.zeros(
    (len(fc_acts_dict) * len(fc_acts_dict[0]), fc_acts_dict[0].shape[1]))

for b in range(len(fc_acts_dict)):
    acts[b * len(fc_acts_dict[0]):(b + 1) *
         len(fc_acts_dict[0]), :] = fc_acts_dict[b]

acts_mean = acts.mean(axis=0)
acts_std = acts.std(axis=0)
