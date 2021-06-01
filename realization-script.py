from inceptionv1 import model as inceptionv1
from prior_optimization import create_optimized_image, input_to_img
import pickle

target = inceptionv1.inception5b.branch4[1].conv
lr = 1e-4
reg = 0
histories = []

for channel_idx in range(1, 50):
    while True:
        input_vec, loss_history, _class_vec = create_optimized_image(
            target, channel_idx, n_steps=256, lr=lr, reg=reg)
        histories.append(loss_history.cpu().detach().numpy())
        img = input_to_img(input_vec)

    img.save(
        f"outputs/realization/channel-{str(channel_idx)}-reg-{str(reg)}.png")


with open("outputs/realization/loss_histories.pickle", "wb") as f:
    pickle.dump(histories, f)
