from inceptionv1 import model as inceptionv1
from prior_optimization import create_optimized_image, input_to_img
import pickle

target = inceptionv1.inception5b
lr = 0.007
histories = []

for channel_idx in range(1):
    input_vec, loss_history = create_optimized_image(
        target, channel_idx, n_steps=512, lr=lr)
    histories.append(loss_history.cpu().detach().numpy())
    img = input_to_img(input_vec)

    img.save(f"channel-{str(channel_idx)}.png")


with open("loss_histories.pickle", "wb") as f:
    pickle.dump(histories, f)
