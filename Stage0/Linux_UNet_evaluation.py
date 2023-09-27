import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras import models
import cv2
from cv2 import INTER_AREA

"""
dice(ground truth 分割結果)
"""


def dice_coef_test(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    if union == 0:
        return 1
    intersection = np.sum(y_true_f * y_pred_f)
    return 2.0 * intersection / union


"""
隨機取樣
"""


def random_sample(x_val, y_val, preds, patient_index):

    all_ind = [5, 15, 30, 45, 60]  # random samples [5, 15, 30, 45, 60]
    all_ind = np.array(all_ind)

    fig, ax = plt.subplots(5, 3, figsize=(20, 15))
    for i in range(len(all_ind)):
        # pred = (pred > 0.5).astype(np.uint8).reshape(IMAGE_H, IMAGE_W)
        ax[i, 0].imshow(np.uint16(x_val[all_ind[i]].squeeze()), cmap="gray")
        ax[i, 1].imshow(y_val[all_ind[i]].squeeze(), cmap="gray")
        ax[i, 2].imshow(preds[all_ind[i]].squeeze(), cmap="gray")
        # ax[i, 3].text(0.5, 0.5, str(np.round(dice_coef_test(y_val[all_ind[i]], preds[all_ind[i]]), decimals=3)), fontsize=20, ha='center')
    plt.savefig("/home/a1095557/UNet_1/result/41_5.png")

    for i in range(5):
        image = preds[all_ind[i]].squeeze()
        image = cv2.resize(image, (512, 512), interpolation=INTER_AREA)
        cv2.imwrite(
            "/home/a1095557/UNet_1/result/" + str(patient_index) + "/" +
            str(i + 1) + ".tif",
            image,
        )


"""
全部取樣
"""


def all_sample(x_val, preds, patient_index):

    fig, ax = plt.subplots(len(x_val), 2, figsize=(5, 50))
    for i, pred in enumerate(preds):
        # pred = (pred > 0.5).astype(np.uint8).reshape(IMAGE_H, IMAGE_W)
        ax[i, 0].imshow(np.uint16(x_val[i].squeeze()), cmap="gray")
        # ax[i, 1].imshow(y_val[i].squeeze(), cmap='gray')
        ax[i, 1].imshow(pred.squeeze(), cmap="gray")
        # ax[i, 3].text(0.5, 0.5, str(np.round(dice_coef_test(y_val[i], pred), decimals=3)), fontsize=20, ha='center')
    plt.savefig("/home/a1095557/UNet_1/result/" + str(patient_index) + ".png")
    """
    for i, pred in enumerate(preds):
        image = pred.squeeze()
        image = cv2.resize(image, (512,512), interpolation=INTER_AREA)
        cv2.imwrite("/home/a1095557/UNet_1/result/" + str(patient_index) + "/" + str(i + 1) + ".tif", image)
    """


for i in range(1):
    model = models.load_model("/home/a1095557/UNet_1/model/model_plus3.h5")

    patient_index = i + 1
    print(patient_index)

    x_val = np.load("/home/a1095557/UNet_1/data/test_plus/" +
                    str(patient_index) + "/x_val.npy")
    # y_val = np.load('/home/a1095557/UNet_1/data/test_plus/' + str(patient_index) + '/y_val.npy')

    preds = model.predict(x_val)
    preds[preds > 0.5] = 1.0
    preds[preds < 0.5] = 0.0

    # print(np.round(dice_coef_test(y_val, preds), decimals=3)) #overall dice

    all_sample(x_val, preds, patient_index)
