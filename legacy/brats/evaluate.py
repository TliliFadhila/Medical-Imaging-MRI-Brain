import numpy as np
import nibabel as nib
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pathlib
from nilearn.image import resample_img, reorder_img
from unet3d.utils.utils import resize


def get_whole_tumor_mask(data):
    return data > 0


def get_tumor_core_mask(data):
    return np.logical_or(data == 1, data == 4)


def get_enhancing_tumor_mask(data):
    return data == 4


def dice_coefficient(truth, prediction):
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))


def main():
    header = ("WholeTumor", "TumorCore", "EnhancingTumor")
    masking_functions = (get_whole_tumor_mask, get_tumor_core_mask, get_enhancing_tumor_mask)
    rows = list()
    subject_ids = list()
    for case_folder in glob.glob("C:\\Users\\Fadhi\\OneDrive\\Bureau\\unet\\data\\Brats17TrainingData\\**\\**"):
        if not os.path.isdir(case_folder):
            continue
        truth_file = next(iter(glob.glob(os.path.join(case_folder, "*seg*.nii"))), None)
        truth_image = nib.load(truth_file)
        truth_image = resize(truth_image, (128, 128, 128))
        truth = truth_image.get_data()
        prediction_file = str(pathlib.Path(__file__).parent.parent / "BraTS2017_Validation_predictions" / os.path.basename(case_folder)) + ".nii"
        if not os.path.isfile(prediction_file):
            continue
        subject_ids.append(os.path.basename(case_folder))
        prediction_image = nib.load(prediction_file)
        prediction = prediction_image.get_fdata()
        rows.append([dice_coefficient(func(truth), func(prediction)) for func in masking_functions])

    df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
    df.to_csv("brats_scores.csv")
    scores = dict()
    for index, score in enumerate(df.columns):
        values = df.values.T[index]
        scores[score] = values[np.isnan(values) == False]
        print(index, score)

    plt.plot(list(scores.values()))
    plt.ylabel("Dice Coefficient")
    plt.savefig("validation_scores_boxplot.png")
    plt.close()

    if os.path.exists("./training.log"):
        training_df = pd.read_csv("./training.log").set_index('epoch')

        plt.plot(training_df['loss'].values, label='training loss')
        plt.plot(training_df['val_loss'].values, label='validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.xlim((0, len(training_df.index)))
        plt.legend(loc='upper right')
        plt.savefig('loss_graph.png')


if __name__ == "__main__":
    main()
