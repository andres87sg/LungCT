from .Covid19ConstanteManager import COVID19_PATH_SAVE_VISUAL_RESPONSE
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import math
matplotlib.use('Agg')

plt.style.use('dark_background')
plt.rc_context({'ytick.color': 'red'})


def generate_visual_result(prediction, original_image, file_name):

    output_path = COVID19_PATH_SAVE_VISUAL_RESPONSE + file_name.split('.')[0] + '_ai_diagnosis' + '.' + file_name.split('.')[1]

    mark_paths = generate_mark(prediction)

    merge_image_mark(original_image, mark_paths, output_path)

    return output_path


def generate_mark(predictions):
    output_path = COVID19_PATH_SAVE_VISUAL_RESPONSE + "/covid.png"

    f, ax = plt.subplots(figsize=(24, 2))
    plt.yticks(fontsize=30)

    df = pd.DataFrame.from_dict({'Condition': list(predictions.keys()), 'Probability': list(predictions.values()), 'Max': [1]*len(list(predictions.values()))}).sort_values(
        "Probability", ascending=False)
    sns.set_color_codes("pastel")
    sns.barplot(x="Max", y="Condition", data=df, color="r")
    sns.set_color_codes("dark")
    g = sns.barplot(x="Probability", y="Condition", data=df, color="r")

    for index, iter in enumerate(df.iterrows()):
        g.text(iter[1].Probability + 0.03, index + 0.20, str(round(iter[1].Probability * 100, 1)) + '%', color='red',
               ha="center", fontsize=22)

    ax.set(xlim=(0, 1))
    ax.set_xticks([])
    ax.xaxis.set_label_text("")
    ax.yaxis.set_label_text("")

    sns.despine(left=True, bottom=True)
    f.savefig(output_path, dpi=400)
    plt.close(f)
    return output_path


def merge_image_mark(image, mark_path, output_path):
    oH, oW = image.shape[:2]
    ovr = np.zeros((oH, oW, 3), dtype="uint8")
    image = np.dstack([image, np.ones((oH, oW), dtype="uint8") * 255])

    lgo_img = cv2.imread(mark_path, cv2.IMREAD_UNCHANGED)

    scl = math.floor((oW/lgo_img.shape[1])*100)
    w = int(lgo_img.shape[1] * scl / 100)
    h = int(lgo_img.shape[0] * scl / 100)
    dim = (w, h)

    lgo = cv2.resize(lgo_img, dim)
    lH, lW = lgo.shape[:2]

    ovr[oH - lH - 30:oH - 30, 15:lW + 15] = lgo[:, :, :3]

    original_color_mean = np.mean(ovr, axis=2)

    final = image.copy()[:, :, :3]
    final[np.where(original_color_mean > 0)] = 0
    final = final + ovr

    cv2.putText(final, text='Stella AI Report', org=(round(lW / 2.7), final.shape[0] - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1 * (final.shape[0] / 1024), color=(0, 0, 255), thickness=2)

    cv2.imwrite(output_path, final)

    cv2.destroyAllWindows()

    return output_path
