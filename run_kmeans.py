# %% [markdown]
# #### imports

# %%
# SSIM para os histogramas
# constancia de cor, erro angular
# universal constancia cor

# %%
# !conda install numba -y

# %%
import os
from sklearn.cluster import k_means, KMeans

# from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
import numpy as np
from numba import njit

import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics.cluster import adjusted_rand_score  # , ran
from cmeans import *
import cv2
from IPython.display import display
from PIL import Image  # , ImageDraw, ImageFont


# from sklearn import datasets
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# %% [markdown]
# ### colect data

# %% [markdown]
# #### eBDtheque_database_v3 Page CYB_COSMOZONE

# %%
metricas_resultados = []


def save_model(model, filename):
    # Save the model in json file
    json_object = json.dumps({"cluster_centers_": model.cluster_centers_.tolist(),}, indent=2)  #
    # Writing to sample.json
    folder = "/".join(filename.split("/")[:-1])
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)

    with open(filename, "w") as outfile:
        outfile.write(json_object)


# %%
# image 출력 함수 정의
def show_img_bbox(X, y):
    plt.figure(figsize=(20, 20))
    for i, img in enumerate([X, y]):
        plt.subplot(1, 2, i + 1)
        plt.imshow(img)


def show_img(X):
    plt.figure(figsize=(5, 5))
    result = plt.imshow(X)
    print(result)
    plt.waitforbuttonpress(1)


# %%
def get_data(folder, index=10, lenght=1):
    # from Pill import Image
    # 10 : 22
    # 25 : 34
    # 40 : 42
    # 50 : 62
    # 63 : 69
    # 73 : 76
    # 81 : 76
    # 96

    img_path = "./datasets/eBDtheque_database_v3/Pages/CYB_COSMOZONE_012.jpg"
    arquivos = os.listdir(folder)[index : index + lenght]  # l + 1
    imgs = []
    # arquivos
    for i, img_file in enumerate(arquivos):
        img_path = f"{folder}{img_file}"
        img = np.array(Image.open(img_path).convert("RGB"))
        # display(Image.fromarray(img))# [:500, :500]

        imgs.append(img.reshape(-1, img.shape[2]))
    # image.shape = 320, 320, 3
    data = np.concatenate(imgs)
    return data / 255  # .astype(np.float32)


# %%
# from Pill import Image
# from PIL import Image #, ImageDraw, ImageFont
img_path = "./datasets/eBDtheque_database_v3/Pages/CYB_COSMOZONE_012.jpg"

image = np.array(Image.open(img_path).convert("RGB"))
# image = image.reshape(-1, image.shape[-1])
# image.shape = 320, 320, 3

# %%
dir = "./datasets/eBDtheque_database_v3/Pages/"

X = get_data(dir, lenght=4)

# %% [markdown]
# #### c = 20

# %% [markdown]
# ##### fit

# %%
model = KMeans(n_clusters=20)
model.fit(X)

# %%
# image = image.reshape(-1, image.shape[-1])

y = model.predict(image.reshape(-1, image.shape[-1]) / 255)

# %%
# model.cluster_centers_

# %%
# model.sigma

# %%
y_image = (model.cluster_centers_[y].reshape(image.shape) * 255).astype(np.uint8)

# %%
ssim, ssim_value, uqi_value = metricas(image, y_image)
metricas_resultados.append([ssim, ssim_value, uqi_value])

filename = img_path.split("/")[-1]

img_name = "results/k_means/comics/imgs/20/"


os.makedirs(img_name, exist_ok=True)


model_name = "results/k_means/comics/model/20/"


os.makedirs(model_name, exist_ok=True)


Image.fromarray(y_image).save(f"{img_name}{filename}")
save_model(model, f"{model_name}model.json")

# %% [markdown]
# ##### show

# %%
# show_img_bbox(image, y_image)

# %% [markdown]
# #### c = 10

# %% [markdown]
# ##### fit

# %%
model = KMeans(n_clusters=10)
model.fit(X)

# %%
# model.cluster_centers_

# %%
# model.sigma

# %%
y = model.predict(image.reshape(-1, image.shape[-1]) / 255)

# %%
y_image = (model.cluster_centers_[y].reshape(image.shape) * 255).astype(np.uint8)

# %%
ssim, ssim_value, uqi_value = metricas(image, y_image)
metricas_resultados.append([ssim, ssim_value, uqi_value])


filaname = img_path.split("/")[-1]

img_name = "results/k_means/comics/imgs/10/"


os.makedirs(img_name, exist_ok=True)


model_name = "results/k_means/comics/model/10/"


os.makedirs(model_name, exist_ok=True)


Image.fromarray(y_image).save(f"{img_name}{filaname}")
save_model(model, f"{model_name}model.json")

# %% [markdown]
# ##### show

# %%
# show_img_bbox(image, y_image)

# %%
# time change
# new row
# new column
# pure grid
# ecs_insert
# whole_row
# whole_column


# %% [markdown]
# #### wikiart High_Renaissance leonardo da vinci mona lisa

# %%
# from Pill import Image
from PIL import Image, ImageDraw, ImageFont
import numpy as np

img_path = "./datasets/wikiart/High_Renaissance/leonardo-da-vinci_mona-lisa.jpg"
# image_shape = 640, 640, 3
img = Image.open(img_path).convert("RGB")
# img = img.resize(image_shape[:2])
image = np.array(img)

X = image.reshape(-1, image.shape[2]) / 255


# X = X.astype(np.float32)
# %%
# image 출력 함수 정의
def show_img_bbox(X, y):
    plt.figure(figsize=(20, 20))
    for i, img in enumerate([X, y]):
        plt.subplot(1, 2, i + 1)
        plt.imshow(img)


"""
# %% [markdown]
# #### c = 20

# %% [markdown]
# ##### fit
# %%
model = KMeans(n_clusters=20)
model.fit(X)

# %%
# model.cluster_centers_

# %%
# model.sigma

# %%
y = model.predict(image.reshape(-1, image.shape[-1])/255)

# %%
y_image = (model.cluster_centers_[y].reshape(image.shape)*255).astype(np.uint8)

# %%
ssim, ssim_value, uqi_value =  metricas(image, y_image)
metricas_resultados.append([
  ssim, ssim_value, uqi_value
])

filaname = img_path.split("/")[-1]

img_name = "results/k_means/art/imgs/20/"


os.makedirs(img_name, exist_ok=True)
except Exception:
  pass

model_name = "results/k_means/art/model/20/"


   os.makedirs(model_name, exist_ok=True)
except Exception:
  pass


Image.fromarray(y_image).save(f"{img_name}{filaname}")
save_model(model,f"{model_name}model.json")

# %% [markdown]
# ##### show

# %%
# show_img_bbox(image, y_image)
"""
# %% [markdown]
# #### c = 10

# %% [markdown]
# ##### fit

# %%
model = KMeans(n_clusters=10)
model.fit(X)

# %%
# model.cluster_centers_

# %%
# model.sigma

# %%
y = model.predict(image.reshape(-1, image.shape[-1]) / 255)

# %%
y_image = (model.cluster_centers_[y].reshape(image.shape) * 255).astype(np.uint8)

# %%
ssim, ssim_value, uqi_value = metricas(image, y_image)
metricas_resultados.append([ssim, ssim_value, uqi_value])

filaname = img_path.split("/")[-1]

img_name = "results/k_means/art/imgs/10/"


os.makedirs(img_name, exist_ok=True)


model_name = "results/k_means/art/model/10/"


os.makedirs(model_name, exist_ok=True)


Image.fromarray(y_image).save(f"{img_name}{filaname}")
save_model(model, f"{model_name}model.json")

# %% [markdown]
# ##### show

# %%
# show_img_bbox(image, y_image)

# %%
import pandas as pd

pd.DataFrame(
    np.array(metricas_resultados).T, index=["ssim histrograma", "ssim", "uqi"]
).T.to_csv(f"./results/k_means/metricas.csv")
print(metricas_resultados)

# %%
# time change
# new row
# new column
# pure grid
# ecs_insert
# whole_row
# whole_column

# c = 20 epoca: 50 funcão objetivo: 12248.27
# SSIM : 0.0040, SSIM entre histogramas: 0.0949, UQI: 0.9861
