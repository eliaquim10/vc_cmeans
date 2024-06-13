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

from scipy.optimize import minimize

import cv2
from sklearn.metrics.cluster import adjusted_rand_score #, ran
# from sklearn import datasets
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def save_model(model, filename):
    # Save the model in json file
    json_object = json.dumps({"cluster_centers_": model.cluster_centers_.tolist(),}, indent=2)  #
    # Writing to sample.json
    folder = "/".join(filename.split("/")[:-1])
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)

    with open(filename, "w") as outfile:
        outfile.write(json_object)

# %% [markdown]
# #### Pre-processing

# %%
class Cleaner(BaseEstimator, TransformerMixin):
  def __init__(self):
    super()
  
  def read_dataset(self, filename):
    X = pd.read_csv(filename, header=None, delim_whitespace=True)
    y = pd.DataFrame({X.shape[1]: np.arange(0, X.shape[0])//200})
    return X, y
  def fit(self, X, y):
    return self
  
  def transform(self, X: pd.DataFrame, y=pd.DataFrame):
    y_lenth = len(y)
    if y_lenth:
      # key = "class"
      # y = pd.DataFrame({key: y})
      values = pd.concat([X, y], axis=1)
    else:
      values = X.copy()

    values = values.dropna()
    values = values.drop_duplicates()
    if y_lenth:
      return values[X.columns.values], values[values.columns.values[-1]]
    return values
  
  def fit_transform(self, X, y=pd.DataFrame):
      return self.fit(X, y).transform(X, y)
  
class Transformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    super()
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    if type(X) == np.ndarray:
      return X
    return X.values
  def fit_transform(self, X, y=None):
    return self.fit(X, y).transform(X, y)

# %% [markdown]
# ### Model

# %% [markdown]
# ##### Metrics

# %%
def uqi(img1, img2):
    """
    Calculate the Universal Quality Image Index (UQI) between two images.

    Parameters:
    img1 (numpy array): First image (reference image).
    img2 (numpy array): Second image (image to be compared).

    Returns:
    float: UQI value between -1 and 1.
    """
    assert img1.shape == img2.shape, "Images must have the same dimensions"
    
    N = img1.size
    
    # Means
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    
    # Variances and covariance
    var1 = np.var(img1)
    var2 = np.var(img2)
    covar = np.mean((img1 - mean1) * (img2 - mean2))
    
    # UQI calculation
    numerator = 4 * covar * mean1 * mean2
    denominator = (var1 + var2) * (mean1**2 + mean2**2)
    
    if denominator == 0:
        return 1 if numerator == 0 else 0
    
    uqi_value = numerator / denominator
    return uqi_value

def calculate_histogram(image, bins=256):
    """
    Calculate the histogram of an image.
    
    Parameters:
    image (numpy array): Input image.
    bins (int): Number of bins for the histogram.
    
    Returns:
    numpy array: Histogram of the image.
    """
    hist = [cv2.calcHist(images=[image], channels=[channel], mask=None, histSize = [bins], ranges=[0, 256]) \
            for channel in range(3)]
    # hist = cv2.normalize(hist, hist).flatten()
    return hist

def ssim_histogram(hist1, hist2, C1=1e-4, C2=9e-4):
    """
    Calculate SSIM between two histograms.
    
    Parameters:
    hist1 (numpy array): Histogram of the first image.
    hist2 (numpy array): Histogram of the second image.
    C1 (float): Constant to avoid division by zero.
    C2 (float): Constant to avoid division by zero.
    
    Returns:
    float: SSIM value.
    """
    def hist_ssim(hist1, hist2):
        mu_x = np.mean(hist1)
        mu_y = np.mean(hist2)
        sigma_x = np.var(hist1)
        sigma_y = np.var(hist2)
        sigma_xy = np.mean((hist1 - mu_x) * (hist2 - mu_y))
        
        return ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    
 
    ssim_value = [hist_ssim(hist1[channel], hist2[channel]) for channel in range(3)]
    
    return np.mean(ssim_value)

def metricas(image1, image2):
    uqi_value = uqi(image1, image2)

    hist1 = calculate_histogram(image1)
    hist2 = calculate_histogram(image2)

    # Calcular SSIM entre histogramas
    ssim = ssim_histogram(image1, image2)
    ssim_value = ssim_histogram(hist1, hist2)
    print(f"SSIM : {ssim:.4f}, SSIM entre histogramas: {ssim_value:.4f}, UQI: {uqi_value:.4f}")
    return ssim, ssim_value, uqi_value
# %%
class Gaussian:
  def distance(self, g, x, sigma):
    if x.shape[0] == 0 or sigma.shape[0] == 0:
      raise Exception
    distance = np.square(np.subtract(x, g)) #**2
    distance_sigma = distance / sigma
    summatory_distance = (-1/2)*np.sum(distance_sigma, 1)
    return np.exp(summatory_distance)

class Metrics:
  def mpc(u: np.array):
    # modified partition coefficient
    n, k = u.shape
    f = np.trace(u.dot(u.T)) / n

    mpc = 1 - (k/(k - 1))*(1 - f)
    return mpc
  def ars(y_true, y_pred):
    # adjusted rand score
    return adjusted_rand_score(y_true, y_pred)

# %% [markdown]
# ##### Shower

# %%
class Shower:
    def show_grafico(self, x):
        _, axes = plt.subplots()
        markers = ["^", "x", "+"]
        p = self.predict(self.G, x)

        for c, (x_1, x_2) in zip(p, x):
            r, g, b, a = (
                0.9 * ((c + 1) / self.num_class),
                0.9 * ((c * 1.1) / self.num_class),
                0.9 * ((c * 1.2) / self.num_class),
                0.1,
            )

            axes.plot(x_1, x_2, "ro", marker=markers[c])

        raio = self.distance_classes() / 2

        for i, (x_1, x_2) in enumerate(self.G):

            plt.plot(x_1, x_2, "go", scalex=0.01, scaley=0.01)
            r, g, b, a = (
                0.9 * ((i + 1) / self.num_class),
                0.9 * ((i * 1.1) / self.num_class),
                0.9 * ((i * 1.2) / self.num_class),
                0.9,
            )
            c = plt.Circle((x_1, x_2), raio, color=(r, g, b, a), fill=False)

            axes.set_aspect(aspect=1, adjustable="datalim", anchor="SW")
            axes.add_artist(c)
        plt.show()

    def show_function_objetive(self, data_f, params):
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

        # plot box plot
        axs[0].boxplot(data_f)
        # axs[0].set_title('funcão objetivo e m ')
        axs[0].yaxis.grid(True)
        axs[0].set_xticks([y + 1 for y in range(len(data_f))], labels=params)
        axs[0].set_xlabel("m")
        axs[0].set_ylabel("f")

        plt.show()

    def show_labels(self, x, filename):
        # Data
        k, n = x.shape
        plt.figure(figsize=(20, 6))

        # Labels
        xlabs = np.arange(k)
        ylabs = np.arange(n)

        # Heat map
        _, ax = plt.subplots(constrained_layout=True)

        # ax.figure()
        ax.set_axis_off()  # = [0.1, 0.1, 0.8, 0.8] #.add_axes()
        ax.imshow(x)
        if n < 10:
            # Add the labels
            ax.set_xticks(ylabs)
            ax.set_yticks(xlabs)

            # Add the values to each cell
            for i in range(len(xlabs)):
                for j in range(len(ylabs)):
                    _ = ax.text(
                        j,
                        i,
                        round(x[i, j], 1),
                        ha="center",
                        va="center",
                        color=(0, 0, 0),
                    )  # "w"
        if filename:
            plt.savefig(filename, dpi=150)
        plt.show()

    def show_G_S(self, folder):
        # show the values of Gs and S(widths)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        print("Gs")
        self.show_labels(self.G, f"{folder}/G")
        print("S2s")
        self.show_labels(self.sigma, f"{folder}/sigma")

# %% [markdown]
# ##### Base Model

# %%
class Model:
    def __init__(self, loss=Gaussian(), var_type=np.float32) -> None:
        # model base
        self.loss = loss
        np.float32 = var_type

    def predict(self, X, is_preprocessing=False, argmax=True):
        # here we make predict in x
        n = X.shape[0]
        k = self.G.shape[0]

        if is_preprocessing:
            X = self.preprocesssing(X)
        u = np.zeros((n, k), np.float32)

        distances = np.array(
            [2 - 2 * self.loss.distance(self.G[i], X, self.sigma[i]) for i in range(k)]
        ).T

        expoente = 1 / (self.m - 1)

        for i in range(n):

            ds = distances[i] == 0
            if (ds).any():
                summation_c_i_1 = np.zeros((k))
                summation_c_i_1[ds] = 1
            else:
                summation_c = np.array(
                    [((distances[i] / distances[i, h]) ** expoente) for h in range(k)]
                )  # .T

                summation_c_i = np.sum(summation_c, 0)
                summation_c_i_1 = np.power(summation_c_i, -1)

            u[i] = summation_c_i_1.copy()
        if argmax:
            return u.argmax(1)
        return u
        # if argmax:

    def set_g(self, g):
        self.G = g.copy()

    def set_sigma(self, sigma):
        self.sigma = sigma.copy()

    def preprocesssing(self, X):
        pipeline = Pipeline(
            [
                ("MinMaxScaler", MinMaxScaler()),
                ("Transformer", Transformer()),
            ]
        )

        return pipeline.transform(X)

    def save_model(self, filename):
        # Save the model in json file
        json_object = json.dumps(
            {"G": self.G.tolist(), "sigma": self.sigma.tolist(), "m": self.m}, indent=2
        )  #
        # Writing to sample.json
        folder = "/".join(filename.split("/")[:-1])
        if not os.path.isdir(folder):
            os.makedirs(folder)

        with open(filename, "w") as outfile:
            outfile.write(json_object)

    def load_model(self, filename):
        # Reading from json file
        with open(filename, "r") as openfile:
            json_object = json.load(openfile)
            self.G = np.array(json_object["G"])
            self.sigma = np.array(json_object["sigma"])
            self.m = json_object["m"]



# %%
# tá funcionando bonitinho

class FuzzyCMeansGaussianS2(Shower, Model):
    def __init__(
        self,
        num_class,
        epochs,
        threshold,
        loss=Gaussian(),
        m=1.1,
        seeder=None,
        verbose=False,
    ) -> None:
        # load the hiperparams
        super().__init__(loss=loss, var_type=np.float32)
        self.num_class = num_class
        self.m = m
        self.verbose = verbose
        if seeder:
            self.seeder = seeder
            np.random.seed(seeder)

        self.threshold = threshold
        self.epochs = epochs
    # @njit
    def init_g(self, x):
        # here we get random prototipes to train
        args = np.random.choice(x.shape[0], self.num_class, replace=False)
        self.G = x[args].copy()
    # @njit
    def init_u(self, x):
        # we calcule initial matrix of membership
        g = self.G.copy()
        sigma = self.sigma.copy()
        self.n = x.shape[0]
        self.k = g.shape[0]

        self.U = np.zeros((self.n, self.k), np.float32)

        distances = np.array(
            [2 - 2 * self.loss.distance(g[i], x, sigma[i]) for i in range(self.k)]
        ).T

        expoente = 1 / (self.m - 1)

        for i in range(self.n):
            # summation_c = np.array([(distances[i, h] / distances[i])**expoente for h in range(self.k)])
            # np.divide
            # summation_c = np.array([(distances[i] / distances[i, h])**expoente for h in range(self.k)]) #.T
            ds = distances[i] == 0
            if (ds).any():
                summation_c_i_1 = np.zeros((self.k))
                summation_c_i_1[ds] = 1
                self.U[i] = summation_c_i_1.copy()

            else:
                summation_c = np.array(
                    [
                        ((distances[i] / distances[i, h]) ** expoente)
                        for h in range(self.k)
                    ]
                )  # .T

                summation_c_i = np.sum(summation_c, 0)
                summation_c_i_1 = np.power(summation_c_i, -1)

                membership_is_nan = np.isnan(summation_c_i_1)

                summation_u = np.sum(summation_c_i_1, where=membership_is_nan == False)

                summation_u_1 = 1 - summation_u
                abs_summation_u_1 = np.abs(summation_u_1)

                if abs_summation_u_1 > 0.01:
                    arg = summation_c_i_1.argmax()
                    summation_c_i_1[arg] = summation_c_i_1[arg] + summation_u_1

                    self.U[i] = summation_c_i_1.copy()
                else:
                    self.U[i] = summation_c_i_1.copy()
    # @njit
    def init_sigma(self, x):
        # we calcule initial matrix of widths each 1
        self.p = x.shape[1]
        self.k = self.G.shape[0]
        self.sigma = np.ones((self.k, self.p), dtype=np.float32)

    # passo 1
    # @njit 
    def otm_prototivo(self, g, u, x, sigma):
        new_g = g.copy()
        u_i = u**self.m

        for i in range(self.k):
            n_g = self.loss.distance(g[i], x, sigma[i])

            u_m_g = u_i[:, i] * n_g  # *
            u_m_gx = ((x.T) * u_m_g).T

            s_umg = np.sum(u_m_g)
            s_umgx = np.sum(u_m_gx, 0)

            new_g[i] = s_umgx / s_umg
        return new_g

    # passo 2
    # @njit 
    def otm_u(self, g, x, sigma):
        new_u = np.zeros((self.n, self.k), dtype=np.float32)

        distances = np.array(
            [2 - 2 * self.loss.distance(g[i], x, sigma[i]) for i in range(self.k)]
        ).T

        expoente = 1 / (self.m - 1)
        for i in range(self.n):

            ds = distances[i] == 0
            if (ds).any():
                summation_c_i_1 = np.zeros((self.k))
                summation_c_i_1[ds] = 1
                new_u[i] = summation_c_i_1.copy()

            else:
                summation_c = np.array(
                    [
                        ((distances[i] / distances[i, h]) ** expoente)
                        for h in range(self.k)
                    ]
                )  # .T

                summation_c_i = np.sum(summation_c, 0)

                summation_c_i_1 = np.power(summation_c_i, -1)

                membership_is_nan = np.isnan(summation_c_i_1)

                summation_u = np.sum(summation_c_i_1, where=membership_is_nan == False)

                summation_u_1 = 1 - summation_u
                abs_summation_u_1 = np.abs(summation_u_1)
                # if

                if summation_u == 0 or membership_is_nan.any():
                    count = np.sum(membership_is_nan)  # total nan
                    new_u[i] = np.nan_to_num(summation_c_i_1, nan=summation_u_1 / count)
                elif abs_summation_u_1 > 0.01:
                    arg = summation_c_i_1.argmax()
                    summation_c_i_1[arg] = summation_c_i_1[arg] + summation_u_1

                    new_u[i] = summation_c_i_1.copy()
                else:
                    new_u[i] = summation_c_i_1.copy()
        return new_u
    
    # @njit 
    def otm_sigma(self, x, g, u):
        s = self.sigma.copy()

        ui = u ** (self.m)

        for i in range(self.k):
            g_i = g[i]
            ui_per = ui[:, i]

            g_d = self.loss.distance(g_i, x, s[i])
            e_d = (x - g_i) ** 2

            s_cn = e_d.T * (ui_per * g_d)

            if s_cn.shape[1] != 1:
                ss_cn = np.sum(s_cn, 1).copy()
            else:
                ss_cn = np.squeeze(s_cn, 1).copy()

            s_p = np.prod(ss_cn)
            s_p = s_p ** (1 / self.p)

            s_ij = s_p / ss_cn

            r_r = s_ij[np.isnan(s_ij) == False]
            p_r = np.prod(r_r)

            s_r = np.abs(1 - p_r)

            # verificacao do produtorio para a classe que deve ser 1
            if p_r == 0:
                teste = False

                if teste:
                    m_s = np.sum(ss_cn) / (self.p)

                # ss_cn = ss_cn*(1/m_s)
                if teste:
                    im_s = 1 / (m_s)
                    new_ss_cn = ss_cn * im_s
                else:
                    new_ss_cn = 1 - ss_cn  # **(1/(self.p))

                new_s_p = np.prod(new_ss_cn)

                new_s_p = new_s_p ** (1 / self.p)

                new_r = new_s_p / new_ss_cn

                prod = np.prod(new_r)

                if np.isnan(prod) == False:
                    s[i] = new_r.copy()

            elif s_r > 0.1:
                # para quando o produtorio de Ss dá diferente de 1
                if r_r.shape[0] == s_ij.shape[0]:
                    s[i] = s_ij / (p_r ** (1 / s_ij.shape[0]))
                else:
                    s_prod = 1 / (p_r ** (1 / s_ij.shape[0]))
                    s[i] = np.nan_to_num(s_ij, copy=True, nan=s_prod)
            else:
                s[i] = s_ij.copy()

        return s
    
    # @njit 
    def funcao_objetivo(self, x, g, u, sigma):
        # k = g.shape[0]
        u_i = u**self.m
        f = np.zeros(self.k, dtype=np.float32)

        s_d = np.zeros(x.shape[0], dtype=np.float32)
        w = np.zeros(x.shape[0], dtype=np.float32)
        for i in range(self.k):
            s_d = 2 - 2 * self.loss.distance(g[i], x, sigma[i])

            w = s_d.T * (u_i[:, i])
            f[i] = np.sum(w)

        return np.sum(f)
    
    # @njit 
    def atualiza_GUS2(self, g, u, sigma):
        self.G = g
        self.U = u
        self.sigma = sigma
    
    # @njit 
    def fit(self, x, y, verbose=False):
        self.redu = True

        self.init_g(x)
        self.init_sigma(x)
        self.init_u(x)

        j = self.funcao_objetivo(x, self.G, self.U, self.sigma)
        if self.verbose:
            print("epoca:", 0, "funcão objetivo:", "{:.2f}".format(j))  # , new_f

        for epoch in range(self.epochs):
            # "passo 1"

            new_sigma = self.otm_sigma(x, self.G, self.U)

            # "passo 2"
            new_g = self.otm_prototivo(self.G, self.U, x, new_sigma)

            # "passo 3"
            new_u = self.otm_u(new_g, x, new_sigma)

            new_j = self.funcao_objetivo(x, new_g, new_u, new_sigma)
            if self.verbose:
                print(f"epoca: {epoch + 1} funcão objetivo:", "{:.2f}".format(new_j))  # f,
                # print(
                #     f"Modified partition coefficient  {Metrics.mpc(self.U)},  {Metrics.ars(y, self.U.argmax(1))}"
                # )  # f,

            # silhouette_score
            if verbose:
                self.show_grafico(x)

            dif = np.abs(new_j - j)
            # new_j > j
            if np.isnan(new_j) == True or dif < self.threshold:
                return
            else:
                j = new_j
                self.atualiza_GUS2(new_g, new_u, new_sigma)


# %% [markdown]
# ##### Trainer

# %%
class Trainer(FuzzyCMeansGaussianS2):
    def __init__(
        self,
        num_class,
        epochs,
        threshold,
        model_creator,
        filename,
        loss=Gaussian(),
        m=1.1,
        verbose=False,
        repeat=1,
    ) -> None:
        # load the hiperparams
        super().__init__(num_class, epochs, threshold, loss=loss, m=m)
        self.num_class = num_class
        np.float32 = np.float32
        self.verbose = verbose
        self.threshold = threshold
        self.epochs = epochs
        self.repeat = repeat
        self.model_creator = model_creator
        self.filename = filename
        self.metrics = {
            "objetive_function": [],
            "mpc": [],
            "ars": [],
        }

    def fit(self, X, y):
        j = np.inf

        for i in range(self.repeat):
            model = self.model_creator(
                num_class=self.num_class,
                epochs=self.epochs,
                threshold=self.threshold,
                loss=self.loss,
                m=self.m,
                verbose=self.verbose,
            )
            model._only = False
            model.fit(X, y)
            new_j = model.funcao_objetivo(X, model.G, model.U, model.sigma)
            new_mpc = Metrics.mpc(model.U)
            new_ars = Metrics.ars(y, model.U.argmax(1))

            print(f"repeation: {i + 1} funcão objetivo: {new_j}")
            print(
                f"Modified partition coefficient: {new_mpc},  adjusted rand score: {new_ars}"
            )

            self.metrics["objetive_function"].append(new_j)
            self.metrics["mpc"].append(new_mpc)
            self.metrics["ars"].append(new_ars)

            if new_j < j:
                self.G = model.G
                self.U = model.U
                self.sigma = model.sigma
                j = new_j

        self.save_model(f"./model/{self.m}/{self.filename}")
        self.save(f"./metrics/{self.m}/{self.filename}")

    def save(self, filename):
        if not len(self.metrics["objetive_function"]):
            print("the models is not fit")
            return
        # Save the model in json file
        json_object = json.dumps(self.metrics, indent=2)  #
        # Writing to sample.json
        folder = "/".join(filename.split("/")[:-1])
        if not os.path.isdir(folder):
            os.makedirs(folder)

        with open(filename, "w") as outfile:
            outfile.write(json_object)
