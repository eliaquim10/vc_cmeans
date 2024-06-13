from os import makedirs
from cmeans import *



import matplotlib.pyplot as plt
import numpy as np

# Definindo uma paleta de 10 cores
colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]
def cores(colors, file_name):
    # cond = model.sigma.max(1)
    
    if ((colors.max(1) > 1).any()):
      colors_formating = (colors.T/colors.max(1)).T
    else:
      colors_formating = colors.copy()
    # Criando uma figura
    fig, ax = plt.subplots(figsize=(colors.shape[0], 1))

    # Desenhando retângulos coloridos
    for i, color in enumerate(colors_formating):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))

    # Configurando o eixo
    ax.set_xlim(0, colors.shape[0])
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(.5, colors.shape[0] + .5))
    formata_numeros = lambda numeros: ",".join(['{:.1f}'.format(numero) for numero in numeros])
    ax.set_xticklabels([f"{formata_numeros(colors[i])}" for i in range(colors.shape[0])], fontsize=7)
    ax.set_yticks([])
    ax.set_aspect('equal')

    # plt.show()
    # print("->", file_name.split("/")[:-1])
    # para criar o diretorio
    os.makedirs("/".join(file_name.split("/")[:-1]), exist_ok=True)
    # Remover o arquivo se ele existir para substituir o arquivo
    if os.path.exists(file_name):
        os.remove(file_name)
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()

cs = [10, 20]
models = ["cmeans/", "k_means/", "minimal/cmeans/", "minimal/k_means/"]
modos = ["comics", "art"]

model = FuzzyCMeansGaussianS2(10, 50, .5)

def load_model(filename):
  # Reading from json file
  with open(filename, "r") as openfile:
      json_object = json.load(openfile)
      return np.array(json_object["cluster_centers_"])
  return None


for c in cs:
  for i, algortim in enumerate(models):
    for modo in modos:
      try:
        set_name = f"./results/{algortim}{modo}/"
        if i & 1:
          g = load_model(f"{set_name}model/{c}/model.json")
          cores(g, f"{set_name}metricas/{c}/g.jpg")
        else:
          model.load_model(f"{set_name}model/{c}/model.json")
          cores(model.G, f"{set_name}metricas/{c}/g.jpg")
          # print((model.sigma.max(1) > 1).any())
          # raise Exception
          cores(model.sigma, f"{set_name}metricas/{c}/sigma.jpg")
          # cores((model.sigma.T/model.sigma.max(1)).T, f"{set_name}metricas/{c}/sigma.jpg")
        
      except Exception:
        print(set_name, c)
        # pass
   
