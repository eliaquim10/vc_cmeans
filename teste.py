

f = open("c:/Users/Eliaquim/Documents/mestrado/proposta/datasets/dcm-dataset/original_paper/train.txt", "r")
w = ' '
lista = []
text = f.read()
print(text)
for line in text.split("\n"):
  first = line.split("/")[0]
  if first not in lista:
    lista.append(line.split("/")[0])

# while(w is not None):
#   w = f.readline()
#   print(w)
print(len(lista))