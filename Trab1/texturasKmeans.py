import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Carrega imagem em escala de cinza
img = cv2.imread("/home/thauan/Faculdade/VisaoComputacional/Trab1/imagens/agro2.jpeg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Imagem não encontrada. Verifique o caminho.")
img = cv2.resize(img, (256, 256))  # Padroniza o tamanho para facilitar

# Define filtros de textura simples
filtros = {
    "horizontal": np.array([[1, -1]]),
    "vertical": np.array([[1], [-1]]),
    "diag45": np.array([[0, 1], [-1, 0]]),
    "diag135": np.array([[1, 0], [0, -1]]),
    "circular": cv2.getGaborKernel((5, 5), 2.0, 0, 5.0, 1.0, 0, ktype=cv2.CV_32F)
}

# Aplica filtros na imagem original
imagens_filtradas = []
for nome, kernel in filtros.items():
    filtrada = cv2.filter2D(img, -1, kernel)
    imagens_filtradas.append(filtrada)

# Extrai vetor de textura por média dos blocos
def extrair_vetores(imagens_filtradas, tamanho_bloco=16):
    h, w = imagens_filtradas[0].shape
    num_blocos_h = h // tamanho_bloco
    num_blocos_w = w // tamanho_bloco

    vetores = []
    for i in range(num_blocos_h):
        for j in range(num_blocos_w):
            vetor_bloco = []
            for img_f in imagens_filtradas:
                bloco = img_f[i*tamanho_bloco:(i+1)*tamanho_bloco,
                              j*tamanho_bloco:(j+1)*tamanho_bloco]
                media = np.mean(bloco)
                vetor_bloco.append(media)
            vetores.append(vetor_bloco)

    return np.array(vetores), num_blocos_h, num_blocos_w

# Extrai os vetores
vetores_textura, h_blocos, w_blocos = extrair_vetores(imagens_filtradas)

# Aplica KMeans para agrupar regiões com textura semelhante
kmeans = KMeans(n_clusters=4, random_state=0)
labels = kmeans.fit_predict(vetores_textura)

# Constrói imagem com os rótulos dos clusters
img_clusters = labels.reshape(h_blocos, w_blocos)
img_clusters_resized = cv2.resize(img_clusters.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

# Mostra resultado
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Imagem Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Segmentação por Textura (KMeans)")
plt.imshow(img_clusters_resized, cmap='tab10')
plt.axis('off')

plt.tight_layout()
plt.show()
