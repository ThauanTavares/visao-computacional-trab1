import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

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

def resize_image(img, h_blocos, w_blocos, labels):
    img_clusters = labels.reshape(h_blocos, w_blocos)
    return cv2.resize(img_clusters.astype(np.uint8), (img.shape[1], img.shape[0]),
                      interpolation=cv2.INTER_NEAREST)

# Carrega imagem em escala de cinza
img_original = cv2.imread("imagens/agro2.jpeg", cv2.IMREAD_GRAYSCALE)
if img_original is None:
    raise FileNotFoundError("Imagem não encontrada. Verifique o caminho.")
img_original = cv2.resize(img_original, (256, 256))  # Tamanho base

# Parâmetros
n_pyramid_levels = 3  # número de níveis de escala
tamanho_bloco = 16
n_clusters_kmeans = 4
eps_dbscan = 10
min_samples_dbscan = 5

# Define filtros de textura
filtros = {
    "horizontal": np.array([[1, -1]]),
    "vertical": np.array([[1], [-1]]),
    "diag45": np.array([[0, 1], [-1, 0]]),
    "diag135": np.array([[1, 0], [0, -1]]),
    "circular": cv2.getGaborKernel((5, 5), 2.0, 0, 5.0, 1.0, 0, ktype=cv2.CV_32F)
}

# Armazena resultados
resultados_kmeans = []
resultados_dbscan = []
imagens_pyramid = []

img = img_original.copy()

for level in range(n_pyramid_levels):
    imagens_pyramid.append(img)

    # Aplica filtros
    imagens_filtradas = []
    for kernel in filtros.values():
        filtrada = cv2.filter2D(img, -1, kernel)
        imagens_filtradas.append(filtrada)

    # Extrai vetores de textura
    vetores_textura, h_blocos, w_blocos = extrair_vetores(imagens_filtradas, tamanho_bloco)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=0)
    labels_kmeans = kmeans.fit_predict(vetores_textura)
    resultado_kmeans = resize_image(img, h_blocos, w_blocos, labels_kmeans)
    resultados_kmeans.append(resultado_kmeans)

    # DBSCAN
    dbscan = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan, metric='manhattan')
    labels_dbscan = dbscan.fit_predict(vetores_textura)
    resultado_dbscan = resize_image(img, h_blocos, w_blocos, labels_dbscan)
    resultados_dbscan.append(resultado_dbscan)

    # Reduz para próximo nível da pirâmide
    img = cv2.pyrDown(img)

# Exibe os resultados
plt.figure(figsize=(15, 5 * n_pyramid_levels))

for i in range(n_pyramid_levels):
    plt.subplot(n_pyramid_levels, 3, i * 3 + 1)
    plt.imshow(imagens_pyramid[i], cmap='gray')
    plt.title(f"Nível {i} - Original")
    plt.axis('off')

    plt.subplot(n_pyramid_levels, 3, i * 3 + 2)
    plt.imshow(resultados_kmeans[i], cmap='tab10')
    plt.title(f"Nível {i} - KMeans")
    plt.axis('off')

    plt.subplot(n_pyramid_levels, 3, i * 3 + 3)
    plt.imshow(resultados_dbscan[i], cmap='tab10')
    plt.title(f"Nível {i} - DBSCAN (Manhattan)")
    plt.axis('off')

plt.tight_layout()
plt.show()
