import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carrega a imagem em tons de cinza
img = cv2.imread('/home/thauan/Faculdade/VisaoComputacional/Trab1/imagens/agro2.jpeg', cv2.IMREAD_GRAYSCALE)

# Filtros direcionais simples (horizontal e vertical)
filtro_horizontal = np.array([[1, -1]])
filtro_vertical = np.array([[1], [-1]])

# Aplica a convolução com os filtros
resposta_horizontal = cv2.filter2D(img, -1, filtro_horizontal)
resposta_vertical = cv2.filter2D(img, -1, filtro_vertical)

# Mostra as respostas em preto e branco
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(resposta_horizontal, cmap='gray')
plt.title('Resposta Horizontal')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(resposta_vertical, cmap='gray')
plt.title('Resposta Vertical')
plt.axis('off')

plt.tight_layout()
plt.show()
