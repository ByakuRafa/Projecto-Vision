"""
Visão Computacional - Detecção de peças e cálculo de coordenadas
Arquivo convertido a partir do notebook Colab.

Objetivo:
- Detectar peças em uma imagem.
- Calcular coordenadas X/Y no plano da bancada usando homografia.
- Opcionalmente estimar Z quando houver mapa de profundidade.
"""

# ===============================================================
# IMPORTANTE
# ===============================================================
# Este arquivo foi organizado para leitura/execução como script.
# Algumas partes originalmente interativas do Colab, como upload
# de imagem, podem exigir ajuste dos caminhos dos arquivos.
# ===============================================================


# ======================================================================
# CÉLULA 1 - MARKDOWN
# ======================================================================
#
# # Visão Computacional — Detecção de peças e coordenadas na bancada
#
# Este notebook foi montado para substituir a lógica quebrada do arquivo original e deixar o experimento defensável para uma disciplina de visão computacional.
#
# ## O que este notebook faz
#
# 1. Carrega uma imagem RGB da bancada.
# 2. Segmenta as peças por contraste/saturação ou, se houver, por mapa de profundidade.
# 3. Calcula o centro de cada peça em pixels.
# 4. Converte o centro da peça para coordenadas reais da bancada usando **homografia**.
# 5. Se houver um depth map válido, estima também a altura/profundidade relativa da peça.
#
# ## Ponto conceitual importante
#
# Uma única imagem com luz direcional **não gera profundidade métrica confiável**. Ela cria sombras e aparência de relevo, mas não informa automaticamente a altura real de cada objeto.
#
# Para coordenadas reais, existem três caminhos:
#
# - **Mais simples e defensável:** imagem RGB + homografia → coordenadas X/Y na bancada.
# - **Melhor para 3D:** RGB + depth map real → coordenadas X/Y/Z.
# - **Mais avançado:** photometric stereo → várias imagens da mesma cena com luzes em direções conhecidas.


# ======================================================================
# CÉLULA 2 - MARKDOWN
# ======================================================================
#
# ## 1. Imports


# ======================================================================
# CÉLULA 3 - CODE
# ======================================================================
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

try:
    from google.colab import files
    IN_COLAB = True
except Exception:
    IN_COLAB = False

plt.rcParams["figure.figsize"] = (10, 7)

def mostrar(img, titulo="", cmap=None, tamanho=(10, 7)):
    """Mostra imagem BGR, RGB, grayscale ou máscara usando matplotlib."""
    plt.figure(figsize=tamanho)
    if img.ndim == 2:
        plt.imshow(img, cmap=cmap or "gray")
    else:
        # OpenCV lê em BGR; matplotlib espera RGB
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(titulo)
    plt.axis("on")
    plt.show()

def garantir_ksize_impar(valor):
    valor = int(valor)
    if valor < 3:
        valor = 3
    if valor % 2 == 0:
        valor += 1
    return valor


# ======================================================================
# CÉLULA 4 - MARKDOWN
# ======================================================================
#
# ## 2. Upload das imagens
#
# Envie pelo menos uma imagem RGB.
#
# Opcionalmente, envie também um depth map. Se você estiver usando Blender, o ideal é exportar um depth map verdadeiro, não uma imagem colorida apenas visualmente.


# ======================================================================
# CÉLULA 5 - CODE
# ======================================================================
if IN_COLAB:
    enviados = files.upload()
    NOMES_ARQUIVOS = list(enviados.keys())
else:
    # Fora do Colab, coloque os arquivos na mesma pasta do notebook e edite os nomes abaixo.
    NOMES_ARQUIVOS = [p.name for p in Path(".").glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]]

print("Arquivos disponíveis:")
for i, nome in enumerate(NOMES_ARQUIVOS):
    print(f"{i}: {nome}")


# ======================================================================
# CÉLULA 6 - MARKDOWN
# ======================================================================
#
# ## 3. Configuração principal
#
# Edite esta célula.
#
# - `RGB_PATH`: imagem principal da bancada.
# - `DEPTH_PATH`: coloque `None` se você não tiver depth map.
# - `LARGURA_BANCADA_CM` e `PROFUNDIDADE_BANCADA_CM`: medidas reais da área útil da bancada.
# - `PONTOS_BANCADA_PX`: quatro cantos da bancada na imagem, na ordem:
#   1. frente-esquerda
#   2. frente-direita
#   3. fundo-direita
#   4. fundo-esquerda
#
# Se não souber os pontos, rode a próxima célula para visualizar a imagem com eixos e estimar as coordenadas dos cantos.


# ======================================================================
# CÉLULA 7 - CODE
# ======================================================================
# Ajuste os nomes depois do upload.


base  = os.path.dirname(os.path.abspath(__file__))
pasta = os.path.join(base, "test_depth")
RGB_PATH   = os.path.join(pasta, "blenderTest2.png")
DEPTH_PATH =  os.path.join(pasta, "img2.png")

# Se tiver depth map, troque None pelo nome do arquivo.
# Exemplo: DEPTH_PATH = "depth.png"
DEPTH_PATH = None

LARGURA_BANCADA_CM = 100.0
PROFUNDIDADE_BANCADA_CM = 60.0

# Ordem: frente-esquerda, frente-direita, fundo-direita, fundo-esquerda.
# Comece com a imagem inteira como aproximação. Depois ajuste manualmente.
PONTOS_BANCADA_PX = np.array([
    [0, 0],
    [999, 0],
    [999, 599],
    [0, 599],
], dtype=np.float32)

# Segmentação RGB
MIN_AREA_PX = 500
MAX_AREA_PX = None          # Exemplo: 50000 ou None
BG_KERNEL = 81              # tamanho para estimar fundo; aumente se objetos forem grandes
DIFF_THRESHOLD = None       # None usa Otsu; ou coloque número como 15, 20, 30...
USAR_SATURACAO = True
SAT_THRESHOLD = 35

# Segmentação por depth, se DEPTH_PATH não for None
USAR_DEPTH_SE_EXISTIR = True
DEPTH_CLIP_START_M = 0.1
DEPTH_CLIP_END_M = 10.0
DEPTH_INVERTIDO = False     # troque para True se seu depth estiver invertido
ALTURA_MINIMA_M = 0.02      # 2 cm acima da bancada
DEPTH_BG_KERNEL = 81

print("RGB_PATH:", RGB_PATH)
print("DEPTH_PATH:", DEPTH_PATH)


# ======================================================================
# CÉLULA 8 - MARKDOWN
# ======================================================================
#
# ## 4. Carregar e visualizar a imagem
#
# Use esta visualização para ajustar os quatro cantos da bancada na célula anterior.


# ======================================================================
# CÉLULA 9 - CODE
# ======================================================================
img_bgr = cv2.imread(RGB_PATH, cv2.IMREAD_COLOR)

if img_bgr is None:
    raise FileNotFoundError(f"Não consegui carregar RGB_PATH={RGB_PATH}. Verifique o nome do arquivo.")

h, w = img_bgr.shape[:2]
print(f"Imagem carregada: {w} x {h} pixels")

# Se os pontos ainda estiverem no exemplo 999x599, ajusta para o tamanho real da imagem.
if np.allclose(PONTOS_BANCADA_PX, np.array([[0,0],[999,0],[999,599],[0,599]], dtype=np.float32)):
    PONTOS_BANCADA_PX = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ], dtype=np.float32)

img_pontos = img_bgr.copy()
for i, (x, y) in enumerate(PONTOS_BANCADA_PX.astype(int), start=1):
    cv2.circle(img_pontos, (x, y), 8, (0, 255, 255), -1)
    cv2.putText(img_pontos, str(i), (x + 8, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

cv2.polylines(img_pontos, [PONTOS_BANCADA_PX.astype(int)], isClosed=True, color=(0, 255, 255), thickness=2)
mostrar(img_pontos, "Imagem RGB com cantos da bancada")


# ======================================================================
# CÉLULA 10 - MARKDOWN
# ======================================================================
#
# ## 5. Homografia: converter pixel para coordenada real da bancada
#
# A homografia é a forma mais simples e correta de obter X/Y reais quando os objetos estão sobre um plano conhecido, como uma mesa ou bancada.
#
# Aqui o referencial adotado é:
#
# - X = largura da bancada, da esquerda para a direita.
# - Y = profundidade da bancada, da frente para o fundo.
# - Unidade: centímetros.


# ======================================================================
# CÉLULA 11 - CODE
# ======================================================================
def construir_homografia(pontos_px, largura_cm, profundidade_cm):
    src = np.asarray(pontos_px, dtype=np.float32)
    dst = np.array([
        [0, 0],
        [largura_cm, 0],
        [largura_cm, profundidade_cm],
        [0, profundidade_cm],
    ], dtype=np.float32)
    H, status = cv2.findHomography(src, dst)
    if H is None:
        raise RuntimeError("Não foi possível calcular a homografia. Verifique os 4 pontos da bancada.")
    return H

def pixel_para_bancada_cm(x, y, H):
    pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
    convertido = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(convertido[0]), float(convertido[1])

H_px_para_cm = construir_homografia(PONTOS_BANCADA_PX, LARGURA_BANCADA_CM, PROFUNDIDADE_BANCADA_CM)

print("Homografia pixel -> bancada:")
print(H_px_para_cm)

# Teste nos cantos
for i, (x, y) in enumerate(PONTOS_BANCADA_PX, start=1):
    Xcm, Ycm = pixel_para_bancada_cm(x, y, H_px_para_cm)
    print(f"Canto {i}: pixel=({x:.1f},{y:.1f}) -> bancada=({Xcm:.1f} cm, {Ycm:.1f} cm)")


# ======================================================================
# CÉLULA 12 - MARKDOWN
# ======================================================================
#
# ## 6. Segmentação RGB das peças
#
# Esta etapa tenta separar peças da bancada usando contraste local, saturação e morfologia.
#
# Não existe um parâmetro universal. Para uma demonstração limpa, use:
#
# - fundo o mais uniforme possível;
# - peças sem contato entre si;
# - iluminação razoável;
# - câmera fixa;
# - imagem com boa resolução.
#
# Se a máscara ficar pegando muita sombra, aumente `DIFF_THRESHOLD`, aumente `MIN_AREA_PX` ou desative `USAR_SATURACAO`.


# ======================================================================
# CÉLULA 13 - CODE
# ======================================================================
def segmentar_rgb(img_bgr, bg_kernel=81, diff_threshold=None, usar_saturacao=True, sat_threshold=35):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    bg_kernel = garantir_ksize_impar(bg_kernel)
    fundo = cv2.medianBlur(gray, bg_kernel)

    diff = cv2.absdiff(gray, fundo)

    if diff_threshold is None:
        _, mask_diff = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask_diff = cv2.threshold(diff, int(diff_threshold), 255, cv2.THRESH_BINARY)

    if usar_saturacao:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        _, mask_sat = cv2.threshold(sat, int(sat_threshold), 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(mask_diff, mask_sat)
    else:
        mask = mask_diff

    # Limpeza morfológica
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    return mask, diff, fundo

mask_rgb, diff_rgb, fundo_rgb = segmentar_rgb(
    img_bgr,
    bg_kernel=BG_KERNEL,
    diff_threshold=DIFF_THRESHOLD,
    usar_saturacao=USAR_SATURACAO,
    sat_threshold=SAT_THRESHOLD,
)

mostrar(diff_rgb, "Mapa de diferença local usado na segmentação", cmap="gray")
mostrar(mask_rgb, "Máscara RGB inicial", cmap="gray")


# ======================================================================
# CÉLULA 14 - MARKDOWN
# ======================================================================
#
# ## 7. Opcional: segmentação por depth map
#
# Use esta etapa apenas se você tiver um depth map verdadeiro.
#
# A ideia é:
#
# 1. Converter o depth bruto para uma escala contínua.
# 2. Estimar a profundidade local da bancada.
# 3. Considerar como objeto tudo que estiver mais próximo da câmera do que a bancada.
# 4. Usar essa diferença como altura/profundidade relativa.
#
# Se `DEPTH_PATH = None`, o notebook segue apenas com a máscara RGB.


# ======================================================================
# CÉLULA 15 - CODE
# ======================================================================
def carregar_depth_metros(depth_path, shape_rgb, clip_start_m=0.1, clip_end_m=10.0, invertido=False):
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Não consegui carregar DEPTH_PATH={depth_path}")

    if depth_raw.ndim == 3:
        # Se for uma imagem colorida, converte para cinza.
        # Atenção: depth colorizado não é depth real; é apenas uma visualização.
        depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)

    dtype_original = depth_raw.dtype
    raw = depth_raw.astype(np.float32)

    if np.issubdtype(dtype_original, np.integer):
        maxv = float(np.iinfo(dtype_original).max)
        norm = raw / maxv
    else:
        minv, maxv = float(np.nanmin(raw)), float(np.nanmax(raw))
        norm = (raw - minv) / (maxv - minv + 1e-9)

    if invertido:
        norm = 1.0 - norm

    depth_m = clip_start_m + norm * (clip_end_m - clip_start_m)

    # Ajusta para o tamanho da imagem RGB, se necessário.
    h_rgb, w_rgb = shape_rgb[:2]
    if depth_m.shape[:2] != (h_rgb, w_rgb):
        depth_m = cv2.resize(depth_m, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)

    return depth_m

def segmentar_por_depth(depth_m, altura_minima_m=0.02, bg_kernel=81):
    bg_kernel = garantir_ksize_impar(bg_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_kernel, bg_kernel))

    # Como objetos estão mais perto da câmera, eles tendem a ter depth menor.
    # O fechamento morfológico ajuda a "preencher" objetos com a profundidade do fundo.
    fundo_depth = cv2.morphologyEx(depth_m.astype(np.float32), cv2.MORPH_CLOSE, kernel)

    altura_m = fundo_depth - depth_m
    altura_m = np.maximum(altura_m, 0)

    mask = (altura_m >= float(altura_minima_m)).astype(np.uint8) * 255

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    return mask, altura_m, fundo_depth

depth_m = None
altura_m = None
mask_depth = None

if DEPTH_PATH is not None and USAR_DEPTH_SE_EXISTIR:
    depth_m = carregar_depth_metros(
        DEPTH_PATH,
        img_bgr.shape,
        clip_start_m=DEPTH_CLIP_START_M,
        clip_end_m=DEPTH_CLIP_END_M,
        invertido=DEPTH_INVERTIDO,
    )
    mask_depth, altura_m, fundo_depth = segmentar_por_depth(
        depth_m,
        altura_minima_m=ALTURA_MINIMA_M,
        bg_kernel=DEPTH_BG_KERNEL,
    )

    mostrar(depth_m, "Depth convertido para metros", cmap="turbo")
    mostrar(altura_m, "Altura relativa estimada: fundo_depth - depth", cmap="turbo")
    mostrar(mask_depth, "Máscara por depth", cmap="gray")
else:
    print("Sem DEPTH_PATH. Usando somente segmentação RGB.")


# ======================================================================
# CÉLULA 16 - MARKDOWN
# ======================================================================
#
# ## 8. Escolher máscara final
#
# Por padrão:
#
# - se houver depth map válido, usa `mask_depth`;
# - caso contrário, usa `mask_rgb`.
#
# Você pode forçar manualmente usando:
#
# ```python
# mask_final = mask_rgb
# ```
#
# ou
#
# ```python
# mask_final = mask_depth
# ```


# ======================================================================
# CÉLULA 17 - CODE
# ======================================================================
if mask_depth is not None:
    mask_final = mask_depth.copy()
    origem_mascara = "depth"
else:
    mask_final = mask_rgb.copy()
    origem_mascara = "rgb"

print("Máscara final:", origem_mascara)
mostrar(mask_final, "Máscara final usada para detectar as peças", cmap="gray")


# ======================================================================
# CÉLULA 18 - MARKDOWN
# ======================================================================
#
# ## 9. Detectar peças, calcular coordenadas e montar tabela
#
# A saída principal é uma tabela com:
#
# - `id`: identificador da peça;
# - `pixel_x`, `pixel_y`: centro da peça na imagem;
# - `x_cm`, `y_cm`: coordenada real estimada na bancada;
# - `z_cm`: altura/profundidade relativa, se houver depth map;
# - `area_px`: área do contorno em pixels;
# - `bbox`: caixa delimitadora em pixels.


# ======================================================================
# CÉLULA 19 - CODE
# ======================================================================
def detectar_pecas(mask, H_px_para_cm, img_bgr, min_area_px=500, max_area_px=None, altura_m=None):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    registros = []
    contornos_validos = []

    max_area_val = float("inf") if max_area_px is None else float(max_area_px)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < float(min_area_px) or area > max_area_val:
            continue

        M = cv2.moments(contour)
        if abs(M["m00"]) < 1e-9:
            continue

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        x_cm, y_cm = pixel_para_bancada_cm(cX, cY, H_px_para_cm)

        x, y, bw, bh = cv2.boundingRect(contour)

        z_cm = np.nan
        if altura_m is not None:
            mask_obj = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(mask_obj, [contour], -1, 255, thickness=-1)
            valores = altura_m[mask_obj > 0]
            valores = valores[np.isfinite(valores)]
            valores = valores[valores > 0]
            if len(valores) > 0:
                z_cm = float(np.median(valores) * 100.0)

        registros.append({
            "id": None,
            "pixel_x": cX,
            "pixel_y": cY,
            "x_cm": x_cm,
            "y_cm": y_cm,
            "z_cm": z_cm,
            "area_px": float(area),
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": bw,
            "bbox_h": bh,
        })
        contornos_validos.append(contour)

    # Ordena de cima para baixo e esquerda para direita para IDs ficarem estáveis.
    ordem = sorted(range(len(registros)), key=lambda i: (registros[i]["y_cm"], registros[i]["x_cm"]))

    registros_ord = []
    contornos_ord = []
    for novo_id, idx in enumerate(ordem, start=1):
        reg = registros[idx].copy()
        reg["id"] = novo_id
        registros_ord.append(reg)
        contornos_ord.append(contornos_validos[idx])

    df = pd.DataFrame(registros_ord)
    return df, contornos_ord

df_pecas, contornos_pecas = detectar_pecas(
    mask_final,
    H_px_para_cm,
    img_bgr,
    min_area_px=MIN_AREA_PX,
    max_area_px=MAX_AREA_PX,
    altura_m=altura_m if mask_depth is not None else None,
)

mostrar(df_pecas.round(2))
print(f"Total de peças detectadas: {len(df_pecas)}")


# ======================================================================
# CÉLULA 20 - MARKDOWN
# ======================================================================
#
# ## 10. Visualização final com coordenadas na imagem


# ======================================================================
# CÉLULA 21 - CODE
# ======================================================================
def desenhar_resultado(img_bgr, df, contornos):
    saida = img_bgr.copy()

    for _, row in df.iterrows():
        idx = int(row["id"])
        contour = contornos[idx - 1]

        cX, cY = int(row["pixel_x"]), int(row["pixel_y"])
        x_cm, y_cm = float(row["x_cm"]), float(row["y_cm"])
        z_cm = row["z_cm"]

        cv2.drawContours(saida, [contour], -1, (0, 255, 0), 2)
        cv2.circle(saida, (cX, cY), 5, (0, 0, 255), -1)

        if pd.isna(z_cm):
            texto = f"#{idx} X={x_cm:.1f} Y={y_cm:.1f} cm"
        else:
            texto = f"#{idx} X={x_cm:.1f} Y={y_cm:.1f} Z={z_cm:.1f} cm"

        cv2.putText(
            saida,
            texto,
            (cX + 8, cY - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            saida,
            texto,
            (cX + 8, cY - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return saida

img_resultado = desenhar_resultado(img_bgr, df_pecas, contornos_pecas)
mostrar(img_resultado, "Resultado final: peças detectadas e coordenadas")


# ======================================================================
# CÉLULA 22 - MARKDOWN
# ======================================================================
#
# ## 11. Exportar resultados
#
# Gera um CSV simples com as coordenadas das peças.


# ======================================================================
# CÉLULA 23 - CODE
# ======================================================================
saida_csv = "coordenadas_pecas.csv"
df_pecas.to_csv(saida_csv, index=False)
print(f"Arquivo salvo: {saida_csv}")

if IN_COLAB:
    # Descomente a linha abaixo se quiser baixar automaticamente.
    # files.download(saida_csv)
    pass


# ======================================================================
# CÉLULA 24 - MARKDOWN
# ======================================================================
#
# ## 12. Texto curto para explicar o método à professora
#
# Você pode usar este resumo na apresentação:
#
# > O objetivo foi detectar peças sobre uma bancada e estimar suas coordenadas. Primeiro, a imagem foi pré-processada para separar objetos do fundo por contraste local, saturação e operações morfológicas. Em seguida, os contornos foram extraídos, e o centroide de cada peça foi calculado por momentos de imagem. Para converter pixels em coordenadas reais da bancada, usei uma homografia calculada a partir dos quatro cantos conhecidos da área de trabalho. Quando existe depth map válido, o programa também estima uma altura relativa comparando a profundidade local da peça com a profundidade estimada da bancada. Assim, a saída final informa o centro de cada peça em pixels e sua posição aproximada em centímetros no plano da bancada.
#
# ## Limitações honestas
#
# - Com uma única imagem RGB, o programa estima X/Y no plano, mas não estima Z real.
# - Sombras de luz direcional podem ajudar visualmente, mas também podem atrapalhar a segmentação.
# - Para Z real, é necessário depth map, estéreo, sensor de profundidade ou photometric stereo com várias imagens e luzes conhecidas.
# - A calibração dos quatro cantos da bancada é decisiva: pontos errados geram coordenadas erradas.
