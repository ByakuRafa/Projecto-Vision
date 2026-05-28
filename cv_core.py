"""
cv_core.py — Pipeline de Estéreo Fotométrico · Modo Blender
============================================================

PRINCÍPIO DO ESTÉREO FOTOMÉTRICO (Photometric Stereo):
    Tiramos 4 fotos do mesmo objeto, cada uma com luz vindo de uma
    direção diferente (N, S, L, O a 45° de elevação). Como a câmara
    não se move, cada pixel recebe intensidades diferentes dependendo
    de como a superfície está orientada em relação a cada luz.

    O modelo físico usado é o de Lambert (superfície difusa, sem brilho):
        I = albedo · (N · L)
    Onde:
        I      = intensidade medida (0–1)
        albedo = reflectividade da superfície (escalar por pixel)
        N      = vetor normal da superfície (o que queremos calcular)
        L      = vetor unitário em direção à luz (conhecido)

    Com 4 equações (uma por imagem) e 3 incógnitas (Nx, Ny, Nz),
    o sistema é sobredeterminado e resolvemos por mínimos quadrados.
    O albedo aparece como a magnitude do vetor solução.

    Do normal map, derivamos o depth map integrando os gradientes
    de superfície (∂Z/∂x = -Nx/Nz, ∂Z/∂y = -Ny/Nz).

ORGANIZAÇÃO DO ARQUIVO:
    Seção 1  — Importações e tipos
    Seção 2  — Vetores de luz (fórmulas de direção da luz)
    Seção 3  — Campo de visão da câmara (tamanho físico visível)
    Seção 4  — Máscaras ROI (região de interesse)
    Seção 5  — Carregamento e normalização de imagens
    Seção 6  — Subtração de luz ambiente
    Seção 7  — Máscara de pixels válidos
    Seção 8  — Resolução fotométrica (lstsq + drop-darkest)
    Seção 9  — Rejeição de pixels ruins por resíduo
    Seção 10 — Detecção de sombras cast
    Seção 11 — Construção do normal map
    Seção 12 — Suavização de normais (opcional)
    Seção 13 — Codificação do normal map em RGB
    Seção 14 — Gradientes de superfície (normal → slope)
    Seção 15 — Integração de gradientes (slope → altura)
    Seção 16 — Normalização e visualização do depth map
    Seção 17 — Construção do albedo map
    Seção 18 — Salvar resultados
    Seção 19 — Pipeline principal (orquestra todas as etapas)
"""

# ==============================================================
# SEÇÃO 1 — IMPORTAÇÕES
# ==============================================================

import os

import cv2
import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter, uniform_filter


# ==============================================================
# SEÇÃO 2 — VETORES DE LUZ
# ==============================================================
# Um vetor de luz descreve de onde a luz vem: é um vetor unitário
# apontando DA superfície EM DIREÇÃO à fonte de luz.
# Lx = componente leste/oeste, Ly = norte/sul, Lz = vertical.
# ==============================================================

def angulos_para_vetor_unitario(lx: float, ly: float, lz: float) -> list[float]:
    """
    Normaliza qualquer vetor (lx, ly, lz) para comprimento 1.

    Por quê normalizar: o modelo de Lambert usa N·L onde ambos devem
    ser unitários. Se L não for unitário, o albedo calculado fica errado.

    Retorna: [lx, ly, lz] com lx²+ly²+lz² = 1.
    """
    mag = np.sqrt(lx**2 + ly**2 + lz**2)
    return [float(lx / mag), float(ly / mag), float(lz / mag)]


def vetor_luz_solar(azimute_graus: float, elevacao_graus: float = 45.0) -> list[float]:
    """
    Calcula o vetor de luz para uma fonte Solar/Direcional do Blender.

    A luz solar não tem posição — só direção. Com azimute e elevação:
        Lx = sin(az) · cos(el)   ← componente leste (az=90°)
        Ly = cos(az) · cos(el)   ← componente norte (az=0°)
        Lz = sin(el)             ← componente vertical (sempre >0)

    Convenção de azimute (0°=Norte, sentido horário):
        N  (az=0°)   → [ 0.000,  0.707, 0.707]
        E  (az=90°)  → [ 0.707,  0.000, 0.707]
        S  (az=180°) → [ 0.000, -0.707, 0.707]
        W  (az=270°) → [-0.707,  0.000, 0.707]

    Nota: os 4 vetores com az=0/90/180/270 e elevacao=45° são
    simétricos, o que torna o sistema de equações bem condicionado.
    """
    az = np.radians(azimute_graus)
    el = np.radians(elevacao_graus)
    lx = np.sin(az) * np.cos(el)
    ly = np.cos(az) * np.cos(el)
    lz = np.sin(el)
    return angulos_para_vetor_unitario(lx, ly, lz)


def vetor_luz_pontual(azimute_graus: float,
                      altura_m: float,
                      distancia_m: float) -> list[float]:
    """
    Calcula o vetor de luz para uma fonte pontual a posição conhecida.

    Útil quando a luz tem posição (não direcional). A direção é
    calculada a partir das coordenadas cartesianas da fonte:
        Lx =  distancia · sin(az)   ← posição horizontal leste
        Ly =  distancia · cos(az)   ← posição horizontal norte
        Lz =  altura                ← posição vertical

    O vetor resultante aponta da origem (objeto) para a luz.
    """
    rad = np.radians(azimute_graus)
    lx  = distancia_m * np.sin(rad)
    ly  = distancia_m * np.cos(rad)
    lz  = altura_m
    return angulos_para_vetor_unitario(lx, ly, lz)


# ==============================================================
# SEÇÃO 3 — CAMPO DE VISÃO DA CÂMARA
# ==============================================================
# "Quanto do mundo físico (em mm) a câmara enxerga?"
# Essa resposta é diferente para câmara perspectiva e ortográfica.
# ==============================================================

def campo_visao_perspectiva(camera_h_mm: float,
                             sensor_w_mm: float,
                             focal_mm: float,
                             img_largura: int,
                             img_altura: int) -> tuple[float, float]:
    """
    Calcula a área física visível por uma câmara PERSPECTIVA apontada para baixo.

    Fórmula: phys_w = altura_camera × largura_sensor / focal
    Intuição: quanto mais longe a câmara, mais ela enxerga.
              quanto maior a focal, mais ela "zoom in" e enxerga menos.

    Retorna: (largura_fisica_mm, altura_fisica_mm) da cena visível.
    """
    phys_w = camera_h_mm * sensor_w_mm / focal_mm
    phys_h = phys_w * img_altura / img_largura
    return phys_w, phys_h


def campo_visao_ortografica(ortho_scale_mm: float,
                             img_largura: int,
                             img_altura: int) -> tuple[float, float]:
    """
    Calcula a área física visível por uma câmara ORTOGRÁFICA do Blender.

    Numa câmara ortográfica NÃO há perspectiva — o campo de visão
    é definido diretamente pelo parâmetro "Orthographic Scale" do Blender,
    independente da distância, focal ou sensor.

    O valor no Blender está em metros. Converta para mm antes de passar
    (ex: Blender mostra 1.28 → passe 1280.0).

    Retorna: (largura_fisica_mm, altura_fisica_mm) da cena visível.
    """
    phys_w = ortho_scale_mm
    phys_h = ortho_scale_mm * img_altura / img_largura
    return phys_w, phys_h


# ==============================================================
# SEÇÃO 4 — MÁSCARAS ROI (Região de Interesse)
# ==============================================================
# A ROI define quais pixels da imagem correspondem à bancada/objeto.
# Pixels fora da ROI são ignorados no processamento.
# ==============================================================

def calcular_roi_em_pixels(phys_w: float,
                            phys_h: float,
                            wb_larg_mm: float,
                            wb_prof_mm: float,
                            img_largura: int,
                            img_altura: int) -> tuple[float, float]:
    """
    Converte as dimensões físicas da bancada em meia-largura e meia-altura em pixels.

    A câmara está centrada sobre a bancada, então a ROI é um retângulo
    centrado na imagem. Calculamos metade do tamanho (half-width, half-height)
    para facilitar o desenho a partir do centro.

    Se a bancada for maior que o campo de visão, limitamos ao tamanho da imagem.

    Retorna: (roi_hw_px, roi_hh_px) — meias dimensões em pixels.
    """
    roi_hw = min((wb_larg_mm / phys_w) * img_largura / 2.0, img_largura / 2.0)
    roi_hh = min((wb_prof_mm / phys_h) * img_altura  / 2.0, img_altura  / 2.0)
    return roi_hw, roi_hh


def desenhar_mascara_retangular(img_largura: int,
                                 img_altura: int,
                                 roi_hw: float,
                                 roi_hh: float,
                                 erosao_px: int = 8) -> np.ndarray:
    """
    Desenha uma máscara booleana com um retângulo centrado na imagem.

    A erosão (encolhimento das bordas) exclui os pixels de borda da ROI,
    que costumam ter iluminação mais ruidosa por difração e reflexos.

    Retorna: array bool (img_altura × img_largura), True = dentro da ROI.
    """
    cx, cy = img_largura / 2.0, img_altura / 2.0
    pts = np.array([
        [cx - roi_hw, cy - roi_hh],   # canto superior esquerdo
        [cx + roi_hw, cy - roi_hh],   # canto superior direito
        [cx + roi_hw, cy + roi_hh],   # canto inferior direito
        [cx - roi_hw, cy + roi_hh],   # canto inferior esquerdo
    ], dtype=np.int32)

    mascara = np.zeros((img_altura, img_largura), dtype=np.uint8)
    cv2.fillPoly(mascara, [pts], 255)

    if erosao_px > 0:
        kernel  = np.ones((erosao_px, erosao_px), np.uint8)
        mascara = cv2.erode(mascara, kernel, iterations=1)

    return mascara.astype(bool)


def roi_automatica_blender(img_largura: int,
                            img_altura: int,
                            wb_larg_mm: float,
                            wb_prof_mm: float,
                            camera_h_mm: float,
                            focal_mm: float,
                            sensor_w_mm: float,
                            ortho_scale_mm: float | None = None,
                            erosao_px: int = 8) -> np.ndarray:
    """
    Gera máscara ROI automaticamente a partir dos parâmetros da câmara Blender.

    Escolhe automaticamente entre câmara ortográfica e perspectiva:
      - Se ortho_scale_mm for fornecido → usa campo_visao_ortografica()
      - Se não → usa campo_visao_perspectiva() como fallback

    Retorna: máscara bool (img_altura × img_largura).
    """
    if ortho_scale_mm is not None:
        phys_w, phys_h = campo_visao_ortografica(ortho_scale_mm, img_largura, img_altura)
    else:
        phys_w, phys_h = campo_visao_perspectiva(
            camera_h_mm, sensor_w_mm, focal_mm, img_largura, img_altura
        )

    roi_hw, roi_hh = calcular_roi_em_pixels(
        phys_w, phys_h, wb_larg_mm, wb_prof_mm, img_largura, img_altura
    )
    return desenhar_mascara_retangular(img_largura, img_altura, roi_hw, roi_hh, erosao_px)


def roi_manual_por_cantos(plano_coords: list[dict],
                           altura: int,
                           largura: int,
                           erosao_px: int = 8) -> np.ndarray:
    """
    Cria máscara ROI a partir de 4 cantos marcados manualmente pelo usuário.

    plano_coords: lista de 4 dicionários com chaves 'x' e 'y' em pixels,
                  marcando os cantos da bancada na imagem.

    Retorna: máscara bool (altura × largura).
    """
    pts     = np.array([[p['x'], p['y']] for p in plano_coords], dtype=np.int32)
    mascara = np.zeros((altura, largura), dtype=np.uint8)
    cv2.fillPoly(mascara, [pts], 255)
    if erosao_px > 0:
        kernel  = np.ones((erosao_px, erosao_px), np.uint8)
        mascara = cv2.erode(mascara, kernel, iterations=1)
    return mascara.astype(bool)


# ==============================================================
# SEÇÃO 5 — CARREGAMENTO DE IMAGENS
# ==============================================================

def carregar_imagem_cinza(filepath: str) -> np.ndarray:
    """
    Carrega uma imagem do disco e converte para escala de cinza normalizada.

    Por que escala de cinza: o modelo de Lambert trabalha com intensidade
    luminosa (escalar), não cor. Usar grayscale é equivalente a usar a
    luminância da imagem.

    Por que normalizar para [0, 1]: evita problemas numéricos com valores
    inteiros e mantém o albedo numa escala consistente.

    Retorna: array float64 (altura × largura), valores em [0.0, 1.0].
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {filepath}")
    return img.astype(np.float64) / 255.0


# ==============================================================
# SEÇÃO 6 — SUBTRAÇÃO DE LUZ AMBIENTE
# ==============================================================

def estimar_luz_ambiente(imgs: list[np.ndarray]) -> np.ndarray:
    """
    Estima a componente de luz ambiente pixel a pixel.

    Método: o mínimo entre as N imagens em cada pixel.

    Raciocínio: em pelo menos uma das 4 direções, cada pixel recebe
    a iluminação mínima (ou está em sombra própria). Esse mínimo
    representa a luz que "vaza" de todas as direções — a luz ambiente.

    Retorna: array (altura × largura) com a estimativa da luz ambiente.
    """
    stack = np.stack(imgs, axis=0)   # (N, H, W)
    return np.min(stack, axis=0)     # menor valor por pixel entre as N imagens


def subtrair_luz_ambiente(imgs: list[np.ndarray]) -> list[np.ndarray]:
    """
    Remove a componente de luz ambiente de todas as imagens.

    Subtrai a estimativa da luz ambiente de cada imagem e garante
    que nenhum pixel fique negativo (clip em 0).

    Por que fazer isso: sem essa subtração, o albedo calculado inclui
    a componente constante de luz ambiente, o que distorce as normais.

    Retorna: lista de imagens corrigidas, mesmas dimensões.
    """
    ambiente = estimar_luz_ambiente(imgs)
    return [np.clip(img - ambiente, 0.0, 1.0) for img in imgs]


# ==============================================================
# SEÇÃO 7 — MÁSCARA DE PIXELS VÁLIDOS
# ==============================================================

def calcular_variacao_entre_imagens(imgs: list[np.ndarray]) -> np.ndarray:
    """
    Calcula o desvio padrão pixel a pixel entre as N imagens.

    Um pixel com variação alta foi iluminado diferentemente em cada
    foto — sinal de que sua normal tem direção identificável.

    Um pixel com variação baixa pode estar:
      - Sempre em sombra (preto em todas as imagens)
      - Sempre saturado (branco em todas as imagens)
      - Em área plana perfeitamente horizontal (pouca variação mesmo assim)

    Retorna: array (altura × largura) com o desvio padrão por pixel.
    """
    stack = np.stack(imgs, axis=0)   # (N, H, W)
    return np.std(stack, axis=0)


def mascara_pixels_saturados(imgs: list[np.ndarray],
                               thresh_sat: float = 0.95) -> np.ndarray:
    """
    Cria máscara True para pixels que NÃO estão saturados em nenhuma imagem.

    Pixels saturados (valor ≥ thresh_sat) têm seu valor "cortado" pelo
    sensor — a intensidade real poderia ser qualquer coisa acima do máximo.
    Isso introduz erro sistemático no cálculo da normal.

    Retorna: máscara bool (altura × largura), True = pixel não saturado.
    """
    mascara = np.ones(imgs[0].shape, dtype=bool)
    for img in imgs:
        mascara &= (img < thresh_sat)
    return mascara


def criar_mascara_valida(imgs: list[np.ndarray],
                          mascara_roi: np.ndarray | None = None,
                          thresh_variacao: float = 0.04,
                          thresh_sat: float = 0.95) -> np.ndarray:
    """
    Combina todos os critérios para identificar pixels processáveis.

    Um pixel é válido se:
      1. Tem variação suficiente entre as imagens (std > thresh_variacao)
      2. Não está saturado em nenhuma das imagens
      3. Está dentro da ROI (se fornecida)

    Parâmetros:
      thresh_variacao: aumente (0.08) se aparecerem artefatos no fundo;
                       diminua (0.02) se objetos uniformes desaparecerem.
      thresh_sat:      pixels com intensidade > este valor são excluídos.

    Retorna: máscara bool (altura × largura).
    """
    variacao       = calcular_variacao_entre_imagens(imgs)
    mask_variacao  = variacao > thresh_variacao
    mask_saturacao = mascara_pixels_saturados(imgs, thresh_sat)

    mascara = mask_variacao & mask_saturacao
    if mascara_roi is not None:
        mascara &= mascara_roi
    return mascara


# ==============================================================
# SEÇÃO 8 — RESOLUÇÃO FOTOMÉTRICA (LSTSQ + DROP-DARKEST)
# ==============================================================
# O coração do algoritmo: resolver I = albedo · (N · L) para N e albedo.
# Com 4 imagens, temos I = L · G onde G = albedo · N (vetor escalonado).
# Resolvemos por pseudoinversa: G = pinv(L) · I
# ==============================================================

def normalizar_vetores_luz(vetores_luz: list[list[float]]) -> np.ndarray:
    """
    Converte a lista de vetores de luz em matriz e renormaliza cada linha.

    Garante que todos os vetores sejam unitários mesmo se o usuário
    passou valores aproximados ou com pequeno erro de arredondamento.

    Retorna: array (4 × 3) com os vetores de luz como linhas.
    """
    L = np.array(vetores_luz, dtype=np.float64)
    L = L / np.linalg.norm(L, axis=1, keepdims=True)
    return L


def preparar_pseudoinversas(L: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Pré-calcula a pseudoinversa de L com todas as 4 luzes e as 4 variações
    com cada uma das luzes removida (para o método Drop-Darkest).

    Por que pré-calcular: a pseudoinversa é cara de calcular. Como usamos
    a mesma L para todos os pixels, calculamos uma vez e reutilizamos.

    Retorna:
      L_inv4  : pseudoinversa (3×4) de L completo — usada para pixels sem sombra
      L_invs  : lista de 4 pseudoinversas (3×3), cada uma sem a luz k
    """
    L_inv4 = np.linalg.pinv(L)                                        # (3, 4)
    L_invs = [np.linalg.pinv(np.delete(L, k, axis=0)) for k in range(4)]  # 4× (3, 3)
    return L_inv4, L_invs


def detectar_sombra_propria(I_val: np.ndarray,
                             razao_sombra: float = 0.60) -> tuple[np.ndarray, np.ndarray]:
    """
    Identifica quais pixels têm uma luz muito mais escura que as demais (sombra própria).

    Método Drop-Darkest: se o valor mínimo de um pixel for menor que
    razao_sombra × mediana, aquela luz provavelmente está na sombra própria
    (a superfície está "de costas" para aquela luz naquele ponto).

    Parâmetros:
      razao_sombra: limiar para considerar sombra. 0.60 = descarta se
                    min < 60% da mediana. Aumente para ser mais agressivo.

    Retorna:
      deve_drop : bool (N_val,) — True se o pixel tem sombra evidente
      min_idx   : int  (N_val,) — qual das 4 luzes é a mais escura
    """
    min_idx   = np.argmin(I_val, axis=0)                          # (N_val,)
    med_val   = np.median(I_val, axis=0)                          # (N_val,)
    min_val   = I_val[min_idx, np.arange(I_val.shape[1])]         # (N_val,)
    deve_drop = min_val < razao_sombra * (med_val + 1e-9)
    return deve_drop, min_idx


def resolver_lstsq_pixels_validos(I_val: np.ndarray,
                                   L: np.ndarray,
                                   L_inv4: np.ndarray,
                                   L_invs: list[np.ndarray],
                                   deve_drop: np.ndarray,
                                   min_idx: np.ndarray) -> np.ndarray:
    """
    Resolve o sistema fotométrico I = L · G para cada pixel válido.

    Para pixels SEM sombra própria → usa todas as 4 luzes (mais estável).
    Para pixels COM sombra própria → remove a luz mais escura e usa 3 luzes.

    G = albedo · N é o vetor "normal escalonado". Para obter a normal:
        albedo = ||G||
        N      = G / albedo

    Retorna: G_val array (3, N_val) com os vetores escalonados por pixel.
    """
    G_val = np.zeros((3, I_val.shape[1]), dtype=np.float64)

    # Pixels sem sombra: resolve com as 4 luzes
    idx_sem = np.where(~deve_drop)[0]
    if idx_sem.size:
        G_val[:, idx_sem] = L_inv4 @ I_val[:, idx_sem]

    # Pixels com sombra: remove a luz mais escura e resolve com 3
    for k in range(4):
        sel = np.where(deve_drop & (min_idx == k))[0]
        if sel.size == 0:
            continue
        I_sub = np.delete(I_val[:, sel], k, axis=0)   # remove linha k → (3, |sel|)
        G_val[:, sel] = L_invs[k] @ I_sub

    n_drop  = deve_drop.sum()
    pct_drop = 100.0 * n_drop / max(1, I_val.shape[1])
    print(f"  [sombra própria]  {n_drop} px com luz descartada ({pct_drop:.1f}%)")

    return G_val


def extrair_albedo_e_normal(G_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Separa o vetor G = albedo · N em albedo (escalar) e normal (unitário).

    albedo = ||G||     → magnitude do vetor = reflectividade da superfície
    N      = G / ||G|| → direção normalizada = orientação da superfície

    Pixels com albedo ≈ 0 (sem sinal) recebem normal zero — serão
    tratados como inválidos nas etapas posteriores.

    Retorna:
      albedo_val : array (N_val,)    — reflectividade por pixel
      N_val      : array (3, N_val)  — normais unitárias por pixel
    """
    albedo_val = np.linalg.norm(G_val, axis=0)
    N_val      = np.divide(
        G_val, albedo_val,
        out=np.zeros_like(G_val),
        where=albedo_val > 1e-9
    )
    return albedo_val, N_val


# ==============================================================
# SEÇÃO 9 — REJEIÇÃO POR RESÍDUO
# ==============================================================

def calcular_residuo_fotometrico(I_val: np.ndarray,
                                  L: np.ndarray,
                                  G_val: np.ndarray,
                                  albedo_val: np.ndarray) -> np.ndarray:
    """
    Calcula o resíduo relativo entre intensidades medidas e preditas.

    Se a solução for perfeita (superfície Lambertiana sem ruído),
    I_predito = L · G deve bater exatamente com I_medido.

    Na prática, sombras, reflexos especulares e ruído fazem I_predito
    divergir de I_medido. O resíduo relativo indica o quão "inexplicável"
    é o padrão de iluminação daquele pixel.

        I_pred    = L · G_val       (intensidades preditas pelo modelo)
        resíduo   = mean|I - I_pred| / albedo   (erro relativo médio)

    Retorna: array (N_val,) com o resíduo relativo por pixel.
    """
    I_pred      = L @ G_val                                   # (4, N_val)
    residuo_abs = np.mean(np.abs(I_val - I_pred), axis=0)    # (N_val,)
    residuo_rel = residuo_abs / (albedo_val + 1e-6)
    return residuo_rel


def filtrar_por_residuo(residuo_rel: np.ndarray,
                         thresh_residuo: float = 0.20) -> np.ndarray:
    """
    Marca como válidos apenas os pixels cujo resíduo está abaixo do limiar.

    Pixels com resíduo alto têm padrão de iluminação que o modelo
    Lambertiano não consegue explicar — geralmente causado por:
      - Reflexo especular
      - Sombra cast não detectada pelo Drop-Darkest
      - Superfície inter-reflexiva (recebe luz refletida de outro objeto)

    thresh_residuo: 0.10=agressivo (remove mais), 0.40=suave (mantém mais).

    Retorna: máscara bool (N_val,), True = pixel confiável.
    """
    return residuo_rel < thresh_residuo


# ==============================================================
# SEÇÃO 10 — DETECÇÃO DE SOMBRAS CAST
# ==============================================================
# Sombra cast = sombra projetada por um objeto sobre outro (ou sobre o chão).
# É diferente da sombra própria (a parte do próprio objeto que não "vê" a luz).
# O Drop-Darkest lida bem com sombra própria mas mal com sombra cast,
# pois esta pode afetar múltiplas luzes simultaneamente.
# ==============================================================

def detectar_sombra_por_coeficiente_variacao(I_val: np.ndarray,
                                              limiar_cv: float = 0.65,
                                              limiar_min: float = 0.08) -> np.ndarray:
    """
    Detecta sombras cast pelo coeficiente de variação (CV = std/mean) das intensidades.

    Raciocínio:
      - Superfície real iluminada: valores de I variam com a inclinação,
        mas nenhuma luz vai a zero. CV moderado (~0.3–0.5).
      - Sombra cast: 1 ou 2 luzes chegam perto de zero enquanto as outras
        estão normais. CV alto (>0.65) combinado com min baixo (<0.08).

    Essa combinação (CV alto E mínimo muito baixo) é a "assinatura" da
    sombra cast que nenhuma normal Lambertiana real produz.

    Retorna: máscara bool (N_val,), True = provável sombra cast.
    """
    media_val = np.mean(I_val, axis=0)
    std_val   = np.std(I_val,  axis=0)
    cv        = std_val / (media_val + 1e-6)
    min_val   = np.min(I_val, axis=0)
    return (cv > limiar_cv) & (min_val < limiar_min)


def calcular_normais_medias_locais(G_val: np.ndarray,
                                    albedo_val: np.ndarray,
                                    idx_val: np.ndarray,
                                    altura: int,
                                    largura: int,
                                    raio_px: int = 5) -> np.ndarray:
    """
    Calcula a normal média de cada pixel olhando para seus vizinhos próximos.

    Monta o mapa de normais na resolução original da imagem, aplica
    um filtro de média (uniform_filter) com o raio especificado, e
    extrai os valores filtrados só nos pixels válidos.

    Usado para comparar a normal calculada de um pixel com o contexto
    dos vizinhos — se divergir muito, é sinal de erro (sombra cast, ruído).

    Retorna: array (3, N_val) com as normais médias locais.
    """
    N_bruta = G_val / (albedo_val + 1e-9)   # normais sem renormalizar

    N_mapa  = np.zeros((3, altura * largura))
    N_mapa[:, idx_val] = N_bruta

    N_local = np.zeros_like(N_mapa)
    tamanho_filtro = raio_px * 2 + 1
    for i in range(3):
        canal        = N_mapa[i].reshape(altura, largura)
        canal_smooth = uniform_filter(canal, size=tamanho_filtro)
        N_local[i]   = canal_smooth.flatten()

    return N_local[:, idx_val]   # só os pixels válidos


def detectar_sombra_por_inconsistencia_espacial(G_val: np.ndarray,
                                                 albedo_val: np.ndarray,
                                                 idx_val: np.ndarray,
                                                 altura: int,
                                                 largura: int,
                                                 raio_px: int = 5,
                                                 thresh_diff: float = 0.35) -> np.ndarray:
    """
    Detecta sombras cast por inconsistência entre a normal calculada e a média local.

    A normal de um pixel em sombra cast é calculada a partir de intensidades
    erradas (algumas com sombra, outras não) — então ela aponta numa direção
    diferente de todos os vizinhos ao redor.

    Métrica: ||N_pixel - N_media_vizinhos|| > thresh_diff

    thresh_diff: 0.20=agressivo (remove mais), 0.50=suave (mantém mais).

    Retorna: máscara bool (N_val,), True = provável sombra cast.
    """
    N_bruta = G_val / (albedo_val + 1e-9)
    N_local = calcular_normais_medias_locais(
        G_val, albedo_val, idx_val, altura, largura, raio_px
    )
    diff = np.linalg.norm(N_bruta - N_local, axis=0)   # (N_val,)
    return diff > thresh_diff


def detectar_sombras_cast(I_val: np.ndarray,
                           G_val: np.ndarray,
                           albedo_val: np.ndarray,
                           idx_val: np.ndarray,
                           altura: int,
                           largura: int,
                           thresh_sombra_cast: float = 0.35,
                           raio_contexto_px: int = 5) -> np.ndarray:
    """
    Combina os dois critérios de detecção de sombra cast.

    Critério A (coeficiente de variação): rápido, detecta o caso típico
    onde 1-2 luzes vão a zero enquanto as outras estão normais.

    Critério B (inconsistência espacial): mais sutil, detecta casos
    onde a sombra é menos extrema mas ainda produz normal aberrante.

    A união (A OR B) maximiza a detecção. Falsos positivos são
    possíveis mas controlados pelos limiares.

    Retorna: máscara bool (N_val,), True = sombra cast detectada.
    """
    sombra_cv = detectar_sombra_por_coeficiente_variacao(I_val)
    sombra_sp = detectar_sombra_por_inconsistencia_espacial(
        G_val, albedo_val, idx_val, altura, largura,
        raio_px=raio_contexto_px, thresh_diff=thresh_sombra_cast
    )
    # INTERSEÇÃO: exige os dois critérios simultaneamente.
    # Usar | (união) rejeitava superfícies inclinadas legítimas que têm
    # CV alto naturalmente. Exigir ambos garante que só são descartados
    # pixels onde TANTO o padrão de intensidade QUANTO a normal calculada
    # são inconsistentes — a "assinatura dupla" da sombra cast real.
    sombra_cast = sombra_cv & sombra_sp

    pct = 100.0 * sombra_cast.sum() / max(1, I_val.shape[1])
    print(f"  [sombra cast]     {sombra_cast.sum()} px identificados ({pct:.1f}%)")
    return sombra_cast


# ==============================================================
# SEÇÃO 11 — CONSTRUÇÃO DO NORMAL MAP
# ==============================================================

def montar_normal_map_flat(N_val: np.ndarray,
                            albedo_val: np.ndarray,
                            idx_final: np.ndarray,
                            validos_final: np.ndarray,
                            n_pixels: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Distribui as normais calculadas nos índices corretos do array plano.

    O algoritmo calcula normais só para os pixels válidos (idx_final),
    mas o mapa final precisa ter a mesma resolução da imagem.
    Esta função "espalha" os resultados nos lugares certos,
    deixando zeros onde não há normal calculada.

    Retorna:
      N_flat      : array (3, n_pixels) com normais distribuídas
      albedo_flat : array (n_pixels,)   com albedos distribuídos
    """
    N_flat      = np.zeros((3, n_pixels), dtype=np.float64)
    albedo_flat = np.zeros(n_pixels,      dtype=np.float64)

    N_flat[0, idx_final] = N_val[0, validos_final]
    N_flat[1, idx_final] = N_val[1, validos_final]
    N_flat[2, idx_final] = N_val[2, validos_final]
    albedo_flat[idx_final] = albedo_val[validos_final]

    return N_flat, albedo_flat


def preencher_fundo_com_normal_para_cima(N_flat: np.ndarray,
                                          mascara_roi: np.ndarray | None) -> np.ndarray:
    """
    Atribui normal [0, 0, 1] (superfície horizontal) aos pixels sem normal calculada.

    Pixels sem normal válida (dentro da ROI mas sem variação suficiente,
    ou rejeitados por sombra) recebem a normal do "chão plano".

    Em RGB isso codifica como [128, 128, 255] — o azul característico
    que representa uma superfície perpendicular à câmara.

    Retorna: máscara bool (n_pixels,) indicando quais pixels foram preenchidos.
    """
    normas     = np.linalg.norm(N_flat, axis=0)
    sem_normal = normas < 1e-9
    if mascara_roi is not None:
        sem_normal = mascara_roi.flatten() & sem_normal
    N_flat[2, sem_normal] = 1.0   # Z=1 → normal aponta para cima
    return sem_normal


# ==============================================================
# SEÇÃO 12 — SUAVIZAÇÃO DE NORMAIS (OPCIONAL)
# ==============================================================

def suavizar_normal_map(N_flat: np.ndarray,
                         altura: int,
                         largura: int,
                         sigma: float = 1.0) -> np.ndarray:
    """
    Aplica filtro Gaussiano em cada canal do normal map para reduzir ruído.

    Útil quando as imagens têm ruído de sensor ou a superfície tem
    textura fina que não é de interesse (papel, tecido, etc.).

    Após suavizar, renormaliza cada vetor para comprimento 1,
    pois o filtro pode encurtar os vetores nas bordas de objetos.

    sigma: raio da suavização em pixels. 1.0=suave, 3.0=agressivo.

    Retorna: N_flat suavizado e renormalizado (mesmas dimensões).
    """
    for i in range(3):
        canal       = N_flat[i].reshape(altura, largura)
        N_flat[i]   = gaussian_filter(canal, sigma=sigma).flatten()

    normas = np.linalg.norm(N_flat, axis=0)
    N_flat = np.divide(N_flat, normas,
                       out=np.zeros_like(N_flat),
                       where=normas > 1e-9)
    return N_flat


# ==============================================================
# SEÇÃO 13 — CORREÇÃO DE PERSPECTIVA (MODO MANUAL)
# ==============================================================

def estimar_rotacao_por_homografia(pontos_imagem: list[dict],
                                    largura_mm: float = 210.0,
                                    altura_mm: float = 297.0) -> np.ndarray:
    """
    Estima a matriz de rotação 3D da câmara a partir de 4 pontos de correspondência.

    Quando a câmara não está perfeitamente apontada para baixo, as normais
    calculadas estão no referencial da câmara, não do mundo. Esta função
    calcula a rotação necessária para corrigir isso.

    Método: calcula a homografia entre os 4 cantos marcados na imagem e
    os 4 cantos físicos conhecidos da bancada. Da homografia, extrai os
    vetores de rotação usando decomposição SVD.

    Retorna: matriz de rotação R (3×3).
    """
    w, h = float(largura_mm), float(altura_mm)
    pts_mundo = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    pts_img   = np.array([[p['x'], p['y']] for p in pontos_imagem], dtype=np.float32)

    H, _ = cv2.findHomography(pts_img, pts_mundo)
    if H is None:
        return np.eye(3)   # fallback: sem rotação

    # Extrai colunas da homografia como vetores de rotação aproximados
    h1  = H[:, 0]
    h2  = H[:, 1]
    lam = 1.0 / (np.linalg.norm(h1) + 1e-12)
    r1  = h1 * lam
    r2  = h2 * lam
    r3  = np.cross(r1, r2)
    R_raw = np.stack([r1, r2, r3], axis=1)

    # Projeta na matriz de rotação mais próxima via SVD
    U, _, Vt = np.linalg.svd(R_raw)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def aplicar_rotacao_nas_normais(N_flat: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Rotaciona todas as normais do referencial da câmara para o referencial do mundo.

    Multiplica cada vetor normal pela matriz de rotação R e renormaliza.

    Retorna: N_flat rotacionado (mesmas dimensões).
    """
    N_rot  = R @ N_flat
    normas = np.linalg.norm(N_rot, axis=0)
    return np.divide(N_rot, normas,
                     out=np.zeros_like(N_rot),
                     where=normas > 1e-9)


# ==============================================================
# SEÇÃO 14 — CODIFICAÇÃO DO NORMAL MAP EM RGB
# ==============================================================

def codificar_normal_map_rgb(nx: np.ndarray,
                              ny: np.ndarray,
                              nz: np.ndarray) -> np.ndarray:
    """
    Converte as componentes da normal (float em [-1, 1]) para imagem RGB uint8.

    Convenção OpenGL (padrão para normal maps de textura):
        R = (Nx + 1) / 2  →  +X aponta para direita   → vermelho
        G = (-Ny + 1) / 2 →  +Y aponta para cima      → verde   (flip Y!)
        B = (Nz + 1) / 2  →  +Z aponta para a câmara  → azul

    O flip em Y é necessário porque o eixo Y da imagem (pixels) cresce
    para BAIXO, mas a convenção de normal map espera Y crescendo para CIMA.

    Uma superfície horizontal (normal = [0,0,1]) resulta em RGB = [128,128,255]
    — o azul médio característico dos normal maps.

    Retorna: array uint8 (altura × largura × 3) em espaço RGB.
    """
    N_rgb = np.stack([
        np.clip(( nx + 1.0) * 127.5, 0, 255),    # R = Nx → direita
        np.clip((-ny + 1.0) * 127.5, 0, 255),    # G = -Ny → flip Y para cima
        np.clip(( nz + 1.0) * 127.5, 0, 255),    # B = Nz → câmara
    ], axis=-1).astype(np.uint8)
    return N_rgb


# ==============================================================
# SEÇÃO 15 — GRADIENTES DE SUPERFÍCIE
# ==============================================================
# Para integrar o depth map, precisamos dos gradientes ∂Z/∂x e ∂Z/∂y.
# Da equação da normal unitária N = [-∂Z/∂x, -∂Z/∂y, 1] / ||N||:
#     ∂Z/∂x = p = -Nx / Nz
#     ∂Z/∂y = q = -Ny / Nz
# ==============================================================

def identificar_pixels_chao(nx: np.ndarray,
                              ny: np.ndarray,
                              mascara_depth: np.ndarray,
                              thresh_horizontal: float = 0.08) -> np.ndarray:
    """
    Identifica pixels que são o chão/fundo (normal perfeitamente vertical).

    Um pixel é "chão" se:
      1. Sua normal aponta para cima: |Nx| e |Ny| são pequenos (próximos de zero)
      2. NÃO pertence a nenhum objeto detectado (não está em mascara_depth)

    A condição 2 é crucial: tampas planas de objetos também têm normal
    vertical, mas são PARTE do objeto — não devem ser tratadas como chão.

    thresh_horizontal: tolerância para considerar "vertical". 0.08 ≈ 4.6°.

    Retorna: máscara bool (altura × largura), True = pixel é chão.
    """
    normal_horizontal = (np.abs(nx) < thresh_horizontal) & (np.abs(ny) < thresh_horizontal)
    return normal_horizontal & ~mascara_depth


def identificar_bordas_perpendiculares(nz: np.ndarray,
                                        thresh_nz: float = 0.05) -> np.ndarray:
    """
    Identifica bordas com superfície quase perpendicular à câmara (nz ≈ 0).

    Quando Nz é muito pequeno, a divisão -Nx/Nz e -Ny/Nz produz gradientes
    enormes que corrompem o solver de integração. Excluímos esses pixels
    dos gradientes.

    thresh_nz: pixels com |nz| < este valor são excluídos.
    0.05 ≈ 87° de inclinação — só exclui superfícies quase verticais.
    Manter em 0.05 (não 0.20) garante que faces laterais a 45-70°
    contribuam com seus gradientes ao solver (importante para reconstrução).

    Retorna: máscara bool (altura × largura), True = borda problemática.
    """
    return np.abs(nz) < thresh_nz


def calcular_gradientes_de_superficie(nx: np.ndarray,
                                       ny: np.ndarray,
                                       nz: np.ndarray,
                                       mascara_depth: np.ndarray,
                                       mask_fundo: np.ndarray,
                                       mask_bordas: np.ndarray,
                                       clip_max: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula os gradientes de superfície p = ∂Z/∂x e q = ∂Z/∂y.

    Fórmulas:
        p = -Nx / Nz   (inclinação em X)
        q = -Ny / Nz   (inclinação em Y)

    Calculamos gradientes APENAS onde temos:
      - normal calculada (mascara_depth)
      - não é chão (não é mask_fundo)
      - não é borda perpendicular (não é mask_bordas)

    Onde não calculamos, o gradiente fica zero (superfície plana).
    O solver de integração propaga a altura dos lados para essas áreas.

    clip_max: limite superior dos gradientes para evitar artefatos
    quando Nz é pequeno mas acima do limiar.

    Retorna: (p_grad, q_grad) — arrays (altura × largura).
    """
    mascara_grad = mascara_depth & ~mask_fundo & ~mask_bordas

    # nz_safe: clamp nos pixels COM gradiente (evita divisão por zero).
    # Usar 1.0 seria errado: produziria -nx/1 = -nx como gradiente mesmo
    # para superfícies quase verticais, introduzindo valores falsos.
    # O clamp para ±0.05 limita o gradiente máximo a ±clip_max.
    nz_safe = np.where(np.abs(nz) < 0.05,
                       np.sign(nz + 1e-12) * 0.05,
                       nz)

    p_grad = np.zeros_like(nx)
    q_grad = np.zeros_like(ny)
    p_grad[mascara_grad] = -nx[mascara_grad] / nz_safe[mascara_grad]
    q_grad[mascara_grad] = -ny[mascara_grad] / nz_safe[mascara_grad]

    p_grad = np.clip(p_grad, -clip_max, clip_max)
    q_grad = np.clip(q_grad, -clip_max, clip_max)
    return p_grad, q_grad


# ==============================================================
# SEÇÃO 16 — INTEGRAÇÃO DE GRADIENTES → DEPTH MAP
# ==============================================================
# Dado p = ∂Z/∂x e q = ∂Z/∂y, queremos Z(x,y).
# Isso é resolver uma equação de Poisson: ∇²Z = ∂p/∂x + ∂q/∂y
# Dois métodos disponíveis: Frankot-Chellappa (FFT) e Poisson (DCT).
# ==============================================================

def calcular_divergencia(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Calcula a divergência ∂p/∂x + ∂q/∂y usando diferenças finitas forward.

    A divergência é o lado direito da equação de Poisson que precisamos resolver.
    Diferença forward: f[i+1] - f[i] (o sinal correto para reconstrução exata).

    Retorna: array (altura × largura) com a divergência.
    """
    rows, cols = p.shape
    div = np.zeros((rows, cols))
    div[:-1, :] += q[1:,  :] - q[:-1, :]    # ∂q/∂y  forward
    div[:, :-1] += p[:,  1:] - p[:,  :-1]   # ∂p/∂x  forward
    return div


def integrar_gradientes_frankot_chellappa(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Integra gradientes para obter altura usando o método de Frankot-Chellappa (1988).

    Trabalha no espaço de Fourier: os gradientes são transformados por FFT,
    a equação de Poisson é resolvida analiticamente nas frequências,
    e o resultado é transformado de volta.

    Vantagem: rápido e robusto com boundary conditions periódicas.
    Desvantagem: pode criar "ringing" (ondulações) perto de bordas abruptas.

    Fórmula no domínio da frequência:
        Z_fft = -j(u·P + v·Q) / (u² + v²)
    onde u, v são as frequências espaciais.

    Retorna: array (altura × largura) com alturas relativas.
    """
    rows, cols = p.shape
    u = np.fft.fftfreq(cols) * 2 * np.pi
    v = np.fft.fftfreq(rows) * 2 * np.pi
    U, V = np.meshgrid(u, v)

    den         = U**2 + V**2
    den[0, 0]   = 1.0                      # evita divisão por zero na DC
    Z_fft       = -1j * (U * np.fft.fft2(p) + V * np.fft.fft2(q)) / den
    Z_fft[0, 0] = 0.0                      # componente DC = 0 (altura relativa)
    return np.real(np.fft.ifft2(Z_fft))


def integrar_gradientes_poisson_dct(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Integra gradientes para obter altura usando solver de Poisson com DCT.

    Usa a Transformada Discreta de Cosseno (DCT) que implementa
    boundary conditions de Neumann (∂Z/∂n = 0 nas bordas) — mais
    adequadas para objetos sobre uma bancada plana.

    Vantagem: menos ringing que FFT, boundary conditions melhores.
    Complexidade: O(N log N), comparável ao FFT.

    Passos:
      1. Calcula a divergência ∂p/∂x + ∂q/∂y
      2. Transforma para domínio DCT
      3. Divide pelos autovalores do Laplaciano (resolve Poisson analiticamente)
      4. Transforma de volta

    Retorna: array (altura × largura) com alturas relativas.
    """
    div = calcular_divergencia(p, q)

    rows, cols = p.shape
    ii  = np.arange(rows).reshape(-1, 1)
    jj  = np.arange(cols).reshape(1, -1)

    # Autovalores do Laplaciano discreto com BCN de Neumann
    lam         = (2.0 * np.cos(np.pi * ii / rows) - 2.0) + \
                  (2.0 * np.cos(np.pi * jj / cols) - 2.0)
    lam[0, 0]   = 1.0   # evita divisão por zero

    Z_dct       = dctn(div, norm='ortho') / lam
    Z_dct[0, 0] = 0.0   # componente DC = 0 (altura relativa)
    return idctn(Z_dct, norm='ortho')


# ==============================================================
# SEÇÃO 17 — NORMALIZAÇÃO E VISUALIZAÇÃO DO DEPTH MAP
# ==============================================================

def fixar_datum_no_chao(depth_raw: np.ndarray, mask_fundo: np.ndarray) -> np.ndarray:
    """
    Ajusta o offset vertical para que o chão fique em Z=0.

    O solver produz alturas relativas — pode sair com o chão em Z=5
    e objetos em Z=8, quando o correto seria chão=0 e objetos=3.
    Subtraímos a mediana do chão de toda a imagem para corrigir isso.

    Depois da subtração, zeramos o chão explicitamente para eliminar
    qualquer ruído residual (ringing do solver).

    Retorna: depth_raw ajustado.
    """
    if np.any(mask_fundo):
        base_z = np.median(depth_raw[mask_fundo])
        depth_raw = depth_raw - base_z
    depth_raw[mask_fundo] = 0.0
    depth_raw = np.maximum(depth_raw, 0.0)   # elimina valores negativos residuais
    return depth_raw


def normalizar_depth_para_255(depth_raw: np.ndarray,
                               mask_fundo: np.ndarray,
                               mascara_depth: np.ndarray,
                               percentil_max: float = 98.0) -> np.ndarray:
    """
    Normaliza os valores de altura para a faixa [0, 255] para visualização.

    Usa o percentil 98 dos pixels de OBJETOS (não do chão) como máximo.
    Isso evita que um pico pontual (artefato) escureça toda a visualização.

    O chão é mantido em zero após a normalização.

    Retorna: array float (altura × largura) em [0, 255].
    """
    pixels_objeto = depth_raw[~mask_fundo & mascara_depth]
    if len(pixels_objeto) == 0:
        return np.zeros_like(depth_raw)

    d_max      = np.percentile(pixels_objeto, percentil_max)
    depth_norm = np.clip(depth_raw / (d_max + 1e-9) * 255.0, 0, 255)
    depth_norm[mask_fundo] = 0.0
    return depth_norm


def aplicar_colormap_jet(depth_norm: np.ndarray) -> np.ndarray:
    """
    Converte o mapa de profundidade para visualização colorida com colormap JET.

    Aplica um leve blur antes para suavizar artefatos de pixelamento.
    Com o chão em 0 e os objetos indo até 255:
      - Azul escuro = chão (mais baixo)
      - Verde/amarelo = alturas intermediárias
      - Vermelho = ponto mais alto

    Retorna: imagem BGR uint8 (altura × largura × 3) pronta para salvar.
    """
    depth_suave    = cv2.GaussianBlur(depth_norm.astype(np.uint8), (5, 5), sigmaX=1.0)
    depth_colormap = cv2.applyColorMap(depth_suave, cv2.COLORMAP_JET)
    return depth_colormap


# ==============================================================
# SEÇÃO 18 — ALBEDO MAP
# ==============================================================

def construir_albedo_map(albedo_flat: np.ndarray,
                          mascara_depth: np.ndarray,
                          altura: int,
                          largura: int) -> np.ndarray:
    """
    Constrói a imagem de albedo normalizada para visualização.

    O albedo é a reflectividade da superfície — independente da iluminação.
    É calculado como a magnitude do vetor G = albedo · N.

    Para visualização, estiramos o contraste usando os percentis 1% e 99%
    dos pixels de objetos (ignora ruído de fundo e outliers extremos).

    Retorna: imagem uint8 (altura × largura) com albedo normalizado.
    """
    albedo_2d = albedo_flat.reshape(altura, largura)
    regiao    = mascara_depth if mascara_depth.any() else np.ones_like(mascara_depth)
    valores   = albedo_2d[regiao]

    a_min, a_max = np.percentile(valores, 1), np.percentile(valores, 99)
    albedo_vis   = np.clip(
        (albedo_2d - a_min) / (a_max - a_min + 1e-9) * 255.0, 0, 255
    ).astype(np.uint8)
    return albedo_vis


# ==============================================================
# SEÇÃO 19 — SALVAR RESULTADOS
# ==============================================================

def salvar_resultados(diretorio_saida: str,
                       N_rgb: np.ndarray,
                       depth_colormap: np.ndarray,
                       albedo_vis: np.ndarray) -> tuple[str, str, str]:
    """
    Salva os três mapas gerados como arquivos PNG no diretório de saída.

    Normal map: convertido de RGB para BGR antes de salvar (OpenCV usa BGR).
    Depth map:  já está em BGR (saída do applyColorMap).
    Albedo map: escala de cinza (single channel).

    Retorna: tupla com os nomes dos arquivos salvos.
    """
    nome_normal = 'resultado_normal.png'
    nome_depth  = 'resultado_depth.png'
    nome_albedo = 'resultado_albedo.png'

    cv2.imwrite(os.path.join(diretorio_saida, nome_normal),
                cv2.cvtColor(N_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(diretorio_saida, nome_depth),  depth_colormap)
    cv2.imwrite(os.path.join(diretorio_saida, nome_albedo), albedo_vis)

    return nome_normal, nome_depth, nome_albedo


# ==============================================================
# SEÇÃO 20 — PIPELINE PRINCIPAL
# ==============================================================
# Esta função orquestra todas as etapas acima em ordem.
# Cada comentário indica qual seção está sendo executada.
# ==============================================================

def processar_mapas(
    caminhos_imagens:   list[str],
    vetores_luz:        list[list[float]],
    diretorio_saida:    str,
    # ── ROI manual ───────────────────────────────────────────
    plano_coords:       list[dict] | None = None,
    largura_mm:         float = 210.0,
    altura_mm:          float = 297.0,
    # ── ROI automático (modo Blender) ────────────────────────
    camera_ortogonal:   bool  = False,
    wb_larg_mm:         float | None = None,
    wb_prof_mm:         float | None = None,
    camera_h_mm:        float | None = None,
    focal_mm:           float = 50.0,
    sensor_w_mm:        float = 36.0,
    ortho_scale_mm:     float | None = None,
    # ── Qualidade ────────────────────────────────────────────
    metodo_depth:       str   = 'poisson',
    suavizar_normais:   bool  = False,
    sigma_suavizacao:   float = 1.0,
    thresh_variacao:    float = 0.04,
    thresh_sat:         float = 0.95,
    thresh_residuo:     float = 0.20,
    razao_sombra:       float = 0.60,
    # ── Detecção de sombra cast ──────────────────────────────
    detectar_sombra:    bool  = True,
    thresh_sombra_cast: float = 0.35,
    raio_contexto_px:   int   = 5,
) -> tuple[str, str, str]:
    """
    Pipeline completo: 4 imagens com luz direcional → normal map + depth map + albedo map.

    Fluxo resumido:
      1. Carrega imagens e define ROI
      2. Remove luz ambiente, cria máscara de pixels úteis
      3. Resolve o sistema fotométrico (normal + albedo) por pixel
      4. Rejeita pixels ruins (sombra própria, resíduo alto, sombra cast)
      5. Constrói e salva os mapas de saída

    Parâmetros principais a ajustar:
      thresh_variacao    : sensibilidade da máscara. Suba se houver ruído no fundo.
      razao_sombra       : agressividade do Drop-Darkest para sombra própria.
      thresh_residuo     : tolerância para o modelo Lambertiano. Baixe para
                           excluir mais reflexos; suba se objetos desaparecerem.
      thresh_sombra_cast : sensibilidade da detecção de sombra cast.
      ortho_scale_mm     : Orthographic Scale do Blender em mm (valor do Blender × 1000).
    """

    # ── ETAPA 1: Carregar imagens ────────────────────────────
    print("[1/9] Carregando imagens...")
    imgs            = [carregar_imagem_cinza(c) for c in caminhos_imagens]
    altura, largura = imgs[0].shape
    n_pixels        = altura * largura

    # ── ETAPA 2: Definir ROI ─────────────────────────────────
    print("[2/9] Calculando ROI...")
    if camera_ortogonal and all(v is not None for v in [wb_larg_mm, wb_prof_mm, camera_h_mm]):
        mascara_roi = roi_automatica_blender(
            img_largura    = largura,
            img_altura     = altura,
            wb_larg_mm     = wb_larg_mm,
            wb_prof_mm     = wb_prof_mm,
            camera_h_mm    = camera_h_mm,
            focal_mm       = focal_mm,
            sensor_w_mm    = sensor_w_mm,
            ortho_scale_mm = ortho_scale_mm,
        )
    elif plano_coords and len(plano_coords) >= 4:
        mascara_roi = roi_manual_por_cantos(plano_coords, altura, largura)
    else:
        mascara_roi = None

    # ── ETAPA 3: Pré-processar imagens ───────────────────────
    print("[3/9] Subtraindo luz ambiente e criando máscara de pixels válidos...")
    imgs    = subtrair_luz_ambiente(imgs)
    mascara = criar_mascara_valida(imgs, mascara_roi, thresh_variacao, thresh_sat)
    idx_val = np.where(mascara.flatten())[0]

    pct_val = 100.0 * len(idx_val) / n_pixels
    print(f"  [máscara]  {len(idx_val)} px válidos ({pct_val:.1f}%)")

    if len(idx_val) == 0:
        raise ValueError(
            "Nenhum pixel válido após mascaramento. "
            "Reduza thresh_variacao ou verifique os parâmetros da câmara/bancada."
        )

    # ── ETAPA 4: Preparar sistema fotométrico ────────────────
    print("[4/9] Preparando vetores de luz e resolvendo sistema fotométrico...")
    L             = normalizar_vetores_luz(vetores_luz)
    L_inv4, L_invs = preparar_pseudoinversas(L)

    I_stack = np.stack([img.flatten() for img in imgs])
    I_val   = I_stack[:, idx_val]    # (4, N_val) — intensidades só nos pixels válidos

    # ── ETAPA 5: Resolver lstsq + drop darkest ───────────────
    deve_drop, min_idx = detectar_sombra_propria(I_val, razao_sombra)
    G_val              = resolver_lstsq_pixels_validos(
        I_val, L, L_inv4, L_invs, deve_drop, min_idx
    )
    albedo_val, N_val  = extrair_albedo_e_normal(G_val)

    # ── ETAPA 6: Rejeitar por resíduo e sombra cast ──────────
    print("[5/9] Filtrando pixels por resíduo e sombra cast...")
    residuo_rel   = calcular_residuo_fotometrico(I_val, L, G_val, albedo_val)
    mask_residuo  = filtrar_por_residuo(residuo_rel, thresh_residuo)

    if detectar_sombra:
        mask_sombra_cast = detectar_sombras_cast(
            I_val, G_val, albedo_val, idx_val, altura, largura,
            thresh_sombra_cast, raio_contexto_px
        )
        validos_final = mask_residuo & ~mask_sombra_cast
    else:
        validos_final = mask_residuo

    idx_final = idx_val[validos_final]
    pct_final = 100.0 * len(idx_final) / n_pixels
    print(f"  [aceitos]  {len(idx_final)} px finais ({pct_final:.1f}%)")

    # ── ETAPA 7: Construir normal map ────────────────────────
    print("[6/9] Construindo normal map...")
    N_flat, albedo_flat = montar_normal_map_flat(
        N_val, albedo_val, idx_final, validos_final, n_pixels
    )
    sem_normal = preencher_fundo_com_normal_para_cima(N_flat, mascara_roi)

    # Correção de perspectiva (só modo manual com câmara inclinada)
    if not camera_ortogonal and plano_coords and len(plano_coords) >= 4:
        R      = estimar_rotacao_por_homografia(plano_coords, largura_mm, altura_mm)
        N_flat = aplicar_rotacao_nas_normais(N_flat, R)

    # Suavização opcional
    if suavizar_normais:
        N_flat     = suavizar_normal_map(N_flat, altura, largura, sigma_suavizacao)
        sem_normal = preencher_fundo_com_normal_para_cima(N_flat, mascara_roi)

    # Separar componentes para uso nas próximas etapas
    nx = N_flat[0].reshape(altura, largura)
    ny = N_flat[1].reshape(altura, largura)
    nz = N_flat[2].reshape(altura, largura)

    N_rgb = codificar_normal_map_rgb(nx, ny, nz)

    # ── ETAPA 8: Construir depth map ─────────────────────────
    print("[7/9] Integrando gradientes para depth map...")

    # SEPARAÇÃO DE RESPONSABILIDADES:
    #   mascara_objetos : só idx_final — pixels com normal CALCULADA e confiável.
    #                     Usada para: identificar o chão, normalizar depth, albedo.
    #   mascara_depth   : TODOS os pixels com qualquer normal (calculada ou preenchida).
    #                     Usada para: calcular gradientes → solver Poisson.
    #
    # Por quê separar: o solver Poisson precisa de um campo de gradientes DENSO.
    # Pixels preenchidos com [0,0,1] têm gradiente = 0 (plano) — isso é correto
    # e necessário para que o solver propague a altura das bordas para o interior.
    # Se usarmos só idx_final (bordas), o solver recebe slivers e produz profundidade nula.

    mascara_objetos = np.zeros(n_pixels, dtype=bool)
    mascara_objetos[idx_final] = True
    mascara_objetos = mascara_objetos.reshape(altura, largura)

    mascara_depth = (np.linalg.norm(N_flat, axis=0) > 1e-9).reshape(altura, largura)

    # mask_fundo usa mascara_objetos para NÃO confundir tampas planas de objetos
    # com chão: se a tampa NÃO está em idx_final (pouca variação de cor), é
    # marcada como chão — errado. Usando mascara_objetos, a tampa fica excluída
    # do "chão" mesmo que tenha pouca variação.
    mask_fundo  = identificar_pixels_chao(nx, ny, mascara_objetos)
    mask_bordas = identificar_bordas_perpendiculares(nz)

    p_grad, q_grad = calcular_gradientes_de_superficie(
        nx, ny, nz, mascara_depth, mask_fundo, mask_bordas
    )

    if metodo_depth == 'poisson':
        depth_raw = integrar_gradientes_poisson_dct(p_grad, q_grad)
    else:
        depth_raw = integrar_gradientes_frankot_chellappa(p_grad, q_grad)

    depth_raw      = fixar_datum_no_chao(depth_raw, mask_fundo)
    depth_norm     = normalizar_depth_para_255(depth_raw, mask_fundo, mascara_objetos)
    depth_colormap = aplicar_colormap_jet(depth_norm)

    # ── ETAPA 9: Construir albedo map e salvar ───────────────
    print("[8/9] Construindo albedo map...")
    albedo_vis = construir_albedo_map(albedo_flat, mascara_objetos, altura, largura)

    print("[9/9] Salvando resultados...")
    nomes = salvar_resultados(diretorio_saida, N_rgb, depth_colormap, albedo_vis)

    print("  Concluído.")
    return nomes