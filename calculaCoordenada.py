"""
calculaCoordenada.py
====================
Detecção de objetos por PROFUNDIDADE (depth map) — independente de cor.

Lógica principal:
  1. O depth map revela a superfície da bancada como um plano de profundidade
     uniforme (os pixels do "chão" concentram-se num valor próximo).
  2. Objetos sobre a bancada ficam MAIS PRÓXIMOS da câmara → menor valor de depth.
  3. Subtraindo o "chão" estimado, obtemos uma máscara de tudo que "sobressai".
  4. Encontramos contornos e calculamos X, Y, Z reais para cada objeto.

Uso:
  python calculaCoordenada.py [opções]

  --rgb   <path>    Imagem RGB
  --depth <path>    Depth map (PNG 8 ou 16 bit)
  --help            Lista todos os parâmetros configuráveis
"""

import cv2
import numpy as np
import os
import argparse

# ==============================================================================
# CONFIGURAÇÃO — edite aqui, ou passe argumentos via linha de comando
# ==============================================================================

CONFIG = {

    # ── Câmara (bater com as propriedades da câmara no Blender) ────────────
    "focal_length_mm" : 50.0,   # Lens > Focal Length
    "sensor_width_mm" : 36.0,   # Lens > Sensor Width (padrão Blender = 36 mm)
    "clip_start_m"    : 0.1,    # Camera > Clip Start (metros)
    "clip_end_m"      : 10.0,   # Camera > Clip End   (metros)

    # ── Workbench (bancada real) ────────────────────────────────────────────
    "workbench": {
        "largura_m"  : 1.0,    # eixo X — largura da bancada
        "profund_m"  : 0.6,    # eixo Y — profundidade da bancada
        "volume_z_m" : 0.5,    # eixo Z — altura máxima de objetos
        # Posição da câmara em relação ao canto frontal-esquerdo da bancada
        "offset_x_m" : 0.5,    # metade da largura (câmara centralizada em X)
        "offset_y_m" : 0.3,    # câmara deslocada para dentro em Y
        "offset_z_m" : 0.8,    # câmara a 80 cm acima da superfície
    },

    # ── Detecção por depth ─────────────────────────────────────────────────
    # O "chão" da bancada é estimado automaticamente pelo histograma do depth.
    # "floor_percentile" define o percentil do histograma usado como referência
    # da superfície (80 = o valor que 80% dos pixels não ultrapassam).
    "floor_percentile"   : 80,

    # Um pixel é considerado objeto se estiver este tanto de metros
    # MAIS PRÓXIMO do que o chão estimado.
    # Aumente se detectar ruído; diminua se perder objetos baixos.
    "object_threshold_m" : 0.03,

    # Filtros de contorno
    "min_area_px"        : 500,   # área mínima em px² (filtra ruído)
    "max_area_px"        : 0,     # 0 = sem limite máximo
    "depth_patch_px"     : 4,     # raio da janela de mediana no depth (px)
}

# ==============================================================================
# CÂMARA
# ==============================================================================

def build_K(cfg, img_w, img_h):
    fx = (cfg["focal_length_mm"] / cfg["sensor_width_mm"]) * img_w
    fy = fx
    cx = img_w / 2.0
    cy = img_h / 2.0
    K  = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=np.float64)
    print(f"  [K] fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")
    return K


def raw_to_meters(depth_raw, img_depth, cfg):
    """Converte valor bruto do pixel para metros (mapeamento linear Blender)."""
    maxv = float(np.iinfo(img_depth.dtype).max) if np.issubdtype(img_depth.dtype, np.integer) else 1.0
    norm = np.asarray(depth_raw, dtype=np.float64) / maxv
    return cfg["clip_start_m"] + norm * (cfg["clip_end_m"] - cfg["clip_start_m"])


def pixel_to_3d(u, v, depth_m, K):
    X = (u - K[0, 2]) * depth_m / K[0, 0]
    Y = (v - K[1, 2]) * depth_m / K[1, 1]
    return X, Y, depth_m


def cam_to_bench(Xc, Yc, Zc, wb):
    """Transforma coordenadas da câmara para o referencial da bancada."""
    return (
        Xc + wb["offset_x_m"],
        Yc + wb["offset_y_m"],
        wb["offset_z_m"] - Zc,
    )

# ==============================================================================
# SEGMENTAÇÃO POR DEPTH
# ==============================================================================

def estimate_floor_depth(depth_raw_gray, percentile):
    """
    Estima o nível do 'chão' pelo percentil dominante do histograma do depth.
    Valores altos de depth = longe = superfície da bancada.
    """
    flat = depth_raw_gray.flatten().astype(np.float64)
    return np.percentile(flat, percentile)


def depth_object_mask(depth_raw_gray, img_depth, cfg):
    """
    Retorna máscara onde pixels correspondem a objetos ACIMA da bancada.

    Passos:
      1. Converte depth bruto para metros
      2. Estima profundidade do chão (percentil alto do histograma)
      3. Pixels com depth < floor - threshold → objeto
      4. Morfologia para limpar ruído e fechar buracos
    """
    depth_m    = raw_to_meters(depth_raw_gray.astype(np.float64), img_depth, cfg)
    floor_raw  = estimate_floor_depth(depth_raw_gray, cfg["floor_percentile"])
    floor_m    = float(raw_to_meters(floor_raw, img_depth, cfg))
    thresh_m   = cfg["object_threshold_m"]

    print(f"  [Depth] Chão estimado: {floor_m:.4f} m  |  Threshold: {thresh_m:.3f} m")

    # Pixels que estão mais próximos que (chão - threshold) = objetos
    obj_mask = (depth_m < (floor_m - thresh_m)).astype(np.uint8) * 255

    # Morfologia: remove ruído pequeno e fecha buracos internos
    k_open  = np.ones((5, 5), np.uint8)
    k_close = np.ones((15, 15), np.uint8)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN,  k_open)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, k_close)

    return obj_mask, floor_m

# ==============================================================================
# VISUALIZAÇÃO
# ==============================================================================

PALETTE = [
    (255, 80,  80),  (80, 200, 255), (255, 220, 50),
    (180, 255, 100), (255, 130, 255),(100, 255, 210),
    (255, 170, 80),  (120, 120, 255),
]

def draw_objects(img_vis, contours_data):
    """Desenha contornos, centróides e labels em img_vis."""
    for obj in contours_data:
        color = PALETTE[obj["id"] % len(PALETTE)]
        cX, cY = obj["pixel"]
        cv2.drawContours(img_vis, [obj["contour"]], -1, color, 2)
        cv2.circle(img_vis, (cX, cY), 6, (0, 255, 0), -1)

        Xw, Yw, Zw = obj["workbench_m"]
        l1 = f"#{obj['id']}  Z={Zw:.2f}m"
        l2 = f"X={Xw:.2f}  Y={Yw:.2f}"
        cv2.putText(img_vis, l1, (cX-45, cY-28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,255,255), 2)
        cv2.putText(img_vis, l2, (cX-45, cY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255,255,255), 1)


def draw_depth_colormap(depth_raw_gray, mask, floor_m, cfg):
    """Visualização colorida do depth com máscara de objetos sobreposta."""
    norm = cv2.normalize(depth_raw_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cmap = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    # Borda branca nos objetos detectados
    contours_mask, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(cmap, contours_mask, -1, (255, 255, 255), 2)
    return cmap


def draw_hud(img, wb, n_objs, floor_m):
    h = img.shape[0]
    cv2.rectangle(img, (0, h-30), (img.shape[1], h), (20,20,20), -1)
    txt = (f"Bancada {wb['largura_m']*100:.0f}x{wb['profund_m']*100:.0f}cm  "
           f"Cam Z={wb['offset_z_m']:.2f}m  "
           f"Chão={floor_m:.3f}m  "
           f"Objetos={n_objs}")
    cv2.putText(img, txt, (8, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200,200,200), 1)

# ==============================================================================
# MAIN
# ==============================================================================

def main(cfg, rgb_path, depth_path):

    print(f"\n{'='*62}")
    print(f"  RGB        : {rgb_path}")
    print(f"  Depth      : {depth_path}")
    print(f"{'='*62}\n")

    # ── Carregar ───────────────────────────────────────────────────────────
    img_rgb   = cv2.imread(rgb_path)
    img_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if img_rgb is None or img_depth is None:
        print("[ERRO] Não foi possível carregar as imagens.")
        print("  Verifique os caminhos: --rgb  e  --depth")
        return []

    h, w = img_rgb.shape[:2]
    print(f"  Resolução  : {w}×{h}   |   Depth dtype: {img_depth.dtype}\n")

    # Depth deve ser escala de cinza
    depth_gray = img_depth if img_depth.ndim == 2 else cv2.cvtColor(img_depth, cv2.COLOR_BGR2GRAY)

    # ── Câmara ─────────────────────────────────────────────────────────────
    print("[Câmara]")
    K  = build_K(cfg, w, h)
    wb = cfg["workbench"]

    # ── Segmentação por depth ───────────────────────────────────────────────
    print("\n[Segmentação por depth]")
    mask, floor_m = depth_object_mask(depth_gray, img_depth, cfg)

    # ── Contornos ──────────────────────────────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    resultados   = []
    img_vis      = img_rgb.copy()
    max_area     = cfg["max_area_px"] if cfg["max_area_px"] > 0 else float("inf")

    print("\n[Objetos detectados]")

    idx = 0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < cfg["min_area_px"] or area > max_area:
            continue

        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Mediana do depth na região do centróide
        r  = cfg["depth_patch_px"]
        y1, y2 = max(0, cY-r), min(h, cY+r+1)
        x1, x2 = max(0, cX-r), min(w, cX+r+1)
        med_raw = float(np.median(depth_gray[y1:y2, x1:x2]))
        depth_m = float(raw_to_meters(med_raw, img_depth, cfg))

        Xc, Yc, Zc     = pixel_to_3d(cX, cY, depth_m, K)
        Xw, Yw, Zw     = cam_to_bench(Xc, Yc, Zc, wb)
        bx, by, bwpx, bhpx = cv2.boundingRect(contour)

        obj = {
            "id"         : idx + 1,
            "pixel"      : (cX, cY),
            "cam_m"      : (round(Xc,4), round(Yc,4), round(Zc,4)),
            "workbench_m": (round(Xw,4), round(Yw,4), round(Zw,4)),
            "depth_m"    : round(depth_m, 4),
            "area_px"    : int(area),
            "bbox_px"    : (bx, by, bwpx, bhpx),
            "contour"    : contour,
        }
        resultados.append(obj)

        print(f"\n  Objeto #{idx+1}")
        print(f"    Píxel       : ({cX}, {cY})   área={int(area)} px²")
        print(f"    Depth       : {depth_m:.4f} m")
        print(f"    Câmara      : X={Xc:.4f}  Y={Yc:.4f}  Z={Zc:.4f}")
        print(f"    Bancada     : X={Xw:.4f}  Y={Yw:.4f}  Z={Zw:.4f}  ← enviar ao braço")
        idx += 1

    # ── Desenhar e exibir ──────────────────────────────────────────────────
    draw_objects(img_vis, resultados)
    draw_hud(img_vis, wb, len(resultados), floor_m)
    depth_vis = draw_depth_colormap(depth_gray, mask, floor_m, cfg)

    print(f"\n{'='*62}")
    print(f"  Total: {len(resultados)} objeto(s) detectado(s)")
    print(f"{'='*62}")
    print("\n[COMANDOS BRAÇO ROBÓTICO]")
    for obj in resultados:
        Xw, Yw, Zw = obj["workbench_m"]
        print(f"  GOTO #{obj['id']}  X={Xw:.4f}  Y={Yw:.4f}  Z={Zw:.4f}")

    cv2.imshow("RGB — Objetos Detectados", img_vis)
    cv2.imshow("Depth Colormap + Mascara", depth_vis)
    cv2.imshow("Mascara de Objetos",       mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return resultados

# ==============================================================================
# ARGUMENTOS
# ==============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Detecta objetos por depth (independente de cor) e retorna coordenadas 3D.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Arquivos
    ap.add_argument("--rgb",   default=None, help="Caminho da imagem RGB")
    ap.add_argument("--depth", default=None, help="Caminho do depth map")

    # Câmara
    ap.add_argument("--focal",       type=float, default=CONFIG["focal_length_mm"], help="Focal length (mm)")
    ap.add_argument("--sensor",      type=float, default=CONFIG["sensor_width_mm"],  help="Sensor width (mm)")
    ap.add_argument("--clip-start",  type=float, default=CONFIG["clip_start_m"],     help="Clip Start (m)")
    ap.add_argument("--clip-end",    type=float, default=CONFIG["clip_end_m"],       help="Clip End (m)")

    # Workbench
    ap.add_argument("--larg",   type=float, default=CONFIG["workbench"]["largura_m"],  help="Largura bancada (m)")
    ap.add_argument("--prof",   type=float, default=CONFIG["workbench"]["profund_m"],  help="Profundidade bancada (m)")
    ap.add_argument("--vol-z",  type=float, default=CONFIG["workbench"]["volume_z_m"], help="Altura máx. objetos (m)")
    ap.add_argument("--cam-x",  type=float, default=CONFIG["workbench"]["offset_x_m"], help="Offset câmara X (m)")
    ap.add_argument("--cam-y",  type=float, default=CONFIG["workbench"]["offset_y_m"], help="Offset câmara Y (m)")
    ap.add_argument("--cam-z",  type=float, default=CONFIG["workbench"]["offset_z_m"], help="Altura câmara (m)")

    # Detecção
    ap.add_argument("--floor-pct",   type=int,   default=CONFIG["floor_percentile"],   help="Percentil para estimar chão (0-100)")
    ap.add_argument("--threshold",   type=float, default=CONFIG["object_threshold_m"], help="Diferença mínima de depth para ser objeto (m)")
    ap.add_argument("--min-area",    type=int,   default=CONFIG["min_area_px"],         help="Área mínima de contorno (px²)")
    ap.add_argument("--max-area",    type=int,   default=CONFIG["max_area_px"],         help="Área máxima de contorno em px² (0=sem limite)")

    args = ap.parse_args()

    # Aplicar no CONFIG
    CONFIG["focal_length_mm"]          = args.focal
    CONFIG["sensor_width_mm"]          = args.sensor
    CONFIG["clip_start_m"]             = args.clip_start
    CONFIG["clip_end_m"]               = args.clip_end
    CONFIG["workbench"]["largura_m"]   = args.larg
    CONFIG["workbench"]["profund_m"]   = args.prof
    CONFIG["workbench"]["volume_z_m"]  = args.vol_z
    CONFIG["workbench"]["offset_x_m"]  = args.cam_x
    CONFIG["workbench"]["offset_y_m"]  = args.cam_y
    CONFIG["workbench"]["offset_z_m"]  = 10
    CONFIG["floor_percentile"]         = args.floor_pct
    CONFIG["object_threshold_m"]       = args.threshold
    CONFIG["min_area_px"]              = args.min_area
    CONFIG["max_area_px"]              = args.max_area

    # Caminhos padrão
    base  = os.path.dirname(os.path.abspath(__file__))
    pasta = os.path.join(base, "test_depth")
    rgb_path   = args.rgb   or os.path.join(pasta, "blenderTest2.png")
    depth_path = args.depth or os.path.join(pasta, "img2.png")

    main(CONFIG, rgb_path, depth_path)