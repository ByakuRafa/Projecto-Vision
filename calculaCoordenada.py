"""
calculaCoordenada_v2.py
=======================
Detecção de objetos por cor em imagem RGB + depth map.
Saída formatada para o KUKA LBR iiwa (E6POS / KRL).

Uso básico:
    python calculaCoordenada_v2.py

Uso com argumentos:
    python calculaCoordenada_v2.py --cor vermelho azul
    python calculaCoordenada_v2.py --cor amarelo --picker
    python calculaCoordenada_v2.py --dry-run      (sem imagens reais, simula dados)
"""

import cv2
import numpy as np
import os
import argparse
import json
import socket
from datetime import datetime

# ==============================================================================
#  CONFIGURAÇÕES — EDITE AQUI CONFORME SEU SETUP
# ==============================================================================

CAMERA = {
    "focal_mm":     50.0,   # Lens > Focal Length (mm)
    "sensor_w_mm":  36.0,   # Lens > Sensor Width  (mm, padrão Blender = 36)
    "clip_start_m":  0.1,   # Camera > Clip Start  (metros)
    "clip_end_m":   10.0,   # Camera > Clip End    (metros)
}

# Área de trabalho física (metros) — o Workbench do Blender
WORKBENCH = {
    "width_m":      1.20,   # Tamanho em X (largura)
    "depth_m":      0.80,   # Tamanho em Y (profundidade)
    "height_m":     0.60,   # Altura máxima esperada dos objetos em Z
    # Deslocamento da origem do workbench em relação à base do robô (metros)
    "origin_x_m":   0.40,
    "origin_y_m":   0.50,
    "origin_z_m":   0.00,
}

# Configuração do KUKA LBR iiwa
KUKA = {
    # Posição de HOME segura do robô (mm, graus)
    "home": {"X": 0.0, "Y": 0.0, "Z": 800.0, "A": 0.0, "B": 90.0, "C": 0.0},
    # Velocidade de aproximação (% da máxima)
    "vel_percent": 20,
    # Offset de Z para pegar o objeto (o robô desce mais N mm abaixo do centróide)
    "grasp_z_offset_mm": -30.0,
    # Envio via socket (True = tenta conectar, False = só imprime)
    "send_socket": False,
    "host": "192.168.0.100",
    "port": 30001,
}

# Área mínima (px²) para um contorno ser considerado objeto real
MIN_CONTOUR_AREA = 300

# ==============================================================================
#  PALETA DE CORES — faixas HSV para cada cor
#  Formato: { "nome": [(H_low,S_low,V_low), (H_high,S_high,V_high)], ... }
#  Vermelho tem DUAS faixas por causa do wrap-around no canal H.
# ==============================================================================
CORES = {
    "vermelho": [
        ((0,   80,  50), (10, 255, 255)),
        ((160, 80,  50), (180,255, 255)),
    ],
    "azul": [
        ((100, 80,  50), (135, 255, 255)),
    ],
    "amarelo": [
        ((20, 80,  50), (35, 255, 255)),
    ],
    "verde": [
        ((36, 50,  50), (85, 255, 255)),
    ],
    "laranja": [
        ((10, 100, 50), (22, 255, 255)),
    ],
    "roxo": [
        ((130, 50, 50), (160, 255, 255)),
    ],
    "ciano": [
        ((80, 50,  50), (100, 255, 255)),
    ],
    "branco": [
        ((0,   0, 200), (180, 40, 255)),
    ],
    "cinza": [
        ((0,   0,  80), (180, 40, 200)),
    ],
}


# ==============================================================================
#  FUNÇÕES AUXILIARES
# ==============================================================================

def build_K(focal_mm, sensor_w_mm, w_px, h_px):
    """Constrói a matriz intrínseca K da câmara."""
    fx = (focal_mm / sensor_w_mm) * w_px
    fy = fx
    cx = w_px / 2.0
    cy = h_px / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def depth_to_meters(pixel_val, dtype, clip_start, clip_end):
    """Converte valor bruto do depth map para metros lineares."""
    max_val = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
    norm = float(pixel_val) / float(max_val)
    return clip_start + norm * (clip_end - clip_start)


def pixel_to_3d(u, v, depth_m, K):
    """Reprojecta píxel (u,v) + profundidade para coordenadas 3D (metros)."""
    X = (u - K[0, 2]) * depth_m / K[0, 0]
    Y = (v - K[1, 2]) * depth_m / K[1, 1]
    return X, Y, depth_m


def camera_to_robot(X_cam, Y_cam, Z_cam, wb):
    """
    Transforma coordenadas da câmara para o referencial da base do robô.
    Assume câmara olhando de cima para baixo, eixo Z da câmara = -Z do robô.
    Ajuste a matriz de rotação se a câmara estiver numa pose diferente.
    """
    # Translação simples (workbench origin + offset)
    X_robot = X_cam + wb["origin_x_m"]
    Y_robot = Y_cam + wb["origin_y_m"]
    Z_robot = (wb["height_m"] - Z_cam) + wb["origin_z_m"]
    return X_robot, Y_robot, Z_robot


def build_mask(hsv, cor_nome):
    """Constrói máscara binária para a cor solicitada."""
    if cor_nome not in CORES:
        raise ValueError(f"Cor '{cor_nome}' não encontrada. Disponíveis: {list(CORES.keys())}")
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (low, high) in CORES[cor_nome]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(low), np.array(high)))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def format_e6pos(X_mm, Y_mm, Z_mm, A=0.0, B=90.0, C=0.0):
    """Formata posição no padrão KRL E6POS do KUKA."""
    return f"{{X {X_mm:.2f}, Y {Y_mm:.2f}, Z {Z_mm:.2f}, A {A:.2f}, B {B:.2f}, C {C:.2f}}}"


def send_to_kuka(positions, kuka_cfg):
    """Envia lista de posições para o robô via socket (protocolo simples JSON)."""
    payload = json.dumps({"positions": positions, "vel": kuka_cfg["vel_percent"]})
    try:
        with socket.create_connection((kuka_cfg["host"], kuka_cfg["port"]), timeout=3) as s:
            s.sendall(payload.encode())
            resp = s.recv(1024).decode()
            print(f"  [KUKA] Resposta: {resp}")
    except Exception as e:
        print(f"  [KUKA] Falha na conexão: {e}")


# ==============================================================================
#  COLOR PICKER INTERATIVO (clique na imagem para amostrar uma cor)
# ==============================================================================

_sampled_hsv = []

def _mouse_sample(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_img = param
        h, s, v = hsv_img[y, x]
        print(f"  [Picker] Clicado ({x},{y}) → H={h} S={s} V={v}")
        _sampled_hsv.append((h, s, v))
        if len(_sampled_hsv) >= 2:
            print("  [Picker] Duas amostras coletadas. Pressione qualquer tecla para continuar.")


def interactive_color_picker(img_rgb):
    """Mostra a imagem e deixa o usuário clicar para amostrar a cor desejada."""
    print("\n[Picker] Clique em 2 pontos do objeto para definir a faixa de cor.")
    print("         Pressione qualquer tecla depois para continuar.\n")
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    cv2.namedWindow("Color Picker — clique no objeto")
    cv2.setMouseCallback("Color Picker — clique no objeto", _mouse_sample, hsv)
    cv2.imshow("Color Picker — clique no objeto", img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(_sampled_hsv) < 2:
        print("  [Picker] Menos de 2 amostras. Usando valores padrão.")
        return None

    h_vals = [p[0] for p in _sampled_hsv]
    s_vals = [p[1] for p in _sampled_hsv]
    v_vals = [p[2] for p in _sampled_hsv]
    margin_h, margin_s, margin_v = 10, 40, 40
    low  = (max(0,   min(h_vals)-margin_h), max(0,   min(s_vals)-margin_s), max(0,   min(v_vals)-margin_v))
    high = (min(180, max(h_vals)+margin_h), min(255, max(s_vals)+margin_s), min(255, max(v_vals)+margin_v))
    print(f"  [Picker] Faixa gerada → low={low}  high={high}")
    # Registra como nova cor "custom" na paleta
    CORES["custom"] = [(low, high)]
    return "custom"


# ==============================================================================
#  PIPELINE PRINCIPAL
# ==============================================================================

def processar(img_rgb, img_depth, cores_alvo, K, wb, kuka_cfg, saida_krl=None):
    h, w = img_rgb.shape[:2]
    hsv   = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    vis   = img_rgb.copy()

    cores_alvo = list(dict.fromkeys(cores_alvo))  # remove duplicatas mantendo ordem
    resultados = []

    # Cores de exibição por nome (BGR)
    display_bgr = {
        "vermelho": (0, 0, 220),  "azul": (220, 100, 0),
        "amarelo":  (0, 210, 220),"verde": (0, 180, 50),
        "laranja":  (0, 140, 255),"roxo":  (180, 0, 180),
        "ciano":    (200, 180, 0),"branco": (200, 200, 200),
        "cinza":    (130, 130, 130), "custom": (0, 255, 200),
    }

    for cor in cores_alvo:
        try:
            mask = build_mask(hsv, cor)
        except ValueError as e:
            print(f"  [AVISO] {e}")
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cor_bgr = display_bgr.get(cor, (200, 200, 200))

        for contour in contours:
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Leitura mediana do depth (janela 7×7 para mais robustez)
            y1, y2 = max(0, cY-3), min(h, cY+4)
            x1, x2 = max(0, cX-3), min(w, cX+4)
            depth_bruto = float(np.median(img_depth[y1:y2, x1:x2].astype(np.float64)))
            depth_m     = depth_to_meters(depth_bruto, img_depth.dtype,
                                          CAMERA["clip_start_m"], CAMERA["clip_end_m"])

            X_cam, Y_cam, Z_cam = pixel_to_3d(cX, cY, depth_m, K)
            X_rob, Y_rob, Z_rob = camera_to_robot(X_cam, Y_cam, Z_cam, wb)

            # Converte para mm (KUKA usa mm)
            X_mm = X_rob * 1000
            Y_mm = Y_rob * 1000
            Z_mm = Z_rob * 1000

            obj_id = len(resultados) + 1
            e6pos  = format_e6pos(X_mm, Y_mm, Z_mm + kuka_cfg["grasp_z_offset_mm"])
            resultado = {
                "id":       obj_id,
                "cor":      cor,
                "pixel":    (cX, cY),
                "depth_m":  round(depth_m, 4),
                "X_mm":     round(X_mm, 2),
                "Y_mm":     round(Y_mm, 2),
                "Z_mm":     round(Z_mm, 2),
                "e6pos":    e6pos,
            }
            resultados.append(resultado)

            # --- Visualização ---
            cv2.drawContours(vis, [contour], -1, cor_bgr, 2)
            cv2.circle(vis, (cX, cY), 7, (0, 255, 0), -1)
            cv2.putText(vis, f"#{obj_id} {cor}", (cX-40, cY-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_bgr, 2)
            cv2.putText(vis, f"X{X_mm:.0f} Y{Y_mm:.0f} Z{Z_mm:.0f}mm",
                        (cX-40, cY-12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)

    # --- Saída no terminal ---
    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  Objetos detetados: {len(resultados)}")
    print(sep)
    for r in resultados:
        print(f"\n  #{r['id']}  Cor: {r['cor'].upper()}")
        print(f"       Píxel : {r['pixel']}")
        print(f"       Depth : {r['depth_m']:.4f} m")
        print(f"       3D    : X={r['X_mm']:.1f}mm  Y={r['Y_mm']:.1f}mm  Z={r['Z_mm']:.1f}mm")
        print(f"       E6POS : {r['e6pos']}")

    # --- Gerar arquivo KRL opcional ---
    if saida_krl and resultados:
        _gerar_krl(resultados, kuka_cfg, saida_krl)

    # --- Enviar ao robô ---
    if kuka_cfg["send_socket"] and resultados:
        positions = [{"X": r["X_mm"], "Y": r["Y_mm"],
                      "Z": r["Z_mm"] + kuka_cfg["grasp_z_offset_mm"]} for r in resultados]
        send_to_kuka(positions, kuka_cfg)

    return vis, resultados


def _gerar_krl(resultados, kuka_cfg, caminho):
    """Gera um arquivo .src KRL básico para o KUKA Sunrise."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    home = kuka_cfg["home"]
    linhas = [
        f"&ACCESS RVP",
        f"&REL 1",
        f"; Gerado automaticamente por calculaCoordenada_v2.py",
        f"; {ts}",
        f"DEF DetectAndPick()",
        f"  INI",
        f"  ; Ir para HOME",
        f"  PTP {format_e6pos(**home)} C_PTP",
        f"",
        f"  ; --- Sequência de pega ---",
    ]
    for r in resultados:
        approach_z = r["Z_mm"] + 80  # aproximação 80mm acima
        linhas += [
            f"",
            f"  ; Objeto #{r['id']} — {r['cor']}",
            f"  LIN {format_e6pos(r['X_mm'], r['Y_mm'], approach_z)}  ; Aproximação",
            f"  LIN {r['e6pos']}  ; Pega",
            f"  ; FECHAR GARRA",
            f"  LIN {format_e6pos(r['X_mm'], r['Y_mm'], approach_z)}  ; Recuar",
        ]
    linhas += [
        f"",
        f"  ; Retornar ao HOME",
        f"  PTP {format_e6pos(**home)} C_PTP",
        f"END",
    ]
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))
    print(f"\n[KRL] Arquivo gerado: {caminho}")


# ==============================================================================
#  MODO DRY-RUN (sem imagens, apenas testa a lógica)
# ==============================================================================

def dry_run(cores_alvo, K, wb, kuka_cfg):
    print("\n[DRY-RUN] Simulando 3 objetos fictícios...")
    fake_objects = [
        {"cor": cores_alvo[0] if cores_alvo else "vermelho", "cX": 640, "cY": 360, "depth_m": 1.5},
        {"cor": cores_alvo[1] if len(cores_alvo) > 1 else "azul",   "cX": 300, "cY": 200, "depth_m": 2.0},
        {"cor": cores_alvo[0] if cores_alvo else "vermelho", "cX": 900, "cY": 540, "depth_m": 1.2},
    ]
    resultados = []
    for i, obj in enumerate(fake_objects):
        X_c, Y_c, Z_c = pixel_to_3d(obj["cX"], obj["cY"], obj["depth_m"], K)
        X_r, Y_r, Z_r = camera_to_robot(X_c, Y_c, Z_c, wb)
        X_mm, Y_mm, Z_mm = X_r * 1000, Y_r * 1000, Z_r * 1000
        e6pos = format_e6pos(X_mm, Y_mm, Z_mm + kuka_cfg["grasp_z_offset_mm"])
        resultados.append({"id": i+1, "cor": obj["cor"], "pixel": (obj["cX"], obj["cY"]),
                            "depth_m": obj["depth_m"], "X_mm": X_mm, "Y_mm": Y_mm,
                            "Z_mm": Z_mm, "e6pos": e6pos})
        print(f"  #{i+1} {obj['cor'].upper():10} → {e6pos}")
    return resultados


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Detecção de objetos por cor + saída KUKA LBR iiwa")
    parser.add_argument("--cor", nargs="+", default=["vermelho"],
                        metavar="COR",
                        help=f"Cores a detectar. Disponíveis: {list(CORES.keys())}")
    parser.add_argument("--picker", action="store_true",
                        help="Ativa o seletor interativo de cor por clique na imagem")
    parser.add_argument("--dry-run", action="store_true",
                        help="Executa sem imagens reais (teste de lógica/KUKA)")
    parser.add_argument("--krl", metavar="ARQUIVO.src", default=None,
                        help="Gera arquivo KRL para o KUKA Sunrise")
    parser.add_argument("--rgb",   default=None, help="Caminho da imagem RGB")
    parser.add_argument("--depth", default=None, help="Caminho do depth map")
    # Parâmetros sobreponíveis via linha de comando
    parser.add_argument("--focal",      type=float, default=None)
    parser.add_argument("--sensor",     type=float, default=None)
    parser.add_argument("--clip-start", type=float, default=None)
    parser.add_argument("--clip-end",   type=float, default=None)
    parser.add_argument("--wb-width",   type=float, default=None)
    parser.add_argument("--wb-depth",   type=float, default=None)
    parser.add_argument("--wb-height",  type=float, default=None)
    args = parser.parse_args()

    # Aplicar overrides de linha de comando
    if args.focal:      CAMERA["focal_mm"]     = args.focal
    if args.sensor:     CAMERA["sensor_w_mm"]  = args.sensor
    if args.clip_start: CAMERA["clip_start_m"] = args.clip_start
    if args.clip_end:   CAMERA["clip_end_m"]   = args.clip_end
    if args.wb_width:   WORKBENCH["width_m"]   = args.wb_width
    if args.wb_depth:   WORKBENCH["depth_m"]   = args.wb_depth
    if args.wb_height:  WORKBENCH["height_m"]  = args.wb_height

    # Resolução padrão para K no dry-run
    render_w, render_h = 1280, 720

    if args.dry_run:
        K = build_K(CAMERA["focal_mm"], CAMERA["sensor_w_mm"], render_w, render_h)
        dry_run(args.cor, K, WORKBENCH, KUKA)
        return

    # Caminhos das imagens
    dir_atual = os.path.dirname(os.path.abspath(__file__))
    pasta     = os.path.join(dir_atual, "test_depth")
    ruta_rgb   = args.rgb   or os.path.join(pasta, "blenderTest2.png")
    ruta_depth = args.depth or os.path.join(pasta, "img2.png")

    img_rgb   = cv2.imread(ruta_rgb)
    img_depth = cv2.imread(ruta_depth, cv2.IMREAD_UNCHANGED)

    if img_rgb is None or img_depth is None:
        print(f"[ERRO] Não foi possível abrir as imagens.")
        print(f"  RGB  : {ruta_rgb}")
        print(f"  Depth: {ruta_depth}")
        return

    if img_depth.ndim == 3:
        img_depth = cv2.cvtColor(img_depth, cv2.COLOR_BGR2GRAY)

    h, w = img_rgb.shape[:2]
    K = build_K(CAMERA["focal_mm"], CAMERA["sensor_w_mm"], w, h)
    print(f"[INFO] Imagens: {w}×{h}  |  Depth dtype: {img_depth.dtype}")
    print(f"[INFO] K = fx={K[0,0]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")

    cores_alvo = list(args.cor)

    if args.picker:
        cor_custom = interactive_color_picker(img_rgb)
        if cor_custom:
            cores_alvo.append(cor_custom)

    vis, resultados = processar(img_rgb, img_depth, cores_alvo, K, WORKBENCH, KUKA, args.krl)

    cv2.imshow("Deteção + Coordenadas KUKA", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()