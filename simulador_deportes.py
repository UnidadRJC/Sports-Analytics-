"""
Simulador de partidos de tenis, bádminton y ping pong, con "bola extra":
puntos normales vs puntos decisivos y, solo en tenis, la paradoja de ganar
un partido con menos puntos que el rival.

Modelo
------
  p1 = probabilidad de que el Jugador 1 gane un punto normal.
  p2 = probabilidad de que el Jugador 1 gane un punto decisivo.
       Se define como p2 = p1 + delta (delta > 0: el Jugador 1 rinde mejor en
       los puntos clave; delta < 0: peor; delta = 0: no hay efecto, p2 = p1).

  Un punto es DECISIVO cuando, al ganarlo, alguno de los dos jugadores cierra
  el juego (punto de juego, break point, ventaja...). En tenis es el juego; en
  bádminton y ping pong no hay juegos dentro del set, así que el "juego" es el
  propio set. En el tie-break cuenta el punto que cierra el tie-break.

Reglas
------
  - Tenis:     al mejor de 5 sets (gana quien llegue a 3).
               Juego: 4 puntos con diferencia de 2 (15-30-40-ventaja).
               Set: 6 juegos con diferencia de 2; en 6-6 tie-break a 7 puntos
               con diferencia de 2.
  - Bádminton: al mejor de 3 sets (gana quien llegue a 2).
               Set a 21 puntos con diferencia de 2 y tope en 30.
  - Ping pong: al mejor de 5 sets (gana quien llegue a 3).
               Set a 11 puntos con diferencia de 2.

Paradoja (solo tenis)
---------------------
  Se cuenta el % de partidos de tenis en los que el GANADOR consiguió menos
  puntos totales que el perdedor.

Uso
---
    python simulador_deportes.py                      # te pregunta cuántos partidos
    python simulador_deportes.py -n 5000              # 5000 partidos por cada p1
    python simulador_deportes.py -n 5000 --delta -0.05 0.05
    python simulador_deportes.py -n 2000 --pmin 0.3 --pmax 0.7 --puntos-p 41
    python simulador_deportes.py -n 2000 --semilla 1 --salida mis_graficas
"""

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np

# Tope de puntos en un set de bádminton (en 29-29 gana el 30.º punto).
TOPE_BADMINTON = 30

ESTILOS = {"Tenis": "o-", "Bádminton": "s-", "Ping Pong": "^-"}


# ----------------------------------------------------------------------------
# Bloques básicos (cada punto se juega con p1 o p2 según sea normal o decisivo)
# ----------------------------------------------------------------------------
def jugar_unidad(p1, p2, rng, tot, objetivo, tope=None):
    """Juega un juego de tenis, un tie-break o un set de bádminton/ping pong:
    gana quien llegue a `objetivo` puntos con diferencia de 2 (o a `tope`).
    Suma los puntos a `tot` = [puntos J1, puntos J2].
    Devuelve True si gana el Jugador 1.
    Un punto es decisivo si, al ganarlo, alguien cierra la unidad."""
    a = b = 0
    hay_clave = p1 != p2
    while True:
        p = p1
        if hay_clave:
            cierra_a = (a >= objetivo - 1 and a - b >= 1) or a + 1 == tope
            cierra_b = (b >= objetivo - 1 and b - a >= 1) or b + 1 == tope
            if cierra_a or cierra_b:
                p = p2
        if rng.random() < p:
            a += 1
            tot[0] += 1
        else:
            b += 1
            tot[1] += 1
        if tope is not None:
            if a == tope:
                return True
            if b == tope:
                return False
        if a >= objetivo and a - b >= 2:
            return True
        if b >= objetivo and b - a >= 2:
            return False


# ----------------------------------------------------------------------------
# Sets de cada deporte
# ----------------------------------------------------------------------------
def set_tenis(p1, p2, rng, tot):
    ga = gb = 0
    while True:
        if jugar_unidad(p1, p2, rng, tot, 4):        # juego
            ga += 1
        else:
            gb += 1
        if ga >= 6 and ga - gb >= 2:
            return True
        if gb >= 6 and gb - ga >= 2:
            return False
        if ga == 6 and gb == 6:                      # tie-break
            return jugar_unidad(p1, p2, rng, tot, 7)


def set_badminton(p1, p2, rng, tot):
    return jugar_unidad(p1, p2, rng, tot, 21, TOPE_BADMINTON)


def set_pingpong(p1, p2, rng, tot):
    return jugar_unidad(p1, p2, rng, tot, 11)


# nombre -> (función que simula un set, sets necesarios para ganar)
DEPORTES = {
    "Tenis": (set_tenis, 3),
    "Bádminton": (set_badminton, 2),
    "Ping Pong": (set_pingpong, 3),
}


# ----------------------------------------------------------------------------
# Partidos y estimaciones
# ----------------------------------------------------------------------------
def simular_partido(p1, p2, rng, simular_set, sets_para_ganar):
    """Devuelve (gana_J1, sets_J1, sets_J2, puntos_J1, puntos_J2)."""
    tot = [0, 0]
    a = b = 0
    while a < sets_para_ganar and b < sets_para_ganar:
        if simular_set(p1, p2, rng, tot):
            a += 1
        else:
            b += 1
    return a == sets_para_ganar, a, b, tot[0], tot[1]


def estimar(p1, p2, n_partidos, rng, simular_set, sets_para_ganar):
    """Devuelve (% victorias del J1, % de partidos con paradoja)."""
    victorias = paradojas = 0
    for _ in range(n_partidos):
        gana1, _, _, pts1, pts2 = simular_partido(p1, p2, rng, simular_set, sets_para_ganar)
        victorias += gana1
        paradojas += (pts1 < pts2) if gana1 else (pts2 < pts1)
    return 100.0 * victorias / n_partidos, 100.0 * paradojas / n_partidos


def analizar(n_partidos, pmin, pmax, n_puntos_p, deltas, semilla=None):
    """Devuelve valores_p y un dict (deporte, delta) -> (% victorias, % paradoja).
    El % de paradoja solo se guarda para el tenis (None en el resto)."""
    rng = random.Random(semilla)
    valores_p = np.linspace(pmin, pmax, n_puntos_p)
    resultados = {}

    for nombre, (simular_set, sets_para_ganar) in DEPORTES.items():
        for delta in deltas:
            print(f"Simulando {nombre} con p2 = p1 {delta:+.2f} ...", flush=True)
            victorias, paradojas = [], []
            for p1 in valores_p:
                p1 = float(p1)
                p2 = min(1.0, max(0.0, p1 + delta))
                v, par = estimar(p1, p2, n_partidos, rng, simular_set, sets_para_ganar)
                victorias.append(v)
                paradojas.append(par)
            resultados[(nombre, delta)] = (
                np.array(victorias),
                np.array(paradojas) if nombre == "Tenis" else None,
            )

    return valores_p, resultados


def imprimir_tablas(valores_p, resultados, deltas):
    for nombre in DEPORTES:
        con_paradoja = nombre == "Tenis"
        titulo = "% victorias J1 / % partidos con paradoja" if con_paradoja else "% victorias J1"
        print(f"\n== {nombre}: {titulo} ==")
        print("   p1  " + "".join(f"|  p2=p1{d:+.2f}   " for d in deltas))
        for i, p in enumerate(valores_p):
            fila = f" {p:.3f} "
            for d in deltas:
                v, par = resultados[(nombre, d)]
                if con_paradoja:
                    fila += f"| {v[i]:6.2f} /{par[i]:5.2f}  "
                else:
                    fila += f"| {v[i]:6.2f}          "
            print(fila)


def ejemplo_paradoja(rng, p=0.5, max_intentos=200000):
    """Busca un partido de tenis (con p1 = p2 = p) en el que el ganador tuvo menos puntos."""
    simular_set, sets_para_ganar = DEPORTES["Tenis"]
    for _ in range(max_intentos):
        gana1, s1, s2, pts1, pts2 = simular_partido(p, p, rng, simular_set, sets_para_ganar)
        if (gana1 and pts1 < pts2) or (not gana1 and pts2 < pts1):
            ganador = "Jugador 1" if gana1 else "Jugador 2"
            sets_g, sets_p = (s1, s2) if gana1 else (s2, s1)
            pts_g, pts_p = (pts1, pts2) if gana1 else (pts2, pts1)
            return (f"Tenis: ganó el {ganador} por {sets_g}-{sets_p} en sets "
                    f"con {pts_g} puntos frente a {pts_p} del rival")
    return "Tenis: no se encontró ningún ejemplo"


# ----------------------------------------------------------------------------
# Gráficas
# ----------------------------------------------------------------------------
def etiqueta_delta(delta):
    return "p2 = p1" if delta == 0 else f"p2 = p1 {delta:+.2f}"


def grafica_comparacion(valores_p, resultados, archivo):
    """Gráfica de referencia: % victorias vs p1 en los tres deportes (p2 = p1)."""
    plt.figure(figsize=(8, 6))
    for nombre in DEPORTES:
        plt.plot(valores_p, resultados[(nombre, 0.0)][0], ESTILOS[nombre], label=nombre)
    plt.xlabel("Probabilidad de ganar un punto (Jugador 1)")
    plt.ylabel("% Victorias Jugador 1")
    plt.title("Probabilidad vs % Victorias Jugador 1")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(archivo, dpi=150)
    print(f"Gráfica guardada en: {archivo}")


def grafica_victorias_p2(valores_p, resultados, deltas, archivo):
    """Una gráfica por deporte con una curva de % victorias para cada delta."""
    fig, ejes = plt.subplots(1, len(DEPORTES), figsize=(15, 4.8), sharey=True)
    for eje, nombre in zip(ejes, DEPORTES):
        for k, delta in enumerate(deltas):
            eje.plot(valores_p, resultados[(nombre, delta)][0], "o-", ms=4,
                     color=f"C{k}", label=etiqueta_delta(delta))
        eje.set_title(nombre)
        eje.set_xlabel("p1: probabilidad de punto normal (Jugador 1)")
        eje.grid(True)
        eje.legend()
    ejes[0].set_ylabel("% Victorias Jugador 1")
    fig.suptitle("Efecto del punto decisivo: % victorias Jugador 1 vs p1")
    fig.tight_layout()
    fig.savefig(archivo, dpi=150)
    print(f"Gráfica guardada en: {archivo}")


def grafica_paradoja_tenis(valores_p, resultados, deltas, archivo):
    """% de partidos de tenis ganados por el jugador con menos puntos totales."""
    plt.figure(figsize=(8, 6))
    for k, delta in enumerate(deltas):
        plt.plot(valores_p, resultados[("Tenis", delta)][1], "o-", ms=4,
                 color=f"C{k}", label=etiqueta_delta(delta))
    plt.xlabel("p1: probabilidad de punto normal (Jugador 1)")
    plt.ylabel("% de partidos con paradoja")
    plt.title("Tenis: partidos ganados por quien hizo menos puntos")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(archivo, dpi=150)
    print(f"Gráfica guardada en: {archivo}")


# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def pedir_partidos():
    while True:
        try:
            n = int(input("¿Cuántos partidos quieres simular por cada valor de p? [1000]: ") or 1000)
            if n > 0:
                return n
        except ValueError:
            pass
        print("Introduce un número entero positivo.")


def main():
    parser = argparse.ArgumentParser(description="Tenis vs bádminton vs ping pong con puntos decisivos")
    parser.add_argument("-n", "--partidos", type=int, help="partidos a simular por cada combinación")
    parser.add_argument("--pmin", type=float, default=0.40, help="p1 mínima (por defecto 0.40)")
    parser.add_argument("--pmax", type=float, default=0.60, help="p1 máxima (por defecto 0.60)")
    parser.add_argument("--puntos-p", type=int, default=21,
                        help="cuántos valores de p1 probar entre pmin y pmax (por defecto 21)")
    parser.add_argument("--delta", type=float, nargs="+", default=[-0.05, 0.05],
                        help="valores de delta con p2 = p1 + delta (por defecto -0.05 0.05); "
                             "delta = 0 (p2 = p1) se incluye siempre")
    parser.add_argument("--semilla", type=int, default=None, help="semilla aleatoria (opcional)")
    parser.add_argument("--salida", default="resultados",
                        help="prefijo de los archivos de gráficas (por defecto 'resultados')")
    args = parser.parse_args()

    if not (0 <= args.pmin < args.pmax <= 1):
        parser.error("Se requiere 0 <= pmin < pmax <= 1")

    deltas = sorted(set(args.delta) | {0.0})
    n = args.partidos if args.partidos else pedir_partidos()

    valores_p, resultados = analizar(n, args.pmin, args.pmax, args.puntos_p, deltas, args.semilla)

    imprimir_tablas(valores_p, resultados, deltas)

    print("\n== Ejemplo de la paradoja en tenis (p1 = p2 = 0.5) ==")
    print(ejemplo_paradoja(random.Random(args.semilla)))

    print()
    grafica_comparacion(valores_p, resultados, f"{args.salida}_victorias.png")
    grafica_victorias_p2(valores_p, resultados, deltas, f"{args.salida}_victorias_p2.png")
    grafica_paradoja_tenis(valores_p, resultados, deltas, f"{args.salida}_paradoja_tenis.png")
    plt.show()


if __name__ == "__main__":
    main()
