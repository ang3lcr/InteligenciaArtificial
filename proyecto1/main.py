import pygame
import heapq
import random
import sys

# Constantes
ANCHO = 800
ALTO = 600
FILAS = 20
COLUMNAS = 20
TAMANO_CELDA = min(ANCHO // COLUMNAS, ALTO // FILAS)

# Colores
NEGRO = (0, 0, 0)
BLANCO = (255, 255, 255)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
AZUL = (0, 0, 255)
GRIS = (128, 128, 128)
AMARILLO = (255, 255, 0)

class Nodo:
    def __init__(self, x, y, g=0, h=0, padre=None):
        self.x = x
        self.y = y
        self.g = g  # Costo desde el inicio
        self.h = h  # Heurística hasta el destino
        self.f = g + h  # Costo total
        self.padre = padre

    def __lt__(self, otro):
        return self.f < otro.f

def a_estrella(inicio, destino, grid):
    movimientos = [(0, 1), (1, 0), (0, -1), (-1, 0), 
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]  # Incluye diagonales
    
    filas, cols = len(grid), len(grid[0])
    
    def es_valida(x, y):
        return 0 <= x < filas and 0 <= y < cols and grid[x][y] == 0
    
    lista_abierta = []
    cerrados = set()
    
    nodo_inicio = Nodo(inicio[0], inicio[1], h=heuristica(inicio, destino))
    heapq.heappush(lista_abierta, nodo_inicio)
    
    while lista_abierta:
        nodo_actual = heapq.heappop(lista_abierta)
        
        if (nodo_actual.x, nodo_actual.y) == destino:
            camino = []
            while nodo_actual:
                camino.append((nodo_actual.x, nodo_actual.y))
                nodo_actual = nodo_actual.padre
            return camino[::-1]
        
        cerrados.add((nodo_actual.x, nodo_actual.y))
        
        for dx, dy in movimientos:
            nx, ny = nodo_actual.x + dx, nodo_actual.y + dy
            
            if not es_valida(nx, ny) or (nx, ny) in cerrados:
                continue
            
            # Costo diferente para diagonales
            costo_movimiento = 1.4 if abs(dx) == 1 and abs(dy) == 1 else 1
            nuevo_g = nodo_actual.g + costo_movimiento
            nuevo_nodo = Nodo(nx, ny, nuevo_g, heuristica((nx, ny), destino), nodo_actual)
            
            existe = False
            for nodo in lista_abierta:
                if (nodo.x, nodo.y) == (nx, ny) and nodo.g <= nuevo_g:
                    existe = True
                    break
            
            if not existe:
                heapq.heappush(lista_abierta, nuevo_nodo)
    
    return None

def heuristica(a, b):
    # Distancia Euclidiana para movimientos diagonales
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def generar_posiciones_aleatorias(grid):
    """Genera posiciones inicial y final aleatorias que no sean obstáculos"""
    celdas_libres = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0:
                celdas_libres.append((i, j))
    
    if len(celdas_libres) < 2:
        return (0, 0), (FILAS-1, COLUMNAS-1)
    
    inicio, destino = random.sample(celdas_libres, 2)
    return inicio, destino

def dibujar_grid(screen, grid, inicio, destino, camino=None):
    screen.fill(BLANCO)
    
    # Dibujar grid
    for i in range(FILAS):
        for j in range(COLUMNAS):
            rect = pygame.Rect(j * TAMANO_CELDA, i * TAMANO_CELDA, TAMANO_CELDA, TAMANO_CELDA)
            
            if grid[i][j] == 1:  # Obstáculo
                pygame.draw.rect(screen, NEGRO, rect)
            else:
                pygame.draw.rect(screen, BLANCO, rect)
                pygame.draw.rect(screen, GRIS, rect, 1)
            
            # Dibujar inicio
            if (i, j) == inicio:
                pygame.draw.rect(screen, VERDE, rect)
            
            # Dibujar destino
            if (i, j) == destino:
                pygame.draw.rect(screen, ROJO, rect)
    
    # Dibujar camino si existe
    if camino:
        for i, j in camino:
            if (i, j) != inicio and (i, j) != destino:
                rect = pygame.Rect(j * TAMANO_CELDA, i * TAMANO_CELDA, TAMANO_CELDA, TAMANO_CELDA)
                pygame.draw.rect(screen, AZUL, rect)
    
    pygame.display.flip()

def main():
    pygame.init()
    screen = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("Algoritmo A* - Haz clic para obstáculos")
    
    # Inicializar grid vacío
    grid = [[0 for _ in range(COLUMNAS)] for _ in range(FILAS)]
    
    # Generar posiciones iniciales aleatorias
    inicio, destino = generar_posiciones_aleatorias(grid)
    
    camino = None
    ejecutando = True
    
    # Texto de instrucciones
    font = pygame.font.SysFont(None, 24)
    
    while ejecutando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                ejecutando = False
            
            elif evento.type == pygame.MOUSEBUTTONDOWN:
                # Agregar/eliminar obstáculo con clic
                x, y = pygame.mouse.get_pos()
                fila = y // TAMANO_CELDA
                columna = x // TAMANO_CELDA
                
                if 0 <= fila < FILAS and 0 <= columna < COLUMNAS:
                    # No permitir obstáculos en inicio o destino
                    if (fila, columna) != inicio and (fila, columna) != destino:
                        grid[fila][columna] = 1 - grid[fila][columna]  # Alternar entre 0 y 1
            
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE:
                    # Ejecutar A*
                    camino = a_estrella(inicio, destino, grid)
                    if not camino:
                        print("No se encontró camino!")
                
                elif evento.key == pygame.K_r:
                    # Reiniciar con nuevas posiciones aleatorias
                    inicio, destino = generar_posiciones_aleatorias(grid)
                    camino = None
                
                elif evento.key == pygame.K_c:
                    # Limpiar grid
                    grid = [[0 for _ in range(COLUMNAS)] for _ in range(FILAS)]
                    inicio, destino = generar_posiciones_aleatorias(grid)
                    camino = None
        
        # Dibujar
        dibujar_grid(screen, grid, inicio, destino, camino)
        
        # Dibujar instrucciones
        instrucciones = [
            "Clic: Colocar/eliminar obstáculo",
            "Espacio: Ejecutar A*",
            "R: Nuevas posiciones aleatorias",
            "C: Limpiar todo"
        ]
        
        for i, texto in enumerate(instrucciones):
            texto_surface = font.render(texto, True, NEGRO)
            screen.blit(texto_surface, (10, ALTO - 100 + i * 25))
        
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()