import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Variables para el cuadrado
square_size = 100
square_angle = 0
square_center = None

# Función para dibujar un cuadrado rotado
def dibujar_cuadrado_rotado(frame, center, size, angle, color=(0, 255, 0), thickness=2):
    # Calcular los vértices del cuadrado
    half_size = size // 2
    vertices = np.array([[-half_size, -half_size],
                         [half_size, -half_size],
                         [half_size, half_size],
                         [-half_size, half_size]], dtype=np.float32)
    
    # Matriz de rotación
    angle_rad = np.radians(angle)
    cos_val = np.cos(angle_rad)
    sin_val = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_val, -sin_val],
                                [sin_val, cos_val]])
    
    # Rotar y trasladar los vértices
    rotated_vertices = np.dot(vertices, rotation_matrix.T) + center
    
    # Dibujar el cuadrado
    for i in range(4):
        pt1 = tuple(rotated_vertices[i].astype(int))
        pt2 = tuple(rotated_vertices[(i + 1) % 4].astype(int))
        cv2.line(frame, pt1, pt2, color, thickness)

# Función para detectar gestos y controlar el cuadrado
def controlar_cuadrado(hand_landmarks, frame):
    global square_size, square_angle
    
    h, w, _ = frame.shape
    
    # Obtener coordenadas de los puntos clave en píxeles
    dedos = [(int(hand_landmarks.landmark[i].x * w), 
              int(hand_landmarks.landmark[i].y * h)) for i in range(21)]
    
    # Puntas de los dedos
    pulgar, indice, medio, anular, menique = dedos[4], dedos[8], dedos[12], dedos[16], dedos[20]
    muneca = dedos[0]
    
    # Mostrar los números de los landmarks en la imagen
    for i, (x, y) in enumerate(dedos):
        cv2.circle(frame, (x, y), 5, (233, 23, 0), -1)
        cv2.putText(frame, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Dibujar línea entre pulgar e índice para el gesto de pellizco
    cv2.line(frame, pulgar, indice, (244, 34, 12), 2)
    
    # Calcular distancia entre pulgar e índice (gesto de pellizco)
    distancia_pulgar_indice = np.linalg.norm(np.array(pulgar) - np.array(indice))
    
    # Mostrar distancia
    cv2.putText(frame, f'Distancia: {int(distancia_pulgar_indice)}', (pulgar[0]-40, pulgar[1] - 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Gesto de pellizco para escalar
    if distancia_pulgar_indice < 50:  # Umbral para detectar pellizco
        # El tamaño cambia según la distancia (invertido para que al juntar dedos se haga más pequeño)
        scale_factor = distancia_pulgar_indice / 50.0
        square_size = max(50, min(300, int(100 * scale_factor)))
    
    # Gesto de rotación (pulgar y meñique extendidos, otros dedos cerrados)
    # Verificar si el pulgar y meñique están extendidos y los otros doblados
    def dedo_extendido(punta, base):
        return punta[1] < base[1]  # La punta está por encima de la base
    
    # Puntos de referencia para verificar extensión de dedos
    base_indice = dedos[5]  # Base del índice
    base_medio = dedos[9]   # Base del medio
    base_anular = dedos[13] # Base del anular
    
    pulgar_extendido = dedo_extendido(pulgar, dedos[2])
    indice_extendido = dedo_extendido(indice, base_indice)
    medio_extendido = dedo_extendido(medio, base_medio)
    anular_extendido = dedo_extendido(anular, base_anular)
    menique_extendido = dedo_extendido(menique, dedos[17])
    
    # Gesto de rotación: solo pulgar y meñique extendidos
    if pulgar_extendido and menique_extendido and not indice_extendido and not medio_extendido and not anular_extendido:
        # Calcular ángulo basado en la posición de la muñeca y el meñique
        dx = menique[0] - muneca[0]
        dy = menique[1] - muneca[1]
        square_angle = (np.degrees(np.arctan2(dy, dx)) + 90) % 360

# Captura de video en tiempo real
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener dimensiones del frame
    h, w, _ = frame.shape
    if square_center is None:
        square_center = (w // 2, h // 2)

    # Convertir a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe
    results = hands.process(frame_rgb)

    # Dibujar puntos de la mano y controlar el cuadrado
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Controlar el cuadrado con gestos
            controlar_cuadrado(hand_landmarks, frame)

    # Dibujar el cuadrado en el centro
    dibujar_cuadrado_rotado(frame, square_center, square_size, square_angle)
    
    # Mostrar información del cuadrado
    cv2.putText(frame, f"Tamano: {square_size}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Angulo: {int(square_angle)}", (50, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Pellizco: escalar | Pulgar+Menique: rotar", (50, h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Mostrar el video
    cv2.imshow("Control de Cuadrado con Gestos", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()