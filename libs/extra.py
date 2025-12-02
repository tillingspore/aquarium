# Definir a escala (exemplo: 10 pixels = 1 cm)
PIXELS_PER_CM = 15
SENSORS_PER_ROW = 8
LINE_SPACING = 20  # Espaço entre as linhas

def pixels_to_cm(pixels):
    return pixels / PIXELS_PER_CM

def cm_to_pixels(cm):
    return int(cm * PIXELS_PER_CM)  # Retorna um valor inteiro para compatibilidade com Pygame
