import pygame
import math

# Инициализация Pygame
pygame.init()

# Константы
WIDTH, HEIGHT = 800, 600
FOV = math.pi / 3  # Поле зрения 60 градусов
HALF_FOV = FOV / 2
NUM_RAYS = 120
MAX_DEPTH = 20
DELTA_ANGLE = FOV / NUM_RAYS
DIST = WIDTH / (2 * math.tan(HALF_FOV))
PROJ_COEFF = 3 * DIST * 50
SCALE = WIDTH // NUM_RAYS

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)
DARKGRAY = (50, 50, 50)

# Карта уровня
MAP = [
    "########",
    "#......#",
    "#..#...#",
    "#..#...#",
    "#......#",
    "########",
]

TILE = 100
MAP_ROWS = len(MAP)
MAP_COLS = len(MAP[0])
WORLD_WIDTH = MAP_COLS * TILE
WORLD_HEIGHT = MAP_ROWS * TILE

# Игрок
player_pos = [150, 150]
player_angle = 0
player_speed = 3

# Окно
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

def ray_casting(screen, player_pos, player_angle):
    ox, oy = player_pos
    cur_angle = player_angle - HALF_FOV
    for ray in range(NUM_RAYS):
        sin_a = math.sin(cur_angle)
        cos_a = math.cos(cur_angle)

        for depth in range(MAX_DEPTH):
            x = ox + depth * cos_a
            y = oy + depth * sin_a
            i, j = int(x // TILE), int(y // TILE)
            if 0 <= i < MAP_COLS and 0 <= j < MAP_ROWS and MAP[j][i] == '#':
                depth *= math.cos(player_angle - cur_angle)  # Устранение рыбьего глаза
                proj_height = PROJ_COEFF / (depth + 0.0001)
                color = GRAY if depth < MAX_DEPTH // 2 else DARKGRAY
                pygame.draw.rect(screen, color,
                                 (ray * SCALE, HEIGHT // 2 - proj_height // 2, SCALE, proj_height))
                break
        cur_angle += DELTA_ANGLE

# Основной игровой цикл
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        player_pos[0] += player_speed * math.cos(player_angle)
        player_pos[1] += player_speed * math.sin(player_angle)
    if keys[pygame.K_s]:
        player_pos[0] -= player_speed * math.cos(player_angle)
        player_pos[1] -= player_speed * math.sin(player_angle)
    if keys[pygame.K_a]:
        player_angle -= 0.05
    if keys[pygame.K_d]:
        player_angle += 0.05

    # Рендеринг
    screen.fill(BLACK)
    ray_casting(screen, player_pos, player_angle)
    pygame.display.flip()
    clock.tick(60)


pygame.quit()
