import numpy as np
import pygame
from engine import MineSweeper, MINE, HIDDEN, FLAGGED
from config import *
import random
from time import sleep


rows, cols, mines = LEVEL

CELL_SIZE = 30
PAD = 20
HEADER = 70
WINDOW_WIDTH = CELL_SIZE*cols + PAD * 2
WINDOW_HEIGHT = CELL_SIZE*rows + PAD * 2 + HEADER

BOARD_X = PAD
BOARD_Y = HEADER + PAD

GAME_END_WAIT = 5

NUMBER_COLORS = {
    1: (0, 0, 255), 2: (0, 128, 0), 3: (255, 0, 0), 4: (0, 0, 128),
    5: (128, 0, 0), 6: (0, 128, 128), 7: (0, 0, 0), 8: (128, 128, 128),
}
GRID_COLOR = (110, 110, 110)



reset_rect = pygame.Rect(WINDOW_WIDTH // 2 - 25, 10, 50, 50)


pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
font = pygame.font.SysFont("consolas", 28, bold=True)
big_font = pygame.font.SysFont("consolas", 48, bold=True)

pygame.display.set_caption("minesweeper")




mineGame = MineSweeper(LEVEL)
clock = pygame.time.Clock()

   


def draw_board(screen, game, game_over=False):
    for r in range(rows):
        for c in range(cols):
            rect = pygame.Rect(BOARD_X + c * CELL_SIZE, BOARD_Y + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            cell = game.board[r, c]
            is_mine = game.solved_board[r, c] == MINE

            if game_over and cell == HIDDEN and is_mine:
                bg = (255, 0, 0) if (r, c) in game.exploded else (220, 220, 220)
                pygame.draw.rect(screen, bg, rect)
                draw_mine(screen, rect)
            elif game_over and cell == FLAGGED and not is_mine:
                # wrong flag: draw an X over it
                pygame.draw.rect(screen, (160, 160, 160), rect)
                draw_flag(screen, rect)
                pygame.draw.line(screen, (0, 0, 0), rect.topleft, rect.bottomright, 2)
                pygame.draw.line(screen, (0, 0, 0), rect.topright, rect.bottomleft, 2)
            elif cell != HIDDEN:
                pygame.draw.rect(screen, (220, 220, 220), rect)
                if cell == FLAGGED:
                    pygame.draw.rect(screen, (160, 160, 160), rect)
                    draw_flag(screen, rect)
                elif cell != 0:
                    img = font.render(str(cell), True, NUMBER_COLORS[cell])
                    screen.blit(img, img.get_rect(center=rect.center))
            else:
                pygame.draw.rect(screen, (160, 160, 160), rect)
                
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)

def draw_mine(screen, rect):
    pygame.draw.circle(screen, (0, 0, 0), rect.center, rect.width // 4)

def draw_flag(screen, rect):
    x, y, s = rect.x, rect.y, rect.width
    pole_x = x + s * 0.55

    # base
    pygame.draw.rect(screen, (0, 0, 0), (x + s * 0.28, y + s * 0.75, s * 0.44, s * 0.1))
    # pole
    pygame.draw.line(screen, (0, 0, 0), (pole_x, y + s * 0.2), (pole_x, y + s * 0.75), 2)
    # red pennant
    pygame.draw.polygon(screen, (220, 0, 0), [
        (pole_x, y + s * 0.2),
        (pole_x, y + s * 0.5),
        (x + s * 0.2, y + s * 0.35),
    ])

def draw_header(screen, mines_left, seconds):
    # reset button
    pygame.draw.rect(screen, (200, 200, 0), reset_rect, border_radius=8)
    pygame.draw.rect(screen, (0, 0, 0), reset_rect, 2, border_radius=8)
    face = font.render(":)", True, (0, 0, 0))
    screen.blit(face, face.get_rect(center=reset_rect.center))

    # mine counter 
    left = font.render(f"{mines_left:03d}", True, (255, 0, 0))
    screen.blit(left, (PAD, 20))

    # timer
    t = font.render(f"{min(seconds, 999):03d}", True, (255, 0, 0))
    screen.blit(t, (WINDOW_WIDTH - PAD - t.get_width(), 20))
    
isplaying = False
running = True
start_ticks = None
elapsed = 0
mine_left = mines
game_over = False

mineGame.reset()
isplaying = True

game_over_start = None


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
    if isplaying:
        col = random.randint(0, cols-1)
        row = random.randint(0, rows-1)
        result = mineGame.open(col, row, True)
        if not result:
            isplaying = False
            game_over = True

    if isplaying:
        if start_ticks == None:
            start_ticks = pygame.time.get_ticks()
        current_tick = pygame.time.get_ticks()
        elapsed = (current_tick - start_ticks) // 1000
    
    
    screen.fill((192, 192, 192))
    draw_board(screen, mineGame, game_over)
    draw_header(screen, mine_left, elapsed)
    
   
        
    if game_over or mineGame.win:
        if game_over_start == None:
                    game_over_start = pygame.time.get_ticks()
        if game_over:
            img = big_font.render("GAME OVER", True, (200, 0, 0))
        else:
            img = big_font.render("YOU WIN!", True, (0, 160, 0))
        box = img.get_rect(center=screen.get_rect().center)
        pygame.draw.rect(screen, (255, 255, 255), box.inflate(30, 20), border_radius=8)
        pygame.draw.rect(screen, (0, 0, 0), box.inflate(30, 20), 2, border_radius=8)
        screen.blit(img, box)
        
        if (pygame.time.get_ticks() - game_over_start)//1000:
            mineGame.reset()
            start_ticks = None
            elapsed = 0
            mine_left = mines
            isplaying = True
            game_over = False
            game_over_start = None

    pygame.display.flip()
    clock.tick(60)

pygame.quit()


