import numpy as np

MINE = -1   
FLAGGED = -1 
HIDDEN = -2   


# there are 3 boards in total
# one will be the solved with all correct flag and mine count
# one will be the current state of board user can see
# one will be the flag, which will count mine when user place a flag

def new_game(rows, cols, mines):
    board = np.zeros((rows, cols)).astype(int)
    idx = np.random.choice(rows * cols, size=mines, replace=False)
    board.flat[idx] = MINE

    mines = (board == MINE).astype(int)
    p = np.pad(mines, 1)


    h, w = mines.shape
    counts = sum(p[dy: dy+h, dx: dx+w]
                 for dy in (0, 1, 2) for dx in (0, 1, 2)
                 if (dx, dy) != (1, 1))


    solved_board = np.where(mines == 1, MINE, counts)
    return np.full((rows, cols), HIDDEN), solved_board, np.zeros((rows, cols)).astype(int)

def flag(board, flag_board, x, y):
    if board[y, x] == HIDDEN:
        board[y, x] = FLAGGED

        h, w = board.shape
        y0, y1 = max(y - 1, 0), min(y + 2, h)
        x0, x1 = max(x - 1, 0), min(x + 2, w)
        flag_board[y0:y1, x0:x1] += 1
        flag_board[y, x] -= 1

    elif board[y, x] == FLAGGED:
        board[y, x] = HIDDEN

        h, w = board.shape
        y0, y1 = max(y - 1, 0), min(y + 2, h)
        x0, x1 = max(x - 1, 0), min(x + 2, w)
        flag_board[y0:y1, x0:x1] -= 1
        flag_board[y, x] += 1



def open(board, solved_board, flag_board, x, y, chording = False):
    h, w = board.shape
    if x < 0 or y < 0 or x >= w or y >= h:
        return True
    if board[y, x] != HIDDEN:
        if board[y, x] == FLAGGED:
            pass
        elif flag_board[y, x] == solved_board[y, x] and chording:
            safe = True
            for i in range(max(y - 1, 0), min(y + 2, h)):
                for j in range(max(x - 1, 0), min(x + 2, w)):
                    if not open(board, solved_board, flag_board, j, i):
                        safe = False

            return safe

    elif solved_board[y, x] == MINE:
        return False
    else:
        stack = [(y, x)]
        while stack:
            cy, cx = stack.pop()
            if board[cy, cx] != HIDDEN: continue
            board[cy, cx] = solved_board[cy, cx]
            if solved_board[cy, cx] == 0:
                for i in range(max(cy - 1, 0), min(cy + 2, h)):
                    for j in range(max(cx - 1, 0), min(cx + 2, w)):
                        if (i, j) != (cy, cx):
                            stack.append((i, j))

    return True


if __name__ == "__main__":
    board, solved_board, flag_board = new_game(3, 4, 4)
    print(board)
    print(solved_board)
    flag(board, flag_board, 1, 1)
    print(board)
    print(flag_board)
