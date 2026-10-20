import numpy as np

MINE = -1   
FLAGGED = -1 
HIDDEN = -2   


# there are 3 boards in total
# one will be the solved with all correct flag and mine count
# one will be the current state of board user can see
# one will be the flag, which will count mine when user place a flag

class MineSweeper:
    def __init__(self, detail):
        self.rng = np.random.default_rng()
        self.rows = detail[0]
        self.cols = detail[1]
        self.mines = detail[2]
        self.board = np.full((self.rows, self.cols), HIDDEN)
        self.solved_board = self.board.copy()
        self.flag_board = self.board.copy()
        self.exploded = set()
        self.win = False
        self.hidden = 0

    def reset(self):
        self.win = False
        self.hidden = self.rows * self.cols
        temp_board = np.zeros((self.rows, self.cols)).astype(int)
        idx = self.rng.choice(self.rows * self.cols, size=self.mines, replace=False)
        temp_board.flat[idx] = MINE

        mines = (temp_board == MINE).astype(int)
        p = np.pad(mines, 1)


        h, w = mines.shape
        counts = sum(p[dy: dy+h, dx: dx+w]
                    for dy in (0, 1, 2) for dx in (0, 1, 2)
                    if (dx, dy) != (1, 1))

        self.board = np.full((self.rows, self.cols), HIDDEN)
        self.solved_board = np.where(mines == 1, MINE, counts)
        self.flag_board = np.zeros((self.rows, self.cols)).astype(int)
        self.exploded = set()
        
        
    def flag(self, x, y):
        if self.board[y, x] == HIDDEN:
            self.board[y, x] = FLAGGED

            h, w = self.board.shape
            y0, y1 = max(y - 1, 0), min(y + 2, h)
            x0, x1 = max(x - 1, 0), min(x + 2, w)
            self.flag_board[y0:y1, x0:x1] += 1
            self.flag_board[y, x] -= 1
            return 1

        elif self.board[y, x] == FLAGGED:
            self.board[y, x] = HIDDEN

            h, w = self.board.shape
            y0, y1 = max(y - 1, 0), min(y + 2, h)
            x0, x1 = max(x - 1, 0), min(x + 2, w)
            self.flag_board[y0:y1, x0:x1] -= 1
            self.flag_board[y, x] += 1
            return -1
        
        return 0



    def open(self, x, y, chording = False):
        h, w = self.board.shape
        if x < 0 or y < 0 or x >= w or y >= h:
            return True
        if self.board[y, x] != HIDDEN:
            if self.board[y, x] == FLAGGED:
                pass
            elif self.board[y, x] != HIDDEN and self.flag_board[y, x] == self.solved_board[y, x] and chording:
                safe = True
                for i in range(max(y - 1, 0), min(y + 2, h)):
                    for j in range(max(x - 1, 0), min(x + 2, w)):
                        if not self.open(j, i):
                            safe = False

                return safe

        elif self.solved_board[y, x] == MINE:
            self.exploded.add((y, x))
            return False
        else:
            stack = [(y, x)]
            while stack:
                cy, cx = stack.pop()
                if self.board[cy, cx] != HIDDEN: continue
                self.board[cy, cx] = self.solved_board[cy, cx]
                self.hidden -= 1
                if self.solved_board[cy, cx] == 0:
                    for i in range(max(cy - 1, 0), min(cy + 2, h)):
                        for j in range(max(cx - 1, 0), min(cx + 2, w)):
                            if (i, j) != (cy, cx):
                                stack.append((i, j))
                                
        # win when the only unopened cells (hidden or flagged) left are the mines
        if self.hidden == self.mines: self.win = True
        return True


if __name__ == "__main__":
    game = MineSweeper([16, 30, 99])
    
    game.reset()
    print(game.board)
    print(game.solved_board)
    print(game.flag_board)