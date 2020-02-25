from numba import jit, jitclass
from numba import boolean, int8, int32, float32
import numpy as np

block_I = np.array([
    [0,0,0,0],
    [1,1,1,1],
    [0,0,0,0],
    [0,0,0,0]
],dtype=np.int8)

block_O = np.array([
    [0,0,0,0],
    [0,1,1,0],
    [0,1,1,0],
    [0,0,0,0]
],dtype=np.int8)

block_T = np.array([
    [0,0,0,0],
    [0,1,0,0],
    [1,1,1,0],
    [0,0,0,0]
],dtype=np.int8)

block_S = np.array([
    [0,0,0,0],
    [0,1,1,0],
    [1,1,0,0],
    [0,0,0,0]
],dtype=np.int8)

block_Z = np.array([
    [0,0,0,0],
    [1,1,0,0],
    [0,1,1,0],
    [0,0,0,0]
],dtype=np.int8)

block_J = np.array([
    [0,0,0,0],
    [0,0,1,0],
    [1,1,1,0],
    [0,0,0,0]
],dtype=np.int8)

block_L = np.array([
    [0,0,0,0],
    [0,1,0,0],
    [0,1,1,1],
    [0,0,0,0]
],dtype=np.int8)

block_proto = np.stack([
    block_I,
    block_O,
    block_T,
    block_S,
    block_Z,
    block_J,
    block_L
])

blocks = np.stack([
    np.stack([np.rot90(b,k) for k in range(4)]) for b in block_proto
])

filled = np.asarray([
    [np.where(b == 1) for b in btypes] for btypes in blocks
])


@jit(nopython=True, cache=True)
def array_equal(a, b):
    if a.shape != b.shape:
        return False
    for ai, bi in zip(a.flat, b.flat):
        if ai != bi:
            return False
    return True


block_spec = [
    ('position', int32[:]),
    ('block_type', int32),
    ('rot_idx', int32),
    ('state', int32[:]),
]


@jitclass(block_spec)
class Block:

    def __init__(self, pos=(0, 0), block_type=0, rot_idx=0):

        self.position = np.zeros(2, dtype=np.int32)
        self.position[0] = pos[0]
        self.position[1] = pos[1]

        self.block_type = block_type
        self.rot_idx = rot_idx

        self.state = np.zeros(4, dtype=np.int32)

    def rotate(self, direction):
        """
        Rotate in given direction ( 0 for CCW, 1 for CW )
        """
        if direction == 0:
            self.rot_idx = (self.rot_idx + 1) % 4
        elif direction == 1:
            self.rot_idx = (self.rot_idx - 1) % 4

    def move(self, direction):
        """
        Move in given direction
        """
        self.position[0] += direction[0]
        self.position[1] += direction[1]

    def getFilled(self):

        _p = self.position

        _tmp = filled[self.block_type][self.rot_idx]

        idx = (_tmp[0] + _p[0], _tmp[1] + _p[1])

        return idx

    def equal(self, state):
        for i in range(4):
            if self.state[i] != state[i]:
                return False
        return True

    def getState(self):

        self.state[0] = self.position[0]
        self.state[1] = self.position[1]
        self.state[2] = self.block_type
        self.state[3] = self.rot_idx

        return self.state


block_itype = Block.class_type.instance_type
@jit((block_itype, block_itype), nopython=True)
def fillBlock(b1, b2):
    b1.position[:] = b2.position
    b1.block_type = b2.block_type
    b1.rot_idx = b2.rot_idx


@jit((block_itype, block_itype), nopython=True)
def equalBlock(b1, b2):
    return (array_equal(b1.position, b2.position) and
           b1.block_type == b2.block_type and
           b1.rot_idx == b2.rot_idx)


@jit(nopython=True, cache=True)
def inside(indices, boardsize):
    h, w = boardsize
    l = len(indices[0])
    for i in range(l):
        x, y = indices[0][i], indices[1][i]
        if x < 0 or x >= h or y < 0 or y >= w:
            return False
    return True


board_spec = [
    ('boardsize', int8[:]),
    ('board', int8[:, :]),
]


@jitclass(board_spec)
class Board:

    def __init__(self, boardsize=(22, 10)):
        self.boardsize = np.zeros(2, dtype=np.int8)
        self.boardsize[0] = boardsize[0]
        self.boardsize[1] = boardsize[1]
        self.board = np.zeros(boardsize, dtype=np.int8)

    def reset(self):
        self.board.fill(0)

    def clearLines(self):
        n_rows, n_cols = self.board.shape
        filled_lines = []
        for _r in range(1, n_rows):
            isFilled = True
            for _c in range(n_cols):
                if self.board[_r][_c] != 1:
                    isFilled = False
            if isFilled:
                filled_lines.append(_r)
                self.board[1:_r+1] = self.board[0:_r]
                self.board[0] = 0
        return len(filled_lines)

    def checkFilled(self, indices):
        """
        if 1 in self.board[indices]:
            return True
        return False
        """
        l = len(indices[0])
        for i in range(l):
            if self.board[indices[0][i]][indices[1][i]] == 1:
                return True
        return False

    def fillBoard(self, indices):

        for i in range(len(indices[0])):
            self.board[indices[0][i]][indices[1][i]] = 1

    def checkLegal(self, indices):
        """
        Check if given indices can be filled, return True if legal, False otherwise
        """
        #if not self.inside(indices):
        h, w = self.boardsize
        for i in range(len(indices[0])):
            x, y = indices[0][i], indices[1][i]
            if x < 0 or x >= h or y < 0 or y >= w:
                return False

        if self.checkFilled(indices):
            return False

        return True

    def equal(self, state):
        return array_equal(self.board, state)

    def getState(self):
        return self.board


board_itype = Board.class_type.instance_type
@jit((board_itype, board_itype), nopython=True)
def fillBoard(b1, b2):
    b1.board[:, :] = b2.board
    b1.boardsize[:] = b2.boardsize


@jit((board_itype, board_itype), nopython=True)
def equalBoard(b1, b2):
    return array_equal(b1.board, b2.board)


t_spec = [
    ('action_count', int32),
    ('actions_per_drop', int32),
    ('block', Block.class_type.instance_type),
    ('board', Board.class_type.instance_type),
    ('boardsize', int8[:]),
    ('b_seq', int32[:]),
    ('b_seq_idx', int32),
    ('b2b_tetris', boolean),
    ('combo', int32),
    ('lines', int32),
    ('init_pos', int32[:]),
    ('score', int32),
    ('end', boolean),
    ('line_stats', int32[:]),
]


@jitclass(t_spec)
class T:

    def __init__(self, boardsize=(22, 10), actions_per_drop=3):
        self.boardsize = np.zeros(2, dtype=np.int8)
        self.boardsize[0] = boardsize[0]
        self.boardsize[1] = boardsize[1]
        self.board = Board(boardsize)

        self.init_pos = np.zeros(2, dtype=np.int32)
        self.init_pos[0] = 0
        self.init_pos[1] = boardsize[1] // 2 - 2

        self.actions_per_drop = actions_per_drop
        self.b_seq = np.zeros(len(block_proto), dtype=np.int32)
        self.b_seq[:] = np.arange(len(block_proto))

        self.line_stats = np.zeros(4, dtype=np.int32)
        self.reset()

    def reset(self):

        self.board.reset()

        self.shuffle_block_seq()

        self.action_count = 0

        self.b2b_tetris = False

        self.combo = 0
        self.lines = 0
        self.score = 0

        self.line_stats.fill(0)

        self.spawnBlock()

        self.end = False

    def shuffle_block_seq(self):
        """
        Shuffle block sequence and reset index
        """
        np.random.shuffle(self.b_seq)
        self.b_seq_idx = 0

    def spawnBlock(self):
        """
        Spawn a new block, return True if success, False otherwise
        """
        self.block = Block(self.init_pos, self.b_seq[self.b_seq_idx], 0)

        self.b_seq_idx += 1

        if self.b_seq_idx == len(block_proto):
            self.shuffle_block_seq()

        f_idx = self.block.getFilled()

        if self.board.checkFilled(f_idx):
            return False

        return True

    def detachBlock(self):
        """
        Detach the current block, clear lines and spawn a new block
        """

        f_idx = self.block.getFilled()

        self.board.fillBoard(f_idx)

        cl = self.board.clearLines()

        if cl == 0:
            self.combo = 0
        else:
            self.score += 50 * self.combo
            self.combo += 1
            if cl < 4:
                self.b2b_tetris = False
                self.score += 200 * cl - 100
            elif cl == 4:
                if self.b2b_tetris:
                    self.score += 1200
                else:
                    self.b2b_tetris = True
                self.score += 800

        self.lines += cl
        self.line_stats[cl-1] += 1

        self.end = not self.spawnBlock()

    def move(self, direction):
        """
        Move block in given direction (2-tuple), return True if success, False otherwise.
        """

        _tmp = self.block.position.copy()

        self.block.move(direction)

        f_idx = self.block.getFilled()

        if not self.board.checkLegal(f_idx):
            self.block.position = _tmp
            return False

        return True

    def rotate(self, direction):
        """
        Rotate block in given direction(0 or 1), return True if success, False otherwise.
        """
        _tmp = self.block.rot_idx

        self.block.rotate(direction)

        f_idx = self.block.getFilled()

        if not self.board.checkLegal(f_idx):
            self.block.rot_idx = _tmp
            return False

        return True

    def play(self, action):
        """
        Play the given action
        0 : Rotate counter-clockwise
        1 : Rotate clockwise
        2 : Move left
        3 : Move down
        4 : Move right
        5 : Hard drop
        6 : pass
        return True if success, False otherwise
        """
        if action == 0:
            success = self.rotate(0)
        elif action == 1:
            success = self.rotate(1)
        elif action == 2:
            success = self.move((0, -1))
        elif action == 3:
            success = self.move((1, 0))
            if not success:
                self.detachBlock()
            else:
                self.score += 1
        elif action == 4:
            success = self.move((0, 1))
        elif action == 5:
            while self.move((1, 0)):
                self.score += 2
            self.detachBlock()
        else:
            success = False

        self.action_count = (self.action_count + 1) % self.actions_per_drop

        if self.action_count == 0:
            if not self.move((1, 0)):
                self.detachBlock()

        #return success

    def getState(self):
        """
        Return state (board + block)
        """
        f_idx = self.block.getFilled()
        b = np.copy(self.board.board)
        for i in range(len(f_idx[0])):
            b[f_idx[0][i], f_idx[1][i]] = -1

        return b

    def printState(self):
        print(self.getState())

    def getScore(self):
        """
        Return score
        """
        return self.score

    def to_array(self):
        _arr = (self.board.getState(), self.block.getState(), self.b_seq, self.b_seq_idx, self.score)
        return _arr

    def hash(self):
        _h = 0

        _board = self.board.getState()
        for i in range(self.boardsize[0] * self.boardsize[1]):
            _h = 31 * _h + _board.flat[i]

        _block = self.block.getState()

        for i in range(len(_block)):
            _h = 31 * _h + _block[i]

        for i in range(len(self.b_seq)):
            _h = 31 * _h + self.b_seq[i]

        _h = 31 * _h + self.b_seq_idx
        _h = 31 * _h + self.score
        _h = 31 * _h + self.lines
        _h = 31 * _h + self.combo

        return _h


t_itype = T.class_type.instance_type
@jit((t_itype, t_itype), nopython=True)
def fillT(t1, t2):
    t1.action_count = t2.action_count
    t1.actions_per_drop = t2.actions_per_drop
    fillBlock(t1.block, t2.block)
    fillBoard(t1.board, t2.board)
    t1.boardsize[:] = t2.boardsize
    t1.b_seq[:] = t2.b_seq[:]
    t1.b_seq_idx = t2.b_seq_idx
    t1.b2b_tetris = t2.b2b_tetris
    t1.combo = t2.combo
    t1.lines = t2.lines
    t1.line_stats[:] = t2.line_stats[:]
    t1.init_pos[:] = t2.init_pos
    t1.end = t2.end
    t1.score = t2.score


@jit((t_itype, t_itype), nopython=True)
def equalT(t1, t2):
    return (t1.action_count == t2.action_count and
        equalBlock(t1.block, t2.block) and
        equalBoard(t1.board, t2.board) and
        array_equal(t1.b_seq, t2.b_seq) and
        t1.b_seq_idx == t2.b_seq_idx and
        t1.b2b_tetris == t2.b2b_tetris and
        t1.combo == t2.combo and
        t1.lines == t2.lines and
        array_equal(t1.line_stats, t2.line_stats) and
        t1.score == t2.score)


class Tetris:

    def __init__(self, boardsize=(22, 10), actions_per_drop=3):
        self.boardsize = boardsize
        self.actions_per_drop = actions_per_drop
        self.tetris = T(boardsize, actions_per_drop)

    @property
    def end(self):
        return self.tetris.end

    @property
    def line_stats(self):
        return self.tetris.line_stats

    def reset(self):
        self.tetris.reset()

    def play(self, action):
        """
        Play the given action
        0 : Rotate counter-clockwise
        1 : Rotate clockwise
        2 : Move left
        3 : Move down
        4 : Move right
        5 : Hard drop
        6 : pass
        """
        self.tetris.play(action)

    def getState(self):
        """
        Return state (board + block)
        """
        return self.tetris.getState()

    def printState(self):
        print(self.getState())

    def getCombo(self):
        """
        Return current combo count
        """
        return self.tetris.combo

    def getLines(self):
        """
        Return lines cleared
        """
        return self.tetris.lines

    def getScore(self):
        """
        Return score
        """
        return self.tetris.score

    def clone(self):
        """
        Make a clone of self
        """

        _tmp = Tetris(self.boardsize, self.actions_per_drop)
        fillT(_tmp.tetris, self.tetris)
        return _tmp

    def copy_from(self, other):

        fillT(self.tetris, other.tetris)

    def __hash__(self):

        h = self.tetris.hash()
        return h

    def __eq__(self, other):

        return equalT(self.tetris, other.tetris)
