from numba import jit
import numpy as np

block_I = np.array([
    [0,0,0,0],
    [1,1,1,1],
    [0,0,0,0],
    [0,0,0,0]
],dtype=np.int8)

block_O = np.array([
    [1,1],
    [1,1]
],dtype=np.int8)

block_T = np.array([
    [0,1,0],
    [1,1,1],
    [0,0,0]
],dtype=np.int8)

block_S = np.array([
    [0,1,1],
    [1,1,0],
    [0,0,0]
],dtype=np.int8)

block_Z = np.array([
    [1,1,0],
    [0,1,1],
    [0,0,0]
],dtype=np.int8)

block_J = np.array([
    [0,0,1],
    [1,1,1],
    [0,0,0]
],dtype=np.int8)

block_L = np.array([
    [1,0,0],
    [1,1,1],
    [0,0,0]
],dtype=np.int8)

blocks = [
    [np.rot90(block_I,k) for k in range(4)],
    [np.rot90(block_O,k) for k in range(4)],
    [np.rot90(block_T,k) for k in range(4)],
    [np.rot90(block_S,k) for k in range(4)],
    [np.rot90(block_Z,k) for k in range(4)],
    [np.rot90(block_J,k) for k in range(4)],
    [np.rot90(block_L,k) for k in range(4)],
]

filled = [
    [np.where( b ==1 ) for b in btypes] for btypes in blocks        
]

class Block:

    def __init__(self,pos=(0,0),block_type=0,rot_idx=0):
        self.position = pos
        self.block_type = block_type
        self.rot_idx = rot_idx

    def rotate(self,direction):
        """
        Rotate in given direction ( 0 for CCW, 1 for CW )
        """
        if direction == 0:
            self.rot_idx = ( self.rot_idx + 1 ) % 4
        elif direction == 1:
            self.rot_idx = ( self.rot_idx - 1 ) % 4

    def move(self,direction):
        """
        Move in given direction
        """
        p = ( self.position[0] + direction[0], self.position[1] + direction[1] )
        self.position = p

    def getFilled(self):
        """
        b = blocks[self.block_type][self.rot_idx]
        ones = np.where( b == 1 )
        """
        _p = self.position

        _tmp = filled[self.block_type][self.rot_idx]

        idx = (_tmp[0] + _p[0], _tmp[1] + _p[1])
        #idx = [ones[i] + self.position[i] for i in range(2)]
        return idx
    
    def clone(self):

        _tmp = Block.__new__(Block)
        for k, v in self.__dict__.items():
            setattr(_tmp,k,v)
        return _tmp

    def equiv(self,b):
        if (self.position == b.position and 
            self.block_type == b.block_type and 
            self.rot_idx == b.rot_idx):
            return True
        return False

@jit(nopython=True)
def inside(indices,boardsize):
    h, w = boardsize
    l = len(indices[0])
    for i in range(l):
        x, y = indices[0][i], indices[1][i]
        if x < 0 or x >= h or y < 0 or y >= w :
            return False
    return True

@jit(nopython=True)
def checkFilled(indices,board):
    l = len(indices[0])
    for i in range(l):
        if board[indices[0][i]][indices[1][i]] == 1:
            return True
    return False

class Board:

    def __init__(self,boardsize=(22,10)):
        self.boardsize = boardsize
        self.board = np.zeros(boardsize,dtype=np.int8)

    def reset(self):
        self.board.fill(0)

    def clearLines(self):
        n_rows, n_cols = self.boardsize 
        count = 0
        for c in range(n_rows-1):
            if np.sum(self.board[-1-c]) == n_cols:
                count += 1
                self.board[1:n_rows-c] = self.board[0:n_rows-c-1]
                self.board[0] = 0
        return count

    def checkFilled(self,indices):
        """
        if 1 in self.board[indices]:
            return True
        return False
        """
        return checkFilled(indices,self.board)

    def fillBoard(self,indices):
        self.board[indices] = 1
    """
    def inside(self,indices):
        for i in indices[0]:
            if i < 0 or i >= self.boardsize[0]:
                return False
        for i in indices[1]:
            if i < 0 or i >= self.boardsize[1]:
                return False
        return True
    """

    def checkLegal(self,indices):
        """
        Check if given indices can be filled, return True if legal, False otherwise
        """
        #if not self.inside(indices):
        if not inside(indices,self.boardsize):
            return False

        if self.checkFilled(indices):
            return False

        return True

    def clone(self):
        _tmp = Board.__new__(Board)
        _tmp.boardsize = self.boardsize
        _tmp.board = np.copy(self.board)
        return _tmp
    
    def equiv(self,b):
        """
        Determine if two Boards are equiv, return True if yes, False otherwise
        """
        if np.array_equal(self.board,b.board):
            return True
        return False

class Tetris:

    def __init__(self,boardsize=(22,10),actions_per_drop=3):
        self.boardsize = boardsize
        self.board = Board(boardsize)
        
        self.init_pos = (0,boardsize[1] // 2 - 2)
        
        self.actions_per_drop = actions_per_drop

        self.reset()

    def reset(self):
        
        self.board.reset()

        self.b_seq = np.random.permutation(len(blocks))
        self.b_seq_idx = 0

        self.action_count = 0

        self.score = 0

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
        self.block = Block(pos=self.init_pos,block_type=self.b_seq[self.b_seq_idx])

        self.b_seq_idx += 1

        if self.b_seq_idx == len(blocks):
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
        
        self.score += cl

        self.end = not self.spawnBlock()

    def move(self,direction):
        """
        Move block in given direction (2-tuple), return True if success, False otherwise.
        """

        _tmp = self.block.position

        self.block.move(direction)

        f_idx = self.block.getFilled()
        
        if not self.board.checkLegal(f_idx):
            self.block.position = _tmp
            return False

        return True

    def rotate(self,direction):
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

    def play(self,action):
        """
        Play the given action
        0 : Rotate counter-clockwise
        1 : Rotate clockwise
        2 : Move left
        3 : Move down
        4 : Move right
        5 : pass
        return True if success, False otherwise
        """
        if action == 0:
            success = self.rotate(0)
        elif action == 1:
            success = self.rotate(1)
        elif action == 2:
            success = self.move((0,-1))
        elif action == 3:
            success = self.move((1,0))
            if not success:
                self.detachBlock()
        elif action == 4:
            success = self.move((0,1))
        else:
            success = False
       
        self.action_count = ( self.action_count + 1 ) % self.actions_per_drop

        if self.action_count == 0:
            if not self.move((1,0)):
                self.detachBlock()

        return success

    def getState(self):
        """
        Return state (board + block)
        """
        f_idx = self.block.getFilled()
        b = np.copy(self.board.board)
        b[f_idx] = -1

        return b

    def printState(self):
        print(self.getState())

    def getScore(self):
        """
        Return score
        """
        return self.score

    def clone(self):
        """
        Make a clone of self
        """
        _tmp = Tetris.__new__(Tetris)
        for k, v in self.__dict__.items():
            if k is 'board':
                _tmp.board = self.board.clone()
            elif k is 'block':
                _tmp.block = self.block.clone()
            else:
                setattr(_tmp,k,v)
        return _tmp
    
    def equiv(self,env):
        if (self.block.equiv(env.block) and 
            self.board.equiv(env.board) and 
            np.array_equal(self.b_seq,env.b_seq) and 
            self.b_seq_idx == env.b_seq_idx):
            return True
        return False
