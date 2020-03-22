#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <blocks.h>
#include <algorithm>
#include <vector>
#include <ctime>
#define P 919
#define N 4

namespace py = pybind11;

class Block{
    public:
    int position[2];
    int block_type;
    int rotation_index;

    Block(){
        position[0] = 0;
        position[1] = 0;
        block_type = 0;
        rotation_index = 0;
    }

    Block(int init_pos[2], int init_type){ 
        position[0] = init_pos[0]; 
        position[1] = init_pos[1];
        block_type = init_type;
    } 

    Block(const Block &other){
        position[0] = other.position[0];
        position[1] = other.position[1];
        block_type = other.block_type;
        rotation_index = other.rotation_index;
    }

    void move(int direction[2]){
        position[0] += direction[0];
        position[1] += direction[1];
    }

    void getFilled(int filled[N][2]){
        for(int i=0;i<N;++i){
            int row = blocks[block_type][rotation_index][i] / 4;
            int col = blocks[block_type][rotation_index][i] % 4;
            filled[i][0] = row + position[0];
            filled[i][1] = col + position[1];
        }
    }

    void rotate(int direction){
        if(direction == 0)
            rotation_index = (rotation_index + 1) % 4;
        else
            rotation_index = (rotation_index - 1) % 4;
    }

    bool operator==(const Block &other){
        return (
            position[0] != other.position[0] and
            position[1] != other.position[1] and
            block_type != other.block_type and 
            rotation_index != other.rotation_index);
    }

    int hash(){
        int _h = 0;
        _h = P * _h + position[0];
        _h = P * _h + position[1];
        _h = P * _h + block_type;
        _h = P * _h + rotation_index;

        return _h;
    }
};

class Board{
    public:
    int boardsize[2];
    std::vector<short> board;

    Board(){
        boardsize[0] = 22;
        boardsize[1] = 10;
        board.resize(boardsize[0] * boardsize[1]);
    }

    Board(int init_bs[2]){
        boardsize[0] = init_bs[0];
        boardsize[1] = init_bs[1];
        board.resize(boardsize[0] * boardsize[1]);
    }

    Board(const Board &other){
        boardsize[0] = other.boardsize[0];
        boardsize[1] = other.boardsize[1];
        board = other.board;
    }

    void reset(){
        std::fill(board.begin(), board.end(), 0);
    }

    int clearLines(){
        int clears = 0;
        for(int r=1;r<boardsize[0];++r){
            bool isFilled = true;

            int r_offset = r * boardsize[1];
            for(int c=0;c<boardsize[1]&&isFilled;++c)
                if(board[r_offset + c] != 1) isFilled = false;
            if(isFilled){
                clears += 1;
                for(int _r=1;_r<=r;++_r){
                    int _r_offset = _r * boardsize[1];
                    int _rp_offset = (_r - 1) * boardsize[1];
                    for(int _c=0;_c<boardsize[1];++_c)
                        board[_r_offset + _c] = board[_rp_offset + _c];
                }
                for(int _c=0;_c<boardsize[1];++_c)
                    board[_c] = 0;
            }
        }
        return clears;
    }

    bool checkFilled(int indices[N][2]){
        for(int i=0;i<N;++i){
            if(board[boardsize[1] * indices[i][0] + indices[i][1]] == 1)
                return true;
        }
        return false;
    }

    void fillBoard(int indices[N][2]){
        for(int i=0;i<N;++i)
            board[boardsize[1] * indices[i][0] + indices[i][1]] = 1;
    }

    bool checkLegal(int indices[N][2]){
        for(int i=0;i<N;++i){
            if( indices[i][0] < 0 ||
                indices[i][0] >= boardsize[0] ||
                indices[i][1] < 0 ||
                indices[i][1] >= boardsize[1])
                return false;
        }

        if(checkFilled(indices))
            return false;

        return true;
    }

    bool operator==(const Board &other){
        if(boardsize[0] != other.boardsize[0] || boardsize[1] != other.boardsize[1])
            return false;

        for(int i=0;i<boardsize[0] * boardsize[1];++i)
            if(board[i] != other.board[i])
                return false;
            
        return true;
    }

    int hash(){
        int _h = 0;
        for(int i: board)
            _h = P * _h + i;
        return _h;
    }
};

class Tetris{
    public:
    int action_count;
    int actions_per_drop;
    Block block;
    Board board;
    int boardsize[2];
    int init_pos[2];
    bool b2b_tetris;
    int combo;
    int max_combo;
    int line_clears;
    int score;
    bool end;
    int line_stats[4];
    int b_seq[7];
    int b_seq_idx;

    Tetris(std::vector<int> _bs, int _apd){
        boardsize[0] = _bs[0];
        boardsize[1] = _bs[1];
        board = Board(boardsize);
        init_pos[0] = 0;
        init_pos[1] = boardsize[1] / 2 - 2;

        actions_per_drop = _apd;
        for(int i=0;i<7;++i)
            b_seq[i] = i;

        std::srand(std::time(0));

        reset();
    }

    void reset(){
        board.reset();
        shuffle_block_sequence();
        action_count = 0;
        b2b_tetris = false;
        combo = 0;
        max_combo = 0;
        line_clears = 0;
        score = 0;

        for(int i=0;i<4;++i)
            line_stats[i] = 0;

        spawnBlock();

        end = false;
    }

    void shuffle_block_sequence(){
        std::random_shuffle(b_seq, b_seq+7);
        b_seq_idx = 0;
    }

    bool spawnBlock(){
        block = Block(init_pos, b_seq[b_seq_idx]);
        b_seq_idx += 1;

        if(b_seq_idx == 7)
            shuffle_block_sequence();

        int filled[N][2];
        block.getFilled(filled);

        if(board.checkFilled(filled))
            return false;

        return true;
    }

    void detachBlock(){
        int filled[N][2];
        block.getFilled(filled);

        board.fillBoard(filled);

        int cl = board.clearLines();

        if(cl == 0){
            combo = 0;
        }else{
            score += 50 * combo;
            combo += 1;
            max_combo = combo > max_combo? combo:max_combo;
            if(cl < 4){
                b2b_tetris = false;
                score += 200 * cl - 100;
            }else if(cl == 4){
                if(b2b_tetris)
                    score += 1200;
                else
                    b2b_tetris = true;
                score += 800;
            }
            line_clears += cl;
            line_stats[cl-1] += 1;
        }

        end = !spawnBlock();
    }

    bool move(int direction[2]){
    
        int _tmp[2];
        _tmp[0] = block.position[0];
        _tmp[1] = block.position[1];

        block.move(direction);

        int filled[N][2];
        block.getFilled(filled);

        if(!board.checkLegal(filled)){
            block.position[0] = _tmp[0];
            block.position[1] = _tmp[1];
            return false;
        }

        return true;
    }

    bool rotate(int direction){
    
        int _tmp = block.rotation_index;
        block.rotate(direction);
        
        int filled[N][2];
        block.getFilled(filled);

        if(!board.checkLegal(filled)){
            block.rotation_index = _tmp;
            return false;
        }

        return true;
    }

    void play(int action){

        if(action == 0){
            rotate(0);
        }else if(action == 1){
            rotate(1);
        }else if(action == 2){
            int _tmp[2] = {0, -1};
            move(_tmp);
        }else if(action == 3){
            int _tmp[2] = {1, 0};
            bool success = move(_tmp);
            if(success)
                score += 1;
            else
                detachBlock();
        }else if(action == 4){
            int _tmp[2] = {0, 1};
            move(_tmp);
        }else if(action == 5){
            int _tmp[2] = {1, 0};
            while(move(_tmp))
                score += 2;
            detachBlock();
        }

        action_count = (action_count + 1) % actions_per_drop;

        if(action_count == 0){
            int _tmp[2] = {1, 0};
            if(!move(_tmp))
                detachBlock();
        }
    }

    py::array_t<short> getState(){
        int filled[N][2];
        block.getFilled(filled);

        std::vector<short> b_tmp(board.board);

        for(int i=0;i<N;++i)
            b_tmp[filled[i][0] * boardsize[1] + filled[i][1]] = -1;
        

        int size = int(sizeof(short));

        return py::array_t<short>({boardsize[0], boardsize[1]}, {boardsize[1] * size, size}, &b_tmp[0]);
    }    

    bool operator==(const Tetris &other){
        return (
            action_count == other.action_count and
            block == other.block and
            board == other.board and
            std::equal(b_seq, b_seq + 7, other.b_seq) and
            b_seq_idx == other.b_seq_idx and
            b2b_tetris == other.b2b_tetris and
            combo == other.combo and
            max_combo == other.max_combo and
            line_clears == other.line_clears and
            std::equal(line_stats, line_stats + 4, other.line_stats) and
            score == other.score);
    }

    int hash(){
        int _h = action_count;
        _h = P * _h + block.hash();
        _h = P * _h + board.hash();
        _h = P * _h + b_seq_idx;
        _h = P * _h + combo;
        _h = P * _h + line_clears;
        _h = P * _h + score;
        return _h; 
    }
};

PYBIND11_MODULE(pyTetris, m){
    py::class_<Tetris>(m, "Tetris")
        .def_readonly("line_clears", &Tetris::line_clears)
        .def_readonly("score", &Tetris::score)
        .def_readonly("combo", &Tetris::combo)
        .def_readonly("max_combo", &Tetris::max_combo)
        .def("reset", &Tetris::reset)      
        .def("play", &Tetris::play) 
        .def(py::init<const std::vector<int> &, int>(),
                py::arg("boardsize") = py::make_tuple(22, 10), 
                py::arg("actions_per_drop") = 1)
        .def("getState", &Tetris::getState)
        .def("__eq__", &Tetris::operator==)
        .def("__hash__", &Tetris::hash);
}
