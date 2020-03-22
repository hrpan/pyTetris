#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <blocks.h>
#include <algorithm>
#include <vector>
#include <ctime>
#include <iostream>
#include <cstdio>
#include <numeric>
#define P 919
#define N 4

namespace py = pybind11;

class Block{
    public:
    int position[2];
    int block_type;
    int rotation_index;

    Block(){
        std::fill_n(position, 2, 0);
        block_type = 0;
        rotation_index = 0;
    }

    Block(int init_pos[2], int init_type){
        std::copy_n(init_pos, 2, position);
        block_type = init_type;
    } 

    Block(const Block &other){
        std::copy_n(other.position, 2, position);
        block_type = other.block_type;
        rotation_index = other.rotation_index;
    }

    void move(const int direction[2]){
        position[0] += direction[0];
        position[1] += direction[1];
    }

    void getFilled(int filled[N][2]){
        for(int i=0;i<N;++i){
            filled[i][0] = blocks[block_type][rotation_index][i][0] + position[0];
            filled[i][1] = blocks[block_type][rotation_index][i][1] + position[1];
        }
    }

    void rotate(int direction){
        if(direction == 0)
            rotation_index = (rotation_index + 1) % 4;
        else
            rotation_index = (rotation_index + 3) % 4;
    }

    bool operator==(const Block &other){
        return (
            std::equal(position, position+2, other.position) &&
            block_type == other.block_type &&
            rotation_index == other.rotation_index);
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
        std::copy_n(init_bs, 2, boardsize);
        board.resize(boardsize[0] * boardsize[1]);
    }

    Board(const Board &other){
        std::copy_n(other.boardsize, 2, boardsize);
        board = other.board;
    }

    void reset(){
        std::fill(board.begin(), board.end(), 0);
    }

    int clearLines(){
        int clears = 0;
        auto begin = board.begin() + boardsize[1];
        for(int r=1;r<boardsize[0];++r){

            auto end = begin + boardsize[1];
            bool isFilled = std::find(begin, end, 0) == end;

            if(isFilled){
                clears += 1;
                for(int _r=r;_r>0;--_r){
                    int _r_offset = _r * boardsize[1];
                    std::copy_n(board.begin() + _r_offset - boardsize[1], boardsize[1], board.begin() + _r_offset);
                }
                std::fill_n(board.begin(), boardsize[1], 0);
            }
            begin += boardsize[1];
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
        return (std::equal(boardsize, boardsize+2, other.boardsize) &&
                std::equal(board.begin(), board.end(), other.board.begin()));
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

    constexpr static int down[2] = {1, 0};
    constexpr static int left[2] = {0, -1};
    constexpr static int right[2] = {0, 1};

    Tetris(std::vector<int> _bs, int _apd){
        std::copy_n(_bs.begin(), 2, boardsize);
        board = Board(boardsize);
        init_pos[0] = 0;
        init_pos[1] = boardsize[1] / 2 - 2;

        actions_per_drop = _apd;

        std::iota(b_seq, b_seq+7, 0);

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

        std::fill_n(line_stats, 4, 0);

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

    bool move(const int direction[2]){
    
        int _tmp[2];
        std::copy_n(block.position, 2, _tmp);

        block.move(direction);

        int filled[N][2];
        block.getFilled(filled);

        if(!board.checkLegal(filled)){
            std::copy_n(_tmp, 2, block.position);
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
            move(left);
        }else if(action == 3){
            bool success = move(down);
            if(success)
                score += 1;
            else
                detachBlock();
        }else if(action == 4){
            move(right);
        }else if(action == 5){
            while(move(down))
                score += 2;
            detachBlock();
        }

        action_count = (action_count + 1) % actions_per_drop;

        if(action_count == 0){
            if(!move(down))
                detachBlock();
        }
    }

    void printState(){

        std::vector<short> b_tmp(board.board);

        if(!end){
            int filled[N][2];
            block.getFilled(filled);
            for(int i=0;i<N;++i)
                b_tmp[filled[i][0] * boardsize[1] + filled[i][1]] = -1;
        }
         
        for(int r=0;r<boardsize[0];++r){
            for(int c=0;c<boardsize[1];++c)
                printf("%2d ", b_tmp[r * boardsize[1] + c]);
            printf("\n");
        }
        fflush(stdout);

    }

    py::array_t<short> getState(){

        std::vector<short> b_tmp(board.board);

        if(!end){
            int filled[N][2];
            block.getFilled(filled);
            for(int i=0;i<N;++i)
                b_tmp[filled[i][0] * boardsize[1] + filled[i][1]] = -1;
        }

        int size = int(sizeof(short));

        return py::array_t<short>({boardsize[0], boardsize[1]}, {boardsize[1] * size, size}, &b_tmp[0]);
    }    

    void copy_from(const Tetris &other){
        action_count = other.action_count;
        actions_per_drop = other.actions_per_drop;
        std::copy_n(other.boardsize, 2, boardsize);
        std::copy_n(other.init_pos, 2, init_pos);
        end = other.end;
        block = other.block;
        board = other.board;
        std::copy_n(other.b_seq, 7, b_seq);
        b_seq_idx = other.b_seq_idx;
        b2b_tetris = other.b2b_tetris;
        combo = other.combo;
        max_combo = other.max_combo;
        line_clears = other.line_clears;
        std::copy_n(other.line_stats, N, line_stats);
        score = other.score;
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
        .def_readonly("end", &Tetris::end)
        .def("copy_from", &Tetris::copy_from)
        .def("reset", &Tetris::reset)      
        .def("play", &Tetris::play) 
        .def("printState", &Tetris::printState) 
        .def(py::init<const std::vector<int> &, int>(),
                py::arg("boardsize") = py::make_tuple(22, 10), 
                py::arg("actions_per_drop") = 1)
        .def("getState", &Tetris::getState)
        .def("__eq__", &Tetris::operator==)
        .def("__hash__", &Tetris::hash);
}
