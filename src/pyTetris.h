#ifndef PYTETRIS_H
#define PYTETRIS_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <blocks.h>
#include <algorithm>
#include <vector>
#include <array>
#include <ctime>
#include <iostream>
#include <cstdio>
#include <numeric>
#include <random>
#include <cstdlib>
#define P 919
#define N 4

namespace py = pybind11;

struct Vec{
    int x, y;

    Vec() : x{0}, y{0}{}

    Vec(int xx, int yy): x{xx}, y{yy} {}

    Vec(const Vec &other){
        x = other.x;
        y = other.y;
    }

    void reset(){x=0; y=0;};

    void operator=(const Vec &other){
        x = other.x;
        y = other.y;
    }

    bool operator==(const Vec &other) const{
        return x == other.x && y == other.y;
    }

    void set(const int xx, const int yy){
        x = xx;
        y = yy;
    }

    void add(const Vec &other){
        x += other.x;
        y += other.y;
    }
};

class Block{
    public:
    Vec position;
    int block_type;
    int rotation_index;
    std::array<Vec, N> filled;

    Block(){
        position.reset();
        block_type = 0;
        rotation_index = 0;
        set_filled();
    }

    Block(const Vec &init_pos, int init_type){
        position = init_pos;
        block_type = init_type;
        rotation_index = 0;
        set_filled();
    } 

    Block(const Block &other){
        position = other.position;
        block_type = other.block_type;
        rotation_index = other.rotation_index;
        set_filled();
    }

    void move(const Vec &direction){
        position.add(direction);
        set_filled();
    }

    void set_filled(){
        for(int i=0;i<N;++i){
            filled[i].x = blocks[block_type][rotation_index][i][0] + position.x;
            filled[i].y = blocks[block_type][rotation_index][i][1] + position.y;
        }
    }

    void rotate(const int direction){
        if(direction == 0)
            rotation_index = (rotation_index + 1) % 4;
        else
            rotation_index = (rotation_index + 3) % 4;
        set_filled();
    }

    bool operator==(const Block &other) const{
        return position == other.position &&
               block_type == other.block_type &&
               rotation_index == other.rotation_index;
    }

    int hash() const{
        int _h = 0;
        _h = P * _h + position.x;
        _h = P * _h + position.y;
        _h = P * _h + block_type;
        _h = P * _h + rotation_index;

        return _h;
    }
};

using Pixel=char;

class Board{
    public:;
    Vec boardsize;
    std::vector<Pixel> board;

    Board(){
        boardsize.set(20, 10);
        board.resize(200);
        reset();
    }

    Board(Vec init_bs){
        boardsize = init_bs;
        board.resize(boardsize.x * boardsize.y);
        reset();
    }

    Board(const Board &other){
        boardsize = other.boardsize;
        board = other.board;
    }

    void reset(){
        std::fill(board.begin(), board.end(), 0);
    }

    int clearLines(){
        int clears = 0;
        for(auto begin=board.begin();begin<board.end();begin+=boardsize.y){

            auto end = begin + boardsize.y;

            if(std::find(begin, end, 0) == end){
                clears += 1;
                for(auto rbegin=begin;rbegin>board.begin();rbegin-=boardsize.y){
                    std::copy_n(rbegin - boardsize.y, boardsize.y, rbegin);
                }
                std::fill_n(board.begin(), boardsize.y, 0);
            }
        }
        return clears;
    }

    template<typename T> bool checkFilled(const T &indices){
        for(auto v: indices){
            if(board[boardsize.y * v.x + v.y] == 1)
                return true;
        }
        return false;
    }

    template<typename T> void fillBoard(const T &indices){
        for(auto v: indices){
            board[boardsize.y * v.x + v.y] = 1;
        }
    }

    template<typename T> bool checkLegal(const T &indices){
        for(auto v: indices){
            if(v.x < 0 || v.x >= boardsize.x || v.y < 0 || v.y >= boardsize.y)
                return false;
        }
        return !checkFilled(indices);
    }

    bool operator==(const Board &other) const{
        return boardsize == other.boardsize && board == other.board;
    }

    int hash() const{
        int _h = 0;
        for(int i: board)
            _h = P * _h + i;
        return _h;
    }
};

enum Scoring{ guideline, lines };
enum Randomizer{ bag, uniform };

class Tetris{
    public:
    int action_count;
    int actions_per_drop;
    Block block;
    Board board;
    Vec boardsize;
    Vec init_pos;
    bool b2b_tetris;
    int combo;
    int max_combo;
    int line_clears;
    int score;
    Scoring scoring;
    bool end;
    std::array<int, N> line_stats;
    std::array<int, 7> b_seq;
    int b_seq_idx;
    Randomizer randomizer;

    static Vec down, left, right;

    Tetris(){
        boardsize.set(20, 10);
        board = Board(boardsize);
        init_pos.set(-1, 3);
        actions_per_drop = 1;
        std::iota(b_seq.begin(), b_seq.end(), 0);

        scoring = (Scoring)0;
        randomizer = (Randomizer)0;
        reset();
    }

    Tetris(const std::vector<int> _bs, int _apd, int _scoring, int _randomizer){
        boardsize.set(_bs[0], _bs[1]);
        board = Board(boardsize);
        init_pos.set(-1, boardsize.y / 2 - 2);

        actions_per_drop = _apd;

        std::iota(b_seq.begin(), b_seq.end(), 0);

        std::srand(std::time(0));
        
        scoring = (Scoring)_scoring;
        randomizer = (Randomizer)_randomizer;
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

        std::fill(line_stats.begin(), line_stats.end(), 0);

        spawnBlock();

        end = false;
    }

    void shuffle_block_sequence(){
        if(randomizer == bag)
            std::random_shuffle(b_seq.begin(), b_seq.end());
        else if(randomizer == uniform)
            b_seq[0] = std::rand() % 7;
        b_seq_idx = 0;
    }

    bool spawnBlock(){
        block = Block(init_pos, b_seq[b_seq_idx]);

        if(randomizer == bag){
            b_seq_idx += 1;

            if(b_seq_idx == 7)
                shuffle_block_sequence();
        }else if(randomizer == uniform){
            shuffle_block_sequence();
        }

        return !board.checkFilled(block.filled);
    }

    void detachBlock(){

        board.fillBoard(block.filled);

        int cl = board.clearLines();

        if(cl == 0){
            combo = 0;
        }else{
            if(scoring == guideline){
                score += 50 * combo;
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
            }else{
                score += cl;
            }
            combo += 1;
            max_combo = combo > max_combo? combo:max_combo;
            line_clears += cl;
            line_stats[cl-1] += 1;
        }

        end = !spawnBlock();
    }

    bool move(const Vec &direction){
    
        Vec _tmp(block.position);

        block.move(direction);

        if(!board.checkLegal(block.filled)){
            block.position = _tmp;
            block.set_filled();
            return false;
        }

        return true;
    }

    bool rotate(int direction){
    
        int _tmp = block.rotation_index;
        block.rotate(direction);
        
        if(!board.checkLegal(block.filled)){
            block.rotation_index = _tmp;
            block.set_filled();
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
            if(move(down)){
                if(scoring == guideline)
                    score += 1;
            }else{
                detachBlock();
            }
        }else if(action == 4){
            move(right);
        }else if(action == 5){
            while(move(down)){
                if(scoring == guideline)
                    score += 2;
            }
            detachBlock();
        }

        action_count = (action_count + 1) % actions_per_drop;

        if(action_count == 0){
            if(!move(down))
                detachBlock();
        }
    }

    void printState(){

        std::vector<Pixel> b_tmp(board.board);

        if(!end){
            for(auto v: block.filled)
                b_tmp[v.x * boardsize.y + v.y] = -1;
        }
         
        for(int r=0;r<boardsize.x;++r){
            for(int c=0;c<boardsize.y;++c)
                printf("%2d ", b_tmp[r * boardsize.y + c]);
            printf("\n");
        }
        fflush(stdout);

    }

    py::array get_line_stats(){
        return py::array(line_stats.size(), line_stats.data()); 
    }

    std::vector<Pixel> _getState(){

        std::vector<Pixel> b_tmp(board.board);

        if(!end){
            for(auto v: block.filled)
                b_tmp[v.x * boardsize.y + v.y] = -1;
        }

        return b_tmp;
    }    

    py::array_t<Pixel> getState(){
        std::vector<Pixel> b_tmp = _getState();

        size_t size = size_t(sizeof(Pixel));

        return py::array_t<Pixel>({boardsize.x, boardsize.y}, {boardsize.y * size, size}, &b_tmp[0]);
    }

    void copy_from(const Tetris &other){
        action_count = other.action_count;
        actions_per_drop = other.actions_per_drop;
        boardsize = other.boardsize;
        init_pos = other.init_pos;
        end = other.end;
        block = other.block;
        board = other.board;
        b_seq = other.b_seq;
        b_seq_idx = other.b_seq_idx;
        b2b_tetris = other.b2b_tetris;
        combo = other.combo;
        max_combo = other.max_combo;
        line_clears = other.line_clears;
        line_stats = other.line_stats;
        score = other.score;
        scoring = other.scoring;
        randomizer = other.randomizer;
    }

    bool operator==(const Tetris &other) const{
        return (
            action_count == other.action_count and
            block == other.block and
            board == other.board and
            b_seq == other.b_seq and
            b_seq_idx == other.b_seq_idx and
            b2b_tetris == other.b2b_tetris and
            combo == other.combo and
            max_combo == other.max_combo and
            line_clears == other.line_clears and
            line_stats == other.line_stats and
            score == other.score);
    }

    int hash() const{
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

Vec Tetris::down = Vec(1, 0);
Vec Tetris::left = Vec(0, -1);
Vec Tetris::right = Vec(0, 1);

#endif
