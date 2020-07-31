#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <vector>
#include <array>
#include <iostream>
#include <cstdio>
#include <numeric>
#include <random>
#include <cstdlib>
#include <pyTetris.h>

namespace py = pybind11;

PYBIND11_MODULE(pyTetris, m){
    py::class_<Tetris>(m, "Tetris", py::buffer_protocol())
        .def_readonly("line_clears", &Tetris::line_clears)
        .def_readonly("score", &Tetris::score)
        .def_readonly("combo", &Tetris::combo)
        .def_readonly("max_combo", &Tetris::max_combo)
        .def_readonly("end", &Tetris::end)
        .def_property_readonly("line_stats", &Tetris::get_line_stats)
        .def("copy_from", &Tetris::copy_from)
        .def("reset", &Tetris::reset)      
        .def("play", &Tetris::play) 
        .def("printState", &Tetris::printState) 
        .def(py::init<const std::vector<int> &, int, int, int>(),
                py::arg("boardsize") = py::make_tuple(20, 10), 
                py::arg("actions_per_drop") = 1,
                py::arg("scoring") = 0,
                py::arg("randomizer") = 0)
        .def("getState", &Tetris::getState)
        .def("__eq__", &Tetris::operator==)
        .def("__hash__", &Tetris::hash)
        .def_buffer([](Tetris &t) -> py::buffer_info{
            return py::buffer_info(
                &t, sizeof(Tetris),
                "pyTetris", 1, {1}, {1});
        });
}
