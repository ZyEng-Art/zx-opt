// SPDX-License-Identifier: Apache-2.0
// Exact integer implementations of the pinned PyZX CFlow score formulas.
// Matching, ordering, tie-breaking, and rewrite execution remain in Python.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cstdint>
#include <new>
#include <unordered_map>
#include <vector>

namespace {
using Word = std::uint64_t;
using Bits = std::vector<Word>;
struct PythonError {};
struct Owned {
    PyObject* p;
    explicit Owned(PyObject* value): p(value) { if (!p) throw PythonError(); }
    ~Owned() { Py_DECREF(p); }
    Owned(const Owned&) = delete;
};
void fail(PyObject* type, const char* message) {
    PyErr_SetString(type, message); throw PythonError();
}
bool truth(PyObject* value) {
    int result = PyObject_IsTrue(value);
    if (result < 0) throw PythonError();
    return result != 0;
}
long long integer(PyObject* value) {
    if (!PyLong_Check(value)) fail(PyExc_TypeError, "CFlow native scoring requires integer vertex IDs");
    auto result = PyLong_AsLongLong(value);
    if (PyErr_Occurred()) throw PythonError();
    return result;
}
long long count(const Bits& bits) {
    long long result = 0;
    for (Word value: bits) result += __builtin_popcountll(value);
    return result;
}
template<class Function> void each(const Bits& bits, Function function) {
    for (std::size_t w = 0; w < bits.size(); ++w) {
        Word value = bits[w];
        while (value) {
            unsigned bit = __builtin_ctzll(value);
            function(w*64+bit); value &= value-1;
        }
    }
}

struct Graph {
    std::unordered_map<long long, std::size_t> indices;
    std::vector<PyObject*> ids, rows; // borrowed; input graph and GIL remain held
    std::vector<Bits> adjacency;
    std::size_t words;

    explicit Graph(PyObject* graph) {
        if (!PyDict_Check(graph)) fail(PyExc_TypeError, "Unsupported CFlow graph backend");
        auto size = PyDict_Size(graph);
        if (size > 8192) fail(PyExc_OverflowError, "Native CFlow bitset budget exceeded");
        words = (size+63)/64;
        PyObject *key, *row; Py_ssize_t position = 0;
        while (PyDict_Next(graph, &position, &key, &row)) {
            if (!PyDict_Check(row)) fail(PyExc_TypeError, "CFlow adjacency rows must be dictionaries");
            indices.emplace(integer(key), ids.size()); ids.push_back(key); rows.push_back(row);
        }
        adjacency.assign(size, Bits(words, 0));
        for (std::size_t i = 0; i < rows.size(); ++i) {
            position = 0; PyObject *other, *type;
            while (PyDict_Next(rows[i], &position, &other, &type)) {
                auto j = index(other);
                if (i == j) fail(PyExc_TypeError, "Self loops use the Python CFlow fallback");
                adjacency[i][j/64] |= Word(1) << (j%64);
            }
        }
    }
    std::size_t index(PyObject* vertex) const {
        auto found = indices.find(integer(vertex));
        if (found == indices.end()) fail(PyExc_TypeError, "CFlow match references an absent vertex");
        return found->second;
    }
    Bits vertices(PyObject* values) const {
        Owned sequence(PySequence_Fast(values, "CFlow vertices must be a sequence"));
        Bits result(words, 0);
        for (Py_ssize_t j = 0; j < PySequence_Fast_GET_SIZE(sequence.p); ++j) {
            auto i = index(PySequence_Fast_GET_ITEM(sequence.p, j));
            result[i/64] |= Word(1) << (i%64);
        }
        return result;
    }
    bool connected(std::size_t a, std::size_t b) const {
        return (adjacency[a][b/64] & (Word(1) << (b%64))) != 0;
    }
};

PyObject* weighted(PyObject* weights, int slot, long long improvement, long long removed, bool identity=false) {
    if (!identity && !(improvement > 0 || (improvement == 0 && removed > 0))) Py_RETURN_NONE;
    if (identity && improvement < 0) fail(PyExc_ValueError, "Negative identity-fusion improvement");
    Owned value(PyLong_FromLongLong(improvement));
    PyObject* result = PyNumber_Multiply(PySequence_Fast_GET_ITEM(weights, slot), value.p);
    if (!result) throw PythonError();
    return result;
}

PyObject* score(const Graph& graph, PyObject* match, PyObject* weights) {
    Owned parts(PySequence_Fast(match, "CFlow matches must have three parts"));
    if (PySequence_Fast_GET_SIZE(parts.p) != 3) fail(PyExc_TypeError, "CFlow matches must have three parts");
    PyObject *lc = PySequence_Fast_GET_ITEM(parts.p, 0), *pivot = PySequence_Fast_GET_ITEM(parts.p, 1),
             *identity = PySequence_Fast_GET_ITEM(parts.p, 2);
    if (truth(lc)) {
        Owned item(PySequence_Fast(lc, "Invalid local-complementation match"));
        if (PySequence_Fast_GET_SIZE(item.p) != 3) fail(PyExc_TypeError, "Invalid LC match");
        Bits remaining = graph.vertices(PySequence_Fast_GET_ITEM(item.p, 1));
        Bits unfused = graph.vertices(PySequence_Fast_GET_ITEM(item.p, 2));
        for (std::size_t w = 0; w < remaining.size(); ++w) remaining[w] &= ~unfused[w];
        long long unfuse = count(unfused) != 0, n = count(remaining)+unfuse, twice_edges = 0;
        each(remaining, [&](std::size_t v) {
            for (std::size_t w = 0; w < remaining.size(); ++w)
                twice_edges += __builtin_popcountll(graph.adjacency[v][w] & remaining[w]);
        });
        return weighted(weights, 0, twice_edges-n*(n-1)/2+n-1, 1-2*unfuse);
    }
    if (truth(pivot)) {
        Owned item(PySequence_Fast(pivot, "Invalid pivot match"));
        if (PySequence_Fast_GET_SIZE(item.p) != 3) fail(PyExc_TypeError, "Invalid pivot match");
        auto v0 = graph.index(PySequence_Fast_GET_ITEM(item.p, 0));
        auto v1 = graph.index(PySequence_Fast_GET_ITEM(item.p, 1));
        Owned unfusions(PySequence_Fast(PySequence_Fast_GET_ITEM(item.p, 2), "Invalid pivot unfusions"));
        if (PySequence_Fast_GET_SIZE(unfusions.p) != 2) fail(PyExc_TypeError, "Invalid pivot unfusions");
        Bits u0 = graph.vertices(PySequence_Fast_GET_ITEM(unfusions.p, 0));
        Bits u1 = graph.vertices(PySequence_Fast_GET_ITEM(unfusions.p, 1));
        Bits a = graph.adjacency[v0], b = graph.adjacency[v1], shared(graph.words);
        a[v1/64] &= ~(Word(1) << (v1%64)); b[v0/64] &= ~(Word(1) << (v0%64));
        for (std::size_t w = 0; w < graph.words; ++w) {
            a[w] &= ~u0[w]; b[w] &= ~u1[w]; shared[w] = a[w] & b[w];
            a[w] &= ~shared[w]; b[w] &= ~shared[w];
        }
        long long f0 = count(u0) != 0, f1 = count(u1) != 0;
        long long na = count(a)+f0, nb = count(b)+f1, ns = count(shared), cross = 0;
        each(a, [&](std::size_t v) {
            for (std::size_t w = 0; w < graph.words; ++w)
                cross += __builtin_popcountll(graph.adjacency[v][w] & (b[w] | shared[w]));
        });
        each(b, [&](std::size_t v) {
            for (std::size_t w = 0; w < graph.words; ++w)
                cross += __builtin_popcountll(graph.adjacency[v][w] & shared[w]);
        });
        long long removed = 2-2*(f0+f1);
        long long edges = 2*cross-(na*nb+na*ns+nb*ns)+na+2*ns+nb+1-2*(f0+f1);
        return weighted(weights, 1, edges-removed, removed);
    }
    if (truth(identity)) {
        Owned item(PySequence_Fast(identity, "Invalid identity-fusion match"));
        if (PySequence_Fast_GET_SIZE(item.p) != 3) fail(PyExc_TypeError, "Invalid identity-fusion match");
        auto center = graph.index(PySequence_Fast_GET_ITEM(item.p, 0));
        auto v0 = graph.index(PySequence_Fast_GET_ITEM(item.p, 1));
        auto v1 = graph.index(PySequence_Fast_GET_ITEM(item.p, 2));
        Bits shared(graph.words);
        for (std::size_t w = 0; w < graph.words; ++w) shared[w] = graph.adjacency[v0][w] & graph.adjacency[v1][w];
        shared[center/64] &= ~(Word(1) << (center%64));
        long long same = 0, extra = 0;
        each(shared, [&](std::size_t v) {
            auto t0 = PyDict_GetItemWithError(graph.rows[v0], graph.ids[v]);
            auto t1 = PyDict_GetItemWithError(graph.rows[v1], graph.ids[v]);
            if (!t0 || !t1) throw PythonError();
            int equal = PyObject_RichCompareBool(t0, t1, Py_EQ);
            if (equal < 0) throw PythonError();
            if (equal) { ++same; if (PyDict_Size(graph.rows[v]) == 2) ++extra; }
        });
        return weighted(weights, 2, same+count(shared)+graph.connected(v0, v1)-extra, 2, true);
    }
    Py_RETURN_NONE;
}

PyObject* score_batch(PyObject*, PyObject* args) {
    PyObject *adjacency, *matches, *weights;
    if (!PyArg_ParseTuple(args, "OOO", &adjacency, &matches, &weights)) return nullptr;
    try {
        Graph graph(adjacency);
        Owned choices(PySequence_Fast(matches, "CFlow matches must be a sequence"));
        Owned factors(PySequence_Fast(weights, "CFlow weights must have three entries"));
        if (PySequence_Fast_GET_SIZE(factors.p) != 3) fail(PyExc_TypeError, "Three CFlow weights required");
        Owned result(PyList_New(PySequence_Fast_GET_SIZE(choices.p)));
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(choices.p); ++i)
            PyList_SET_ITEM(result.p, i, score(graph, PySequence_Fast_GET_ITEM(choices.p, i), factors.p));
        Py_INCREF(result.p); return result.p;
    } catch (const PythonError&) { return nullptr; }
      catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
}
PyMethodDef methods[] = {{"score_batch", score_batch, METH_VARARGS, "Exact CFlow LC/pivot/identity scores in input order."}, {nullptr, nullptr, 0, nullptr}};
PyModuleDef definition = {PyModuleDef_HEAD_INIT, "zxopt_native_scores", nullptr, -1, methods};
} // namespace
PyMODINIT_FUNC PyInit_zxopt_native_scores() { return PyModule_Create(&definition); }
