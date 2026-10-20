// Leonardo cryptic-pocket grid: buried, probe-accessible solvent points and the pockets they form.
//
// For a set of heavy-atom coordinates:
//   1. occupancy grid: cells closer than `radius` to an atom centre are protein;
//   2. candidate points: cells at least `probe` from every atom centre (a water-sized probe fits) and within
//      `probe + shell` of one (the solvent shell around the protein);
//   3. burial: fraction of `ndir` evenly spread rays (Fibonacci sphere) that meet protein within `ray` A;
//   4. pocket points: burial >= min_frac; 26-connected clusters of pocket points are pockets;
//   5. every residue with an atom within `contact` A of a pocket's points gets that pocket's score
//      (volume x mean burial), keeping the best pocket, and accumulates the pocket volumes it lines.
//
// Same algorithm as the numpy reference in leonardo/cryptic_pocket_runtime.py (tests compare them).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

struct Grid {
    double lo[3];
    int n[3];
    double h;
    std::size_t size() const { return static_cast<std::size_t>(n[0]) * n[1] * n[2]; }
    std::size_t idx(int i, int j, int k) const {
        return (static_cast<std::size_t>(i) * n[1] + j) * n[2] + k;
    }
    bool inside(int i, int j, int k) const {
        return i >= 0 && j >= 0 && k >= 0 && i < n[0] && j < n[1] && k < n[2];
    }
};

Grid make_grid(const double* xyz, std::size_t na, double h, double pad) {
    Grid g{};
    g.h = h;
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (std::size_t a = 0; a < na; ++a)
        for (int d = 0; d < 3; ++d) {
            lo[d] = std::min(lo[d], xyz[3 * a + d]);
            hi[d] = std::max(hi[d], xyz[3 * a + d]);
        }
    for (int d = 0; d < 3; ++d) {
        g.lo[d] = lo[d] - pad;
        g.n[d] = static_cast<int>(std::ceil((hi[d] + pad - g.lo[d]) / h)) + 1;
    }
    return g;
}

// mark cells whose centre is within r of an atom centre (atoms snapped to the nearest cell, as numpy does)
void mark(const Grid& g, const double* xyz, std::size_t na, double r, std::vector<uint8_t>& out) {
    out.assign(g.size(), 0);
    const int m = static_cast<int>(std::ceil(r / g.h));
    std::vector<int> off;
    for (int i = -m; i <= m; ++i)
        for (int j = -m; j <= m; ++j)
            for (int k = -m; k <= m; ++k)
                if ((i * i + j * j + k * k) * g.h * g.h <= r * r) {
                    off.push_back(i);
                    off.push_back(j);
                    off.push_back(k);
                }
    for (std::size_t a = 0; a < na; ++a) {
        int c[3];
        for (int d = 0; d < 3; ++d) c[d] = static_cast<int>(std::lround((xyz[3 * a + d] - g.lo[d]) / g.h));
        for (std::size_t o = 0; o < off.size(); o += 3) {
            const int i = c[0] + off[o], j = c[1] + off[o + 1], k = c[2] + off[o + 2];
            if (g.inside(i, j, k)) out[g.idx(i, j, k)] = 1;
        }
    }
}

std::vector<double> fibonacci_dirs(int n) {
    std::vector<double> d(3 * n);
    const double golden = M_PI * (1.0 + std::sqrt(5.0));
    for (int i = 0; i < n; ++i) {
        const double t = i + 0.5;
        const double phi = std::acos(1.0 - 2.0 * t / n), th = golden * t;
        d[3 * i] = std::cos(th) * std::sin(phi);
        d[3 * i + 1] = std::sin(th) * std::sin(phi);
        d[3 * i + 2] = std::cos(phi);
    }
    return d;
}

// half-to-even rounding, as numpy.rint
inline int rint_even(double x) { return static_cast<int>(std::nearbyint(x)); }

}  // namespace

py::dict pocket_grid(py::array_t<double, py::array::c_style | py::array::forcecast> xyz_in,
                     py::array_t<int32_t, py::array::c_style | py::array::forcecast> atom_res_in, int n_res,
                     double spacing, double radius, double probe, double shell, double ray, int ndir,
                     double min_frac, double min_volume, double contact) {
    if (xyz_in.ndim() != 2 || xyz_in.shape(1) != 3) throw std::invalid_argument("xyz must be (N, 3)");
    const std::size_t na = static_cast<std::size_t>(xyz_in.shape(0));
    if (static_cast<std::size_t>(atom_res_in.shape(0)) != na) throw std::invalid_argument("atom_res length");
    const double* xyz = xyz_in.data();
    const int32_t* atom_res = atom_res_in.data();

    Grid g = make_grid(xyz, na, spacing, ray + 2.0);
    std::vector<uint8_t> occ, inner, outer;
    {
        py::gil_scoped_release nogil;
        mark(g, xyz, na, radius, occ);
        mark(g, xyz, na, probe, inner);
        mark(g, xyz, na, probe + shell, outer);
    }

    // candidate points and their burial
    const std::vector<double> dirs = fibonacci_dirs(ndir);
    const int steps = static_cast<int>(ray / spacing);
    const int min_hits = static_cast<int>(std::ceil(min_frac * ndir - 1e-9));
    std::vector<int> cand;
    std::vector<float> burial;
    {
        py::gil_scoped_release nogil;
        for (int i = 0; i < g.n[0]; ++i)
            for (int j = 0; j < g.n[1]; ++j)
                for (int k = 0; k < g.n[2]; ++k) {
                    const std::size_t c = g.idx(i, j, k);
                    if (!outer[c] || inner[c]) continue;
                    int hits = 0;
                    for (int d = 0; d < ndir; ++d) {
                        for (int s = 1; s <= steps; ++s) {
                            const int qi = rint_even(i + dirs[3 * d] * s), qj = rint_even(j + dirs[3 * d + 1] * s),
                                      qk = rint_even(k + dirs[3 * d + 2] * s);
                            if (g.inside(qi, qj, qk) && occ[g.idx(qi, qj, qk)]) {
                                ++hits;
                                break;
                            }
                        }
                    }
                    // compared in hits (21/30 as a float is below 0.7)
                    if (hits >= min_hits) {
                        const float f = static_cast<float>(hits) / ndir;
                        cand.push_back(i);
                        cand.push_back(j);
                        cand.push_back(k);
                        burial.push_back(f);
                    }
                }
    }

    // 26-connected clusters of pocket points
    const std::size_t np = burial.size();
    std::vector<int32_t> where(g.size(), -1);
    for (std::size_t p = 0; p < np; ++p) where[g.idx(cand[3 * p], cand[3 * p + 1], cand[3 * p + 2])] = static_cast<int32_t>(p);
    std::vector<int32_t> label(np, -1);
    std::vector<double> cl_vol, cl_bur;
    std::vector<std::size_t> stack;
    for (std::size_t p = 0; p < np; ++p) {
        if (label[p] >= 0) continue;
        const int32_t id = static_cast<int32_t>(cl_vol.size());
        double vol = 0.0, bur = 0.0;
        stack.assign(1, p);
        label[p] = id;
        while (!stack.empty()) {
            const std::size_t q = stack.back();
            stack.pop_back();
            vol += spacing * spacing * spacing;
            bur += burial[q];
            const int i = cand[3 * q], j = cand[3 * q + 1], k = cand[3 * q + 2];
            for (int di = -1; di <= 1; ++di)
                for (int dj = -1; dj <= 1; ++dj)
                    for (int dk = -1; dk <= 1; ++dk) {
                        if (!di && !dj && !dk) continue;
                        if (!g.inside(i + di, j + dj, k + dk)) continue;
                        const int32_t r = where[g.idx(i + di, j + dj, k + dk)];
                        if (r >= 0 && label[r] < 0) {
                            label[r] = id;
                            stack.push_back(static_cast<std::size_t>(r));
                        }
                    }
        }
        cl_vol.push_back(vol);
        cl_bur.push_back(bur / (vol / (spacing * spacing * spacing)));
    }

    // residue scores: best pocket (volume x burial) lined, and total pocket volume lined
    std::vector<double> score(n_res, 0.0), volume(n_res, 0.0);
    {
        // atoms hashed on a cell grid of size `contact`
        Grid ag = make_grid(xyz, na, contact, contact);
        std::vector<std::vector<int32_t>> bins(ag.size());
        for (std::size_t a = 0; a < na; ++a) {
            const int i = static_cast<int>((xyz[3 * a] - ag.lo[0]) / contact);
            const int j = static_cast<int>((xyz[3 * a + 1] - ag.lo[1]) / contact);
            const int k = static_cast<int>((xyz[3 * a + 2] - ag.lo[2]) / contact);
            bins[ag.idx(i, j, k)].push_back(static_cast<int32_t>(a));
        }
        std::vector<int32_t> seen(n_res, -1);
        const double c2 = contact * contact;
        // points grouped by cluster
        std::vector<std::vector<std::size_t>> members(cl_vol.size());
        for (std::size_t p = 0; p < np; ++p) members[label[p]].push_back(p);
        for (std::size_t c = 0; c < cl_vol.size(); ++c) {
            if (cl_vol[c] < min_volume) continue;
            const double s = cl_vol[c] * cl_bur[c];
            for (std::size_t p : members[c]) {
                const double x = g.lo[0] + cand[3 * p] * spacing, y = g.lo[1] + cand[3 * p + 1] * spacing,
                             z = g.lo[2] + cand[3 * p + 2] * spacing;
                const int bi = static_cast<int>((x - ag.lo[0]) / contact), bj = static_cast<int>((y - ag.lo[1]) / contact),
                          bk = static_cast<int>((z - ag.lo[2]) / contact);
                for (int di = -1; di <= 1; ++di)
                    for (int dj = -1; dj <= 1; ++dj)
                        for (int dk = -1; dk <= 1; ++dk) {
                            if (!ag.inside(bi + di, bj + dj, bk + dk)) continue;
                            for (int32_t a : bins[ag.idx(bi + di, bj + dj, bk + dk)]) {
                                const double dx = xyz[3 * a] - x, dy = xyz[3 * a + 1] - y, dz = xyz[3 * a + 2] - z;
                                if (dx * dx + dy * dy + dz * dz > c2) continue;
                                const int32_t r = atom_res[a];
                                if (r < 0 || r >= n_res || seen[r] == static_cast<int32_t>(c)) continue;
                                seen[r] = static_cast<int32_t>(c);
                                score[r] = std::max(score[r], s);
                                volume[r] += cl_vol[c];
                            }
                        }
            }
        }
    }

    // outputs
    py::array_t<double> pts({static_cast<py::ssize_t>(np), static_cast<py::ssize_t>(3)});
    py::array_t<float> bur_out(static_cast<py::ssize_t>(np));
    py::array_t<int32_t> lab_out(static_cast<py::ssize_t>(np));
    auto P = pts.mutable_unchecked<2>();
    auto B = bur_out.mutable_unchecked<1>();
    auto L = lab_out.mutable_unchecked<1>();
    for (std::size_t p = 0; p < np; ++p) {
        for (int d = 0; d < 3; ++d) P(p, d) = g.lo[d] + cand[3 * p + d] * spacing;
        B(p) = burial[p];
        L(p) = label[p];
    }
    py::dict out;
    out["points"] = pts;
    out["burial"] = bur_out;
    out["label"] = lab_out;
    out["cluster_volume"] = py::array_t<double>(static_cast<py::ssize_t>(cl_vol.size()), cl_vol.data());
    out["cluster_burial"] = py::array_t<double>(static_cast<py::ssize_t>(cl_bur.size()), cl_bur.data());
    out["residue_score"] = py::array_t<double>(static_cast<py::ssize_t>(n_res), score.data());
    out["residue_volume"] = py::array_t<double>(static_cast<py::ssize_t>(n_res), volume.data());
    return out;
}

PYBIND11_MODULE(_cryptic, m) {
    m.doc() = "Leonardo cryptic-pocket grid (buried probe-accessible points, pockets, residue scores)";
    m.def("pocket_grid", &pocket_grid, py::arg("xyz"), py::arg("atom_res"), py::arg("n_res"),
          py::arg("spacing") = 1.0, py::arg("radius") = 1.8, py::arg("probe") = 3.0, py::arg("shell") = 3.5,
          py::arg("ray") = 10.0, py::arg("ndir") = 30, py::arg("min_frac") = 0.7, py::arg("min_volume") = 30.0,
          py::arg("contact") = 4.0);
}
