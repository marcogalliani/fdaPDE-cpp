// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __FDAPDE_BENCH_UTILS_H__
#define __FDAPDE_BENCH_UTILS_H__

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fdapde {
namespace bench {

using vector_t = Eigen::Matrix<double, Dynamic, 1>;
using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;

/* A benchmark differs from a test in what it asserts. A test pins an invariant; a benchmark measures a
behaviour that is expected to have a particular SHAPE -- an ordering between methods, a convergence rate --
and reports the numbers behind it. Each benchmark here therefore does both: it prints the measurements, and
it checks a small number of named claims about them, so a change in behaviour is reported rather than left
for a reader to notice in a table. A failed claim makes the whole run exit non-zero. */

struct options {
    int reps = 4;         // noise replicates where a benchmark averages over them
    bool full = false;    // wider sweeps (more grids, more lambdas, more replicates)
    bool quiet = false;   // suppress tables, keep the claim summary
};

// one measured claim: a named expectation about the shape of the results
struct claim {
    std::string name, detail;
    bool ok;
};

class report {
   public:
    explicit report(const std::string& bench) : bench_(bench) { }
    // record a claim. `detail` should carry the numbers the claim was judged on, so a failure is
    // diagnosable from the log alone.
    void check(const std::string& name, bool ok, const std::string& detail = "") {
        claims_.push_back({name, detail, ok});
    }
    template <typename... Args> void checkf(const std::string& name, bool ok, const char* fmt, Args... args) {
        char buf[512];
        std::snprintf(buf, sizeof(buf), fmt, args...);
        claims_.push_back({name, std::string(buf), ok});
    }
    const std::vector<claim>& claims() const { return claims_; }
    const std::string& name() const { return bench_; }
    int failures() const {
        int n = 0;
        for (const auto& c : claims_) {
            if (!c.ok) { ++n; }
        }
        return n;
    }
   private:
    std::string bench_;
    std::vector<claim> claims_;
};

// fixed-width table printer: benchmarks are read by eye and diffed between runs, so columns must line up
// and numbers must be formatted identically every time.
class table {
   public:
    explicit table(std::vector<std::string> headers) : h_(std::move(headers)) { w_.assign(h_.size(), 0); }
    void row(std::vector<std::string> cells) { rows_.push_back(std::move(cells)); }
    void rule() { rows_.push_back({}); }   // an empty row prints as a horizontal rule
    void print(const std::string& indent = "  ") const {
        std::vector<std::size_t> w(h_.size());
        for (std::size_t i = 0; i < h_.size(); ++i) { w[i] = h_[i].size(); }
        for (const auto& r : rows_) {
            for (std::size_t i = 0; i < r.size() && i < w.size(); ++i) { w[i] = std::max(w[i], r[i].size()); }
        }
        std::size_t total = 0;
        for (std::size_t i = 0; i < w.size(); ++i) { total += w[i] + 2; }
        std::cout << indent;
        for (std::size_t i = 0; i < h_.size(); ++i) { std::cout << pad(h_[i], w[i]) << "  "; }
        std::cout << "\n" << indent << std::string(total, '-') << "\n";
        for (const auto& r : rows_) {
            if (r.empty()) {
                std::cout << indent << std::string(total, '-') << "\n";
                continue;
            }
            std::cout << indent;
            for (std::size_t i = 0; i < r.size(); ++i) { std::cout << pad(r[i], w[i]) << "  "; }
            std::cout << "\n";
        }
    }
   private:
    static std::string pad(const std::string& s, std::size_t w) {
        return s.size() >= w ? s : std::string(w - s.size(), ' ') + s;   // right-aligned: these are numbers
    }
    std::vector<std::string> h_;
    std::vector<std::size_t> w_;
    std::vector<std::vector<std::string>> rows_;
};

// consistent numeric formatting across every benchmark
inline std::string sci(double v, int prec = 4) {
    char b[64];
    std::snprintf(b, sizeof(b), "%.*e", prec, v);
    return b;
}
inline std::string fix(double v, int prec = 3) {
    char b[64];
    std::snprintf(b, sizeof(b), "%.*f", prec, v);
    return b;
}
inline std::string num(long v) { return std::to_string(v); }
// observed convergence order between two errors on grids differing by a factor of two in dt
inline std::string order(double coarse, double fine) {
    if (!(coarse > 0) || !(fine > 0)) { return "--"; }
    return fix(std::log2(coarse / fine), 2);
}

/* Shared fixtures.
Every benchmark in this folder measures the SAME two-component system, so results are comparable across
files and a change in one benchmark's numbers cannot be blamed on a different problem.

    prior     y0' = y0*y1 + sin t
              y1' = y0 - y1^2
    forcing   g(t) = amp * (sin 3t, cos 2t)          the term the prior does NOT contain

The truth is generated from prior + g and fitted with the prior alone, so the smoother must reconstruct g
through its control. amp = 0 recovers a correctly specified problem, where the optimal control is zero. */
struct nonlinear_field {
    vector_t operator()(double t, const vector_t& y) const {
        vector_t out(2);
        out << y[0] * y[1] + std::sin(t), y[0] - y[1] * y[1];
        return out;
    }
    matrix_t state_jacobian(double, const vector_t& y) const {
        matrix_t J(2, 2);
        J << y[1], y[0], 1.0, -2.0 * y[1];
        return J;
    }
};
inline double forcing_0(double t, double amp) { return amp * std::sin(3.0 * t); }
inline double forcing_1(double t, double amp) { return amp * std::cos(2.0 * t); }

struct forced_field {
    double amp = 0.3;
    vector_t operator()(double t, const vector_t& y) const {
        vector_t out = nonlinear_field {}(t, y);
        out[0] += forcing_0(t, amp);
        out[1] += forcing_1(t, amp);
        return out;
    }
    // g carries no state dependence, so the Jacobian is the prior's
    matrix_t state_jacobian(double t, const vector_t& y) const { return nonlinear_field {}.state_jacobian(t, y); }
};

/* Stiff fixture.
    y0' = -k (y0 - sin t) + cos t     fast, relaxation rate k
    y1' = y0 - y1^2                   slow, nonlinear
Eigenvalues -k and -2*y1, so the stiffness ratio is ~k/2. Deliberately chosen so the EQUATION is stiff
while the SOLUTION stays smooth: started at y0(0) = 0 the fast component sits exactly on the slow manifold
(y0 = sin t solves it), so nothing about the trajectory is pathological. Starting off that manifold instead
creates a fast layer of width ~1/k that a coarse grid cannot resolve, which is the genuinely hard case --
Gauss schemes are A-stable but not L-stable, so an unresolved layer is propagated rather than damped. */
struct stiff_field {
    double k = 200.0;
    vector_t operator()(double t, const vector_t& y) const {
        vector_t out(2);
        out << -k * (y[0] - std::sin(t)) + std::cos(t), y[0] - y[1] * y[1];
        return out;
    }
    matrix_t state_jacobian(double, const vector_t& y) const {
        matrix_t J(2, 2);
        J << -k, 0.0, 1.0, -2.0 * y[1];
        return J;
    }
};

// a high-accuracy reference integrator: GL3 with a tight Newton tolerance. Every "truth" in these
// benchmarks comes from this, never from the scheme under measurement.
inline RKIntegrator<3> reference_integrator() { return RKIntegrator<3>(ode_schemes::gauss_legendre_3(), 300, 1e-15); }

// deterministic Gaussian noise: the seed depends only on the replicate index, so two methods compared at
// the same replicate see the SAME data. That pairing is what makes small differences between methods
// meaningful rather than Monte-Carlo scatter.
inline matrix_t add_noise(const matrix_t& Y, double sigma, int replicate) {
    std::mt19937 rng(1234 + 991 * replicate);
    std::normal_distribution<double> nd(0.0, sigma);
    matrix_t out = Y;
    for (int i = 0; i < out.rows(); ++i) {
        for (int j = 0; j < out.cols(); ++j) { out(i, j) += nd(rng); }
    }
    return out;
}

inline double rms(const matrix_t& A, const matrix_t& B) { return std::sqrt((A - B).squaredNorm() / A.size()); }

/* Evaluation grid.
Accuracy is measured on a FIXED dense set of query times through the solver's continuous evaluation, never
at the fit's own grid nodes: the nodes move with m, so a nodal error is not comparable across resolutions.
The reference trajectory is sampled from a refinement that contains every query point exactly, so no
interpolation of the truth enters the comparison. */
struct eval_grid {
    vector_t q;         // query times
    matrix_t truth;     // reference trajectory at those times
    vector_t fine;      // the refinement the truth was integrated on
    matrix_t fine_y;
    // `sub` is how many refinement steps sit between consecutive query points
    template <typename Field>
    eval_grid(const Field& f, const vector_t& y0, double T, int n_query, int sub) {
        fine = vector_t::LinSpaced(sub * (n_query - 1) + 1, 0.0, T);
        fine_y = reference_integrator().integrate(ode_rhs_field {f}, fine, y0);
        q.resize(n_query);
        truth.resize(n_query, y0.size());
        for (int i = 0; i < n_query; ++i) {
            q[i] = fine[sub * i];
            truth.row(i) = fine_y.row(sub * i);
        }
    }
    // the truth at an arbitrary fit grid, read off the same refinement (never re-integrated coarsely)
    matrix_t at(const vector_t& time, double T) const {
        matrix_t Y(time.size(), truth.cols());
        for (int i = 0; i < time.size(); ++i) {
            const int idx = static_cast<int>(std::round(time[i] / T * (fine.size() - 1)));
            Y.row(i) = fine_y.row(idx);
        }
        return Y;
    }
};

// error of a fitted control against the forcing it should have recovered, on the query grid
template <typename Solver> double control_error(Solver& solver, const eval_grid& g, double amp) {
    double s = 0;
    for (int i = 0; i < g.q.size(); ++i) {
        const vector_t u = solver.eval_control(g.q[i]);
        s += std::pow(u[0] - forcing_0(g.q[i], amp), 2) + std::pow(u[1] - forcing_1(g.q[i], amp), 2);
    }
    return std::sqrt(s / (2 * g.q.size()));
}
// largest discontinuity of the fitted control across an interior grid node. Zero only for a C0 space;
// the probe offset is small enough that a continuous control's residual is its slope times that offset.
template <typename Solver> double control_jump(Solver& solver, const vector_t& time) {
    double j = 0;
    for (int k = 1; k + 1 < time.size(); ++k) {
        j = std::max(j, (solver.eval_control(time[k] - 1e-7) - solver.eval_control(time[k] + 1e-7))
                          .cwiseAbs().maxCoeff());
    }
    return j;
}

// Benchmark registry. Each src/*.cpp registers its entries at static-initialisation time; main.cpp
// includes them and dispatches by name.
struct entry {
    std::string name, description;
    std::function<void(const options&, report&)> run;
};
inline std::vector<entry>& registry() {
    static std::vector<entry> r;
    return r;
}
struct registrar {
    registrar(const char* name, const char* desc, std::function<void(const options&, report&)> fn) {
        registry().push_back({name, desc, std::move(fn)});
    }
};
#define FDAPDE_BENCHMARK(id, desc)                                                                        \
    static void id##_run(const fdapde::bench::options&, fdapde::bench::report&);                          \
    static fdapde::bench::registrar id##_registrar(#id, desc, id##_run);                                  \
    static void id##_run(const fdapde::bench::options& opt, fdapde::bench::report& rep)

}   // namespace bench
}   // namespace fdapde

#endif   // __FDAPDE_BENCH_UTILS_H__
