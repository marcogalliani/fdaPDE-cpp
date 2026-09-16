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

// include eigen first to avoid possible linking errors
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fdaPDE/models.h>   // fdaPDE

#include "src/bench_utils.h"

// the benchmark sources are written against the library's unqualified names, as the tests are
using namespace fdapde;

// benchmarks register themselves at static-initialisation time; including them here is all that is needed
#include "src/grid_refinement.cpp"
#include "src/stiff.cpp"

namespace {

void usage() {
    std::cout << "Usage: fdapde_benchmark [options] [name ...]\n\n"
                 "  Runs the named benchmarks, or all of them when none is named. Every benchmark prints\n"
                 "  its measurements and then checks a few named claims about their shape; the process\n"
                 "  exits non-zero if any claim fails, so a change in behaviour is reported rather than\n"
                 "  left to be spotted in a table.\n\n"
                 "  -l, --list         list the available benchmarks and exit\n"
                 "  -r, --reps N       noise replicates where a benchmark averages (default 4)\n"
                 "  -f, --full         wider sweeps: more grids, more lambdas (slower)\n"
                 "  -q, --quiet        suppress the tables, keep the claim summary\n"
                 "  -h, --help         show this message\n";
}

}   // namespace

int main(int argc, char** argv) {
    using namespace fdapde::bench;
    options opt;
    std::vector<std::string> selected;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else if (a == "-l" || a == "--list") {
            for (const auto& e : registry()) { std::cout << "  " << e.name << "\n      " << e.description << "\n"; }
            return 0;
        } else if ((a == "-r" || a == "--reps") && i + 1 < argc) {
            opt.reps = std::max(1, std::atoi(argv[++i]));
        } else if (a == "-f" || a == "--full") {
            opt.full = true;
        } else if (a == "-q" || a == "--quiet") {
            opt.quiet = true;
        } else if (!a.empty() && a[0] == '-') {
            std::cout << "unknown option: " << a << "\n";
            usage();
            return 2;
        } else {
            selected.push_back(a);
        }
    }

    std::vector<report> reports;
    int total_failures = 0;
    for (const auto& e : registry()) {
        if (!selected.empty() && std::find(selected.begin(), selected.end(), e.name) == selected.end()) { continue; }
        std::cout << "\n== " << e.name << " ==\n   " << e.description << "\n\n";
        report rep(e.name);
        const auto t0 = std::chrono::steady_clock::now();
        e.run(opt, rep);
        const auto t1 = std::chrono::steady_clock::now();
        std::cout << "\n  claims:\n";
        for (const auto& c : rep.claims()) {
            std::cout << "    [" << (c.ok ? "  ok  " : " FAIL ") << "] " << c.name << "\n";
            if (!c.detail.empty()) { std::cout << "             " << c.detail << "\n"; }
        }
        std::cout << "  (" << fix(std::chrono::duration<double>(t1 - t0).count(), 1) << " s)\n";
        total_failures += rep.failures();
        reports.push_back(std::move(rep));
    }

    if (reports.empty()) {
        std::cout << "no benchmark matched. Use --list to see the available names.\n";
        return 2;
    }
    std::cout << "\n== summary ==\n";
    int total_claims = 0;
    for (const auto& r : reports) {
        total_claims += static_cast<int>(r.claims().size());
        std::cout << "  " << (r.failures() ? "FAIL  " : "ok    ") << r.name() << "  ("
                  << r.claims().size() - r.failures() << "/" << r.claims().size() << " claims held)\n";
    }
    std::cout << "\n  " << total_claims - total_failures << "/" << total_claims << " claims held across "
              << reports.size() << " benchmark" << (reports.size() == 1 ? "" : "s") << ".\n";
    return total_failures == 0 ? 0 : 1;
}
