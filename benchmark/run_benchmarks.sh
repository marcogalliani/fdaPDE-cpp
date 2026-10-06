#!/bin/sh

# Build and run the benchmark suite. Mirrors test/run_tests.sh, minus the memcheck path: these are timed
# measurements, and running them under valgrind would make the timings meaningless.

SCRIPT_NAME=$(basename "$0")
BUILD_DIR=build/

help()
{
    echo "Usage: ./run_benchmarks.sh [options] [benchmark ...]

       -l --list             list the available benchmarks
       -r --reps N           noise replicates where a benchmark averages (default 4)
       -f --full             wider sweeps: more grids, more lambdas (slower)
       -q --quiet            suppress the tables, keep the claim summary
       -h --help             shows this message

Any remaining arguments name the benchmarks to run; with none, all of them run.
Exits non-zero if any benchmark's claims fail."
    exit 2
}

case " $* " in
    *" -h "*|*" --help "*) help ;;
esac

if [ -d "$BUILD_DIR" ]; then
    rm -f build/CMakeCache.txt
    rm -rf build/CMakeFiles/
else
    mkdir build/
fi
cd build/ || exit 1

cmake -Wno-dev -H../ -B. || exit 1
make || exit 1

./fdapde_benchmark "$@"
BENCH_OUTPUT=$?

rm -f fdapde_benchmark
exit $BENCH_OUTPUT
