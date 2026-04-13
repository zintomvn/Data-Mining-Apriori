# tests/test_correctness.jl
# Unit tests: kiểm tra tính đúng đắn của Apriori (Basic + Hash Tree)
# Chạy: julia --project tests/test_correctness.jl

_SRC = joinpath(@__DIR__, "..", "src", "algorithm")
include(joinpath(_SRC, "structures.jl"))
include(joinpath(_SRC, "utils.jl"))
include(joinpath(_SRC, "apriori.jl"))

using Printf

mutable struct TR; passed::Int; failed::Int; TR()=new(0,0); end
function assert_eq(tr::TR, name, got, exp)
    if got == exp; println("  ✅ $name"); tr.passed += 1
    else; println("  ❌ $name (got=$got exp=$exp)"); tr.failed += 1; end
end
assert_true(tr::TR, name, c) = assert_eq(tr, name, c, true)

# ─── Tests ────────────────────────────────────────────────────────────────────
function test_db1(tr::TR)
    println("\n── DB1: hand example minsup=0.5 ──")
    D = load_database(joinpath(@__DIR__,"..","data","toy","db1.txt"))
    res = apriori_basic(D, 0.5; verbose=false)
    flat = flatten_results(res)
    assert_eq(tr, "#FI = 4", total_fi(res), 4)
    assert_eq(tr, "sup({2}) = 6", get(flat,[2],0), 6)
    assert_eq(tr, "sup({4}) = 5", get(flat,[4],0), 5)
    assert_eq(tr, "sup({2,4}) = 4", get(flat,[2,4],0), 4)
    assert_true(tr, "{1} not frequent", !haskey(flat,[1]))
end

function test_db2_bf(tr::TR)
    println("\n── DB2: vs brute-force minsup=0.4 ──")
    D = load_database(joinpath(@__DIR__,"..","data","toy","db2.txt"))
    assert_eq(tr, "#FI match BF", total_fi(apriori_basic(D,0.4;verbose=false)),
              count_all_frequent_bruteforce(D, 0.4))
end

function test_db3_all(tr::TR)
    println("\n── DB3: identical txns minsup=0.5 ──")
    D = load_database(joinpath(@__DIR__,"..","data","toy","db3.txt"))
    assert_eq(tr, "#FI = 31", total_fi(apriori_basic(D,0.5;verbose=false)), 31)
end

function test_db4_sparse(tr::TR)
    println("\n── DB4: sparse minsup=0.4 ──")
    D = load_database(joinpath(@__DIR__,"..","data","toy","db4.txt"))
    res = apriori_basic(D, 0.4; verbose=false)
    assert_eq(tr, "#FI match BF", total_fi(res), count_all_frequent_bruteforce(D, 0.4))
    assert_true(tr, "No 2-itemsets", !haskey(res, 2))
end

function test_db5_bf(tr::TR)
    println("\n── DB5: vs brute-force minsup=0.5 ──")
    D = load_database(joinpath(@__DIR__,"..","data","toy","db5.txt"))
    assert_eq(tr, "#FI match BF", total_fi(apriori_basic(D,0.5;verbose=false)),
              count_all_frequent_bruteforce(D, 0.5))
end

function test_basic_vs_hashtree(tr::TR)
    println("\n── Basic vs Hash Tree consistency ──")
    for (name, ms) in [("db1",0.5),("db2",0.4),("db3",0.5),("db5",0.5)]
        D = load_database(joinpath(@__DIR__,"..","data","toy","$name.txt"))
        fb = flatten_results(apriori_basic(D, ms; verbose=false))
        fo = flatten_results(apriori_optimized(D, ms; verbose=false))
        assert_eq(tr, "$name: same count", length(fb), length(fo))
        assert_eq(tr, "$name: same support", fb, fo)
    end
end

function test_multiple_minsup(tr::TR)
    println("\n── Multiple minsup on DB2 ──")
    D = load_database(joinpath(@__DIR__,"..","data","toy","db2.txt"))
    for ms in [0.2, 0.4, 0.6, 0.8]
        nb = total_fi(apriori_basic(D, ms; verbose=false))
        bf = count_all_frequent_bruteforce(D, ms)
        assert_eq(tr, "minsup=$(Int(ms*100))%: BF match", nb, bf)
    end
end

# ─── MAIN ─────────────────────────────────────────────────────────────────────
println("=" ^ 50 * "\n  UNIT TESTS – Apriori FIM\n" * "=" ^ 50)
tr = TR()
test_db1(tr); test_db2_bf(tr); test_db3_all(tr)
test_db4_sparse(tr); test_db5_bf(tr)
test_basic_vs_hashtree(tr); test_multiple_minsup(tr)
println("\n" * "=" ^ 50)
@printf("  %d PASS / %d FAIL (total %d)\n", tr.passed, tr.failed, tr.passed+tr.failed)
println("=" ^ 50)
tr.failed > 0 && exit(1)
