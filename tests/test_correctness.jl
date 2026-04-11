# tests/test_correctness.jl
# Unit tests mức Level 2: kiểm tra tính đúng đắn của Apriori
# So sánh với brute-force reference và kết quả tính tay từ Chương 2
#
# Chạy: julia --project tests/test_correctness.jl

_SRC = joinpath(@__DIR__, "..", "src", "algorithm")
include(joinpath(_SRC, "structures.jl"))
include(joinpath(_SRC, "utils.jl"))
include(joinpath(_SRC, "apriori.jl"))

using Printf

# ──────────────────────────────────────────────────────────────────────────────
# Khung test đơn giản (không cần Pkg Test)
# ──────────────────────────────────────────────────────────────────────────────
mutable struct TestResults
    passed::Int
    failed::Int
    TestResults() = new(0, 0)
end

function assert_eq(tr::TestResults, name::String, got, expected)
    if got == expected
        println("  ✅  PASS  $name")
        tr.passed += 1
    else
        println("  ❌  FAIL  $name")
        println("       Got      : $got")
        println("       Expected : $expected")
        tr.failed += 1
    end
end

function assert_true(tr::TestResults, name::String, cond::Bool)
    assert_eq(tr, name, cond, true)
end

# ──────────────────────────────────────────────────────────────────────────────
# Hàm đếm tổng số frequent itemsets từ kết quả Apriori
# ──────────────────────────────────────────────────────────────────────────────
total_fi(res) = sum(length(v) for v in values(res); init=0)

# ──────────────────────────────────────────────────────────────────────────────
# Test 1 – DB1: kết quả tính tay từ Chương 2
# ──────────────────────────────────────────────────────────────────────────────
function test_db1_hand_example(tr::TestResults)
    println("\n── Test 1: DB1 (ví dụ tính tay Chương 2, minsup=0.5) ──")
    D    = load_database(joinpath(@__DIR__, "..", "data", "toy", "db1.txt"))
    res  = apriori_basic(D, 0.5)
    flat = flatten_results(res)

    # Kết quả tính tay: {2}(6), {3}(4), {4}(5), {2,4}(4)
    assert_eq(tr, "Số lượng frequent itemsets = 4",  total_fi(res), 4)
    assert_eq(tr, "sup({2}) = 6",    get(flat, [2], 0), 6)
    assert_eq(tr, "sup({3}) = 4",    get(flat, [3], 0), 4)
    assert_eq(tr, "sup({4}) = 5",    get(flat, [4], 0), 5)
    assert_eq(tr, "sup({2,4}) = 4",  get(flat, [2,4], 0), 4)
    assert_true(tr, "{1} không frequent", !haskey(flat, [1]))
    assert_true(tr, "{2,3} không frequent", !haskey(flat, [2,3]))
end

# ──────────────────────────────────────────────────────────────────────────────
# Test 2 – DB2: so sánh với brute-force
# ──────────────────────────────────────────────────────────────────────────────
function test_db2_vs_bruteforce(tr::TestResults)
    println("\n── Test 2: DB2 vs Brute-force (minsup=0.4) ──")
    D       = load_database(joinpath(@__DIR__, "..", "data", "toy", "db2.txt"))
    minsup  = 0.4
    res     = apriori_basic(D, minsup)
    bf_cnt  = count_all_frequent_bruteforce(D, minsup)
    assert_eq(tr, "Số FI khớp brute-force", total_fi(res), bf_cnt)
end

# ──────────────────────────────────────────────────────────────────────────────
# Test 3 – DB3: tất cả hay toàn bộ 2^5-1 = 31 itemsets
# ──────────────────────────────────────────────────────────────────────────────
function test_db3_identical_transactions(tr::TestResults)
    println("\n── Test 3: DB3 (3 giao dịch giống nhau, minsup=0.5) ──")
    D   = load_database(joinpath(@__DIR__, "..", "data", "toy", "db3.txt"))
    res = apriori_basic(D, 0.5)
    assert_eq(tr, "Tổng FI = 31 (2^5 - 1)", total_fi(res), 31)
    flat = flatten_results(res)
    assert_eq(tr, "sup({1,2,3,4,5}) = 3", get(flat, [1,2,3,4,5], 0), 3)
end

# ──────────────────────────────────────────────────────────────────────────────
# Test 4 – DB4: sparse, chỉ 1-itemsets
# ──────────────────────────────────────────────────────────────────────────────
function test_db4_sparse(tr::TestResults)
    println("\n── Test 4: DB4 (sparse, minsup=0.4) ──")
    D    = load_database(joinpath(@__DIR__, "..", "data", "toy", "db4.txt"))
    res  = apriori_basic(D, 0.4)
    bf   = count_all_frequent_bruteforce(D, 0.4)
    assert_eq(tr, "Số FI khớp brute-force", total_fi(res), bf)
    assert_true(tr, "Không có 2-itemsets frequent", !haskey(res, 2))
end

# ──────────────────────────────────────────────────────────────────────────────
# Test 5 – DB5: vs brute-force
# ──────────────────────────────────────────────────────────────────────────────
function test_db5_vs_bruteforce(tr::TestResults)
    println("\n── Test 5: DB5 vs Brute-force (minsup=0.5) ──")
    D    = load_database(joinpath(@__DIR__, "..", "data", "toy", "db5.txt"))
    minsup = 0.5
    res  = apriori_basic(D, minsup)
    bf   = count_all_frequent_bruteforce(D, minsup)
    assert_eq(tr, "Số FI khớp brute-force", total_fi(res), bf)
end

# ──────────────────────────────────────────────────────────────────────────────
# Test 6 – Kiểm tra: basic vs optimized cho cùng kết quả
# ──────────────────────────────────────────────────────────────────────────────
function test_basic_vs_optimized(tr::TestResults)
    println("\n── Test 6: Basic vs Optimized (DB1, minsup=0.5) ──")
    D     = load_database(joinpath(@__DIR__, "..", "data", "toy", "db1.txt"))
    res_b = apriori_basic(D, 0.5)
    res_o = apriori_optimized(D, 0.5)
    flat_b = flatten_results(res_b)
    flat_o = flatten_results(res_o)
    assert_eq(tr, "Số FI bằng nhau", total_fi(res_b), total_fi(res_o))
    assert_eq(tr, "Support maps giống nhau", flat_b, flat_o)
end

# ──────────────────────────────────────────────────────────────────────────────
# Test 7 – Kiểm tra nhiều minsup trên DB2
# ──────────────────────────────────────────────────────────────────────────────
function test_multiple_minsup(tr::TestResults)
    println("\n── Test 7: Nhiều mức minsup trên DB2 ──")
    D = load_database(joinpath(@__DIR__, "..", "data", "toy", "db2.txt"))
    for ms in [0.2, 0.4, 0.6, 0.8]
        res = apriori_basic(D, ms)
        bf  = count_all_frequent_bruteforce(D, ms)
        assert_eq(tr, "minsup=$(Int(ms*100))%: FI khớp brute-force", total_fi(res), bf)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Test 8 – Kiểm tra support count tuyệt đối đúng
# ──────────────────────────────────────────────────────────────────────────────
function test_support_counts(tr::TestResults)
    println("\n── Test 8: Giá trị support count chính xác (DB5, minsup=0.5) ──")
    D    = load_database(joinpath(@__DIR__, "..", "data", "toy", "db5.txt"))
    res  = apriori_basic(D, 0.5)
    flat = flatten_results(res)
    # DB5 có 6 giao dịch; {1,2,3,4,5} chỉ có trong giao dịch 6 → count=1 → không frequent
    assert_true(tr, "{1,2,3,4,5} không frequent",  !haskey(flat, [1,2,3,4,5]))
    # {1,2,3} xuất hiện trong: T1[1,2,3,4], T2[1,2,3,5], T6[1,2,3,4,5] → count=3 >= ceil(0.5*6)=3
    assert_eq(tr, "sup({1,2,3}) = 3",  get(flat, [1,2,3], 0), 3)
end

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
println("=" ^ 55)
println("  KIỂM TRA TÍNH ĐÚNG ĐẮN – Apriori FIM")
println("=" ^ 55)

tr = TestResults()

test_db1_hand_example(tr)
test_db2_vs_bruteforce(tr)
test_db3_identical_transactions(tr)
test_db4_sparse(tr)
test_db5_vs_bruteforce(tr)
test_basic_vs_optimized(tr)
test_multiple_minsup(tr)
test_support_counts(tr)

println("\n" * "=" ^ 55)
@printf("  KẾT QUẢ: %d PASS  /  %d FAIL  (tổng %d tests)\n",
        tr.passed, tr.failed, tr.passed + tr.failed)
println("=" ^ 55)

tr.failed > 0 && exit(1)
