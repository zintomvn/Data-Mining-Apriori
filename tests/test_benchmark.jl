# tests/test_benchmark.jl
# Level 3: Đo lường và so sánh hiệu năng Basic vs Optimized Apriori
# trên các toy và benchmark datasets
#
# Chạy: julia --project tests/test_benchmark.jl

_SRC = joinpath(@__DIR__, "..", "src", "algorithm")
include(joinpath(_SRC, "structures.jl"))
include(joinpath(_SRC, "utils.jl"))
include(joinpath(_SRC, "apriori.jl"))

using Printf

# ──────────────────────────────────────────────────────────────────────────────
# So sánh tốc độ và bộ nhớ: Basic vs Optimized
# ──────────────────────────────────────────────────────────────────────────────
function benchmark_pair(label::String, filepath::String, minsup::Float64;
                        warmup_runs::Int=1)
    println("\n" * "─"^60)
    println("  Dataset  : $label")
    println("  File     : $filepath")
    println("  Minsup   : $(round(minsup*100, digits=2))%")
    println("─"^60)

    D = load_database(filepath)

    # Warm-up (JIT compilation)
    for _ in 1:warmup_runs
        apriori_basic(D, minsup)
        apriori_optimized(D, minsup)
    end

    # ── Basic ──────────────────────────────────────────────────────
    println("\n  [Basic]")
    res_b, t_b = measure_time(apriori_basic, D, minsup)
    _,    mb_b = measure_memory_mb(apriori_basic, D, minsup)

    # ── Optimized ──────────────────────────────────────────────────
    println("\n  [Optimized]")
    res_o, t_o = measure_time(apriori_optimized, D, minsup)
    _,    mb_o = measure_memory_mb(apriori_optimized, D, minsup)

    # ── Kết quả nhất quán? ─────────────────────────────────────────
    n_b = sum(length(v) for v in values(res_b); init=0)
    n_o = sum(length(v) for v in values(res_o); init=0)
    consistent = (n_b == n_o)

    println("\n  ┌───────────────┬──────────────┬──────────────┐")
    println("  │ Chỉ số        │    Basic     │  Optimized   │")
    println("  ├───────────────┼──────────────┼──────────────┤")
    @printf("  │ Thời gian(ms) │ %12.2f │ %12.2f │\n", t_b, t_o)
    @printf("  │ Bộ nhớ (MB)   │ %12.4f │ %12.4f │\n", mb_b, mb_o)
    @printf("  │ # FI          │ %12d │ %12d │\n", n_b, n_o)
    println("  └───────────────┴──────────────┴──────────────┘")

    speedup = t_b > 0 ? t_b / max(t_o, 0.001) : 1.0
    mem_ratio = mb_b > 0 ? mb_b / max(mb_o, 0.0001) : 1.0
    @printf("  Tốc độ: Optimized nhanh hơn %.2f× | Bộ nhớ: %.2f×\n", speedup, mem_ratio)
    println("  Kết quả nhất quán: $(consistent ? "✅ CÓ" : "❌ KHÔNG")")

    !consistent && println("  ⚠  CẢNH BÁO: Hai phiên bản trả về số FI khác nhau!")

    return consistent, t_b, t_o, mb_b, mb_o
end

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
println("=" ^ 60)
println("  BENCHMARK: Basic vs Optimized Apriori")
println("  (Level 3 – Đo lường cải thiện)")
println("=" ^ 60)

DATA_DIR = joinpath(@__DIR__, "..", "data")

benchmarks = [
    # (label,             filepath,                             minsup)
    ("DB1 (toy)",         joinpath(DATA_DIR, "toy",       "db1.txt"),  0.50),
    ("DB5 (toy)",         joinpath(DATA_DIR, "toy",       "db5.txt"),  0.50),
    ("Chess",             joinpath(DATA_DIR, "benchmark", "chess.txt"),     0.80),
    ("Mushroom",          joinpath(DATA_DIR, "benchmark", "mushroom.txt"),   0.30),
    ("T10I4D100K",        joinpath(DATA_DIR, "benchmark", "T10I4D100K.txt"), 0.01),
    ("Retail",            joinpath(DATA_DIR, "benchmark", "retail.txt"),     0.01),
]

all_consistent = true
summary_rows   = Tuple{String,Float64,Float64,Float64,Float64}[]

for (label, path, ms) in benchmarks
    if !isfile(path)
        println("\n⚠ Bỏ qua '$label': file không tồn tại → $path")
        continue
    end
    ok, tb, to, mb_b, mb_o = benchmark_pair(label, path, ms)
    global all_consistent &= ok
    push!(summary_rows, (label, tb, to, mb_b, mb_o))
end

# ── Bảng tổng hợp ────────────────────────────────────────────────────────────
println("\n" * "=" ^ 60)
println("  BẢNG TỔNG HỢP")
println("=" ^ 60)
@printf("  %-18s %10s %10s  %8s\n", "Dataset", "Basic(ms)", "Opt(ms)", "Speedup")
println("  " * "─"^55)
for (lbl, tb, to, _, _) in summary_rows
    @printf("  %-18s %10.2f %10.2f %8.2f×\n", lbl, tb, to, tb / max(to, 0.001))
end
println("=" ^ 60)
println("  Tính nhất quán: $(all_consistent ? "✅ Tất cả PASS" : "❌ Có SAI LỆCH")")
println("=" ^ 60)
