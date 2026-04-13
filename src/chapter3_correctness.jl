# src/chapter3_correctness.jl
# Chương 3: So sánh kết quả Apriori (basic/optimized) với SPMF
# Xuất CSV → docs/chapter_3/
#
# Chạy: julia --project src/chapter3_correctness.jl

_SRC = joinpath(@__DIR__, "algorithm")
include(joinpath(_SRC, "structures.jl"))
include(joinpath(_SRC, "utils.jl"))
include(joinpath(_SRC, "apriori.jl"))

using Printf

# ─── Paths ────────────────────────────────────────────────────────────────────
const ROOT_DIR   = joinpath(@__DIR__, "..")
const SPMF_JAR   = joinpath(ROOT_DIR, "spmf.jar")
const OUT_DIR    = joinpath(ROOT_DIR, "outputs")
const CSV_DIR    = joinpath(ROOT_DIR, "docs", "chapter_3")
const DATA_TOY   = joinpath(ROOT_DIR, "data", "toy")
const DATA_BENCH = joinpath(ROOT_DIR, "data", "benchmark")
mkpath(OUT_DIR); mkpath(CSV_DIR)

# ─── Datasets ─────────────────────────────────────────────────────────────────
const TOY_DATASETS = [
    ("db1", joinpath(DATA_TOY, "db1.txt"), 0.50),
    ("db2", joinpath(DATA_TOY, "db2.txt"), 0.40),
    ("db3", joinpath(DATA_TOY, "db3.txt"), 0.50),
    ("db4", joinpath(DATA_TOY, "db4.txt"), 0.40),
    ("db5", joinpath(DATA_TOY, "db5.txt"), 0.50),
]

const BENCH_DATASETS = [
    ("chess",      joinpath(DATA_BENCH, "chess.txt"),      0.80),
    ("mushroom",   joinpath(DATA_BENCH, "mushroom.txt"),   0.30),
    ("retail",     joinpath(DATA_BENCH, "retail.txt"),     0.01),
    ("accidents",  joinpath(DATA_BENCH, "accidents.txt"),  0.60),
    ("T10I4D100K", joinpath(DATA_BENCH, "T10I4D100K.txt"), 0.02),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Hàm chính: chạy thuật toán + SPMF + so sánh → CSV row
# ═══════════════════════════════════════════════════════════════════════════════
function run_comparison(name::String, input::String, minsup::Float64)
    println("\n  ─── $name (minsup=$(round(minsup*100,digits=2))%) ───")

    D = load_database(input)
    db_size = length(D)
    println("    DB size: $db_size transactions")

    # 1. Apriori Basic
    res_basic, t_basic = measure_time(() -> apriori_basic(D, minsup; verbose=false))
    n_basic = total_fi(res_basic)
    our_output = joinpath(OUT_DIR, "output_$(name).txt")
    save_results(our_output, res_basic, db_size)

    # 2. Apriori Optimized
    res_opt, t_opt = measure_time(() -> apriori_optimized(D, minsup; verbose=false))
    n_opt = total_fi(res_opt)
    our_output_opt = joinpath(OUT_DIR, "output_$(name)_opt.txt")
    save_results(our_output_opt, res_opt, db_size)

    # 3. SPMF
    spmf_output = joinpath(OUT_DIR, "spmf_output_$(name).txt")
    minsup_pct = minsup * 100.0
    t_spmf = -1.0
    n_spmf = -1
    match_pct = -1.0
    n_match = 0
    only_spmf = 0

    if isfile(SPMF_JAR)
        t_spmf = run_spmf(SPMF_JAR, input, spmf_output, minsup_pct)
        if t_spmf >= 0 && isfile(spmf_output)
            spmf_res = load_spmf_output(spmf_output)
            n_spmf = length(spmf_res)
            our_flat = flatten_results(res_basic)
            _, _, n_match, match_pct, mismatches, only_spmf = compare_results(our_flat, spmf_res)

            if !isempty(mismatches)
                println("    ⚠ $(length(mismatches)) mismatches trong support count:")
                for (is, ours, theirs) in mismatches[1:min(3, length(mismatches))]
                    println("      $is : ours=$ours spmf=$theirs")
                end
            end
        else
            println("    ⚠ SPMF failed hoặc không có output")
        end
    else
        println("    ⚠ spmf.jar không tìm thấy tại $SPMF_JAR")
    end

    basic_opt_match = (n_basic == n_opt) ? "Yes" : "No"

    @printf("    Basic: %d FI (%.1fms) | Opt: %d FI (%.1fms) | Match: %s\n",
            n_basic, t_basic, n_opt, t_opt, basic_opt_match)
    if n_spmf >= 0
        @printf("    SPMF : %d FI (%.1fms) | Match%%: %.1f%% | Only_SPMF: %d\n",
                n_spmf, t_spmf, match_pct, only_spmf)
    end

    return [name, round(minsup*100, digits=2), db_size,
            n_basic, round(t_basic, digits=2),
            n_opt,   round(t_opt, digits=2),
            n_spmf >= 0 ? n_spmf : "N/A",
            t_spmf >= 0 ? round(t_spmf, digits=2) : "N/A",
            n_spmf >= 0 ? n_match : "N/A",
            n_spmf >= 0 ? match_pct : "N/A",
            basic_opt_match]
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
function main()
    println("=" ^ 60)
    println("  CHƯƠNG 3 – SO SÁNH KẾT QUẢ VỚI SPMF")
    println("=" ^ 60)

    headers = ["dataset", "minsup_pct", "db_size",
               "n_fi_basic", "time_basic_ms",
               "n_fi_opt",   "time_opt_ms",
               "n_fi_spmf",  "time_spmf_ms",
               "n_match_spmf", "match_pct_spmf",
               "basic_opt_consistent"]

    # ─── Toy datasets ─────────────────────────────────────────────────────────
    println("\n── TOY DATASETS ──")
    toy_rows = []
    for (name, path, ms) in TOY_DATASETS
        isfile(path) || (println("  Skip: $name"); continue)
        push!(toy_rows, run_comparison(name, path, ms))
    end
    write_csv(joinpath(CSV_DIR, "correctness_toy.csv"), headers, toy_rows)

    # ─── Level 3 comparison (Basic vs OPT) ──────────────────────────────
    println("\n── LEVEL 3: BASIC vs OPT ──")
    l3_headers = ["dataset", "minsup_pct",
                  "time_basic_ms", "mem_basic_mb",
                  "time_opt_ms",   "mem_opt_mb",
                  "speedup", "consistent"]
    l3_rows = []
    for (name, path, ms) in vcat(TOY_DATASETS)
        isfile(path) || continue
        D = load_database(path)
        # Measure Basic
        GC.gc()
        stats_b = @timed apriori_basic(D, ms; verbose=false)
        res_b = stats_b.value
        tb = stats_b.time * 1000 # ms
        mb_b = stats_b.bytes / (1024 * 1024)

        # Measure Opt
        GC.gc()
        stats_o = @timed apriori_optimized(D, ms; verbose=false)
        res_o = stats_o.value
        to = stats_o.time * 1000 # ms
        mb_o = stats_o.bytes / (1024 * 1024)

        consistent = total_fi(res_b) == total_fi(res_o) ? "Yes" : "No"
        speedup = to > 0 ? round(tb / to, digits=2) : 0.0
        push!(l3_rows, [name, round(ms*100,digits=2),
                        round(tb,digits=2), round(mb_b,digits=4),
                        round(to,digits=2), round(mb_o,digits=4),
                        speedup, consistent])
        @printf("    %-14s Basic=%.1fms Opt=%.1fms Speedup=%.2f× %s\n",
                name, tb, to, speedup, consistent)
    end
    write_csv(joinpath(CSV_DIR, "level3_comparison.csv"), l3_headers, l3_rows)

    println("\n" * "=" ^ 60)
    println("  ✅ Chương 3 hoàn thành!")
    println("  CSV → docs/chapter_3/")
    println("=" ^ 60)
end

main()
