# src/experiments.jl
# Chương 4: Thực nghiệm và đánh giá – 6 thực nghiệm
# Mỗi thực nghiệm: (1) thu thập số liệu  (2) ghi CSV  (3) vẽ biểu đồ
#
# Output: docs/chapter_4/{a,b,c,d,e,f}/*.csv + *.png
# Chạy: julia --project src/experiments.jl

_SRC = joinpath(@__DIR__, "algorithm")
include(joinpath(_SRC, "structures.jl"))
include(joinpath(_SRC, "utils.jl"))
include(joinpath(_SRC, "apriori.jl"))

using Plots, Printf, Statistics, Random
gr()

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
const ROOT      = joinpath(@__DIR__, "..")
const BENCH_DIR = joinpath(ROOT, "data", "benchmark")
const SPMF_JAR  = joinpath(ROOT, "spmf.jar")
const OUT_DIR   = joinpath(ROOT, "outputs")
mkpath(OUT_DIR)

const DATASETS = [
    ("Chess",      joinpath(BENCH_DIR, "chess.txt")),
    ("Mushroom",   joinpath(BENCH_DIR, "mushroom.txt")),
    ("Retail",     joinpath(BENCH_DIR, "retail.txt")),
    ("Accidents",  joinpath(BENCH_DIR, "accidents.txt")),
    ("T10I4D100K", joinpath(BENCH_DIR, "T10I4D100K.txt")),
]

const MINSUP_RANGES = Dict(
    "Chess"      => [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60],
    "Mushroom"   => [0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10],
    "Retail"     => [0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
    "Accidents"  => [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30],
    "T10I4D100K" => [0.10, 0.05, 0.03, 0.02, 0.01, 0.007, 0.005],
)

const MINSUP_MED = Dict(
    "Chess" => 0.75, "Mushroom" => 0.25, "Retail" => 0.01,
    "Accidents" => 0.60, "T10I4D100K" => 0.02,
)

function ensure_dir(path)
    mkpath(path); return path
end

function load_all()
    dbs = Dict{String, Vector{Vector{Int}}}()
    for (name, path) in DATASETS
        isfile(path) || (@warn "Missing: $path"; continue)
        print("  Loading $name ... ")
        dbs[name] = load_database(path)
        println("$(length(dbs[name])) txns")
    end
    return dbs
end

# ══════════════════════════════════════════════════════════════════════════════
# EXP A: Correctness + SPMF comparison
# ══════════════════════════════════════════════════════════════════════════════
function exp_a(dbs)
    dir = ensure_dir(joinpath(ROOT, "docs", "chapter_4", "a"))
    println("\n" * "═"^55 * "\n  EXP A – CORRECTNESS\n" * "═"^55)

    headers = ["dataset","minsup_pct","n_fi_basic","n_fi_opt","n_fi_spmf",
               "match_pct","time_ours_ms","time_spmf_ms"]
    rows = []
    names_plt = String[]; n_b_plt = Int[]; n_o_plt = Int[]; n_s_plt = Int[]

    for (name, _) in DATASETS
        haskey(dbs, name) || continue
        D = dbs[name]; ms = MINSUP_MED[name]
        ms_pct = ms * 100

        res_b, tb = measure_time(() -> apriori_basic(D, ms; verbose=false))
        res_o, to = measure_time(() -> apriori_optimized(D, ms; verbose=false))
        nb = total_fi(res_b); no = total_fi(res_o)

        # SPMF
        ns = -1; ts = -1.0; mpct = -1.0
        spmf_out = joinpath(OUT_DIR, "spmf_exp_a_$(lowercase(name)).txt")
        if isfile(SPMF_JAR)
            inp = [p for (n,p) in DATASETS if n==name][1]
            ts = run_spmf(SPMF_JAR, inp, spmf_out, ms_pct)
            if ts >= 0 && isfile(spmf_out)
                spmf_map = load_spmf_output(spmf_out)
                ns = length(spmf_map)
                our_flat = flatten_results(res_b)
                _, _, _, mpct, _, _ = compare_results(our_flat, spmf_map)
            end
        end

        push!(rows, [name, round(ms_pct,digits=2), nb, no,
                     ns >= 0 ? ns : "N/A",
                     mpct >= 0 ? mpct : "N/A",
                     round(min(tb,to),digits=2),
                     ts >= 0 ? round(ts,digits=2) : "N/A"])
        push!(names_plt, name); push!(n_b_plt, nb); push!(n_o_plt, no)
        push!(n_s_plt, ns >= 0 ? ns : 0)

        @printf("  %-12s Basic=%d Opt=%d SPMF=%s Match=%s\n",
                name, nb, no, ns >= 0 ? string(ns) : "N/A",
                mpct >= 0 ? "$(mpct)%" : "N/A")
    end

    write_csv(joinpath(dir, "correctness.csv"), headers, rows)

    # Plot
    if !isempty(names_plt)
        xs = 1:length(names_plt)
        p = bar(xs .- 0.2, n_b_plt, bar_width=0.2, label="Basic", color=:steelblue,
                yscale=:log10, title="Exp A – Correctness", ylabel="#FI",
                xticks=(xs, names_plt), legend=:topright, size=(800,500), dpi=150)
        bar!(xs, n_o_plt, bar_width=0.2, label="HashTree", color=:coral)
        any(x->x>0, n_s_plt) && bar!(xs .+ 0.2, max.(n_s_plt,1), bar_width=0.2, label="SPMF", color=:green)
        savefig(p, joinpath(dir, "correctness.png"))
        println("  ✓ PNG → $(dir)/correctness.png")
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# EXP B: Runtime vs Minsup  (+ SPMF comparison)
# ══════════════════════════════════════════════════════════════════════════════
function exp_b(dbs)
    dir = ensure_dir(joinpath(ROOT, "docs", "chapter_4", "b"))
    println("\n" * "═"^55 * "\n  EXP B – RUNTIME vs MINSUP\n" * "═"^55)

    headers = ["dataset","minsup_pct","time_ours_ms","time_spmf_ms"]
    rows = []
    colors = [:steelblue, :coral, :green, :purple, :orange]

    p_all = plot(title="Exp B – Runtime vs Minsup", xlabel="Minsup (%)",
                 ylabel="Time (ms, log)", yscale=:log10, legend=:topright,
                 size=(900,550), dpi=150)

    for (ci, (name, _)) in enumerate(DATASETS)
        haskey(dbs, name) || continue
        D = dbs[name]; minsups = MINSUP_RANGES[name]
        times_ours = Float64[]; times_spmf = Float64[]

        print("  $name: ")
        for ms in minsups
            _, t = measure_time(() -> apriori_optimized(D, ms; verbose=false))
            push!(times_ours, t)
            # SPMF
            ts = -1.0
            if isfile(SPMF_JAR)
                inp = [p for (n,p) in DATASETS if n==name][1]
                spmf_out = joinpath(OUT_DIR, "spmf_b_$(lowercase(name))_$(ms).txt")
                ts = run_spmf(SPMF_JAR, inp, spmf_out, ms*100)
            end
            push!(times_spmf, ts)
            push!(rows, [name, round(ms*100,digits=3), round(t,digits=2),
                         ts >= 0 ? round(ts,digits=2) : "N/A"])
            @printf("%.0f ", t)
        end
        println()

        ms_pct = minsups .* 100
        plot!(p_all, ms_pct, times_ours, label="$name (ours)",
              marker=:circle, lw=2, color=colors[mod1(ci, length(colors))])
        valid_spmf = [(ms_pct[i], times_spmf[i]) for i in eachindex(times_spmf) if times_spmf[i] >= 0]
        if !isempty(valid_spmf)
            plot!(p_all, [x[1] for x in valid_spmf], [x[2] for x in valid_spmf],
                  label="$name (SPMF)", marker=:diamond, lw=1, ls=:dash,
                  color=colors[mod1(ci, length(colors))])
        end

        # Individual plot
        p_ind = plot(ms_pct, times_ours, title="Exp B – $name",
                     xlabel="Minsup (%)", ylabel="Time (ms)", marker=:circle, lw=2,
                     label="Ours", color=colors[mod1(ci, length(colors))],
                     size=(700,450), dpi=150, legend=:topright)
        if !isempty(valid_spmf)
            plot!(p_ind, [x[1] for x in valid_spmf], [x[2] for x in valid_spmf],
                  label="SPMF", marker=:diamond, lw=1, ls=:dash, color=:gray)
        end
        savefig(p_ind, joinpath(dir, "runtime_$(lowercase(name)).png"))
    end

    write_csv(joinpath(dir, "runtime_vs_minsup.csv"), headers, rows)
    savefig(p_all, joinpath(dir, "runtime_all.png"))
    println("  ✓ CSV + PNG → $dir")
end

# ══════════════════════════════════════════════════════════════════════════════
# EXP C: #FI vs Minsup
# ══════════════════════════════════════════════════════════════════════════════
function exp_c(dbs)
    dir = ensure_dir(joinpath(ROOT, "docs", "chapter_4", "c"))
    println("\n" * "═"^55 * "\n  EXP C – #FI vs MINSUP\n" * "═"^55)

    headers = ["dataset","minsup_pct","n_fi"]
    rows = []; colors = [:steelblue, :coral, :green, :purple, :orange]
    p = plot(title="Exp C – #FI vs Minsup", xlabel="Minsup (%)",
             ylabel="#FI (log)", yscale=:log10, legend=:topright,
             size=(900,550), dpi=150)

    for (ci, (name, _)) in enumerate(DATASETS)
        haskey(dbs, name) || continue
        D = dbs[name]; minsups = MINSUP_RANGES[name]
        counts = Int[]
        print("  $name: ")
        for ms in minsups
            res = apriori_optimized(D, ms; verbose=false)
            n = total_fi(res)
            push!(counts, max(n, 1))
            push!(rows, [name, round(ms*100,digits=3), n])
            @printf("%d ", n)
        end
        println()
        plot!(p, minsups.*100, counts, label=name, marker=:square, lw=2,
              color=colors[mod1(ci, length(colors))])
    end

    write_csv(joinpath(dir, "itemcount_vs_minsup.csv"), headers, rows)
    savefig(p, joinpath(dir, "itemcount.png"))
    println("  ✓ CSV + PNG → $dir")
end

# ══════════════════════════════════════════════════════════════════════════════
# EXP D: Memory (Basic vs HashTree)
# ══════════════════════════════════════════════════════════════════════════════
function exp_d(dbs)
    dir = ensure_dir(joinpath(ROOT, "docs", "chapter_4", "d"))
    println("\n" * "═"^55 * "\n  EXP D – MEMORY\n" * "═"^55)

    headers = ["dataset","minsup_pct","mem_basic_mb","mem_opt_mb","reduction_pct"]
    rows = []; names_plt = String[]; mb_b_plt = Float64[]; mb_o_plt = Float64[]

    for (name, _) in DATASETS
        haskey(dbs, name) || continue
        D = dbs[name]; ms = MINSUP_MED[name]
        _, mb_b = measure_memory_mb(() -> apriori_basic(D, ms; verbose=false))
        _, mb_o = measure_memory_mb(() -> apriori_optimized(D, ms; verbose=false))
        red = mb_b > 0 ? round((mb_b - mb_o)/mb_b*100, digits=1) : 0.0
        push!(rows, [name, round(ms*100,digits=2), round(mb_b,digits=4),
                     round(mb_o,digits=4), red])
        push!(names_plt, name); push!(mb_b_plt, mb_b); push!(mb_o_plt, mb_o)
        @printf("  %-12s Basic=%.4fMB Opt=%.4fMB Red=%.1f%%\n", name, mb_b, mb_o, red)
    end

    write_csv(joinpath(dir, "memory.csv"), headers, rows)

    if !isempty(names_plt)
        xs = 1:length(names_plt)
        p = bar(xs .- 0.15, mb_b_plt, bar_width=0.3, label="Basic", color=:steelblue,
                title="Exp D – Memory", ylabel="MB", xticks=(xs, names_plt),
                legend=:topright, size=(800,500), dpi=150)
        bar!(xs .+ 0.15, mb_o_plt, bar_width=0.3, label="HashTree", color=:coral)
        savefig(p, joinpath(dir, "memory.png"))
        println("  ✓ CSV + PNG → $dir")
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# EXP E: Scalability
# ══════════════════════════════════════════════════════════════════════════════
function exp_e(dbs)
    dir = ensure_dir(joinpath(ROOT, "docs", "chapter_4", "e"))
    println("\n" * "═"^55 * "\n  EXP E – SCALABILITY\n" * "═"^55)

    headers = ["dataset","pct","n_trans","time_ms"]
    rows = []; pcts = [0.10, 0.25, 0.50, 0.75, 1.00]
    targets = [("Retail", 0.01), ("Accidents", 0.60)]
    colors = [:steelblue, :coral]

    p = plot(title="Exp E – Scalability", xlabel="#Transactions",
             ylabel="Time (ms)", legend=:topleft, size=(800,500), dpi=150)

    for (ci, (name, ms)) in enumerate(targets)
        haskey(dbs, name) || continue
        D = dbs[name]; sizes = Int[]; times = Float64[]
        println("  $name (minsup=$(ms*100)%):")
        for pct in pcts
            sub = create_subset_db(D, pct)
            _, t = measure_time(() -> apriori_optimized(sub, ms; verbose=false))
            push!(sizes, length(sub)); push!(times, t)
            push!(rows, [name, round(pct*100,digits=0), length(sub), round(t,digits=2)])
            @printf("    %.0f%% → %d txns : %.0f ms\n", pct*100, length(sub), t)
        end
        plot!(p, sizes, times, label=name, marker=:circle, lw=2,
              color=colors[mod1(ci, length(colors))])
    end

    write_csv(joinpath(dir, "scalability.csv"), headers, rows)
    savefig(p, joinpath(dir, "scalability.png"))
    println("  ✓ CSV + PNG → $dir")
end

# ══════════════════════════════════════════════════════════════════════════════
# EXP F: Transaction length effect
# ══════════════════════════════════════════════════════════════════════════════
function exp_f(; n_trans=5000, n_items=200, minsup=0.10)
    dir = ensure_dir(joinpath(ROOT, "docs", "chapter_4", "f"))
    println("\n" * "═"^55 * "\n  EXP F – TXN LENGTH EFFECT\n" * "═"^55)

    headers = ["avg_len","time_ms","n_fi"]
    rows = []; avg_lens = [5, 10, 15, 20, 25, 30]
    times = Float64[]; n_fis = Int[]

    @printf("  n_trans=%d n_items=%d minsup=%.1f%%\n", n_trans, n_items, minsup*100)
    for al in avg_lens
        D = generate_synthetic_db(n_trans, n_items, al)
        res, t = measure_time(() -> apriori_optimized(D, minsup; verbose=false))
        n = total_fi(res)
        push!(times, t); push!(n_fis, max(n, 1))
        push!(rows, [al, round(t,digits=2), n])
        @printf("  AvgLen=%d : %.2fms, %d FI\n", al, t, n)
    end

    write_csv(joinpath(dir, "txn_length.csv"), headers, rows)

    p = plot(avg_lens, times, title="Exp F – Txn Length Effect",
             xlabel="Avg Txn Length", ylabel="Time (ms)",
             label="Time", marker=:circle, lw=2, color=:steelblue,
             legend=:topleft, size=(800,500), dpi=150)
    p2 = twinx(p)
    plot!(p2, avg_lens, n_fis, label="#FI", marker=:square, lw=2,
          ls=:dash, color=:coral, ylabel="#FI (log)", yscale=:log10, legend=:topright)
    savefig(p, joinpath(dir, "txn_length.png"))
    println("  ✓ CSV + PNG → $dir")
end

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
function main()
    println("=" ^ 55)
    println("  CHƯƠNG 4 – THỰC NGHIỆM VÀ ĐÁNH GIÁ")
    println("=" ^ 55)
    println("\n  Loading benchmark data...")
    dbs = load_all()
    isempty(dbs) && (@warn "No datasets!"; return)

    exp_a(dbs)
    exp_b(dbs)
    exp_c(dbs)
    exp_d(dbs)
    exp_e(dbs)
    exp_f()

    println("\n" * "=" ^ 55)
    println("  ✅ All 6 experiments completed!")
    println("  CSV + PNG → docs/chapter_4/{a,b,c,d,e,f}/")
    println("=" ^ 55)
end

main()
