# src/experiments.jl
# Chương 4: Thực nghiệm và đánh giá – 6 thực nghiệm bắt buộc
#
# Chạy: julia --project src/experiments.jl
#
# Kết quả: biểu đồ PNG lưu tại docs/

_SRC = joinpath(@__DIR__, "algorithm")
include(joinpath(_SRC, "structures.jl"))
include(joinpath(_SRC, "utils.jl"))
include(joinpath(_SRC, "apriori.jl"))

using Plots
using Printf
using Statistics
using Random

# ─── Thiết lập đường dẫn ──────────────────────────────────────────────────────
const BENCH_DIR = joinpath(@__DIR__, "..", "data", "benchmark")
const DOCS_DIR  = joinpath(@__DIR__, "..", "docs")
mkpath(DOCS_DIR)

gr()   # backend GR – không cần display

const DATASETS = [
    ("Chess",      joinpath(BENCH_DIR, "chess.txt")),
    ("Mushroom",   joinpath(BENCH_DIR, "mushroom.txt")),
    ("Retail",     joinpath(BENCH_DIR, "retail.txt")),
    ("Accidents",  joinpath(BENCH_DIR, "accidents.txt")),
    ("T10I4D100K", joinpath(BENCH_DIR, "T10I4D100K.txt")),
]

# Minsup thích hợp cho từng dataset (từ cao xuống thấp)
const MINSUP_RANGES = Dict(
    "Chess"      => [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60],
    "Mushroom"   => [0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10],
    "Retail"     => [0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
    "Accidents"  => [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30],
    "T10I4D100K" => [0.10, 0.05, 0.03, 0.02, 0.01, 0.007, 0.005],
)

# Minsup "trung bình" (cho Exp D)
const MINSUP_MED = Dict(
    "Chess"      => 0.75,
    "Mushroom"   => 0.25,
    "Retail"     => 0.01,
    "Accidents"  => 0.60,
    "T10I4D100K" => 0.02,
)

# ─── TIỆN ÍCH ─────────────────────────────────────────────────────────────────
total_fi(res) = sum(length(v) for v in values(res); init=0)

function load_all()::Dict{String, Vector{Vector{Int}}}
    dbs = Dict{String, Vector{Vector{Int}}}()
    for (name, path) in DATASETS
        if !isfile(path)
            @warn "Không tìm thấy file: $path  →  bỏ qua $name"
            continue
        end
        print("  Đang đọc $name ... ")
        dbs[name] = load_database(path)
        println("$(length(dbs[name])) giao dịch")
    end
    return dbs
end

# ══════════════════════════════════════════════════════════════════════════════
# THC NGHIỆM A: Kiểm tra tính đúng đắn (Correctness)
# ══════════════════════════════════════════════════════════════════════════════
"""
    exp_a_correctness(dbs)

Với mỗi dataset và minsup trung bình:
- Chạy Apriori (Basic và Optimized)
- Báo cáo số FI tìm được + so sánh hai phiên bản
- (Để so sánh với SPMF: chạy SPMF trên cùng input rồi điền cột SPMF)

Biểu đồ: bảng text + bar chart số FI theo dataset.
"""
function exp_a_correctness(dbs::Dict{String,Vector{Vector{Int}}})
    println("\n" * "═"^60)
    println("  THC NGHIỆM A – TÍNH ĐÚNG ĐẮN (Correctness)")
    println("═"^60)

    names    = String[]
    n_basic  = Int[]
    n_opt    = Int[]
    minsups  = Float64[]
    match    = Bool[]

    @printf("  %-14s  %7s  %8s  %8s  %8s\n",
            "Dataset", "Minsup%", "#FI Basic", "#FI Opt", "Khớp?")
    println("  " * "─"^55)

    for (name, _) in DATASETS
        haskey(dbs, name) || continue
        D  = dbs[name]
        ms = MINSUP_MED[name]

        res_b = apriori_basic(D, ms)
        res_o = apriori_optimized(D, ms)
        nb    = total_fi(res_b)
        no    = total_fi(res_o)
        ok    = (nb == no)

        @printf("  %-14s  %7.2f  %8d  %8d  %8s\n",
                name, ms*100, nb, no, ok ? "✅" : "❌")

        push!(names, name); push!(n_basic, nb); push!(n_opt, no)
        push!(minsups, ms); push!(match, ok)
    end

    # Bar chart (dùng 2 series bar thường thay vì groupedbar)
    isempty(names) && return
    xs = 1:length(names)
    p = bar(xs .- 0.15, n_basic, bar_width=0.3, label="Basic",
            color=:steelblue, yscale=:log10,
            title="Exp A – Số FI tìm được (Basic vs Optimized)",
            ylabel="Số Frequent Itemsets", xlabel="Dataset",
            legend=:topright, size=(800,500), dpi=150,
            titlefont=font(13), guidefont=font(11),
            xticks=(xs, names))
    bar!(xs .+ 0.15, n_opt, bar_width=0.3, label="Optimized", color=:coral)
    savefig(p, joinpath(DOCS_DIR, "exp_a_correctness.png"))
    println("\n  ✓ Lưu biểu đồ → docs/exp_a_correctness.png")
    println("  Ghi chú: Để so sánh SPMF, chạy SPMF trên cùng file và điền vào bảng trên.")
end

# ══════════════════════════════════════════════════════════════════════════════
# THC NGHIỆM B: Thời gian chạy theo minsup
# ══════════════════════════════════════════════════════════════════════════════
"""
    exp_b_runtime_vs_minsup(dbs)

Với mỗi dataset: đo thời gian (ms) tại 7 mức minsup giảm dần.
Vẽ line chart thời gian theo minsup (riêng cho từng dataset + tổng hợp).
"""
function exp_b_runtime_vs_minsup(dbs::Dict{String,Vector{Vector{Int}}})
    println("\n" * "═"^60)
    println("  THC NGHIỆM B – THỜI GIAN CHẠY THEO MINSUP")
    println("═"^60)

    colors = [:steelblue, :coral, :green, :purple, :orange]
    p_all  = plot(title="Exp B – Thời gian chạy theo Minsup (Log scale)",
                  xlabel="Minsup (%)", ylabel="Thời gian (ms, log)",
                  yscale=:log10, legend=:topright, size=(900,550), dpi=150,
                  titlefont=font(12), guidefont=font(10))

    for (ci, (name, _)) in enumerate(DATASETS)
        haskey(dbs, name) || continue
        D       = dbs[name]
        minsups = MINSUP_RANGES[name]
        times   = Float64[]

        print("  $name: ")
        for ms in minsups
            _, t = measure_time(apriori_optimized, D, ms)
            push!(times, t)
            @printf("%.0fms ", t)
        end
        println()

        ms_pct = minsups .* 100
        plot!(p_all, ms_pct, times,
              label=name, marker=:circle, lw=2,
              color=colors[mod1(ci, length(colors))])

        # Biểu đồ riêng cho dataset này
        p_ind = plot(ms_pct, times,
                     title="Exp B – $name: Thời gian theo Minsup",
                     xlabel="Minsup (%)", ylabel="Thời gian (ms)",
                     marker=:circle, lw=2, color=colors[mod1(ci, length(colors))],
                     legend=false, size=(700,450), dpi=150,
                     titlefont=font(12), guidefont=font(10))
        savefig(p_ind, joinpath(DOCS_DIR, "exp_b_runtime_$(lowercase(name)).png"))
    end

    savefig(p_all, joinpath(DOCS_DIR, "exp_b_runtime_all.png"))
    println("  ✓ Lưu biểu đồ → docs/exp_b_runtime_*.png")
end

# ══════════════════════════════════════════════════════════════════════════════
# THC NGHIỆM C: Số lượng FI theo minsup
# ══════════════════════════════════════════════════════════════════════════════
"""
    exp_c_itemsets_vs_minsup(dbs)

Vẽ đồ thị số lượng FI sinh ra theo minsup giảm dần.
Nhận xét mối quan hệ minsup ↓ → output size ↑ (phi tuyến, exponential).
"""
function exp_c_itemsets_vs_minsup(dbs::Dict{String,Vector{Vector{Int}}})
    println("\n" * "═"^60)
    println("  THC NGHIỆM C – SỐ LƯỢNG FI THEO MINSUP")
    println("═"^60)

    colors = [:steelblue, :coral, :green, :purple, :orange]
    p_all  = plot(title="Exp C – Số FI theo Minsup",
                  xlabel="Minsup (%)", ylabel="Số Frequent Itemsets (log)",
                  yscale=:log10, legend=:topright, size=(900,550), dpi=150,
                  titlefont=font(12), guidefont=font(10))

    for (ci, (name, _)) in enumerate(DATASETS)
        haskey(dbs, name) || continue
        D       = dbs[name]
        minsups = MINSUP_RANGES[name]
        counts  = Int[]

        print("  $name: ")
        for ms in minsups
            res = apriori_optimized(D, ms)
            n   = total_fi(res)
            push!(counts, max(n, 1))   # log10(0) → 1 fallback
            @printf("%d ", n)
        end
        println()

        ms_pct = minsups .* 100
        plot!(p_all, ms_pct, counts,
              label=name, marker=:square, lw=2,
              color=colors[mod1(ci, length(colors))])
    end

    savefig(p_all, joinpath(DOCS_DIR, "exp_c_itemcount_vs_minsup.png"))
    println("  ✓ Lưu biểu đồ → docs/exp_c_itemcount_vs_minsup.png")
end

# ══════════════════════════════════════════════════════════════════════════════
# THC NGHIỆM D: Bộ nhớ (Memory Usage)
# ══════════════════════════════════════════════════════════════════════════════
"""
    exp_d_memory(dbs)

Đo tổng bytes cấp phát (allocated) tại minsup trung bình cho mỗi dataset.
So sánh Basic vs Optimized.
"""
function exp_d_memory(dbs::Dict{String,Vector{Vector{Int}}})
    println("\n" * "═"^60)
    println("  THC NGHIỆM D – SỬ DỤNG BỘ NHỚ")
    println("═"^60)

    names    = String[]
    mb_basic = Float64[]
    mb_opt   = Float64[]

    @printf("  %-14s  %10s  %10s  %10s\n", "Dataset", "Basic(MB)", "Opt(MB)", "Giảm(%)")
    println("  " * "─"^50)

    for (name, _) in DATASETS
        haskey(dbs, name) || continue
        D  = dbs[name]
        ms = MINSUP_MED[name]

        _, mb_b = measure_memory_mb(apriori_basic,     D, ms)
        _, mb_o = measure_memory_mb(apriori_optimized, D, ms)
        red = mb_b > 0 ? (mb_b - mb_o) / mb_b * 100 : 0.0
        @printf("  %-14s  %10.4f  %10.4f  %9.1f%%\n", name, mb_b, mb_o, red)

        push!(names, name); push!(mb_basic, mb_b); push!(mb_opt, mb_o)
    end

    isempty(names) && return
    xs = 1:length(names)
    p = bar(xs .- 0.15, mb_basic, bar_width=0.3, label="Basic",
            color=:steelblue,
            title="Exp D – Bộ nhớ cấp phát (MB)",
            ylabel="MB cấp phát", xlabel="Dataset",
            legend=:topright, size=(800,500), dpi=150,
            titlefont=font(13), guidefont=font(11),
            xticks=(xs, names))
    bar!(xs .+ 0.15, mb_opt, bar_width=0.3, label="Optimized", color=:coral)
    savefig(p, joinpath(DOCS_DIR, "exp_d_memory.png"))
    println("  ✓ Lưu biểu đồ → docs/exp_d_memory.png")
end

# ══════════════════════════════════════════════════════════════════════════════
# THC NGHIỆM E: Scalability
# ══════════════════════════════════════════════════════════════════════════════
"""
    exp_e_scalability(dbs)

Chọn Retail (và Accidents nếu có): tạo tập con 10%,25%,50%,75%,100%.
Vẽ đồ thị thời gian chạy theo kích thước CSDL.
"""
function exp_e_scalability(dbs::Dict{String,Vector{Vector{Int}}})
    println("\n" * "═"^60)
    println("  THC NGHIỆM E – SCALABILITY")
    println("═"^60)

    pcts    = [0.10, 0.25, 0.50, 0.75, 1.00]
    targets = [("Retail", 0.01), ("Accidents", 0.60)]
    colors  = [:steelblue, :coral]

    p = plot(title="Exp E – Scalability (Thời gian vs Kích thước DB)",
             xlabel="Số giao dịch", ylabel="Thời gian (ms)",
             legend=:topleft, size=(800,500), dpi=150,
             titlefont=font(12), guidefont=font(10))

    for (ci, (name, ms)) in enumerate(targets)
        haskey(dbs, name) || continue
        D = dbs[name]
        sizes = Int[]
        times = Float64[]

        println("  $name (minsup=$(ms*100)%):")
        for pct in pcts
            sub = create_subset_db(D, pct)
            _, t = measure_time(apriori_optimized, sub, ms)
            push!(sizes, length(sub))
            push!(times, t)
            @printf("    %.0f%% → %d giao dịch : %.0f ms\n", pct*100, length(sub), t)
        end

        plot!(p, sizes, times,
              label=name, marker=:circle, lw=2,
              color=colors[mod1(ci, length(colors))])
    end

    savefig(p, joinpath(DOCS_DIR, "exp_e_scalability.png"))
    println("  ✓ Lưu biểu đồ → docs/exp_e_scalability.png")
end

# ══════════════════════════════════════════════════════════════════════════════
# THC NGHIỆM F: Ảnh hưởng độ dài giao dịch
# ══════════════════════════════════════════════════════════════════════════════
"""
    exp_f_transaction_length(; n_trans=5000, n_items=200, minsup=0.10)

Sinh CSDL tổng hợp với avg_len = 5,10,15,20,25,30.
Đo thời gian chạy và số FI. Vẽ hai đường trên cùng biểu đồ.
"""
function exp_f_transaction_length(; n_trans::Int=5000, n_items::Int=200,
                                  minsup::Float64=0.10)
    println("\n" * "═"^60)
    println("  THC NGHIỆM F – ẢNH HƯỞNG ĐỘ DÀI GIAO DỊCH")
    println("═"^60)
    @printf("  Tham số: n_trans=%d  n_items=%d  minsup=%.1f%%\n",
            n_trans, n_items, minsup*100)

    avg_lens = [5, 10, 15, 20, 25, 30]
    times    = Float64[]
    n_fi     = Int[]

    @printf("  %-10s  %10s  %10s\n", "AvgLen", "Time(ms)", "#FI")
    println("  " * "─"^35)

    for al in avg_lens
        D        = generate_synthetic_db(n_trans, n_items, al)
        res, t   = measure_time(apriori_optimized, D, minsup)
        n        = total_fi(res)
        push!(times, t); push!(n_fi, max(n, 1))
        @printf("  %-10d  %10.2f  %10d\n", al, t, n)
    end

    p = plot(avg_lens, times,
             title="Exp F – Ảnh hưởng Độ dài Giao dịch (n=$n_trans, items=$n_items)",
             xlabel="Độ dài giao dịch trung bình", ylabel="Thời gian (ms)",
             label="Thời gian", marker=:circle, lw=2, color=:steelblue,
             legend=:topleft, size=(800,500), dpi=150,
             titlefont=font(12), guidefont=font(10))
    # Trục phụ: số FI
    p2 = twinx(p)
    plot!(p2, avg_lens, n_fi,
          label="#FI", marker=:square, lw=2, linestyle=:dash, color=:coral,
          ylabel="Số Frequent Itemsets (log)", yscale=:log10, legend=:topright)

    savefig(p, joinpath(DOCS_DIR, "exp_f_txn_length.png"))
    println("  ✓ Lưu biểu đồ → docs/exp_f_txn_length.png")
end

# ══════════════════════════════════════════════════════════════════════════════
# MAIN – chạy tất cả 6 thực nghiệm
# ══════════════════════════════════════════════════════════════════════════════
function main()
    println("=" ^ 60)
    println("  CHƯƠNG 4 – THỰC NGHIỆM VÀ ĐÁNH GIÁ")
    println("  Apriori Frequent Itemset Mining")
    println("=" ^ 60)
    println("\n  Đang tải dữ liệu benchmark...")
    dbs = load_all()
    isempty(dbs) && (@warn "Không có dataset nào khả dụng!"; return)

    exp_a_correctness(dbs)
    exp_b_runtime_vs_minsup(dbs)
    exp_c_itemsets_vs_minsup(dbs)
    exp_d_memory(dbs)
    exp_e_scalability(dbs)
    exp_f_transaction_length()

    println("\n" * "=" ^ 60)
    println("  ✅ Hoàn thành 6 thực nghiệm!")
    println("  Biểu đồ đã lưu tại: docs/")
    println("=" ^ 60)
end

main()
