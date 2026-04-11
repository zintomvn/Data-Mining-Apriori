# src/chapter5_market_basket.jl
# Chương 5: Ứng dụng thực tế – Phân tích giỏ hàng (Market Basket Analysis)
# Dùng tập dữ liệu Retail (data/benchmark/retail.txt)
#
# Chạy: julia --project src/chapter5_market_basket.jl

_SRC = joinpath(@__DIR__, "algorithm")
include(joinpath(_SRC, "structures.jl"))
include(joinpath(_SRC, "utils.jl"))
include(joinpath(_SRC, "apriori.jl"))

using Plots
using Printf

# ─── Tham số ──────────────────────────────────────────────────────────────────
const RETAIL_PATH = joinpath(@__DIR__, "..", "data", "benchmark", "retail.txt")
const DOCS_DIR    = joinpath(@__DIR__, "..", "docs")
mkpath(DOCS_DIR)

const MINSUP  = 0.01    # 1% – khoảng 882 / 88162 giao dịch
const MINCONF = 0.30    # Confidence tối thiểu 30%
const TOP_N   = 10      # Hiển thị top-10 luật

# ─── HÀM HIỂN THỊ ─────────────────────────────────────────────────────────────

"""
    print_rule_table(rules::Vector{AssociationRule}, top_n::Int)

In bảng top-N luật kết hợp sắp xếp giảm dần theo lift.
"""
function print_rule_table(rules::Vector{AssociationRule}, top_n::Int)
    n = min(top_n, length(rules))
    println("\n  ┌─────┬──────────────────────────────────────────┬────────┬────────┬────────┐")
    println("  │ Rank│ Luật X ⟹ Y                               │  Sup%  │ Conf%  │  Lift  │")
    println("  ├─────┼──────────────────────────────────────────┼────────┼────────┼────────┤")
    for i in 1:n
        r = rules[i]
        rule_str = "$(r.antecedent) => $(r.consequent)"
        # Cắt chuỗi nếu quá dài
        if length(rule_str) > 42
            rule_str = rule_str[1:39] * "..."
        end
        @printf("  │ %3d │ %-42s │ %6.2f │ %6.2f │ %6.3f │\n",
                i, rule_str, r.support*100, r.confidence*100, r.lift)
    end
    println("  └─────┴──────────────────────────────────────────┴────────┴────────┴────────┘")
end

"""
    analyze_top_rules(rules::Vector{AssociationRule}, db_size::Int)

Phân tích và in giải thích ý nghĩa kinh doanh của top-5 luật theo lift.
"""
function analyze_top_rules(rules::Vector{AssociationRule}, db_size::Int)
    println("\n" * "─"^60)
    println("  Phân tích ý nghĩa kinh doanh (Top-5 luật theo Lift)")
    println("─"^60)

    for i in 1:min(5, length(rules))
        r = rules[i]
        println("\n  Luật #$i:  $(r.antecedent)  ⟹  $(r.consequent)")
        @printf("    • Support   = %.2f%% (xuất hiện cùng nhau trong %.2f%% giao dịch)\n",
                r.support * 100, r.support * 100)
        @printf("    • Confidence= %.2f%% (khi mua %s, %.2f%% cũng mua %s)\n",
                r.confidence * 100, r.antecedent, r.confidence * 100, r.consequent)
        @printf("    • Lift      = %.3f (", r.lift)
        if r.lift > 1.5
            println("Tương quan dương mạnh – nên bày cùng kệ hoặc gợi ý chéo)")
        elseif r.lift > 1.0
            println("Tương quan dương nhẹ – có thể cân nhắc upsell)")
        elseif r.lift ≈ 1.0
            println("Độc lập – không có tương quan đặc biệt)")
        else
            println("Tương quan âm – hai sản phẩm cạnh tranh nhau)")
        end
    end
end

# ─── BIỂU ĐỒ ──────────────────────────────────────────────────────────────────

"""
    plot_top_rules(rules::Vector{AssociationRule}, top_n::Int)

Vẽ biểu đồ scatter Confidence vs Support, kích thước điểm tỉ lệ với Lift.
"""
function plot_top_rules(rules::Vector{AssociationRule}, top_n::Int)
    n   = min(top_n * 3, length(rules))   # lấy nhiều hơn top_n để biểu đồ rõ hơn
    sups  = [r.support    * 100 for r in rules[1:n]]
    confs = [r.confidence * 100 for r in rules[1:n]]
    lifts = [r.lift             for r in rules[1:n]]
    sizes = clamp.(lifts .* 10, 4, 40)

    p = scatter(sups, confs,
        markersize  = sizes,
        zcolor      = lifts,
        xlabel      = "Support (%)",
        ylabel      = "Confidence (%)",
        title       = "Ch.5 – Association Rules Retail\n(màu & kích thước ~ Lift)",
        colorbar_title = "Lift",
        legend      = false,
        color       = :plasma,
        alpha       = 0.7,
        size        = (800, 550),
        dpi         = 150,
        titlefont   = font(11),
        guidefont   = font(10))

    savefig(p, joinpath(DOCS_DIR, "ch5_association_rules_scatter.png"))
    println("  ✓ Lưu biểu đồ scatter → docs/ch5_association_rules_scatter.png")
end

"""
    plot_top10_lift(rules::Vector{AssociationRule}, top_n::Int)

Bar chart Top-10 luật theo Lift.
"""
function plot_top10_lift(rules::Vector{AssociationRule}, top_n::Int)
    n      = min(top_n, length(rules))
    labels = ["Rule $i" for i in 1:n]
    lifts  = [rules[i].lift for i in 1:n]
    confs  = [rules[i].confidence * 100 for i in 1:n]

    p = bar(labels, lifts,
        title     = "Ch.5 – Top-$n Luật theo Lift (Retail)",
        ylabel    = "Lift",
        xlabel    = "Luật (xếp theo Lift giảm dần)",
        color     = :steelblue,
        legend    = false,
        size      = (900, 500),
        dpi       = 150,
        xrotation = 30,
        titlefont = font(12),
        guidefont = font(10))

    savefig(p, joinpath(DOCS_DIR, "ch5_top10_lift.png"))
    println("  ✓ Lưu biểu đồ top-10 lift → docs/ch5_top10_lift.png")
end

# ─── THỐNG KÊ PHÂN PHỐI LUẬT ──────────────────────────────────────────────────

"""
    plot_rules_distribution(rules::Vector{AssociationRule})

Histogram phân phối Confidence và Lift của toàn bộ luật.
"""
function plot_rules_distribution(rules::Vector{AssociationRule})
    isempty(rules) && return
    lifts = [r.lift for r in rules]
    confs = [r.confidence * 100 for r in rules]

    p1 = histogram(lifts, bins=30,
                   title="Phân phối Lift", xlabel="Lift", ylabel="Tần số",
                   color=:steelblue, legend=false, dpi=150, titlefont=font(11))
    p2 = histogram(confs, bins=30,
                   title="Phân phối Confidence", xlabel="Confidence (%)", ylabel="Tần số",
                   color=:coral,     legend=false, dpi=150, titlefont=font(11))

    p = plot(p1, p2, layout=(1,2), size=(1000, 400), dpi=150,
             plot_title="Ch.5 – Phân phối Luật Kết hợp (Retail)")
    savefig(p, joinpath(DOCS_DIR, "ch5_rules_distribution.png"))
    println("  ✓ Lưu biểu đồ phân phối → docs/ch5_rules_distribution.png")
end

# ─── MAIN ─────────────────────────────────────────────────────────────────────
function main()
    gr()

    println("=" ^ 60)
    println("  CHƯƠNG 5 – PHÂN TÍCH GIỎ HÀNG (Market Basket Analysis)")
    println("  Dataset: Retail (data/benchmark/retail.txt)")
    println("=" ^ 60)

    # 1. Kiểm tra file
    if !isfile(RETAIL_PATH)
        error("Không tìm thấy file Retail: $RETAIL_PATH")
    end

    # 2. Load data
    println("\n  Đang tải dữ liệu...")
    D = load_database(RETAIL_PATH)
    db_size = length(D)
    @printf("  ✓ Đã nạp %d giao dịch\n", db_size)

    n_items = length(unique(vcat(D...)))
    avg_len = mean(length.(D))
    @printf("  ✓ Số item khác nhau : %d\n", n_items)
    @printf("  ✓ Độ dài TB giao dịch: %.2f\n", avg_len)

    # 3. Chạy Apriori
    println("\n  Đang khai thác frequent itemsets...")
    @printf("  Minsup = %.2f%% (count = %d)\n", MINSUP*100, ceil(Int, MINSUP*db_size))
    frequent, elapsed = measure_time(apriori_optimized, D, MINSUP)
    total = sum(length(v) for v in values(frequent); init=0)
    @printf("  ✓ Tìm được %d frequent itemsets (%.2f ms)\n\n", total, elapsed)

    # In phân phối theo kích thước
    println("  Phân phối theo kích thước:")
    for k in sort(collect(keys(frequent)))
        @printf("    k=%d: %d itemsets\n", k, length(frequent[k]))
    end

    # 4. Sinh luật kết hợp
    println("\n  Đang sinh luật kết hợp...")
    @printf("  Minconf = %.0f%%\n", MINCONF*100)
    rules = generate_association_rules(frequent, db_size, MINCONF)
    @printf("  ✓ Sinh được %d luật kết hợp\n", length(rules))

    if isempty(rules)
        println("  ⚠ Không có luật nào thỏa mãn minconf. Thử giảm minconf hoặc minsup.")
        return
    end

    # 5. Hiển thị kết quả
    println("\n" * "─"^60)
    @printf("  TOP-%d LUẬT KẾT HỢP (sắp giảm dần theo Lift)\n", TOP_N)
    println("─"^60)
    print_rule_table(rules, TOP_N)
    analyze_top_rules(rules, db_size)

    # Thống kê tổng hợp
    all_lifts  = [r.lift        for r in rules]
    all_confs  = [r.confidence  for r in rules]
    all_sups   = [r.support     for r in rules]
    println("\n  Thống kê tổng quát:")
    @printf("    Lift  : min=%.3f  max=%.3f  mean=%.3f\n", minimum(all_lifts),  maximum(all_lifts),  mean(all_lifts))
    @printf("    Conf  : min=%.2f%% max=%.2f%% mean=%.2f%%\n", minimum(all_confs)*100, maximum(all_confs)*100, mean(all_confs)*100)
    @printf("    Sup   : min=%.4f%% max=%.4f%% mean=%.4f%%\n", minimum(all_sups)*100, maximum(all_sups)*100, mean(all_sups)*100)

    # 6. Vẽ biểu đồ
    println("\n  Đang vẽ biểu đồ...")
    plot_top_rules(rules, TOP_N)
    plot_top10_lift(rules, TOP_N)
    plot_rules_distribution(rules)

    println("\n" * "=" ^ 60)
    println("  ✅ Hoàn thành Chương 5!")
    println("  Biểu đồ đã lưu tại: docs/ch5_*.png")
    println("=" ^ 60)
end

main()
