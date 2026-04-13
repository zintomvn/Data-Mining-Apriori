# src/chapter5_market_basket.jl
# Chương 5: Phân tích giỏ hàng (Market Basket Analysis)
# Dataset: Retail (data/benchmark/retail.txt)
# Output: docs/chapter_5/*.csv + *.png
#
# Chạy: julia --project src/chapter5_market_basket.jl

_SRC = joinpath(@__DIR__, "algorithm")
include(joinpath(_SRC, "structures.jl"))
include(joinpath(_SRC, "utils.jl"))
include(joinpath(_SRC, "apriori.jl"))

using Plots, Printf, Statistics
gr()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
const ROOT        = joinpath(@__DIR__, "..")
const RETAIL_PATH = joinpath(ROOT, "data", "benchmark", "retail.txt")
const CSV_DIR     = joinpath(ROOT, "docs", "chapter_5")
mkpath(CSV_DIR)

const MINSUP  = 0.01    # 1%
const MINCONF = 0.30    # 30%
const TOP_N   = 10

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
function main()
    println("=" ^ 60)
    println("  CHƯƠNG 5 – MARKET BASKET ANALYSIS")
    println("  Dataset: Retail | Minsup=$(MINSUP*100)% | Minconf=$(MINCONF*100)%")
    println("=" ^ 60)

    isfile(RETAIL_PATH) || error("File not found: $RETAIL_PATH")

    # ─── 1. Load data ─────────────────────────────────────────────────────────
    println("\n  Loading data...")
    D = load_database(RETAIL_PATH)
    db_size = length(D)
    n_items = length(unique(vcat(D...)))
    avg_len = mean(length.(D))
    @printf("  ✓ %d transactions, %d items, avg_len=%.2f\n", db_size, n_items, avg_len)

    # ─── 2. Run Apriori ──────────────────────────────────────────────────────
    println("\n  Mining frequent itemsets...")
    frequent, elapsed = measure_time(() -> apriori_optimized(D, MINSUP; verbose=false))
    n_total = total_fi(frequent)
    @printf("  ✓ %d FI found (%.2f ms)\n", n_total, elapsed)

    # Phân phối theo k
    println("  Distribution by k:")
    for k in sort(collect(keys(frequent)))
        @printf("    k=%d: %d itemsets\n", k, length(frequent[k]))
    end

    # ─── 3. Save frequent itemsets CSV ────────────────────────────────────────
    fi_headers = ["itemset", "support_count", "support_pct"]
    fi_rows = []
    for k in sort(collect(keys(frequent)))
        for is in sort(frequent[k])
            push!(fi_rows, [join(is.itemset, " "), is.count,
                            round(relative_support(is.count, db_size)*100, digits=4)])
        end
    end
    write_csv(joinpath(CSV_DIR, "frequent_itemsets.csv"), fi_headers, fi_rows)

    # ─── 4. Generate association rules ────────────────────────────────────────
    println("\n  Generating association rules...")
    rules = generate_association_rules(frequent, db_size, MINCONF)
    @printf("  ✓ %d rules (minconf=%.0f%%)\n", length(rules), MINCONF*100)

    if isempty(rules)
        println("  ⚠ No rules found. Try lower minconf or minsup.")
        return
    end

    # ─── 5. Save rules CSV ────────────────────────────────────────────────────
    rule_headers = ["rank", "antecedent", "consequent", "support_pct",
                    "confidence_pct", "lift"]
    rule_rows = []
    for (i, r) in enumerate(rules)
        push!(rule_rows, [i, join(r.antecedent, " "), join(r.consequent, " "),
                          round(r.support*100, digits=4),
                          round(r.confidence*100, digits=2),
                          round(r.lift, digits=4)])
    end
    write_csv(joinpath(CSV_DIR, "association_rules.csv"), rule_headers, rule_rows)

    # ─── 6. Statistics CSV ────────────────────────────────────────────────────
    all_lifts = [r.lift for r in rules]
    all_confs = [r.confidence for r in rules]
    all_sups  = [r.support for r in rules]

    stat_headers = ["metric", "value"]
    stat_rows = [
        ["db_size",       db_size],
        ["n_items",       n_items],
        ["avg_txn_len",   round(avg_len, digits=2)],
        ["minsup_pct",    MINSUP * 100],
        ["minconf_pct",   MINCONF * 100],
        ["n_frequent_itemsets", n_total],
        ["n_rules",       length(rules)],
        ["time_mining_ms", round(elapsed, digits=2)],
        ["lift_min",      round(minimum(all_lifts), digits=4)],
        ["lift_max",      round(maximum(all_lifts), digits=4)],
        ["lift_mean",     round(mean(all_lifts), digits=4)],
        ["conf_min_pct",  round(minimum(all_confs)*100, digits=2)],
        ["conf_max_pct",  round(maximum(all_confs)*100, digits=2)],
        ["conf_mean_pct", round(mean(all_confs)*100, digits=2)],
        ["sup_min_pct",   round(minimum(all_sups)*100, digits=4)],
        ["sup_max_pct",   round(maximum(all_sups)*100, digits=4)],
        ["sup_mean_pct",  round(mean(all_sups)*100, digits=4)],
    ]
    write_csv(joinpath(CSV_DIR, "statistics.csv"), stat_headers, stat_rows)

    # ─── 7. Print Top-N rules ─────────────────────────────────────────────────
    println("\n  TOP-$TOP_N RULES (by Lift):")
    println("  " * "─"^70)
    @printf("  %-4s %-25s %-25s %6s %6s %6s\n",
            "#", "Antecedent", "Consequent", "Sup%", "Conf%", "Lift")
    println("  " * "─"^70)
    for i in 1:min(TOP_N, length(rules))
        r = rules[i]
        @printf("  %-4d %-25s %-25s %6.2f %6.1f %6.3f\n",
                i, string(r.antecedent), string(r.consequent),
                r.support*100, r.confidence*100, r.lift)
    end

    # Business interpretation
    println("\n  Business Insights:")
    for i in 1:min(5, length(rules))
        r = rules[i]
        println("    Rule #$i: $(r.antecedent) ⟹ $(r.consequent)")
        @printf("      Lift=%.3f → ", r.lift)
        if r.lift > 2.0
            println("Strong positive correlation – cross-sell opportunity")
        elseif r.lift > 1.0
            println("Positive correlation – consider bundling")
        else
            println("Weak/negative correlation")
        end
    end

    # ─── 8. Charts ────────────────────────────────────────────────────────────
    println("\n  Generating charts...")

    # Scatter: Support vs Confidence (color=Lift)
    n_plot = min(length(rules), TOP_N * 5)
    sups  = [r.support * 100 for r in rules[1:n_plot]]
    confs = [r.confidence * 100 for r in rules[1:n_plot]]
    lifts = [r.lift for r in rules[1:n_plot]]
    sizes = clamp.(lifts .* 8, 4, 30)

    p1 = scatter(sups, confs, markersize=sizes, zcolor=lifts,
                 xlabel="Support (%)", ylabel="Confidence (%)",
                 title="Ch.5 – Association Rules (color~Lift)",
                 colorbar_title="Lift", legend=false, color=:plasma,
                 alpha=0.7, size=(800,550), dpi=150)
    savefig(p1, joinpath(CSV_DIR, "scatter.png"))

    # Bar: Top-10 Lift
    n_bar = min(TOP_N, length(rules))
    labels = ["R$i" for i in 1:n_bar]
    lifts_bar = [rules[i].lift for i in 1:n_bar]
    p2 = bar(labels, lifts_bar, title="Ch.5 – Top-$n_bar Rules by Lift",
             ylabel="Lift", color=:steelblue, legend=false,
             size=(900,500), dpi=150, xrotation=30)
    savefig(p2, joinpath(CSV_DIR, "top10_lift.png"))

    # Histogram: Lift + Confidence distribution
    p3a = histogram([r.lift for r in rules], bins=30,
                    title="Lift Distribution", xlabel="Lift", ylabel="Count",
                    color=:steelblue, legend=false, dpi=150)
    p3b = histogram([r.confidence*100 for r in rules], bins=30,
                    title="Confidence Distribution", xlabel="Confidence (%)",
                    ylabel="Count", color=:coral, legend=false, dpi=150)
    p3 = plot(p3a, p3b, layout=(1,2), size=(1000,400), dpi=150)
    savefig(p3, joinpath(CSV_DIR, "distribution.png"))

    println("  ✓ Charts saved to docs/chapter_5/")

    println("\n" * "=" ^ 60)
    println("  ✅ Chapter 5 complete!")
    println("  Output → docs/chapter_5/")
    println("=" ^ 60)
end

main()
