# src/algorithm/apriori.jl
# Thuật toán Apriori FIM – Paper gốc:
#   Agrawal & Srikant, "Fast Algorithms for Mining Association Rules", VLDB 1994
#
# Hai phiên bản:
#   - apriori_basic     : đếm support bằng linear scan (Level 1)
#   - apriori_optimized : đếm support bằng Hash Tree   (Level 3)

# Auto-include dependencies khi chạy trực tiếp từ CLI
if !@isdefined(Itemset)
    include(joinpath(@__DIR__, "structures.jl"))
    include(joinpath(@__DIR__, "utils.jl"))
end

using Printf

# ══════════════════════════════════════════════════════════════════════════════
# APRIORI-GEN  (Join + Prune)
# ══════════════════════════════════════════════════════════════════════════════

"""
    _has_infrequent_subset(candidate, prev_set) -> Bool

Prune step: kiểm tra candidate có (k-1)-subset không thuộc L_{k-1}.
Nếu có → candidate bị loại (Apriori property / downward closure).

# Độ phức tạp: O(k²)
"""
function _has_infrequent_subset(candidate::Vector{Int},
                                 prev_set::Set{Vector{Int}})::Bool
    k = length(candidate)
    @inbounds for pos in 1:k
        sub = Vector{Int}(undef, k - 1)
        idx = 1
        @inbounds for i in 1:k
            i == pos && continue
            sub[idx] = candidate[i]
            idx += 1
        end
        sub ∉ prev_set && return true
    end
    return false
end

"""
    _apriori_gen(prev_level) -> Vector{Vector{Int}}

Sinh candidates C_k từ L_{k-1}:
  - Join: ghép l1, l2 khi prefix (k-2) giống nhau
  - Prune: loại nếu có infrequent subset

# Độ phức tạp: O(|L_{k-1}|² × k)
"""
function _apriori_gen(prev_level::Vector{Vector{Int}})::Vector{Vector{Int}}
    candidates = Vector{Vector{Int}}()
    prev_set   = Set{Vector{Int}}(prev_level)
    n = length(prev_level)
    @inbounds for i in 1:(n-1)
        l1 = prev_level[i]
        @inbounds for j in (i+1):n
            l2 = prev_level[j]
            l1[1:end-1] != l2[1:end-1] && break
            c = vcat(l1, [l2[end]])
            !_has_infrequent_subset(c, prev_set) && push!(candidates, c)
        end
    end
    return candidates
end

# ══════════════════════════════════════════════════════════════════════════════
# ĐẾM SUPPORT – LINEAR SCAN (Basic)
# ══════════════════════════════════════════════════════════════════════════════

"""
    _count_support_basic!(Ck, D, counts)

Đếm support bằng linear scan: với mỗi transaction t, kiểm tra từng candidate.
# Độ phức tạp: O(|D| × |Ck| × k)
"""
function _issubset_sorted(sub::Vector{Int}, arr::Vector{Int})::Bool
    i, j = 1, 1
    m, n = length(sub), length(arr)
    while i <= m && j <= n
        @inbounds if sub[i] == arr[j]
            i += 1
            j += 1
        elseif sub[i] > arr[j]
            j += 1
        else
            return false
        end
    end
    return i > m
end

function _count_support_basic!(Ck::Vector{Vector{Int}},
                                 D::Vector{Vector{Int}},
                                 counts::Vector{Int})
    fill!(counts, 0)
    for t in D
        for ci in eachindex(Ck)
            if _issubset_sorted(Ck[ci], t)
                counts[ci] += 1
            end
        end
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# ĐẾM SUPPORT – HASH TREE (Optimized, paper gốc Section 2.1.2)
# ══════════════════════════════════════════════════════════════════════════════

"""
    _count_support_hashtree!(Ck, D, counts)

Đếm support dùng Hash Tree:
1. Xây Hash Tree từ Ck (một lần).
2. Với mỗi transaction t: tìm subsets trong hash tree → tăng count.

**Tại sao Hash Tree tốt hơn?**
Thay vì quét O(|Ck|) candidates cho mỗi t, hash tree chỉ duyệt
các nhánh tương ứng items trong t → trung bình O(|Ck|^α) với α < 1.

# Độ phức tạp: O(|D| × C(|t|,k)/num_buckets^depth) trung bình
"""
function _count_support_hashtree!(Ck::Vector{Vector{Int}},
                                    D::Vector{Vector{Int}},
                                    counts::Vector{Int})
    fill!(counts, 0)
    isempty(Ck) && return
    # Xây hash tree từ candidates
    ht = build_hash_tree(Ck; max_leaf_size=3, num_buckets=7)
    # Duyệt từng transaction
    @inbounds for t in D
        matched = find_subsets_in_transaction(ht, Ck, t)
        @inbounds for ci in unique!(matched)
            counts[ci] += 1
        end
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# APRIORI BASIC  (Level 1 – Linear scan)
# ══════════════════════════════════════════════════════════════════════════════

"""
    apriori_basic(D, minsup) -> Dict{Int, Vector{Itemset}}

Apriori cơ bản: đếm support bằng Hash Tree (theo thiết kế gốc để tối ưu).
"""
function apriori_basic(D::Vector{Vector{Int}}, minsup::Float64;
                       verbose::Bool=true)::Dict{Int, Vector{Itemset}}
    db_size      = length(D)
    minsup_count = ceil(Int, minsup * db_size)
    all_frequent = Dict{Int, Vector{Itemset}}()

    verbose && println("  ┌─ [Basic (HashTree)] DB=$db_size | minsup=$(round(minsup*100,digits=2))% (count=$minsup_count)")

    # L₁
    item_cnt = Dict{Int,Int}()
    for t in D, item in t
        item_cnt[item] = get(item_cnt, item, 0) + 1
    end
    L_prev = sort!([Itemset([item], cnt) for (item, cnt) in item_cnt if cnt >= minsup_count])
    isempty(L_prev) && (verbose && println("  └─ 0 FI"); return all_frequent)
    all_frequent[1] = L_prev
    verbose && println("  │  k=1 : $(length(L_prev)) FI")

    k = 2
    while !isempty(L_prev)
        Ck = _apriori_gen([is.itemset for is in L_prev])
        isempty(Ck) && break
        counts = zeros(Int, length(Ck))
        _count_support_basic!(Ck, D, counts)
        frequent_k = [Itemset(Ck[ci], counts[ci]) for ci in eachindex(Ck) if counts[ci] >= minsup_count]
        isempty(frequent_k) && break
        all_frequent[k] = frequent_k
        L_prev = frequent_k
        verbose && println("  │  k=$k : $(length(frequent_k)) FI ($(length(Ck)) cands)")
        k += 1
    end
    total = sum(length(v) for v in values(all_frequent))
    verbose && println("  └─ Total: $total FI")
    return all_frequent
end

# ══════════════════════════════════════════════════════════════════════════════
# APRIORI OPTIMIZED  (Level 3 – Hash Tree, paper gốc)
# ══════════════════════════════════════════════════════════════════════════════

"""
    apriori_optimized(D, minsup) -> Dict{Int, Vector{Itemset}}

Apriori tối ưu: đếm support dùng **Hash Tree** (Agrawal & Srikant 1994).

Candidates Ck được lưu trong hash tree. Khi quét mỗi transaction t,
thay vì kiểm tra tuần tự tất cả |Ck| candidates, hash tree chỉ duyệt
các nhánh tương ứng → giảm đáng kể số lần so sánh.
"""
function apriori_optimized(D::Vector{Vector{Int}}, minsup::Float64;
                           verbose::Bool=true)::Dict{Int, Vector{Itemset}}
    db_size      = length(D)
    minsup_count = ceil(Int, minsup * db_size)
    all_frequent = Dict{Int, Vector{Itemset}}()

    verbose && println("  ┌─ [HashTree] DB=$db_size | minsup=$(round(minsup*100,digits=2))% (count=$minsup_count)")

    # L₁
    item_cnt = Dict{Int,Int}()
    for t in D, item in t
        item_cnt[item] = get(item_cnt, item, 0) + 1
    end
    L_prev = sort!([Itemset([item], cnt) for (item, cnt) in item_cnt if cnt >= minsup_count])
    isempty(L_prev) && (verbose && println("  └─ 0 FI"); return all_frequent)
    all_frequent[1] = L_prev
    verbose && println("  │  k=1 : $(length(L_prev)) FI")

    k = 2
    while !isempty(L_prev)
        Ck = _apriori_gen([is.itemset for is in L_prev])
        isempty(Ck) && break
        counts = zeros(Int, length(Ck))
        _count_support_hashtree!(Ck, D, counts)   # ← HASH TREE
        frequent_k = [Itemset(Ck[ci], counts[ci]) for ci in eachindex(Ck) if counts[ci] >= minsup_count]
        isempty(frequent_k) && break
        all_frequent[k] = frequent_k
        L_prev = frequent_k
        verbose && println("  │  k=$k : $(length(frequent_k)) FI ($(length(Ck)) cands)")
        k += 1
    end
    total = sum(length(v) for v in values(all_frequent))
    verbose && println("  └─ Total: $total FI")
    return all_frequent
end

# ══════════════════════════════════════════════════════════════════════════════
# SINH LUẬT KẾT HỢP
# ══════════════════════════════════════════════════════════════════════════════

"""
    generate_association_rules(frequent, db_size, minconf) -> Vector{AssociationRule}

Sinh luật X ⟹ Y từ frequent itemsets. Sắp xếp giảm dần theo lift.
"""
function generate_association_rules(frequent::Dict{Int,Vector{Itemset}},
                                     db_size::Int,
                                     minconf::Float64)::Vector{AssociationRule}
    sup_map = Dict{Vector{Int}, Float64}()
    for (_, items) in frequent
        for is in items
            sup_map[is.itemset] = relative_support(is.count, db_size)
        end
    end

    rules = AssociationRule[]
    for (k, items) in frequent
        k < 2 && continue
        for is in items
            Z = is.itemset; sup_Z = sup_map[Z]
            for mask in 1:(1 << k - 2)
                X = [Z[i] for i in 1:k if (mask >> (i-1)) & 1 == 1]
                Y = [Z[i] for i in 1:k if (mask >> (i-1)) & 1 == 0]
                (isempty(X) || isempty(Y)) && continue
                haskey(sup_map, X) || continue
                sup_X = sup_map[X]; sup_X == 0.0 && continue
                conf = sup_Z / sup_X
                conf >= minconf || continue
                haskey(sup_map, Y) || continue
                sup_Y = sup_map[Y]
                lift = sup_Y == 0.0 ? 0.0 : conf / sup_Y
                push!(rules, AssociationRule(X, Y, sup_Z, conf, lift))
            end
        end
    end
    sort!(rules, rev=true)
    return rules
end

# ══════════════════════════════════════════════════════════════════════════════
# GIAO DIỆN I/O  (Level 4)
# ══════════════════════════════════════════════════════════════════════════════

"""
    run_apriori(input_path, minsup; output_path, optimized, verbose)

Đọc file SPMF → chạy Apriori → lưu kết quả.
"""
function run_apriori(input_path::String, minsup::Float64;
                     output_path::String="", optimized::Bool=false,
                     verbose::Bool=true)::Dict{Int, Vector{Itemset}}
    verbose && println("\n" * "═"^55)
    verbose && println("  Apriori FIM | $(optimized ? "Hash Tree" : "Basic")")
    verbose && println("  Input  : $input_path")
    verbose && println("  Minsup : $(round(minsup*100,digits=3))%")
    verbose && println("═"^55)

    D = load_database(input_path)
    verbose && println("  Loaded $(length(D)) transactions\n")

    algo = optimized ? apriori_optimized : apriori_basic
    result, elapsed_ms = measure_time(algo, D, minsup)
    verbose && @printf("\n  ⏱  Time: %.2f ms\n", elapsed_ms)

    if !isempty(output_path)
        save_results(output_path, result, length(D))
    end
    return result
end

# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("Usage: julia apriori.jl <input> <minsup> [output] [--opt]")
        exit(1)
    end
    inp  = ARGS[1]
    ms   = parse(Float64, ARGS[2])
    outp = length(ARGS) >= 3 && !startswith(ARGS[3], "--") ? ARGS[3] : ""
    opt  = "--opt" ∈ ARGS
    run_apriori(inp, ms; output_path=outp, optimized=opt)
end