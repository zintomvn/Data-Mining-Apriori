
# procedure has_infrequent_subset(c, L_{k-1})
# Trả về TRUE nếu có subset (k-1) của c KHÔNG nằm trong L_{k-1}
function has_infrequent_subset(candidate::Vector{Int}, prev_level::Set{Vector{Int}})
    k = length(candidate)
    for pos in 1:k
        # Tạo subset bằng cách bỏ phần tử tại vị trí pos
        subset = vcat(candidate[1:pos-1], candidate[pos+1:end])
        if !(subset in prev_level)
            return true   # có infrequent subset → prune
        end
    end
    return false
end


# procedure apriori_gen(L_{k-1})
# Join step + Prune step → trả về C_k
function apriori_gen(prev_level::Vector{Vector{Int}})
    candidates = Vector{Vector{Int}}()

    # Dùng Set để kiểm tra nhanh trong prune step
    prev_level_set = Set(prev_level)

    n = length(prev_level)

    # --- Join step ---
    # Ghép l1 và l2 nếu k-1 phần tử đầu giống nhau, l1[k-1] < l2[k-1]
    for i in 1:n
        l1 = prev_level[i]
        for j in (i+1):n
            l2 = prev_level[j]

            # Kiểm tra k-1 phần tử đầu phải giống nhau
            if l1[1:end-1] != l2[1:end-1]
                break  # vì đã sort, không cần xét tiếp j
            end

            # l1[k-1] < l2[k-1] (tự động đúng vì i < j và đã sort)
            c = vcat(l1, [l2[end]])  # join: ghép thêm phần tử cuối của l2

            # --- Prune step ---
            # Xóa c nếu có subset (k-1) nào không frequent
            if !has_infrequent_subset(c, prev_level_set)
                push!(candidates, c)
            end
        end
    end

    return candidates
end


# Tìm các candidates trong C_k là subset của transaction t
# subset(C_k, t) trong pseudocode
function get_subsets_in_transaction(candidates::Vector{Vector{Int}}, transaction::Vector{Int})
    result = Vector{Int}()  # trả về index của candidates khớp
    t_set = Set(transaction)
    for (idx, c) in enumerate(candidates)
        if all(item in t_set for item in c)
            push!(result, idx)
        end
    end
    return result
end

function find_frequent_1_itemsets(database, minsup_count::Int)
    item_count = Dict{Int, Int}()
    for transaction in database
        for item in transaction
            item_count[item] = get(item_count, item, 0) + 1
        end
    end

    frequent1 = [[item] for (item, count) in item_count if count >= minsup_count]
    sort!(frequent1)   # lexical order — bắt buộc!
    return frequent1
end

function apriori(database::Vector{Vector{Int}}, minsup::Float64)
    db_size    = length(database)
    minsup_count = ceil(Int, minsup * db_size)

    println("Database size    : $db_size transactions")
    println("Min support count: $minsup_count")

    L = Dict{Int, Vector{Vector{Int}}}()          # L[k] = frequent k-itemsets
    support = Dict{Vector{Int}, Int}()            # lưu support của từng itemset

    # (1) L_1
    L[1] = find_frequent_1_itemsets(database, minsup_count)
    for itemset in L[1]
        support[itemset] = sum(itemset[1] in t for t in database)
    end

    println("k=1 : $(length(L[1])) frequent itemsets")

    # (2) for (k=2; L_{k-1} ≠ ∅; k++)
    k = 2
    while !isempty(get(L, k-1, []))

        # (3) C_k = apriori_gen(L_{k-1})
        Ck = apriori_gen(L[k-1])

        if isempty(Ck)
            break
        end

        # Khởi tạo đếm support = 0 cho mọi candidate
        count = Dict(c => 0 for c in Ck)

        # (4) for each transaction t ∈ D
        for t in database
            # (5) C_t = subset(C_k, t)
            matched_indices = get_subsets_in_transaction(Ck, t)
            # (6-7) for each c ∈ C_t: c.count++
            for idx in matched_indices
                count[Ck[idx]] += 1
            end
        end

        # (9) L_k = {c ∈ C_k | c.count ≥ min_sup}
        L[k] = [c for c in Ck if count[c] >= minsup_count]
        for c in L[k]
            support[c] = count[c]
        end

        println("k=$k : $(length(L[k])) frequent itemsets  ($(length(Ck)) candidates)")

        k += 1
    end

    # (11) return L = ∪_k L_k
    return L, support
end

database = [
    [1, 2, 3, 4],
    [1, 2, 4],
    [1, 2],
    [2, 3, 4],
    [2, 3],
    [3, 4],
    [2, 4],
]

min_sup = 0.5
L, support = apriori(database, min_sup)

println("\n=== KẾT QUẢ ===")
for k in sort(collect(keys(L)))
    println("\nFrequent $k-itemsets:")
    for itemset in L[k]
        println("  $itemset  #SUP: $(support[itemset])")
    end
end