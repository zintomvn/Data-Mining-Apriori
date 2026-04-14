# ============================================================
#  Struct đại diện cho một phần tử trong L_k / C_k
#  Tương ứng với pseudocode: mỗi c có c.itemset và c.count
# ============================================================
mutable struct Itemset
    itemset :: Vector{Int}
    count   :: Int
end

# Constructors tiện lợi
Itemset(items::Vector{Int}) = Itemset(items, 0)

# Cho phép dùng Itemset trong Set / Dict (hash theo itemset)
Base.hash(x::Itemset, h::UInt)  = hash(x.itemset, h)
Base.:(==)(a::Itemset, b::Itemset) = a.itemset == b.itemset
Base.show(io::IO, x::Itemset)   = print(io, "$(x.itemset)  #SUP=$(x.count)")


# ============================================================
#  has_infrequent_subset(c, L_{k-1})
#  Trả về TRUE nếu có (k-1)-subset của c.itemset không nằm trong L_{k-1}
# ============================================================
function has_infrequent_subset(c::Itemset, prev_level_set::Set{Vector{Int}})
    items = c.itemset
    k     = length(items)
    for pos in 1:k
        subset = vcat(items[1:pos-1], items[pos+1:end])
        if !(subset in prev_level_set)
            return true
        end
    end
    return false
end


# ============================================================
#  apriori_gen(L_{k-1}) → C_k
#  Join step + Prune step
#  Nhận vào Vector{Itemset}, trả về Vector{Itemset} (count = 0)
# ============================================================
function apriori_gen(prev_level::Vector{Itemset})
    candidates = Vector{Itemset}()

    # Set các itemset (Vector{Int}) để kiểm tra nhanh trong prune step
    prev_level_set = Set(p.itemset for p in prev_level)

    n = length(prev_level)

    for i in 1:n
        l1 = prev_level[i].itemset
        for j in (i+1):n
            l2 = prev_level[j].itemset

            # Join condition: k-1 phần tử đầu giống nhau
            if l1[1:end-1] != l2[1:end-1]
                break   # đã sort → không cần xét j tiếp theo
            end

            # c = l1 ⋈ l2
            c = Itemset(vcat(l1, [l2[end]]))   # count = 0 mặc định

            # Prune step: xóa c nếu có infrequent subset
            if !has_infrequent_subset(c, prev_level_set)
                push!(candidates, c)
            end
        end
    end

    return candidates
end


# ============================================================
#  subset(C_k, t): tìm các Itemset trong C_k là subset của t
#  Trả về Vector{Itemset} (tham chiếu, để c.count++ hoạt động)
# ============================================================
function subset(Ck::Vector{Itemset}, transaction::Vector{Int})
    t_set  = Set(transaction)
    result = Vector{Itemset}()
    for c in Ck
        if all(item in t_set for item in c.itemset)
            push!(result, c)
        end
    end
    return result
end


# ============================================================
#  find_frequent_1_itemsets(D, minsup_count) → L_1
#  Trả về Vector{Itemset} đã có count, sắp xếp lexical
# ============================================================
function find_frequent_1_itemsets(D::Vector{Vector{Int}}, minsup_count::Int)
    item_count = Dict{Int,Int}()
    for t in D
        for item in t
            item_count[item] = get(item_count, item, 0) + 1
        end
    end

    L1 = [Itemset([item], count)
          for (item, count) in item_count
          if count >= minsup_count]

    sort!(L1, by = x -> x.itemset)   # lexical order — bắt buộc cho join step
    return L1
end


# ============================================================
#  apriori(D, minsup) → L
# ============================================================
function apriori(D::Vector{Vector{Int}}, minsup::Float64)
    db_size      = length(D)
    minsup_count = ceil(Int, minsup * db_size)

    println("Database size    : $db_size transactions")
    println("Min support count: $minsup_count\n")

    # L[k] = Vector{Itemset} chứa các frequent k-itemsets
    L = Dict{Int, Vector{Itemset}}()

    # (1) L_1
    L[1] = find_frequent_1_itemsets(D, minsup_count)
    println("k=1 : $(length(L[1])) frequent itemsets")

    # (2) for (k=2; L_{k-1} ≠ ∅; k++)
    k = 2
    while !isempty(get(L, k-1, Itemset[]))

        # (3) C_k = apriori_gen(L_{k-1})
        Ck = apriori_gen(L[k-1])
        isempty(Ck) && break

        # (4) for each transaction t ∈ D
        for t in D
            # (5) C_t = subset(C_k, t)
            Ct = subset(Ck, t)

            # (6-7) for each c ∈ C_t: c.count++
            for c in Ct
                c.count += 1
            end
        end

        # (9) L_k = {c ∈ C_k | c.count ≥ min_sup}
        L[k] = filter(c -> c.count >= minsup_count, Ck)

        println("k=$k : $(length(L[k])) frequent itemsets  ($(length(Ck)) candidates)")

        k += 1
    end

    # (11) return L = ∪_k L_k
    return L
end


# ============================================================
#  Test
# ============================================================
D = [
    [1, 2, 3, 4],
    [1, 2, 4],
    [1, 2],
    [2, 3, 4],
    [2, 3],
    [3, 4],
    [2, 4],
]

min_sup = 0.5
L = apriori(D, min_sup)

for k in sort(collect(keys(L)))
    println("\nFrequent $k-itemsets:")
    for itemset in L[k]
        println("  $itemset")
    end
end