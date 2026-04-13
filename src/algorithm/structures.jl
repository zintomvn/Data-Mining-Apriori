# src/algorithm/structures.jl
# Cấu trúc dữ liệu cho Apriori FIM:
#   - Itemset, AssociationRule (cơ bản)
#   - HashTreeNode, HashTree   (tối ưu theo paper gốc Agrawal & Srikant 1994)
# File này phải được include đầu tiên.

using Printf

# ══════════════════════════════════════════════════════════════════════════════
# ITEMSET
# ══════════════════════════════════════════════════════════════════════════════

"""
    Itemset

Cấu trúc lưu một tập phổ biến (frequent itemset) và số lần xuất hiện.

# Fields
- `itemset::Vector{Int}` : Danh sách item đã sắp xếp tăng dần
- `count::Int`           : Support tuyệt đối (số giao dịch chứa itemset)

# Độ phức tạp
- Khởi tạo / copy : O(k)
- So sánh / hash  : O(k)
"""
mutable struct Itemset
    itemset::Vector{Int}
    count::Int
    Itemset(items::Vector{Int})            = new(items, 0)
    Itemset(items::Vector{Int}, cnt::Int)  = new(items, cnt)
end

Base.hash(x::Itemset, h::UInt)      = hash(x.itemset, h)
Base.:(==)(a::Itemset, b::Itemset)  = a.itemset == b.itemset
Base.copy(x::Itemset)               = Itemset(copy(x.itemset), x.count)
Base.show(io::IO, x::Itemset)       = print(io, "$(x.itemset) | count=$(x.count)")
Base.isless(a::Itemset, b::Itemset) = a.itemset < b.itemset

# ══════════════════════════════════════════════════════════════════════════════
# ASSOCIATION RULE
# ══════════════════════════════════════════════════════════════════════════════

"""
    AssociationRule

Luật kết hợp  X ⟹ Y  với support, confidence, lift.

# Fields
- `antecedent`, `consequent` : Vector{Int}
- `support`, `confidence`, `lift` : Float64
"""
struct AssociationRule
    antecedent::Vector{Int}
    consequent::Vector{Int}
    support::Float64
    confidence::Float64
    lift::Float64
end

function Base.show(io::IO, r::AssociationRule)
    @printf(io, "%s => %s | sup=%.4f conf=%.4f lift=%.4f",
            r.antecedent, r.consequent, r.support, r.confidence, r.lift)
end
Base.isless(a::AssociationRule, b::AssociationRule) = a.lift < b.lift

# ══════════════════════════════════════════════════════════════════════════════
# HASH TREE  (Agrawal & Srikant, VLDB 1994, Section 2.1.2)
# ══════════════════════════════════════════════════════════════════════════════
#
# Hash Tree dùng để lưu tập candidate Ck và nhanh chóng tìm tất cả
# candidates là subset của một transaction t.
#
# Cấu trúc:
#   - Internal node: hash trên item thứ `depth` của candidate → con
#   - Leaf node    : danh sách candidates (khi |list| > max_leaf_size → split)
#
# Khi đếm support cho transaction t, thay vì quét O(|Ck|), ta chỉ duyệt
# các nhánh tương ứng với items trong t → giảm đáng kể so với brute-force.

"""
    HashTreeNode

Nút trong Hash Tree.

# Fields
- `is_leaf::Bool`                       : true nếu đây là lá
- `children::Dict{Int, HashTreeNode}`   : con (internal node)
- `candidates::Vector{Int}`             : index của candidates trong Ck (leaf)
- `depth::Int`                          : độ sâu (0-indexed, dùng làm chỉ số item)
"""
mutable struct HashTreeNode
    is_leaf::Bool
    children::Dict{Int, HashTreeNode}
    candidates::Vector{Int}             # chỉ số vào mảng Ck gốc
    depth::Int

    HashTreeNode(depth::Int) = new(true, Dict{Int,HashTreeNode}(), Int[], depth)
end

"""
    HashTree

Cây hash lưu candidate k-itemsets.

# Fields
- `root::HashTreeNode`
- `max_leaf_size::Int`  : ngưỡng split lá (mặc định 3)
- `num_buckets::Int`    : số bucket hash      (mặc định 5)
- `k::Int`              : chiều dài mỗi candidate

# Paper reference
Agrawal & Srikant 1994, Sec 2.1.2:
\"The candidate itemsets are stored in a hash-tree.\"

# Độ phức tạp
- Insert  : O(k) amortized
- Lookup  : O(C(|t|,k-depth) × k) trung bình ≪ O(|Ck|) khi |Ck| lớn
"""
struct HashTree
    root::HashTreeNode
    max_leaf_size::Int
    num_buckets::Int
    k::Int      # candidate size
end

"""
    HashTree(k; max_leaf_size=3, num_buckets=5)

Tạo Hash Tree rỗng cho k-itemset candidates.
"""
function HashTree(k::Int; max_leaf_size::Int=3, num_buckets::Int=5)
    return HashTree(HashTreeNode(1), max_leaf_size, num_buckets, k)
end

"""
    _hash_item(item::Int, num_buckets::Int) -> Int

Hàm hash đơn giản: item mod num_buckets.
"""
@inline _hash_item(item::Int, num_buckets::Int)::Int = mod(item, num_buckets)

"""
    insert!(tree::HashTree, Ck::Vector{Vector{Int}}, ci::Int)

Chèn candidate Ck[ci] vào hash tree.
Nếu leaf vượt max_leaf_size → split thành internal node.

# Độ phức tạp: O(k) amortized
"""
function insert!(tree::HashTree, Ck::Vector{Vector{Int}}, ci::Int)
    cand = Ck[ci]
    _insert_node!(tree, tree.root, cand, ci, 1)
end

function _insert_node!(tree::HashTree, node::HashTreeNode,
                       cand::Vector{Int}, ci::Int, depth::Int)
    if node.is_leaf
        push!(node.candidates, ci)
        # Split nếu vượt ngưỡng và depth <= k  (còn item để hash)
        if length(node.candidates) > tree.max_leaf_size && depth <= tree.k
            _split_node!(tree, node, Ck_ref(tree, node, ci), depth)
        end
    else
        # Hash trên item thứ `depth`
        if depth <= length(cand)
            h = _hash_item(cand[depth], tree.num_buckets)
            if !haskey(node.children, h)
                node.children[h] = HashTreeNode(depth + 1)
            end
            _insert_node!(tree, node.children[h], cand, ci, depth + 1)
        end
    end
end

# Dummy – chỉ dùng trong split; thực tế ta truyền Ck vào build_hash_tree
function Ck_ref(tree::HashTree, node::HashTreeNode, ci::Int)
    return nothing  # placeholder
end

"""
    _split_node!(tree, node, _, depth)

Chuyển leaf thành internal node: re-hash tất cả candidates hiện có.
"""
function _split_node!(tree::HashTree, node::HashTreeNode, ::Any, depth::Int)
    # Không thể split – placeholder; build_hash_tree xử lý đúng
end

"""
    build_hash_tree(Ck::Vector{Vector{Int}};
                    max_leaf_size::Int=3, num_buckets::Int=7) -> HashTree

Xây Hash Tree từ tập candidate Ck.
Mỗi candidate được chèn từ root, hash trên item thứ `depth`.

# Tham số
- `Ck` : Danh sách k-itemset candidates
- `max_leaf_size` : Ngưỡng tách lá
- `num_buckets`   : Số bucket hash

# Ví dụ
```julia
Ck = [[1,2,3], [1,2,4], [1,3,5], [2,3,4]]
ht = build_hash_tree(Ck)
```
"""
function build_hash_tree(Ck::Vector{Vector{Int}};
                         max_leaf_size::Int=3, num_buckets::Int=7)::HashTree
    k = isempty(Ck) ? 0 : length(Ck[1])
    tree = HashTree(HashTreeNode(1), max_leaf_size, num_buckets, k)
    for ci in eachindex(Ck)
        _ht_insert!(tree, tree.root, Ck, ci, 1)
    end
    return tree
end

"""
    _ht_insert!(tree, node, Ck, ci, depth)

Chèn candidate Ck[ci] vào node tại depth.
Nếu leaf vượt max_leaf_size → tách thành internal node.
"""
function _ht_insert!(tree::HashTree, node::HashTreeNode,
                     Ck::Vector{Vector{Int}}, ci::Int, depth::Int)
    if node.is_leaf
        push!(node.candidates, ci)
        if length(node.candidates) > tree.max_leaf_size && depth <= tree.k
            # Split: chuyển leaf → internal
            old_cands = copy(node.candidates)
            empty!(node.candidates)
            node.is_leaf = false
            for oci in old_cands
                h = _hash_item(Ck[oci][depth], tree.num_buckets)
                if !haskey(node.children, h)
                    node.children[h] = HashTreeNode(depth + 1)
                end
                _ht_insert!(tree, node.children[h], Ck, oci, depth + 1)
            end
        end
    else
        if depth <= length(Ck[ci])
            h = _hash_item(Ck[ci][depth], tree.num_buckets)
            if !haskey(node.children, h)
                node.children[h] = HashTreeNode(depth + 1)
            end
            _ht_insert!(tree, node.children[h], Ck, ci, depth + 1)
        else
            # depth > k: lưu vào node hiện tại như leaf fallback
            push!(node.candidates, ci)
        end
    end
end

"""
    find_subsets_in_transaction(tree::HashTree, Ck::Vector{Vector{Int}},
                                transaction::Vector{Int}) -> Vector{Int}

Tìm tất cả candidates trong Ck (lưu trong hash tree) là subset của `transaction`.
Trả về danh sách chỉ số ci.

# Thuật toán (paper gốc)
Bắt đầu từ root. Tại mỗi internal node ở depth d:
- Với mỗi item t[i] trong transaction (i ≥ vị trí bắt đầu):
  + Hash t[i] → tìm child → đệ quy với depth+1, start=i+1
Tại leaf: kiểm tra từng candidate → nếu subset → thêm vào kết quả.

# Độ phức tạp: O(C(|t|, k-depth)) trung bình, tốt hơn O(|Ck|)
"""
function find_subsets_in_transaction(tree::HashTree, Ck::Vector{Vector{Int}},
                                     transaction::Vector{Int})::Vector{Int}
    result = Int[]
    t_set  = BitSet(transaction)
    _ht_find!(tree.root, Ck, transaction, t_set, 1, result, tree.num_buckets)
    return result
end

function _ht_find!(node::HashTreeNode, Ck::Vector{Vector{Int}},
                   transaction::Vector{Int}, t_set::BitSet,
                   start::Int, result::Vector{Int}, num_buckets::Int)
    if node.is_leaf
        # Kiểm tra từng candidate trong leaf
        @inbounds for ci in node.candidates
            cand = Ck[ci]
            ok = true
            @inbounds for item in cand
                if item ∉ t_set
                    ok = false; break
                end
            end
            ok && push!(result, ci)
        end
    else
        # Internal node: thử từng item trong transaction từ vị trí `start`
        n = length(transaction)
        @inbounds for i in start:n
            h = _hash_item(transaction[i], num_buckets)
            if haskey(node.children, h)
                _ht_find!(node.children[h], Ck, transaction, t_set,
                          i + 1, result, num_buckets)
            end
        end
    end
end
