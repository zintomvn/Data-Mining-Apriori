# src/algorithm/structures.jl
# Các cấu trúc dữ liệu chính cho thuật toán Apriori FIM
# File này phải được include đầu tiên

using Printf

"""
    Itemset

Cấu trúc lưu một tập phổ biến (frequent itemset) và số lần xuất hiện.

# Fields
- `itemset::Vector{Int}` : Danh sách item đã sắp xếp tăng dần (lexicographic order)
- `count::Int`           : Support tuyệt đối (số giao dịch chứa itemset)

# Constructors
- `Itemset(items)`       : count khởi tạo = 0
- `Itemset(items, count)`: count cho trước

# Ghi chú về độ phức tạp
- Khởi tạo  : O(1)
- So sánh   : O(k)  với k = |itemset|
- Hash       : O(k)

# Ví dụ
```julia
is = Itemset([1, 2, 3], 5)   # {1,2,3} với count=5
println(is)                   # [1, 2, 3] | count=5
```
"""
mutable struct Itemset
    itemset::Vector{Int}
    count::Int

    Itemset(items::Vector{Int})            = new(items, 0)
    Itemset(items::Vector{Int}, cnt::Int)  = new(items, cnt)
end

Base.hash(x::Itemset, h::UInt)          = hash(x.itemset, h)
Base.:(==)(a::Itemset, b::Itemset)      = a.itemset == b.itemset
Base.copy(x::Itemset)                   = Itemset(copy(x.itemset), x.count)
Base.show(io::IO, x::Itemset)           = print(io, "$(x.itemset) | count=$(x.count)")
Base.isless(a::Itemset, b::Itemset)     = a.itemset < b.itemset


"""
    AssociationRule

Luật kết hợp  X ⟹ Y  cùng các chỉ số đo lường.

# Fields
- `antecedent::Vector{Int}` : Vế trái  X
- `consequent::Vector{Int}` : Vế phải  Y
- `support::Float64`        : sup(X ∪ Y)  (tương đối)
- `confidence::Float64`     : conf = sup(X∪Y) / sup(X)
- `lift::Float64`           : lift = conf / sup(Y)

# Ví dụ
```julia
r = AssociationRule([1,2], [3], 0.4, 0.8, 2.0)
println(r)   # [1, 2] => [3] | sup=0.4000 conf=0.8000 lift=2.0000
```
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
