# src/algorithm/utils.jl
# Các hàm tiện ích: I/O, đo lường hiệu năng, sinh dữ liệu tổng hợp
# Yêu cầu: structures.jl phải được include trước file này

using Random
using Printf

# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────

"""
    load_database(filepath::String) -> Vector{Vector{Int}}

Đọc CSDL giao dịch từ file theo định dạng SPMF:
- Mỗi dòng là một giao dịch; các item cách nhau bằng dấu cách.
- Dòng trống và dòng bắt đầu bằng `#`, `%`, `@` được bỏ qua.
- Items trong mỗi transaction được **sắp xếp tăng dần** (bắt buộc cho Apriori).

# Tham số
- `filepath` : Đường dẫn file định dạng SPMF (`.txt`)

# Giá trị trả về
- `Vector{Vector{Int}}` : Danh sách các giao dịch đã sắp xếp

# Ví dụ
```julia
D = load_database("data/benchmark/chess.txt")
println(length(D))   # 3196
```
"""
function load_database(filepath::String)::Vector{Vector{Int}}
    db = Vector{Vector{Int}}()
    open(filepath, "r") do f
        for line in eachline(f)
            s = strip(line)
            (isempty(s) || s[1] ∈ ('#', '%', '@')) && continue
            push!(db, sort!(parse.(Int, split(s))))
        end
    end
    return db
end


"""
    save_results(filepath::String,
                 results::Dict{Int,Vector{Itemset}},
                 db_size::Int) -> Nothing

Lưu frequent itemsets ra file theo định dạng SPMF:
`item1 item2 ... #SUP: count`

# Tham số
- `filepath`  : Đường dẫn file xuất
- `results`   : Dict  k → [Itemset...]  từ `apriori_basic` / `apriori_optimized`
- `db_size`   : Tổng số giao dịch (để tính support tương đối khi cần)
"""
function save_results(filepath::String,
                      results::Dict{Int,Vector{Itemset}},
                      db_size::Int)::Nothing
    mkpath(dirname(filepath) == "" ? "." : dirname(filepath))
    open(filepath, "w") do f
        for k in sort(collect(keys(results)))
            for is in sort(results[k])
                println(f, join(is.itemset, " ") * " #SUP: $(is.count)")
            end
        end
    end
    total = sum(length(v) for v in values(results))
    println("✓ Đã lưu $total frequent itemsets → $filepath")
    return nothing
end


# ──────────────────────────────────────────────────────────────────────────────
# Đo lường hiệu năng
# ──────────────────────────────────────────────────────────────────────────────

"""
    measure_time(f::Function, args...) -> Tuple{Any, Float64}

Đo thời gian chạy của `f(args...)`.

# Giá trị trả về
- `(result, elapsed_ms)` : kết quả và thời gian chạy (mili giây)
"""
function measure_time(f::Function, args...)
    t0     = time_ns()
    result = f(args...)
    return result, (time_ns() - t0) / 1_000_000.0
end


"""
    measure_memory_mb(f::Function, args...) -> Tuple{Any, Float64}

Đo tổng bộ nhớ được cấp phát (allocated) khi gọi `f(args...)`.

Dùng macro `@allocated` để lấy bytes. Đây là **tổng bytes cấp phát**
(không phải peak live memory), nhưng đủ để so sánh giữa các cài đặt.

# Giá trị trả về
- `(result, mb)` : kết quả và số MB đã cấp phát
"""
function measure_memory_mb(f::Function, args...)
    GC.gc(); GC.gc()
    alloc  = @allocated result = f(args...)
    return result, alloc / 1024 / 1024
end


# ──────────────────────────────────────────────────────────────────────────────
# Kiểm thử
# ──────────────────────────────────────────────────────────────────────────────

"""
    count_all_frequent_bruteforce(D::Vector{Vector{Int}}, minsup::Float64) -> Int

Đếm tổng số frequent itemsets bằng brute-force (liệt kê 2^|I| tập con).
**Chỉ dùng cho toy databases nhỏ** trong unit tests.

# Độ phức tạp: O(2^|I| × |D|)
"""
function count_all_frequent_bruteforce(D::Vector{Vector{Int}}, minsup::Float64)::Int
    all_items = sort(unique(vcat(D...)))
    n_items   = length(all_items)
    n_db      = length(D)
    min_cnt   = ceil(Int, minsup * n_db)
    count     = 0
    T_sets    = [Set{Int}(t) for t in D]          # BitSet cho mỗi transaction

    for mask in 1:(1 << n_items - 1)
        itemset = @inbounds [all_items[i] for i in 1:n_items if (mask >> (i-1)) & 1 == 1]
        sup = sum(all(item ∈ ts for item in itemset) for ts in T_sets)
        sup >= min_cnt && (count += 1)
    end
    return count
end


"""
    flatten_results(results::Dict{Int,Vector{Itemset}}) -> Dict{Vector{Int},Int}

Chuyển đổi  Dict{k → [Itemset...]}  sang  Dict{itemset → count}  phẳng.
"""
function flatten_results(results::Dict{Int,Vector{Itemset}})::Dict{Vector{Int},Int}
    out = Dict{Vector{Int},Int}()
    for (_, items) in results
        for is in items
            out[is.itemset] = is.count
        end
    end
    return out
end


# ──────────────────────────────────────────────────────────────────────────────
# Sinh dữ liệu
# ──────────────────────────────────────────────────────────────────────────────

"""
    create_subset_db(db::Vector{Vector{Int}}, pct::Float64; seed::Int=42)
                    -> Vector{Vector{Int}}

Tạo tập con gồm `pct × 100%` giao dịch đầu tiên của `db` (giữ thứ tự tự nhiên).

# Tham số
- `pct`  : Tỉ lệ giao dịch ∈ (0, 1]
- `seed` : Random seed (đảm bảo tái sản xuất)
"""
function create_subset_db(db::Vector{Vector{Int}}, pct::Float64; seed::Int=42)::Vector{Vector{Int}}
    n = max(1, round(Int, length(db) * pct))
    return db[1:n]
end


"""
    generate_synthetic_db(n_trans::Int, n_items::Int, avg_len::Int; seed::Int=42)
                          -> Vector{Vector{Int}}

Sinh CSDL tổng hợp với độ dài giao dịch trung bình `avg_len`.
Dùng phân phối Gaussian quanh avg_len (σ=2), cắt về [1, n_items].
Kết quả có tính **tái sản xuất** nhờ seed cố định.

# Dùng cho Thực nghiệm F (ảnh hưởng độ dài giao dịch)
"""
function generate_synthetic_db(n_trans::Int, n_items::Int, avg_len::Int;
                                seed::Int=42)::Vector{Vector{Int}}
    rng = MersenneTwister(seed)
    db  = Vector{Vector{Int}}(undef, n_trans)
    for i in 1:n_trans
        len = clamp(round(Int, avg_len + 2 * randn(rng)), 1, n_items)
        db[i] = sort!(unique(rand(rng, 1:n_items, 2 * len))[1:min(len, n_items)])
    end
    return db
end


"""
    relative_support(count::Int, db_size::Int) -> Float64

Tính support tương đối: count / db_size ∈ [0, 1].
"""
@inline relative_support(count::Int, db_size::Int)::Float64 = count / db_size
