# src/algorithm/utils.jl
# Hàm tiện ích: I/O, đo lường, SPMF integration, CSV, sinh dữ liệu
# Yêu cầu: structures.jl phải include trước

using Random, Printf

# ══════════════════════════════════════════════════════════════════════════════
# I/O  –  SPMF FORMAT
# ══════════════════════════════════════════════════════════════════════════════

"""
    load_database(filepath::String) -> Vector{Vector{Int}}

Đọc CSDL giao dịch từ file SPMF. Items sort tăng dần.
Bỏ qua dòng trống / comment (#, %, @).
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
    save_results(filepath, results, db_size) -> Nothing

Lưu frequent itemsets ra file SPMF: `item1 item2 ... #SUP: count`
"""
function save_results(filepath::String,
                      results::Dict{Int,Vector{Itemset}},
                      db_size::Int)::Nothing
    d = dirname(filepath)
    !isempty(d) && mkpath(d)
    open(filepath, "w") do f
        for k in sort(collect(keys(results)))
            for is in sort(results[k])
                println(f, join(is.itemset, " ") * " #SUP: $(is.count)")
            end
        end
    end
    total = sum(length(v) for v in values(results))
    println("  ✓ Saved $total frequent itemsets → $filepath")
    return nothing
end

# ══════════════════════════════════════════════════════════════════════════════
# CSV  –  Ghi file CSV thủ công (không cần package)
# ══════════════════════════════════════════════════════════════════════════════

"""
    write_csv(filepath::String, headers::Vector{String},
              rows::Vector{<:AbstractVector}) -> Nothing

Ghi dữ liệu ra file CSV (dấu phẩy phân cách, UTF-8).

# Tham số
- `filepath` : đường dẫn file .csv
- `headers`  : tên cột
- `rows`     : mỗi phần tử là 1 Vector tương ứng 1 dòng

# Ví dụ
```julia
write_csv("out.csv", ["A","B"], [[1, "x"], [2, "y"]])
```
"""
function write_csv(filepath::String, headers::Vector{String},
                   rows::Vector)::Nothing
    d = dirname(filepath)
    !isempty(d) && mkpath(d)
    open(filepath, "w") do f
        println(f, join(headers, ","))
        for row in rows
            println(f, join(string.(row), ","))
        end
    end
    println("  ✓ CSV → $filepath  ($(length(rows)) rows)")
    return nothing
end

# ══════════════════════════════════════════════════════════════════════════════
# SPMF INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

"""
    run_spmf(jar_path, input, output, minsup_pct; algo="Apriori") -> Float64

Chạy SPMF Apriori qua command line.
`minsup_pct` là phần trăm (0-100), ví dụ 50 cho 50%.

Trả về thời gian chạy (ms) đo bằng Julia. Trả -1.0 nếu lỗi.
"""
function run_spmf(jar_path::String, input::String, output::String,
                  minsup_pct::Float64; algo::String="Apriori")::Float64
    minsup_str = string(round(minsup_pct, digits=4)) * "%"
    # Normalize paths (resolve .., spaces etc.)
    jar  = abspath(jar_path)
    inp  = abspath(input)
    outp = abspath(output)
    mkpath(dirname(outp))
    cmd = `java -jar $jar run $algo $inp $outp $minsup_str`
    t0 = time_ns()
    try
        run(pipeline(cmd, stdout=devnull, stderr=devnull))
        return (time_ns() - t0) / 1_000_000.0
    catch e
        @warn "SPMF failed: $e"
        return -1.0
    end
end

"""
    load_spmf_output(filepath) -> Dict{Vector{Int}, Int}

Parse SPMF output file. Trả về Dict(itemset → support_count).
Format mỗi dòng: `item1 item2 ... #SUP: count`
"""
function load_spmf_output(filepath::String)::Dict{Vector{Int}, Int}
    result = Dict{Vector{Int}, Int}()
    isfile(filepath) || return result
    open(filepath, "r") do f
        for line in eachline(f)
            s = strip(line)
            isempty(s) && continue
            parts = split(s, " #SUP: ")
            length(parts) != 2 && continue
            items = sort(parse.(Int, split(strip(parts[1]))))
            cnt   = parse(Int, strip(parts[2]))
            result[items] = cnt
        end
    end
    return result
end

"""
    load_our_output(filepath) -> Dict{Vector{Int}, Int}

Parse output file của algorithm (cùng format SPMF).
"""
load_our_output(filepath::String) = load_spmf_output(filepath)

"""
    compare_results(ours::Dict{Vector{Int},Int},
                    spmf::Dict{Vector{Int},Int})
    -> (n_ours, n_spmf, n_match, match_pct, mismatches)

So sánh kết quả algorithm vs SPMF.
"""
function compare_results(ours::Dict{Vector{Int},Int},
                         spmf::Dict{Vector{Int},Int})
    n_ours = length(ours)
    n_spmf = length(spmf)
    n_match = 0
    mismatches = Vector{Tuple{Vector{Int}, Int, Int}}()  # (itemset, our_sup, spmf_sup)

    for (itemset, our_sup) in ours
        if haskey(spmf, itemset)
            if spmf[itemset] == our_sup
                n_match += 1
            else
                push!(mismatches, (itemset, our_sup, spmf[itemset]))
            end
        end
    end

    # Itemsets trong SPMF nhưng không có trong ours
    only_spmf = length(setdiff(keys(spmf), keys(ours)))

    match_pct = n_spmf == 0 ? 100.0 : round(n_match / n_spmf * 100, digits=2)
    return n_ours, n_spmf, n_match, match_pct, mismatches, only_spmf
end

# ══════════════════════════════════════════════════════════════════════════════
# ĐO LƯỜNG HIỆU NĂNG
# ══════════════════════════════════════════════════════════════════════════════

"""
    measure_time(f, args...) -> (result, elapsed_ms)
"""
function measure_time(f::Function, args...)
    t0 = time_ns()
    result = f(args...)
    return result, (time_ns() - t0) / 1_000_000.0
end

"""
    measure_memory_mb(f, args...) -> (result, mb_allocated)
"""
function measure_memory_mb(f::Function, args...)
    GC.gc(); GC.gc()
    alloc = @allocated result = f(args...)
    return result, alloc / 1024 / 1024
end

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

"""
    flatten_results(results) -> Dict{Vector{Int}, Int}
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

"""
    total_fi(results) -> Int

Tổng số frequent itemsets.
"""
total_fi(res) = sum(length(v) for v in values(res); init=0)

"""
    count_all_frequent_bruteforce(D, minsup) -> Int

Brute-force đếm FI. Chỉ dùng cho toy databases nhỏ.
"""
function count_all_frequent_bruteforce(D::Vector{Vector{Int}}, minsup::Float64)::Int
    all_items = sort(unique(vcat(D...)))
    n_items = length(all_items)
    n_db = length(D)
    min_cnt = ceil(Int, minsup * n_db)
    count = 0
    T_sets = [Set{Int}(t) for t in D]
    for mask in 1:(1 << n_items - 1)
        itemset = @inbounds [all_items[i] for i in 1:n_items if (mask >> (i-1)) & 1 == 1]
        sup = sum(all(item ∈ ts for item in itemset) for ts in T_sets)
        sup >= min_cnt && (count += 1)
    end
    return count
end

@inline relative_support(count::Int, db_size::Int)::Float64 = count / db_size

# ══════════════════════════════════════════════════════════════════════════════
# SINH DỮ LIỆU
# ══════════════════════════════════════════════════════════════════════════════

"""
    create_subset_db(db, pct) -> Vector{Vector{Int}}

Lấy `pct×100%` giao dịch đầu tiên.
"""
function create_subset_db(db::Vector{Vector{Int}}, pct::Float64)::Vector{Vector{Int}}
    n = max(1, round(Int, length(db) * pct))
    return db[1:n]
end

"""
    generate_synthetic_db(n_trans, n_items, avg_len; seed=42) -> Vector{Vector{Int}}

Sinh CSDL tổng hợp. Dùng cho Exp F.
"""
function generate_synthetic_db(n_trans::Int, n_items::Int, avg_len::Int;
                                seed::Int=42)::Vector{Vector{Int}}
    rng = MersenneTwister(seed)
    db = Vector{Vector{Int}}(undef, n_trans)
    for i in 1:n_trans
        len = clamp(round(Int, avg_len + 2 * randn(rng)), 1, n_items)
        items = sort!(unique(rand(rng, 1:n_items, 2 * len)))
        db[i] = items[1:min(len, length(items))]
    end
    return db
end
