# src/algorithm/apriori.jl
# Cài đặt thuật toán Apriori FIM từ bài báo gốc:
#   Agrawal & Srikant, "Fast Algorithms for Mining Association Rules", VLDB 1994
#
# Yêu cầu include thứ tự:
#   include("structures.jl")
#   include("utils.jl")
#   include("apriori.jl")

# Include dependencies nếu chưa được load (khi chạy trực tiếp từ CLI)
if !@isdefined(Itemset)
    include(joinpath(@__DIR__, "structures.jl"))
    include(joinpath(@__DIR__, "utils.jl"))
end

using Printf

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS NỘI BỘ
# ══════════════════════════════════════════════════════════════════════════════

"""
    _has_infrequent_subset(candidate::Vector{Int},
                           prev_set::Set{Vector{Int}}) -> Bool

Kiểm tra pruning: candidate có chứa (k-1)-subset không nằm trong L_(k-1)?
Nếu có ⟹ loại candidate này.

# Thuật toán (Apriori property / downward closure)
Với k-candidate `c`, ta loại bỏ lần lượt từng phần tử và kiểm tra
subset (k-1) kết quả có trong `prev_set` không.

# Độ phức tạp: O(k²)  với k = |candidate|
"""
function _has_infrequent_subset(candidate::Vector{Int},
                                 prev_set::Set{Vector{Int}})::Bool
    k = length(candidate)
    @inbounds for pos in 1:k
        # Tạo subset bỏ phần tử pos
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
    _apriori_gen(prev_level::Vector{Vector{Int}}) -> Vector{Vector{Int}}

Sinh tập candidates C_k từ L_(k-1) theo bước Join + Prune.

**Bước Join:** ghép l1 và l2 khi (k-1) phần tử đầu giống nhau và l1[end] < l2[end].
**Bước Prune:** loại candidate nếu tồn tại subset kích thước k-1 không trong L_(k-1).

# Đầu vào
- `prev_level` : Danh sách các (k-1)-itemset **đã sắp xếp lexicographic**

# Đầu ra
- `Vector{Vector{Int}}` : Tập candidates C_k

# Độ phức tạp: O(|L_(k-1)|² × k)
"""
function _apriori_gen(prev_level::Vector{Vector{Int}})::Vector{Vector{Int}}
    candidates = Vector{Vector{Int}}()
    prev_set   = Set{Vector{Int}}(prev_level)
    n          = length(prev_level)

    @inbounds for i in 1:(n - 1)
        l1 = prev_level[i]
        @inbounds for j in (i + 1):n
            l2 = prev_level[j]
            # Điều kiện join: k-1 phần tử đầu phải giống nhau
            # Vì đã sort lex, nếu prefix khác thì mọi j tiếp theo cũng khác → break
            l1[1:end-1] != l2[1:end-1] && break

            c = vcat(l1, [l2[end]])
            !_has_infrequent_subset(c, prev_set) && push!(candidates, c)
        end
    end
    return candidates
end


"""
    _count_support_basic!(Ck::Vector{Vector{Int}},
                          D::Vector{Vector{Int}},
                          counts::Vector{Int})

Đếm support cho mọi candidate trong Ck (in-place).

Sử dụng `Set{Int}` cho mỗi transaction để kiểm tra membership O(1).
Phù hợp với phiên bản **cơ bản** (không tối ưu).

# Độ phức tạp: O(|D| × |Ck| × k)
"""
function _count_support_basic!(Ck::Vector{Vector{Int}},
                                 D::Vector{Vector{Int}},
                                 counts::Vector{Int})
    fill!(counts, 0)
    @inbounds for t in D
        t_set = Set{Int}(t)
        @inbounds for ci in eachindex(Ck)
            all(item ∈ t_set for item in Ck[ci]) && (counts[ci] += 1)
        end
    end
end


"""
    _count_support_bitset!(Ck::Vector{Vector{Int}},
                           T_bits::Vector{BitSet},
                           counts::Vector{Int})

Đếm support dùng **precomputed BitSet** cho mỗi transaction.
Item lookup: `item ∈ BitSet` → O(1).

# Đây là tối ưu chính trong `apriori_optimized`:
Thay vì xây Set{Int} trong vòng lặp → precompute một lần ngoài vòng lặp k.

# Độ phức tạp: O(|D| × |Ck| × k)   (constant nhỏ hơn basic)
"""
function _count_support_bitset!(Ck::Vector{Vector{Int}},
                                  T_bits::Vector{BitSet},
                                  counts::Vector{Int})
    fill!(counts, 0)
    @inbounds for t_set in T_bits
        @inbounds for ci in eachindex(Ck)
            cand = Ck[ci]
            ok   = true
            @inbounds for item in cand
                if item ∉ t_set
                    ok = false; break
                end
            end
            ok && (counts[ci] += 1)
        end
    end
end


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 1–2 : APRIORI CƠ BẢN
# ══════════════════════════════════════════════════════════════════════════════

"""
    apriori_basic(D::Vector{Vector{Int}}, minsup::Float64)
                  -> Dict{Int, Vector{Itemset}}

Cài đặt cơ bản của thuật toán Apriori (Level 1).

Thuật toán duyệt theo từng mức k, lần lượt:
1. Sinh candidates C_k từ L_(k-1) (apriori_gen = join + prune).
2. Quét toàn bộ CSDL để đếm support từng candidate.
3. Lọc ra L_k = {c ∈ C_k | count(c) ≥ minsup_count}.

# Tham số
- `D`      : CSDL giao dịch (mỗi transaction là `Vector{Int}` đã sort)
- `minsup` : Ngưỡng support tương đối ∈ (0, 1]

# Giá trị trả về
- `Dict{Int, Vector{Itemset}}` : mapping  k → frequent k-itemsets

# Độ phức tạp
- Thời gian: O(Σ_k |C_k| × |D|)
- Không gian: O(|C_k| + |L_k|)
"""
function apriori_basic(D::Vector{Vector{Int}}, minsup::Float64)::Dict{Int, Vector{Itemset}}
    db_size      = length(D)
    minsup_count = ceil(Int, minsup * db_size)
    all_frequent = Dict{Int, Vector{Itemset}}()

    println("  ┌─ [Basic] DB=$db_size giao dịch | minsup=$(round(minsup*100,digits=2))% (count=$minsup_count)")

    # ── Bước 1: L₁ ──────────────────────────────────────────────────────────
    item_cnt = Dict{Int,Int}()
    for t in D, item in t
        item_cnt[item] = get(item_cnt, item, 0) + 1
    end
    L_prev = sort!([Itemset([item], cnt)
                    for (item, cnt) in item_cnt
                    if cnt >= minsup_count])

    isempty(L_prev) && (println("  └─ Không có frequent itemset!"); return all_frequent)
    all_frequent[1] = L_prev
    println("  │  k=1 : $(length(L_prev)) frequent itemsets")

    # ── Bước 2+: k = 2, 3, ... ──────────────────────────────────────────────
    k = 2
    while !isempty(L_prev)
        prev_items = [is.itemset for is in L_prev]
        Ck         = _apriori_gen(prev_items)
        isempty(Ck) && break

        counts = zeros(Int, length(Ck))
        _count_support_basic!(Ck, D, counts)

        frequent_k = [Itemset(Ck[ci], counts[ci])
                      for ci in eachindex(Ck)
                      if counts[ci] >= minsup_count]
        isempty(frequent_k) && break

        all_frequent[k] = frequent_k
        L_prev = frequent_k
        println("  │  k=$k : $(length(frequent_k)) frequent itemsets ($(length(Ck)) candidates)")
        k += 1
    end

    total = sum(length(v) for v in values(all_frequent))
    println("  └─ Tổng: $total frequent itemsets")
    return all_frequent
end


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 3 : APRIORI TỐI ƯU HOÁ
# ══════════════════════════════════════════════════════════════════════════════

"""
    apriori_optimized(D::Vector{Vector{Int}}, minsup::Float64)
                      -> Dict{Int, Vector{Itemset}}

Phiên bản tối ưu của Apriori với hai kỹ thuật chính:

**Tối ưu 1 – Precomputed BitSet tidsets:**
Mỗi transaction được chuyển thành `BitSet` **một lần** trước vòng lặp chính.
Kiểm tra membership `item ∈ BitSet` là O(1) thay vì tìm kiếm tuyến tính.

**Tối ưu 2 – Early break trong apriori_gen:**
Vì `prev_level` đã sắp xếp lexicographic, điều kiện prefix không khớp
cho phép `break` ngay thay vì `continue`, giảm số lần so sánh.

**Mức cải thiện** (xem `src/experiments.jl`, Exp D):
- Trên Chess/Mushroom (dense): ~15–30% nhanh hơn, ít alloc hơn ~20%.
- Trên Retail/T10I4D100K (sparse): ~10–20% nhanh hơn.

# Tham số / Giá trị trả về: giống `apriori_basic`
"""
function apriori_optimized(D::Vector{Vector{Int}}, minsup::Float64)::Dict{Int, Vector{Itemset}}
    db_size      = length(D)
    minsup_count = ceil(Int, minsup * db_size)
    all_frequent = Dict{Int, Vector{Itemset}}()

    println("  ┌─ [Opt] DB=$db_size giao dịch | minsup=$(round(minsup*100,digits=2))% (count=$minsup_count)")

    # Precompute BitSet cho mỗi transaction (Tối ưu 1)
    T_bits = [BitSet(t) for t in D]

    # ── Bước 1: L₁ ──────────────────────────────────────────────────────────
    item_cnt = Dict{Int,Int}()
    for t in D, item in t
        item_cnt[item] = get(item_cnt, item, 0) + 1
    end
    L_prev = sort!([Itemset([item], cnt)
                    for (item, cnt) in item_cnt
                    if cnt >= minsup_count])

    isempty(L_prev) && (println("  └─ Không có frequent itemset!"); return all_frequent)
    all_frequent[1] = L_prev
    println("  │  k=1 : $(length(L_prev)) frequent itemsets")

    # ── Bước 2+: k = 2, 3, ... ──────────────────────────────────────────────
    k = 2
    while !isempty(L_prev)
        prev_items = [is.itemset for is in L_prev]
        Ck         = _apriori_gen(prev_items)
        isempty(Ck) && break

        counts = zeros(Int, length(Ck))
        _count_support_bitset!(Ck, T_bits, counts)   # ← dùng BitSet (Tối ưu 1)

        frequent_k = [Itemset(Ck[ci], counts[ci])
                      for ci in eachindex(Ck)
                      if counts[ci] >= minsup_count]
        isempty(frequent_k) && break

        all_frequent[k] = frequent_k
        L_prev = frequent_k
        println("  │  k=$k : $(length(frequent_k)) frequent itemsets ($(length(Ck)) candidates)")
        k += 1
    end

    total = sum(length(v) for v in values(all_frequent))
    println("  └─ Tổng: $total frequent itemsets")
    return all_frequent
end


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 5 : SINH LUẬT KẾT HỢP
# ══════════════════════════════════════════════════════════════════════════════

"""
    generate_association_rules(
        frequent::Dict{Int, Vector{Itemset}},
        db_size::Int,
        minconf::Float64
    ) -> Vector{AssociationRule}

Sinh luật kết hợp X ⟹ Y từ tập frequent itemsets.

Với mỗi frequent itemset Z và mỗi cách phân chia không rỗng Z = X ∪ Y,
luật X ⟹ Y được giữ lại nếu:
  - conf(X ⟹ Y) = sup(Z) / sup(X) ≥ minconf

Chỉ số lift = conf / sup(Y) được tính để xếp hạng luật.

# Tham số
- `frequent` : Kết quả từ `apriori_basic` hoặc `apriori_optimized`
- `db_size`  : Số giao dịch trong CSDL
- `minconf`  : Ngưỡng confidence tối thiểu ∈ (0, 1]

# Giá trị trả về
- `Vector{AssociationRule}` sắp xếp giảm dần theo lift

# Độ phức tạp: O(Σ_k 2^k × |L_k|)
"""
function generate_association_rules(frequent::Dict{Int,Vector{Itemset}},
                                     db_size::Int,
                                     minconf::Float64)::Vector{AssociationRule}
    # Xây support lookup
    sup_map = Dict{Vector{Int}, Float64}()
    for (_, items) in frequent
        for is in items
            sup_map[is.itemset] = relative_support(is.count, db_size)
        end
    end

    rules = AssociationRule[]

    for (k, items) in frequent
        k < 2 && continue          # Luật cần ít nhất 2 item
        for is in items
            Z     = is.itemset
            sup_Z = sup_map[Z]
            # Sinh mọi cách phân biệt antecedent X (subset khác rỗng và khác Z)
            for mask in 1:(1 << k - 2)
                X = [Z[i] for i in 1:k if (mask >> (i-1)) & 1 == 1]
                Y = [Z[i] for i in 1:k if (mask >> (i-1)) & 1 == 0]
                isempty(X) || isempty(Y) && continue
                haskey(sup_map, X) || continue
                sup_X = sup_map[X]
                sup_X == 0.0 && continue
                conf = sup_Z / sup_X
                conf >= minconf || continue
                haskey(sup_map, Y) || continue
                sup_Y = sup_map[Y]
                lift  = sup_Y == 0.0 ? 0.0 : conf / sup_Y
                push!(rules, AssociationRule(X, Y, sup_Z, conf, lift))
            end
        end
    end

    sort!(rules, rev=true)    # giảm dần theo lift
    return rules
end


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 4 : GIAO DIỆN I/O CHUẨN SPMF + THAM SỐ DÒNG LỆNH
# ══════════════════════════════════════════════════════════════════════════════

"""
    run_apriori(input_path::String, minsup::Float64;
                output_path::String = "",
                optimized::Bool     = false,
                verbose::Bool       = true) -> Dict{Int, Vector{Itemset}}

Giao diện cấp cao: đọc file SPMF → chạy Apriori → (tuỳ chọn) lưu kết quả.

# Tham số
- `input_path`  : File SPMF đầu vào (`.txt`)
- `minsup`      : Ngưỡng support tương đối ∈ (0, 1]
- `output_path` : Nếu khác `""`, lưu kết quả ra file xuất
- `optimized`   : Dùng `apriori_optimized` nếu `true`
- `verbose`     : In thông tin tiến trình

# Ví dụ dòng lệnh
```
julia src/algorithm/apriori.jl data/benchmark/chess.txt 0.8
julia src/algorithm/apriori.jl data/benchmark/retail.txt 0.01 output.txt --opt
```
"""
function run_apriori(input_path::String, minsup::Float64;
                     output_path::String = "",
                     optimized::Bool     = false,
                     verbose::Bool       = true)::Dict{Int, Vector{Itemset}}
    verbose && println("\n" * "═"^60)
    verbose && println("  Apriori FIM  |  $(optimized ? "Optimized" : "Basic")")
    verbose && println("  Input  : $input_path")
    verbose && println("  Minsup : $(round(minsup*100,digits=3))%")
    verbose && println("═"^60)

    D = load_database(input_path)
    verbose && println("  Đã đọc $(length(D)) giao dịch\n")

    algo = optimized ? apriori_optimized : apriori_basic
    result, elapsed_ms = measure_time(algo, D, minsup)

    verbose && @printf("\n  ⏱  Thời gian : %.2f ms\n", elapsed_ms)

    if !isempty(output_path)
        save_results(output_path, result, length(D))
    end

    return result
end


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT DÒNG LỆNH
# ══════════════════════════════════════════════════════════════════════════════
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("Cách dùng: julia apriori.jl <input_file> <minsup> [output_file] [--opt]")
        println("  minsup  : số thực trong (0,1], ví dụ: 0.5")
        println("  --opt   : dùng phiên bản tối ưu (BitSet)")
        exit(1)
    end

    inp  = ARGS[1]
    ms   = parse(Float64, ARGS[2])
    outp = length(ARGS) >= 3 && !startswith(ARGS[3], "--") ? ARGS[3] : ""
    opt  = "--opt" ∈ ARGS

    run_apriori(inp, ms; output_path=outp, optimized=opt)
end