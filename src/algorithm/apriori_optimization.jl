#=
Input:
    - D, a database of transactions;
    - min_sup, the minimum support count threshold.
Output: L, frequent itemsets in D.

D: database
min_sup: minsup_count
=#

function load_database(filepath::String)
    database = Vector{Vector{Int}}()
    
    open(filepath, "r") do file
        for line in eachline(file)
            # Bỏ qua dòng trống hoặc comment (giống Java)
            if isempty(line) || line[1] in ('#', '%', '@')
                continue
            end
            # Tách các số và parse thành Int
            items = parse.(Int, split(line))
            push!(database, items)
        end
    end
    
    return database
end


# Tìm frequent 1-itemsets: trả về list (item, count) với count >= minsup_count
function find_frequent_1_itemsets(D, minsup_count::Int)
    # Đếm support từng item
    item_count = Dict{Int, Int}()
    
    for transaction in D
        for item in transaction
            item_count[item] = get(item_count, item, 0) + 1
        end
    end
    
    # Lọc những item có support >= minsup
    L1 = [(item, count) for (item, count) in item_count
                  if count >= minsup_count]
    
    # Sort theo thứ tự lexical (quan trọng để Apriori hoạt động đúng!)
    sort!(L1, by = x -> x[1])
    
    return L1
end

function generate_candidates_2(L1::Vector{Int})
    candidates = Vector{Vector{Int}}()
    n = length(L1)
    
    for i in 1:n
        for j in (i+1):n
            push!(candidates, [L1[i], L1[j]])
        end
    end
    
    return candidates
end

function generate_candidates_k(prev_level::Vector{Vector{Int}})
    candidates = Vector{Vector{Int}}()
    n = length(prev_level)
    
    for i in 1:n
        for j in (i+1):n
            s1 = prev_level[i]
            s2 = prev_level[j]
            k = length(s1)
            
            # Kiểm tra k-1 phần tử đầu phải giống nhau
            if s1[1:end-1] != s2[1:end-1]
                continue
            end
            
            # Phần tử cuối s1 phải nhỏ hơn s2 (lexical order)
            if s1[end] >= s2[end]
                continue
            end
            
            # Tạo candidate mới
            new_candidate = vcat(s1, [s2[end]])
            
            # Kiểm tra tất cả subsets (k-1) có trong prev_level không
            if all_subsets_frequent(new_candidate, prev_level)
                push!(candidates, new_candidate)
            end
        end
    end
    
    return candidates
end

function all_subsets_frequent(candidate::Vector{Int}, prev_level::Vector{Vector{Int}})
    k = length(candidate)
    
    # Thử bỏ từng phần tử để tạo subset kích thước k-1
    for pos in 1:k
        subset = vcat(candidate[1:pos-1], candidate[pos+1:end])
        
        # Tìm subset trong prev_level (dùng in đơn giản)
        if !(subset in prev_level)
            return false
        end
    end
    
    return true
end

function count_support(candidates::Vector{Vector{Int}}, database)
    # support_count[i] = số transaction chứa candidates[i]
    support_count = zeros(Int, length(candidates))
    
    for transaction in database
        for (idx, candidate) in enumerate(candidates)
            # Kiểm tra candidate có phải subset của transaction không
            if is_subset(candidate, transaction)
                support_count[idx] += 1
            end
        end
    end
    
    return support_count
end

# Kiểm tra candidate có nằm trong transaction không
# Vì cả 2 đều sorted, dùng cách quét tuần tự (giống Java)
function is_subset(candidate::Vector{Int}, transaction::Vector{Int})
    pos = 1  # vị trí hiện tại trong candidate
    for item in transaction
        if item == candidate[pos]
            pos += 1
            if pos > length(candidate)
                return true
            end
        elseif item > candidate[pos]
            return false  # Tối ưu: transaction đã sort, không cần quét tiếp
        end
    end
    return false
end

function apriori(database, minsup::Float64)
    db_size = length(database)
    minsup_count = ceil(Int, minsup * db_size)
    
    println("Database size: $db_size transactions")
    println("Min support count: $minsup_count")
    
    # Lưu kết quả: Dict từ size -> list of (itemset, support)
    all_frequent = Dict{Int, Vector{Tuple{Vector{Int}, Int}}}()
    
    # === BƯỚC 1: Tìm frequent 1-itemsets ===
    frequent1 = find_frequent_1_itemsets(database, minsup_count)
    
    if isempty(frequent1)
        println("Không có frequent itemset nào!")
        return all_frequent
    end
    
    all_frequent[1] = [(([item], count)) for (item, count) in frequent1]
    frequent1_items = [item for (item, _) in frequent1]
    
    println("Size 1: $(length(frequent1)) frequent itemsets")
    
    # === BƯỚC 2+: Lặp tìm frequent k-itemsets ===
    k = 2
    prev_level_items = frequent1_items  # dùng cho generate_candidates_2
    prev_level = Vector{Vector{Int}}()  # dùng cho generate_candidates_k
    
    while true
        # Sinh candidates
        if k == 2
            candidates = generate_candidates_2(frequent1_items)
        else
            candidates = generate_candidates_k(prev_level)
        end
        
        if isempty(candidates)
            break
        end
        
        # Đếm support
        support_counts = count_support(candidates, database)
        
        # Lọc frequent candidates
        frequent_k = Vector{Tuple{Vector{Int}, Int}}()
        for (i, candidate) in enumerate(candidates)
            if support_counts[i] >= minsup_count
                push!(frequent_k, (candidate, support_counts[i]))
            end
        end
        
        if isempty(frequent_k)
            break
        end
        
        all_frequent[k] = frequent_k
        prev_level = [itemset for (itemset, _) in frequent_k]
        
        println("Size $k: $(length(frequent_k)) frequent itemsets")
        k += 1
    end
    
    return all_frequent
end


