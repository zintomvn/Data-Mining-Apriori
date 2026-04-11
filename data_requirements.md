# Data & Input Requirements – Apriori FIM Lab

## 1. Dữ liệu benchmark (đã có sẵn)

Tất cả 5 file đã tồn tại tại `data/benchmark/`. Không cần tải thêm.

| File | #Trans | #Items | AvgLen | Loại |
|------|--------|--------|--------|------|
| `chess.txt` | 3 196 | 75 | 37.0 | Dense |
| `mushroom.txt` | 8 124 | 119 | 23.0 | Dense |
| `retail.txt` | 88 162 | 16 470 | 10.3 | Sparse |
| `accidents.txt` | 340 183 | 468 | 33.8 | Dense, large |
| `T10I4D100K.txt` | 100 000 | 870 | 10.1 | Sparse, synthetic |

Format: mỗi dòng là 1 giao dịch, items cách nhau bằng dấu cách. Dòng `#`, `%`, `@` bị bỏ qua.

---

## 2. Dữ liệu toy (đã tạo tự động)

Các file tại `data/toy/` được dùng cho unit tests:

| File | #Trans | #Items | Minsup test |
|------|--------|--------|-------------|
| `db1.txt` | 7 | 4 | 0.5 → 4 FI |
| `db2.txt` | 5 | 3 | 0.4 → 7 FI |
| `db3.txt` | 3 | 5 | 0.5 → 31 FI |
| `db4.txt` | 5 | 4 | 0.4 → 4 FI (chỉ 1-itemsets) |
| `db5.txt` | 6 | 5 | 0.5 → nhiều FI |

---

## 3. Cài đặt môi trường Julia

### Bước 1: Cài đặt Plots.jl (chỉ cần làm một lần)

```julia
# Mở Julia REPL trong thư mục project
cd("d:/University/Semester 8/Class - Data mining/Lab/Lab 2/Data-Mining-Apriori")

# Kích hoạt project và cài packages
using Pkg
Pkg.activate(".")
Pkg.instantiate()       # cài tất cả deps từ Project.toml
```

Hoặc chạy script cài đặt nhanh:

```powershell
julia --project -e "using Pkg; Pkg.instantiate()"
```

> **Lưu ý**: Lần đầu tiên Plots.jl sẽ build GR backend (~5-10 phút). Lần sau sẽ nhanh hơn.

### Bước 2: Kiểm tra cài đặt

```julia
using Plots
gr()
plot(1:10, rand(10))   # Kiểm tra backend hoạt động
```

---

## 4. Cách chạy từng phần

### Chương 3 – Unit Tests (Level 2)

```powershell
# Từ thư mục project
julia --project tests/test_correctness.jl
julia --project tests/test_benchmark.jl
```

### Chương 3 – Dòng lệnh SPMF (Level 4)

```powershell
# Basic
julia --project src/algorithm/apriori.jl data/benchmark/chess.txt 0.8

# Optimized + lưu file
julia --project src/algorithm/apriori.jl data/benchmark/retail.txt 0.01 output.txt --opt
```




### Chương 4 – Thực nghiệm (6 Exp → docs/*.png)

```powershell
julia --project src/experiments.jl
```

> ⚠ `accidents.txt` với minsup thấp có thể mất nhiều phút. Tăng minsup trong file nếu cần.

### Chương 5 – Market Basket Analysis

```powershell
julia --project src/chapter5_market_basket.jl
```

---

## 5. Thời gian chạy ước tính

| Script | Dataset | Minsup | Thời gian dự kiến |
|--------|---------|--------|-------------------|
| experiments.jl (Exp B) | Chess | 0.6-0.9 | ~1-5s |
| experiments.jl (Exp B) | Mushroom | 0.1-0.5 | ~2-10s |
| experiments.jl (Exp B) | Retail | 0.001-0.1 | ~5-30s |
| experiments.jl (Exp B) | Accidents | 0.3-0.9 | **5-30 phút** |
| chapter5_market_basket.jl | Retail | 0.01 | ~10-60s |

---

## 6. Output Files

Sau khi chạy đủ, folder `docs/` sẽ có:

```
docs/
├── exp_a_correctness.png
├── exp_b_runtime_chess.png
├── exp_b_runtime_mushroom.png
├── exp_b_runtime_retail.png
├── exp_b_runtime_accidents.png
├── exp_b_runtime_t10i4d100k.png
├── exp_b_runtime_all.png
├── exp_c_itemcount_vs_minsup.png
├── exp_d_memory.png
├── exp_e_scalability.png
├── exp_f_txn_length.png
├── ch5_association_rules_scatter.png
├── ch5_top10_lift.png
└── ch5_rules_distribution.png
```

---

## 7. So sánh với SPMF (Exp A)

Để kiểm tra kết quả với SPMF:
1. Tải SPMF từ https://www.philippe-fournier-viger.com/spmf/index.php?link=download.php
2. Chạy Apriori trong SPMF trên cùng file với cùng minsup
3. So sánh số dòng trong file output với số FI báo cáo từ `experiments.jl`

Kết quả tham chiếu SPMF cho Chess (minsup=0.75): **khoảng 74 frequent itemsets** (phụ thuộc phiên bản).
