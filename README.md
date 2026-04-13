# Apriori Frequent Itemset Mining (FIM)

Đồ án triển khai thuật toán Apriori từ đầu bằng ngôn ngữ Julia, bao gồm phiên bản đếm support tuyến tính (Basic) và phiên bản tối ưu bằng **Hash Tree** (theo paper gốc Agrawal & Srikant 1994). Dự án cũng tự động tích hợp chạy đối chứng so sánh với công cụ **SPMF**, cùng với module chạy các thực nghiệm bắt buộc, xuất báo cáo dưới dạng `.csv` và biểu đồ `.png`.

---

## 1. Yêu cầu hệ thống

1. **Julia** (khuyên dùng bản ≥ 1.9)
2. **Java** (cần có trong `PATH` hệ thống để tự động chạy so sánh với `spmf.jar`)
3. **SPMF**: File `spmf.jar` đã được đính kèm ở thư mục gốc của project.
4. **Package Julia**: `Plots.jl`.

### Cài đặt thư viện Julia
Mở PowerShell ở thư mục gốc dự án, chạy lệnh sau (chỉ cần chạy lần đầu tiên):
```powershell
julia --project -e "using Pkg; Pkg.instantiate()"
```
> **Lưu ý**: Lần đầu tiên chạy có thể mất vài phút để Julia cài đặt và pre-compile `Plots.jl`.

---

## 2. Mô tả dữ liệu (Datasets)

Các tập dữ liệu đã có sẵn trong dự án:

1. **Benchmark Dữ liệu** (`data/benchmark/`)
   - `chess.txt`: 3,196 giao dịch (Dense)
   - `mushroom.txt`: 8,124 giao dịch (Dense)
   - `retail.txt`: 88,162 giao dịch (Sparse) - Tập dữ liệu chính cho bài toán Market Basket Analysis
   - `accidents.txt`: 340,183 giao dịch (Dense, rất lớn)
   - `T10I4D100K.txt`: 100,000 giao dịch (Sparse, tổng hợp)

2. **Toy Dữ liệu** (`data/toy/`)
   - 5 file `db1.txt` đến `db5.txt` dùng để kiểm tra tính đúng đắn (Unit Tests).

---

## 3. Hướng dẫn chạy đồ án

### 3.1 Chạy giao diện dòng lệnh (CLI)
Bạn có thể chạy thuật toán Apriori trực tiếp để xuất ra file định dạng giống SPMF (`item1 item2 ... #SUP: count`).

```powershell
# Chạy phiên bản Basic
julia --project src/algorithm/apriori.jl data/benchmark/chess.txt 0.8

# Chạy phiên bản Optimized và lưu kết quả
julia --project src/algorithm/apriori.jl data/benchmark/retail.txt 0.01 outputs/output_retail.txt --opt
```

### 3.2 Kiểm thử tính đúng đắn (Unit Tests)
Chạy trên các tập toy, so sánh kết quả Basic vs Hash Tree vs Thuật toán vét cạn (Brute-force).
```powershell
julia --project tests/test_correctness.jl
```

### 3.3 Chương 3 – So sánh thực nghiệm với SPMF
Chạy thuật toán trên cả tập Benchmark và Toy, sau đó tự động gọi `spmf.jar` chạy và so sánh số lượng FI, thời gian chạy, và độ chính xác.
```powershell
julia --project src/chapter3_correctness.jl
```
**Output Folder**: `docs/chapter_3/` chứa `correctness_toy.csv`, `correctness_benchmark.csv`, `level3_comparison.csv`.

### 3.4 Chương 4 – 6 Thực nghiệm đo lường hiệu năng
Chạy tự động toàn bộ 6 thực nghiệm (độ chính xác, thời gian, số lượng FI, RAM, scalability, transaction length effect).
```powershell
julia --project src/experiments.jl
```
**Output Folder**: `docs/chapter_4/a` đến `f/` có chứa cả dữ liệu số `.csv` và biểu đồ trực quan `.png` từng phần.

*(Lưu ý: Dataset `accidents.txt` kích cỡ lớn nên việc chạy có thể mất vài phút tuỳ hệ thống).*

### 3.5 Chương 5 – Ứng dụng thực tế: Phân tích giỏ hàng (Market Basket)
Thực hiện chạy Apriori trên tập Retail với cấu hình `Minsup=0.01` và `Minconf=0.3` để sinh ra các Luật kết hợp (Association Rules), phân tích Top10 rules theo chỉ số `Lift`.
```powershell
julia --project src/chapter5_market_basket.jl
```
**Output Folder**: `docs/chapter_5/` chứa các dữ liệu báo cáo `.csv` và các biểu đồ vẽ tự động như Scatter rules `.png`, Histogram `.png`.

---

## 4. Đặc tả kĩ thuật triển khai (Hash Tree)
Dự án đã triển khai việc đếm Support hoàn thiện theo paper gốc:
- `apriori_basic()`: Dùng linear scan duyệt qua database cho từng candidate. Độ phức tạp đếm support là $O(|D| \times |C_k| \times k)$
- `apriori_optimized()`: Xây dựng **Hash Tree** chứa các candidates thuộc $C_k$. Mỗi transaction duyệt qua Hash Tree để đếm xem chứa subsets nào. Giúp bỏ qua các candidates không cần thiết, gia tăng đáng kể tốc độ hệ thống so với phiên bản Basic. 

Đồng thời, đồ án đã đính kèm `SPMF (spmf.jar)` ngay trong source code, và viết script bằng Julia giao tiếp (pipeline CLI) với Java để lấy số liệu đối xứng tự động tạo báo cáo khách quan.
