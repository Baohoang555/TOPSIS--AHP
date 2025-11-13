# Ứng dụng Hỗ trợ Quyết định (AHP + TOPSIS) để Phân tích Cổ phiếu Ngân hàng

Đây là một dự án Hệ thống Hỗ trợ Quyết định (Decision Support System - DSS) được xây dựng bằng Python và Streamlit, nhằm mục đích xếp hạng các cổ phiếu ngân hàng tại Việt Nam dựa trên mô hình lai (hybrid model) AHP-TOPSIS.

Mô hình này cho phép người dùng linh hoạt thiết lập tầm quan trọng của các tiêu chí tài chính thông qua thuật toán **AHP (Analytic Hierarchy Process)**, sau đó sử dụng các trọng số này làm đầu vào cho thuật toán **TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)** để đưa ra bảng xếp hạng cuối cùng.
## 🛠️ Công nghệ sử dụng

* **Ngôn ngữ:** Python 3.11+
* **Giao diện Web App:** Streamlit
* **Phân tích & Xử lý dữ liệu:** Pandas, NumPy
* **Thu thập dữ liệu:** yfinance
* **Mô hình & Thuật toán:** AHP, TOPSIS (code thuần)
* **Môi trường thử nghiệm:** Jupyter Notebook
## ⚙️ Cài đặt

1.  **Clone dự án (hoặc tải về):**
    ```bash
    git clone https://github.com/Baohoang555/TOPSIS--AHP.git
    cd "TOPSIS+ AHP"
    ```

2.  **Tạo môi trường ảo (khuyên dùng):**
    ```bash
    python -m venv venv
    ```
    * Trên Windows: `.\venv\Scripts\activate`
    * Trên macOS/Linux: `source venv/bin/activate`

3.  **Cài đặt các thư viện cần thiết:**
    pip install -r package.txt
    

## 📊 Luồng xử lý Dữ liệu (Bắt buộc)

Trước khi chạy ứng dụng, bạn cần tạo file ma trận quyết định.

1.  **Chạy `src/Data Preprocessing/Input_Data.ipynb`:**
    * Mở và chạy notebook này để tải dữ liệu snapshot thị trường từ `yfinance` và dữ liệu cơ bản từ `Book1.csv`.
    * Kết quả: Sẽ tạo ra 2 file `market_data_snapshot.csv` và `funda_data_2021.csv`.

2.  **Chạy `src/Data Preprocessing/Data_Cleaning.ipynb`:**
    * Mở và chạy notebook này để gộp, tính toán các chỉ số phái sinh (D/E, LDR, v.v.), làm sạch và loại bỏ `NaN`.
    * Kết quả: Sẽ tạo ra file **`DECISION_MATRIX_FOR_TOPSIS.csv`**—đây là file đầu vào cuối cùng cho ứng dụng.

## 🏃 Cách chạy Ứng dụng Streamlit

1.  Mở Terminal của bạn.
2.  **Quan trọng:** Đảm bảo bạn đang ở thư mục **gốc** của dự án (`TOPSIS+ AHP/`), **không** phải bên trong thư mục `app/`.
3.  Chạy lệnh sau:
    streamlit run app/main.py
