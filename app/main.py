import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# --- CÁC HÀM LOGIC (AHP/TOPSIS) ---
# (Các hàm logic không thay đổi, chúng ta chỉ thay đổi giao diện)
def calculate_ahp_weights(pairwise_matrix):
    """
    Tính toán trọng số AHP và Tỷ số Nhất quán (CR)
    từ một ma trận so sánh cặp.
    """
    n = pairwise_matrix.shape[0]
    
    try:
        col_sums = pairwise_matrix.sum(axis=0)
        col_sums[col_sums == 0] = 1e-9
        norm_matrix = pairwise_matrix / col_sums
        weights = norm_matrix.mean(axis=1)
    except Exception as e:
        st.error(f"Lỗi khi chuẩn hóa ma trận: {e}")
        return None, None, "Lỗi khi chuẩn hóa ma trận"

    RI_lookup = {
        1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 
        11: 1.51
    }

    try:
        A_x = pairwise_matrix.dot(weights)
        weights_safe = np.where(weights == 0, 1e-9, weights)
        lambda_max = (A_x / weights_safe).mean()
        CI = (lambda_max - n) / (n - 1) if n > 1 else 0
        RI = RI_lookup.get(n, 1.59)
        CR = CI / RI if RI != 0 else 0
    except Exception as e:
        return weights, None, f"Lỗi khi tính toán CR: {e}"

    return weights, CR, None # Trả về weights, CR, và không có lỗi

def run_topsis_analysis(decision_matrix, ahp_weights, criteria_types):
    """
    Thực hiện phân tích TOPSIS và trả về điểm số.
    """
    matrix = decision_matrix.values.astype(float)
    
    try:
        norm_denominator = np.linalg.norm(matrix, axis=0)
        norm_denominator[norm_denominator == 0] = 1e-9
        norm_matrix = matrix / norm_denominator
    except Exception as e:
        st.error(f"Lỗi khi chuẩn hóa ma trận TOPSIS: {e}")
        return None

    weighted_matrix = norm_matrix * ahp_weights

    ideal_best = np.zeros(matrix.shape[1])
    ideal_worst = np.zeros(matrix.shape[1])

    for j in range(matrix.shape[1]):
        if criteria_types[j] == 'benefit':
            ideal_best[j] = np.max(weighted_matrix[:, j])
            ideal_worst[j] = np.min(weighted_matrix[:, j])
        elif criteria_types[j] == 'cost':
            ideal_best[j] = np.min(weighted_matrix[:, j])
            ideal_worst[j] = np.max(weighted_matrix[:, j])

    dist_best = np.linalg.norm(weighted_matrix - ideal_best, axis=1)
    dist_worst = np.linalg.norm(weighted_matrix - ideal_worst, axis=1)

    epsilon = 1e-9
    closeness_score = dist_worst / (dist_best + dist_worst + epsilon)

    return closeness_score
# --- KẾT THÚC CÁC HÀM LOGIC ---


# --- CẤU HÌNH CỐT LÕI CỦA MÔ HÌNH ---
CRITERIA_GROUPS = {
    "Định giá (Valuation)": ['P/E (TTM)', 'P/B', 'EPS (TTM)'],
    "Khả năng sinh lời (Profitability)": ['ROE', 'ROA', 'NIM'],
    "Sức khỏe tài chính (Risk)": ['D_E', 'LDR', 'NPL_Ratio'],
    "Hiệu quả hoạt động (Efficiency)": ['Asset_Turnover', 'CIR']
}

CRITERIA_TYPES = [
    'cost', 'cost', 'benefit', # Định giá
    'benefit', 'benefit', 'benefit', # Sinh lời
    'cost', 'cost', 'cost', # Rủi ro
    'benefit', 'cost' # Hiệu quả
]

ALL_CRITERIA_ORDERED = [item for sublist in CRITERIA_GROUPS.values() for item in sublist]
GROUP_NAMES = list(CRITERIA_GROUPS.keys())

# --- CẤU HÌNH THANH TRƯỢT (SLIDER) ---
SLIDER_MAP = {
    "Ưu tiên mạnh B (9)": 1/9.0,
    "Ưu tiên B (7)": 1/7.0,
    "Ưu tiên khá B (5)": 1/5.0,
    "Ưu tiên nhẹ B (3)": 1/3.0,
    "Như nhau (1)": 1.0,
    "Ưu tiên nhẹ A (3)": 3.0,
    "Ưu tiên khá A (5)": 5.0,
    "Ưu tiên A (7)": 7.0,
    "Ưu tiên mạnh A (9)": 9.0,
}
SLIDER_LABELS = list(SLIDER_MAP.keys())

# --- HÀM HELPER CHO GIAO DIỆN SLIDER ---
def display_comparison_sliders(items_list, key_prefix):
    """Hiển thị các thanh trượt để so sánh cặp cho một danh sách."""
    for i in range(len(items_list)):
        for j in range(i + 1, len(items_list)):
            item_a = items_list[i]
            item_b = items_list[j]
            
            labels = [
                label.replace("A", item_a).replace("B", item_b) 
                for label in SLIDER_LABELS
            ]
            
            st.select_slider(
                f"So sánh **{item_a}** và **{item_b}**",
                options=labels,
                value="Như nhau (1)".replace("A", item_a).replace("B", item_b),
                key=f"slider_{key_prefix}_{item_a}_{item_b}"
            )

def build_matrix_from_sliders(items_list, key_prefix):
    """Xây dựng ma trận AHP từ giá trị của các thanh trượt."""
    n = len(items_list)
    matrix = np.ones((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            item_a = items_list[i]
            item_b = items_list[j]
            
            slider_key = f"slider_{key_prefix}_{item_a}_{item_b}"
            label_value = st.session_state[slider_key]
            
            original_label = " (1)".join(label_value.split(" (1)")[:-1]) + " (1)"
            for l in SLIDER_LABELS:
                if l.replace("A", item_a).replace("B", item_b) == label_value:
                    original_label = l
                    break
            
            numeric_value = SLIDER_MAP[original_label]
            
            matrix[i, j] = numeric_value
            matrix[j, i] = 1.0 / numeric_value
            
    return matrix
# --- KẾT THÚC HÀM HELPER ---


# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(layout="wide")

# --- THANH BÊN TRÁI (SIDEBAR) CHO AHP ---
with st.sidebar:
    st.title("🎛️ Bảng điều khiển AHP")
    st.write("Thiết lập trọng số cho mô hình.")

    # Cấp 1: So sánh giữa các Nhóm (Trọng số Nhóm)
    st.header("Cấp 1: So sánh Nhóm")
    display_comparison_sliders(GROUP_NAMES, "group")
    
    st.divider()

    # Cấp 2: So sánh trong từng Nhóm (Trọng số Nội bộ)
    st.header("Cấp 2: So sánh Tiêu chí")
    # Dùng expander thay vì tabs cho sidebar
    for group_name in GROUP_NAMES:
        with st.expander(f"Nhóm: {group_name}"):
            criteria_in_group = CRITERIA_GROUPS[group_name]
            if len(criteria_in_group) == 1:
                st.write(f"Nhóm này chỉ có 1 tiêu chí ({criteria_in_group[0]}), không cần so sánh.")
            else:
                display_comparison_sliders(criteria_in_group, group_name)
    
    st.divider()

    # Nút tính toán "Công Thức Vàng"
    if st.button("Tính Trọng số AHP Toàn cục", use_container_width=True):
        all_weights_valid = True
        cr_errors = [] # Danh sách lưu các lỗi CR
        
        # 1. Xây dựng và tính toán Cấp 1 (Nhóm)
        group_matrix = build_matrix_from_sliders(GROUP_NAMES, "group")
        group_weights, group_cr, err = calculate_ahp_weights(group_matrix)
        
        if err:
            all_weights_valid = False
            cr_errors.append(f"Ma trận Nhóm: {err}")
        elif group_cr >= 0.1:
            cr_errors.append(f"Ma trận Nhóm KHÔNG nhất quán (CR = {group_cr:.4f})")
        
        local_weights_dict = {}

        # 2. Xây dựng và tính toán Cấp 2 (Nội bộ nhóm)
        for group_name in GROUP_NAMES:
            criteria_in_group = CRITERIA_GROUPS[group_name]
            if len(criteria_in_group) == 1:
                weights, cr = np.array([1.0]), 0.0
            else:
                local_matrix = build_matrix_from_sliders(criteria_in_group, group_name)
                weights, cr, err = calculate_ahp_weights(local_matrix)
            
            if err:
                all_weights_valid = False
                cr_errors.append(f"Nhóm '{group_name}': {err}")
            elif cr >= 0.1:
                cr_errors.append(f"Ma trận Nhóm '{group_name}' KHÔNG nhất quán (CR = {cr:.4f})")
                
            local_weights_dict[group_name] = weights

        # 3. Tính Trọng số Toàn cục (Global Weight) = Local x Group
        if all_weights_valid and group_weights is not None:
            final_global_weights = []
            for i, group_name in enumerate(GROUP_NAMES):
                group_weight = group_weights[i]
                local_weights = local_weights_dict[group_name]
                
                global_weights_for_group = group_weight * (local_weights if local_weights is not None else 0)
                final_global_weights.extend(global_weights_for_group)
            
            final_weights_array = np.array(final_global_weights)
            
            df_final_weights = pd.DataFrame({
                "Tiêu chí": ALL_CRITERIA_ORDERED,
                "Trọng số Toàn cục": final_weights_array
            })
            
            # Lưu kết quả vào session state để main page hiển thị
            st.session_state['ahp_weights'] = final_weights_array
            st.session_state['df_final_weights'] = df_final_weights
            st.session_state['cr_errors'] = cr_errors
            st.session_state['ahp_run_success'] = True # Báo hiệu đã chạy
        else:
            st.session_state['ahp_run_success'] = False
            st.session_state['cr_errors'] = cr_errors


# --- KHU VỰC CHÍNH (MAIN PAGE) CHO KẾT QUẢ ---
st.title("Kết quả Xếp hạng Cổ phiếu (AHP + TOPSIS) 📈")

# 1. Hiển thị kết quả AHP
st.header("1. Kết quả Trọng số AHP")
if 'ahp_run_success' not in st.session_state:
    st.info("Vui lòng thiết lập và nhấn 'Tính Trọng số AHP Toàn cục' ở thanh bên trái.")
elif not st.session_state['ahp_run_success']:
    st.error("Tính toán AHP thất bại. Vui lòng kiểm tra lỗi ở thanh bên trái.")
else:
    st.write("Đây là Trọng số Toàn cục cuối cùng sẽ được dùng cho TOPSIS:")
    st.dataframe(st.session_state['df_final_weights'])
    
    # Hiển thị thông báo nhất quán
    cr_errors = st.session_state.get('cr_errors', [])
    if not cr_errors:
        st.success("Tất cả các ma trận đều nhất quán (CR < 0.1)")
    else:
        st.warning("Một hoặc nhiều ma trận KHÔNG nhất quán. Vui lòng kiểm tra lại các đánh giá.")
        for error in cr_errors:
            st.error(error)

st.divider()

# 2. Khu vực chạy TOPSIS
st.header("2. Xếp hạng (TOPSIS)")
uploaded_file = st.file_uploader("Tải lên file 'DECISION_MATRIX_FOR_TOPSIS.csv'", type=["csv"])

if uploaded_file is None:
    st.info("Vui lòng tải file 'DECISION_MATRIX_FOR_TOPSIS.csv' để tiếp tục.")
elif 'ahp_weights' not in st.session_state:
    st.warning("Vui lòng tính 'Trọng số AHP Toàn cục' ở thanh bên trái trước khi chạy TOPSIS.")
else:
    st.success("Đã có Trọng số AHP và File Ma trận Quyết định. Sẵn sàng chạy TOPSIS.")
    
    try:
        df_decision = pd.read_csv(uploaded_file)
        st.write("Xem trước Ma trận Quyết định (File CSV):")
        st.dataframe(df_decision.head())

        # Hiển thị loại tiêu chí để xác nhận
        with st.expander("Xem lại Loại Tiêu chí (Benefit/Cost)"):
            st.dataframe(pd.Series(CRITERIA_TYPES, index=ALL_CRITERIA_ORDERED, name="Loại"))
        
        if st.button("Chạy TOPSIS và Xếp hạng", use_container_width=True, type="primary"):
            weights = st.session_state['ahp_weights']
            tickers = df_decision['ticker']
            
            try:
                # Đảm bảo ma trận dữ liệu theo đúng thứ tự
                matrix_data = df_decision[ALL_CRITERIA_ORDERED]
            except KeyError:
                st.error(f"Lỗi: File CSV của bạn thiếu một trong các cột tiêu chí bắt buộc. Vui lòng đảm bảo file có đủ 11 cột: {ALL_CRITERIA_ORDERED}")
                st.stop()
                
            scores = run_topsis_analysis(matrix_data, weights, CRITERIA_TYPES)
            
            if scores is not None:
                df_results = pd.DataFrame({'Ticker': tickers, 'TOPSIS_Score': scores})
                df_results['Rank'] = df_results['TOPSIS_Score'].rank(ascending=False).astype(int)
                df_results = df_results.sort_values(by='Rank')
                
                st.subheader("🎉 Kết quả Xếp hạng Cuối cùng 🎉")
                st.dataframe(df_results)
            else:
                st.error("Không thể chạy phân tích TOPSIS do lỗi trong quá trình tính toán.")

    except Exception as e:
        st.error(f"LỖI: Không thể đọc file CSV. Vui lòng đảm bảo file đúng định dạng. Lỗi: {e}")

st.divider()

# 3. Khu vực Phân tích Nhạy cảm (Sensitivity Analysis)
st.header("3. Phân tích Nhạy cảm (Sensitivity Analysis)")

if 'ahp_weights' not in st.session_state:
    st.warning("Vui lòng tính 'Trọng số AHP Toàn cục' ở thanh bên trái trước khi chạy Phân tích Nhạy cảm.")
else:
    st.success("Đã có Trọng số AHP. Sẵn sàng chạy Phân tích Nhạy cảm.")
    
    try:
        df_decision = pd.read_csv("src/Data Preprocessing/DECISION_MATRIX_FOR_TOPSIS.csv")
        st.write("Ma trận Quyết định (từ file CSV mặc định):")
        st.dataframe(df_decision.head())

        if st.button("Chạy Phân tích Nhạy cảm với 3 Kịch bản", use_container_width=True, type="primary"):
            tickers = df_decision['ticker']
            matrix_data = df_decision[ALL_CRITERIA_ORDERED]

            # Kịch bản 1: Ưu tiên cân bằng (Equal weights)
            st.subheader("Kịch bản 1: Ưu tiên cân bằng")
            equal_weights = np.ones(len(ALL_CRITERIA_ORDERED)) / len(ALL_CRITERIA_ORDERED)
            scores_1 = run_topsis_analysis(matrix_data, equal_weights, CRITERIA_TYPES)
            df_results_1 = pd.DataFrame({'Ticker': tickers, 'TOPSIS_Score': scores_1})
            df_results_1['Rank'] = df_results_1['TOPSIS_Score'].rank(ascending=False).astype(int)
            df_results_1 = df_results_1.sort_values(by='Rank')
            st.dataframe(df_results_1)

            # Kịch bản 2: Ưu tiên mạnh về Sinh lời (Profitability)
            st.subheader("Kịch bản 2: Ưu tiên mạnh về Sinh lời")
            # Giả lập trọng số AHP: Nhóm Sinh lời có trọng số cao
            group_weights = np.array([0.1, 0.6, 0.2, 0.1])  # Định giá, Sinh lời, Rủi ro, Hiệu quả
            local_weights = {
                'Định giá': np.array([0.33, 0.33, 0.34]),
                'Khả năng sinh lời': np.array([0.5, 0.3, 0.2]),
                'Sức khỏe tài chính': np.array([0.33, 0.33, 0.34]),
                'Hiệu quả hoạt động': np.array([0.5, 0.5])
            }
            global_weights_2 = np.concatenate([
                group_weights[0] * local_weights['Định giá'],
                group_weights[1] * local_weights['Khả năng sinh lời'],
                group_weights[2] * local_weights['Sức khỏe tài chính'],
                group_weights[3] * local_weights['Hiệu quả hoạt động']
            ])
            scores_2 = run_topsis_analysis(matrix_data, global_weights_2, CRITERIA_TYPES)
            df_results_2 = pd.DataFrame({'Ticker': tickers, 'TOPSIS_Score': scores_2})
            df_results_2['Rank'] = df_results_2['TOPSIS_Score'].rank(ascending=False).astype(int)
            df_results_2 = df_results_2.sort_values(by='Rank')
            st.dataframe(df_results_2)

            # Kịch bản 3: Ưu tiên mạnh về Rủi ro thấp (Low Risk)
            st.subheader("Kịch bản 3: Ưu tiên mạnh về Rủi ro thấp")
            # Giả lập trọng số AHP: Nhóm Rủi ro có trọng số cao
            group_weights_3 = np.array([0.1, 0.1, 0.6, 0.2])  # Định giá, Sinh lời, Rủi ro, Hiệu quả
            global_weights_3 = np.concatenate([
                group_weights_3[0] * local_weights['Định giá'],
                group_weights_3[1] * local_weights['Khả năng sinh lời'],
                group_weights_3[2] * local_weights['Sức khỏe tài chính'],
                group_weights_3[3] * local_weights['Hiệu quả hoạt động']
            ])
            scores_3 = run_topsis_analysis(matrix_data, global_weights_3, CRITERIA_TYPES)
            df_results_3 = pd.DataFrame({'Ticker': tickers, 'TOPSIS_Score': scores_3})
            df_results_3['Rank'] = df_results_3['TOPSIS_Score'].rank(ascending=False).astype(int)
            df_results_3 = df_results_3.sort_values(by='Rank')
            st.dataframe(df_results_3)

            # So sánh các kịch bản
            st.subheader("So sánh Xếp hạng giữa các Kịch bản")
            comparison = pd.DataFrame({
                'Ticker': tickers,
                'Rank_KB1': df_results_1['Rank'].values,
                'Rank_KB2': df_results_2['Rank'].values,
                'Rank_KB3': df_results_3['Rank'].values
            })
            st.dataframe(comparison)

    except Exception as e:
        st.error(f"LỖI: Không thể đọc file CSV mặc định. Lỗi: {e}")