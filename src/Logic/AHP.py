import numpy as np

def calculate_ahp_weights(pairwise_matrix):
    """
    Tính toán trọng số AHP và Tỷ số Nhất quán (CR)
    từ một ma trận so sánh cặp.
    """
    n = pairwise_matrix.shape[0]

    # 1. Chuẩn hóa và tính trọng số (weights)
    col_sums = pairwise_matrix.sum(axis=0)
    norm_matrix = pairwise_matrix / col_sums
    weights = norm_matrix.mean(axis=1)

    # 2. Kiểm tra tính nhất quán (Consistency Check)
    RI_lookup = {
        1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 
        11: 1.51
    }

    A_x = pairwise_matrix.dot(weights)
    lambda_max = (A_x / weights).mean()
    CI = (lambda_max - n) / (n - 1)
    RI = RI_lookup.get(n, 1.59) # Dùng giá trị lớn nếu n > 15

    CR = CI / RI if RI != 0 else 0

    # Trả về trọng số và CR
    return weights, CR