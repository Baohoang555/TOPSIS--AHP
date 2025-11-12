import numpy as np
import pandas as pd

def run_topsis_analysis(decision_matrix, ahp_weights, criteria_types):
    """
    Thực hiện phân tích TOPSIS và trả về DataFrame đã xếp hạng.
    - decision_matrix: DataFrame của Pandas (đã loại bỏ cột 'ticker')
    - ahp_weights: Mảng NumPy chứa trọng số từ AHP
    - criteria_types: Danh sách ['benefit', 'cost', ...]
    """

    matrix = decision_matrix.values

    # 1. Chuẩn hóa (Vector Normalization)
    norm_matrix = matrix / np.linalg.norm(matrix, axis=0)

    # 2. Tính ma trận trọng số
    weighted_matrix = norm_matrix * ahp_weights

    # 3. Xác định giải pháp lý tưởng (A+) và phi lý tưởng (A-)
    ideal_best = np.zeros(matrix.shape[1])
    ideal_worst = np.zeros(matrix.shape[1])

    for j in range(matrix.shape[1]):
        if criteria_types[j] == 'benefit':
            ideal_best[j] = np.max(weighted_matrix[:, j])
            ideal_worst[j] = np.min(weighted_matrix[:, j])
        elif criteria_types[j] == 'cost':
            ideal_best[j] = np.min(weighted_matrix[:, j])
            ideal_worst[j] = np.max(weighted_matrix[:, j])

    # 4. Tính khoảng cách
    dist_best = np.linalg.norm(weighted_matrix - ideal_best, axis=1)
    dist_worst = np.linalg.norm(weighted_matrix - ideal_worst, axis=1)

    # 5. Tính điểm và xếp hạng
    epsilon = 1e-9 # Tránh chia cho 0
    closeness_score = dist_worst / (dist_best + dist_worst + epsilon)

    return closeness_score