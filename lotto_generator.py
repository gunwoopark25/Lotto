"""
로또 번호 생성기
- 엑셀 파일에서 과거 당첨번호를 읽어와 가중치 기반으로 번호를 예측합니다.
- 5개의 묶음(세트)을 생성합니다.
  - 3개: 적게 나온 번호에 높은 가중치 (역빈도 가중치)
  - 2개: 많이 나온 번호에 높은 가중치 (빈도 가중치)
- 최근 5회 당첨번호에 약간의 불이익을 줍니다.
- 사용자가 구간별(1~10, 11~20, 21~30, 31~40, 41~45) 개수를 지정할 수 있습니다.
- 각 구간 별 최고로 잘나온 경우의 수
1위 : 1~10 : 2, 11~20 : 1, 21~30 : 2, 31~40 : 1
2위 : 1~10 : 2, 11~20 : 2, 21~30 : 1, 31~40 : 1
"""

import pandas as pd
import numpy as np
import random
import sys
import os


# ========================
# 엑셀 데이터 로드 함수
# ========================
def load_lotto_data(filepath):
    """
    CSV/엑셀 파일에서 로또 당첨번호 데이터를 로드합니다.

    파일 형식 (헤더 2줄):
      행1: 일자 | 당첨 숫자 (병합셀)
      행2:      | 1 | 2 | 3 | 4 | 5 | 6 | 보너스
      행3~: 날짜 | 번호1 | 번호2 | ... | 번호6 | 보너스번호

    최신 회차가 위에 있는 구조 → 오래된 순서로 뒤집어서 반환합니다.
    보너스 번호는 무시합니다.
    """
    if not os.path.exists(filepath):
        print(f"[오류] 파일을 찾을 수 없습니다: {filepath}")
        sys.exit(1)

    # CSV와 엑셀 파일 모두 지원
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        # 헤더 2줄 건너뛰고 읽기 (직접 컬럼명 지정)
        df = pd.read_csv(filepath, skiprows=2, header=None, encoding="utf-8-sig")
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath, skiprows=2, header=None)
    else:
        print(f"[오류] 지원하지 않는 파일 형식입니다: {ext}")
        sys.exit(1)

    # 컬럼 구조: 0=일자, 1~6=당첨번호, 7=보너스
    # 당첨번호 6개만 추출 (컬럼 인덱스 1~6)
    number_cols = [1, 2, 3, 4, 5, 6]

    # 빈 행 제거 후 정수 변환
    df_numbers = df[number_cols].dropna().astype(int)
    draws = df_numbers.values.tolist()

    # 최신→오래된 순서를 오래된→최신 순서로 뒤집기
    # (최근 5회 추출 시 마지막 5개가 최신이 되도록)
    draws.reverse()

    print(f"[정보] 총 {len(draws)}회차의 당첨번호를 로드했습니다.")
    print(f"[정보] 가장 오래된 회차 번호: {draws[0]}")
    print(f"[정보] 가장 최근 회차 번호: {draws[-1]}")

    return draws


# ========================
# 번호별 출현 빈도 계산
# ========================
def calculate_frequency(draws):
    """
    전체 회차에서 각 번호(1~45)의 출현 횟수를 계산합니다.
    반환: {번호: 출현횟수} 딕셔너리
    """
    freq = {i: 0 for i in range(1, 46)}
    for draw in draws:
        for num in draw:
            if 1 <= num <= 45:
                freq[num] += 1
    return freq


# ========================
# 최근 5회 출현 번호 추출
# ========================
def get_recent_numbers(draws, recent_count=5):
    """
    최근 N회차에서 나온 번호들의 집합을 반환합니다.
    draws는 오래된 순서 → 최근 순서로 정렬되어 있다고 가정합니다.
    """
    recent_draws = draws[-recent_count:]  # 마지막 N개가 최근 회차
    recent_nums = set()
    for draw in recent_draws:
        for num in draw:
            recent_nums.add(num)
    return recent_nums


# ========================
# 가중치 계산 함수
# ========================
def build_weights(freq, recent_nums, mode="inverse"):
    """
    각 번호(1~45)에 대한 가중치를 계산합니다.

    mode:
      - "inverse": 적게 나온 번호에 높은 가중치 (역빈도)
      - "frequency": 많이 나온 번호에 높은 가중치 (빈도)

    최근 5회 출현 번호에는 작은 불이익(0.95배)을 줍니다.
    """
    numbers = list(range(1, 46))

    # 빈도 값 리스트
    freq_values = np.array([freq[n] for n in numbers], dtype=float)

    # 빈도의 최솟값/최댓값
    min_freq = freq_values.min()
    max_freq = freq_values.max()

    if mode == "inverse":
        # 역빈도 가중치: 적게 나온 번호일수록 높은 가중치
        # 선형 변환으로 부드러운 가중치 생성 (드라마틱한 편차 방지)
        if max_freq == min_freq:
            weights = np.ones(45)
        else:
            # 정규화: 0~1 범위로 변환 후 반전
            normalized = (freq_values - min_freq) / (max_freq - min_freq)
            # 반전: 많이 나온건 낮게, 적게 나온건 높게
            inverted = 1.0 - normalized
            # 기본 가중치 1.0에 약간의 편차만 추가 (0.7 ~ 1.3 범위)
            weights = 0.7 + inverted * 0.6

    elif mode == "frequency":
        # 빈도 가중치: 많이 나온 번호일수록 높은 가중치
        if max_freq == min_freq:
            weights = np.ones(45)
        else:
            normalized = (freq_values - min_freq) / (max_freq - min_freq)
            # 기본 가중치 1.0에 약간의 편차만 추가 (0.7 ~ 1.3 범위)
            weights = 0.7 + normalized * 0.6

    # 최근 5회 출현 번호에 불이익 적용 (0.95배 = 5% 감소)
    for i, num in enumerate(numbers):
        if num in recent_nums:
            weights[i] *= 0.95

    # 가중치를 확률로 변환 (합이 1이 되도록)
    weight_dict = {}
    for i, num in enumerate(numbers):
        weight_dict[num] = weights[i]

    return weight_dict


# ========================
# 구간별 번호 선택 함수
# ========================
def pick_numbers_by_range(weight_dict, range_counts):
    """
    각 구간에서 지정된 개수만큼 번호를 가중치 기반으로 뽑습니다.

    range_counts: {(1,10): 2, (11,20): 1, (21,30): 1, (31,40): 1, (41,45): 1}
    weight_dict: {번호: 가중치} 딕셔너리

    반환: 정렬된 6개 번호 리스트
    """
    selected = []

    for (start, end), count in range_counts.items():
        if count == 0:
            continue

        # 해당 구간의 번호와 가중치 추출
        nums_in_range = list(range(start, end + 1))
        weights_in_range = [weight_dict[n] for n in nums_in_range]

        # 가중치 합이 0이면 균등 확률로 대체
        total = sum(weights_in_range)
        if total == 0:
            probs = [1.0 / len(nums_in_range)] * len(nums_in_range)
        else:
            probs = [w / total for w in weights_in_range]

        # 구간 내에서 count개 비복원 추출
        chosen = np.random.choice(nums_in_range, size=count, replace=False, p=probs)
        selected.extend(chosen.tolist())

    selected.sort()
    return selected


# ========================
# 사용자 입력: 구간별 개수
# ========================
def get_range_counts_from_user():
    """
    터미널에서 사용자에게 각 구간별 원하는 번호 개수를 입력받습니다.
    총합이 6이어야 합니다.
    """
    ranges = [
        (1, 10),
        (11, 20),
        (21, 30),
        (31, 40),
        (41, 45),
    ]

    print("\n====================================")
    print("  각 구간별 원하는 번호 개수를 입력하세요")
    print("  (총합이 반드시 6이어야 합니다)")
    print("====================================")

    while True:
        range_counts = {}
        total = 0

        for start, end in ranges:
            while True:
                try:
                    count = int(input(f"  {start}~{end} 구간에서 몇 개? : "))
                    if count < 0:
                        print("  [오류] 0 이상의 숫자를 입력하세요.")
                        continue
                    # 구간 내 번호 개수보다 많이 요청하면 오류
                    max_in_range = end - start + 1
                    if count > max_in_range:
                        print(
                            f"  [오류] 이 구간에는 최대 {max_in_range}개까지만 가능합니다."
                        )
                        continue
                    break
                except ValueError:
                    print("  [오류] 숫자를 입력하세요.")

            range_counts[(start, end)] = count
            total += count

        # 총합 검증
        if total != 6:
            print(
                f"\n  [오류] 총합이 {total}개입니다. 반드시 6개여야 합니다. 다시 입력하세요."
            )
            continue
        else:
            print(f"  [확인] 구간별 설정 완료! (총 {total}개)")
            return range_counts


# ========================
# 메인 실행 함수
# ========================
def main():
    # 엑셀 파일 경로 (사용자 환경에 맞게 수정하세요)
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = input("엑셀 파일 경로를 입력하세요: ").strip()

    # 1단계: 데이터 로드
    draws = load_lotto_data(filepath)

    # 2단계: 전체 빈도 계산
    freq = calculate_frequency(draws)

    # 3단계: 최근 5회 출현 번호 추출
    recent_nums = get_recent_numbers(draws, recent_count=5)
    print(f"[정보] 최근 5회 출현 번호: {sorted(recent_nums)}")

    # 4단계: 사용자에게 구간별 개수 입력받기
    range_counts = get_range_counts_from_user()

    # 5단계: 가중치 생성
    # 역빈도 가중치 (적게 나온 번호 우대) - 3세트용
    weights_inverse = build_weights(freq, recent_nums, mode="inverse")
    # 빈도 가중치 (많이 나온 번호 우대) - 2세트용
    weights_freq = build_weights(freq, recent_nums, mode="frequency")

    # 6단계: 5세트 번호 생성
    print("\n========================================")
    print("        로또 번호 생성 결과")
    print("========================================")

    for i in range(5):
        if i < 3:
            # 세트 1~3: 역빈도 가중치 (적게 나온 번호 우대)
            numbers = pick_numbers_by_range(weights_inverse, range_counts)
            label = "역빈도"
        else:
            # 세트 4~5: 빈도 가중치 (많이 나온 번호 우대)
            numbers = pick_numbers_by_range(weights_freq, range_counts)
            label = "빈도"

        print(f"  세트 {i+1} [{label}] : {numbers}")

    print("========================================")
    print("  행운을 빕니다!")
    print("========================================\n")


if __name__ == "__main__":
    main()
