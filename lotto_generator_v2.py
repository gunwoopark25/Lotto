"""
로또 번호 생성기 v2
- 마르코프 체인 기반 구간 패턴 예측
- 구간별 출현 패턴 주기 분석
- 빈도/역빈도 가중치 (v1 유지)

5세트 구성:
  세트 1~2 : 마르코프 1위 예측 패턴 + 역빈도/빈도 가중치
  세트 3~4 : 마르코프 2위 예측 패턴 + 역빈도/빈도 가중치
  세트 5   : 마르코프 3위 예측 패턴 + 역빈도 가중치
  (모든 세트에 구간 주기 가중치 반영)
"""

import pandas as pd
import numpy as np
import sys
import os
from collections import defaultdict


# ========================
# 구간 정의 (전역)
# ========================
ZONES = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 45)]


# ========================
# 데이터 로드
# ========================
def load_lotto_data(filepath):
    """
    CSV/엑셀 파일에서 로또 당첨번호 데이터를 로드합니다.
    파일 형식: 헤더 2줄 → 행3~: 날짜 | 번호1~6 | 보너스
    최신 회차가 위에 있는 구조 → 오래된 순으로 뒤집어 반환합니다.
    """
    if not os.path.exists(filepath):
        print(f"[오류] 파일을 찾을 수 없습니다: {filepath}")
        sys.exit(1)

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath, skiprows=2, header=None, encoding="utf-8-sig")
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath, skiprows=2, header=None)
    else:
        print(f"[오류] 지원하지 않는 파일 형식입니다: {ext}")
        sys.exit(1)

    df_numbers = df[[1, 2, 3, 4, 5, 6]].dropna().astype(int)
    draws = df_numbers.values.tolist()
    draws.reverse()  # 오래된 → 최신 순서

    print(f"[정보] 총 {len(draws)}회차 당첨번호 로드 완료")
    print(f"[정보] 가장 오래된 회차: {draws[0]}")
    print(f"[정보] 가장 최근 회차:   {draws[-1]}")
    return draws


# ========================
# 번호 빈도 계산
# ========================
def calculate_frequency(draws):
    freq = {i: 0 for i in range(1, 46)}
    for draw in draws:
        for num in draw:
            if 1 <= num <= 45:
                freq[num] += 1
    return freq


# ========================
# 최근 N회 출현 번호
# ========================
def get_recent_numbers(draws, recent_count=5):
    recent_nums = set()
    for draw in draws[-recent_count:]:
        recent_nums.update(draw)
    return recent_nums


# ========================
# 마르코프 체인
# ========================
def get_zone_pattern(draw):
    """회차 하나를 구간별 개수 tuple로 변환. 예: (2, 1, 1, 2, 0)"""
    return tuple(sum(1 for n in draw if s <= n <= e) for s, e in ZONES)


def build_markov_chain(draws):
    """
    연속된 회차의 구간 패턴 전이 횟수를 계산하고 확률로 변환합니다.
    반환: {현재_패턴: {다음_패턴: 전이_확률}}
    """
    transitions = defaultdict(lambda: defaultdict(int))
    for i in range(len(draws) - 1):
        curr = get_zone_pattern(draws[i])
        nxt  = get_zone_pattern(draws[i + 1])
        transitions[curr][nxt] += 1

    transition_probs = {}
    for curr, nexts in transitions.items():
        total = sum(nexts.values())
        transition_probs[curr] = {p: c / total for p, c in nexts.items()}

    return transition_probs


def predict_next_patterns(draws, transition_probs, top_n=3):
    """
    마르코프 체인으로 다음 회차 구간 패턴 상위 후보를 반환합니다.
    직전 회차 패턴이 없으면 전체에서 가장 흔한 패턴으로 대체합니다.
    """
    last_pattern = get_zone_pattern(draws[-1])

    zone_labels = [f"{s}~{e}" for s, e in ZONES]
    last_dict = dict(zip(zone_labels, last_pattern))
    print(f"\n[마르코프] 직전 회차 구간 패턴: {last_dict}")

    if last_pattern not in transition_probs:
        print("[마르코프] 해당 패턴의 전이 데이터 없음 → 전체 최빈 패턴 사용")
        counter = defaultdict(int)
        for d in draws:
            counter[get_zone_pattern(d)] += 1
        sorted_p = sorted(counter.items(), key=lambda x: -x[1])
        return [p for p, _ in sorted_p[:top_n]]

    sorted_nexts = sorted(transition_probs[last_pattern].items(), key=lambda x: -x[1])

    print(f"[마르코프] 다음 회차 예측 상위 {top_n}개 패턴:")
    for rank, (pattern, prob) in enumerate(sorted_nexts[:top_n], 1):
        p_dict = dict(zip(zone_labels, pattern))
        print(f"  {rank}위 패턴 {p_dict}  (전이 확률 {prob:.1%})")

    return [p for p, _ in sorted_nexts[:top_n]]


# ========================
# 구간별 주기 분석
# ========================
def analyze_zone_cycles(draws):
    """
    각 구간에서 숫자가 출현한 회차의 간격(주기)을 분석합니다.
    - avg_cycle  : 평균 출현 간격 (회차 수)
    - since_last : 마지막 출현 이후 지난 회차 수
    - due        : 평균 주기 이상 지났으면 True (출현 시기 도래)
    - cycle_weight: 주기 초과 비율에 따른 가중치 (0.8 ~ 1.5)
    """
    zone_analysis = {}

    for (s, e) in ZONES:
        active = [i for i, d in enumerate(draws) if any(s <= n <= e for n in d)]

        if len(active) < 2:
            since = len(draws) - 1 - (active[-1] if active else 0)
            zone_analysis[(s, e)] = {
                "avg_cycle": None,
                "std_cycle": None,
                "since_last": since,
                "due": False,
                "cycle_weight": 1.0,
            }
            continue

        intervals  = [active[i + 1] - active[i] for i in range(len(active) - 1)]
        avg_cycle  = float(np.mean(intervals))
        std_cycle  = float(np.std(intervals))
        since_last = len(draws) - 1 - active[-1]
        due        = since_last >= avg_cycle

        ratio        = since_last / avg_cycle if avg_cycle > 0 else 0
        cycle_weight = round(min(1.5, 0.8 + ratio * 0.4), 3)

        zone_analysis[(s, e)] = {
            "avg_cycle":    round(avg_cycle, 2),
            "std_cycle":    round(std_cycle, 2),
            "since_last":   since_last,
            "due":          due,
            "cycle_weight": cycle_weight,
        }

    return zone_analysis


def print_zone_analysis(zone_analysis):
    print("\n[구간 주기 분석]")
    header = f"  {'구간':<8} {'평균주기':>8} {'표준편차':>8} {'마지막후':>8} {'도래':>6} {'가중치':>8}"
    print(header)
    print("  " + "-" * 50)
    for (s, e), info in zone_analysis.items():
        avg = f"{info['avg_cycle']:.1f}" if info["avg_cycle"] is not None else " N/A"
        std = f"{info['std_cycle']:.1f}" if info["std_cycle"] is not None else " N/A"
        due = "O" if info["due"] else "-"
        print(
            f"  {s}~{e:<4} {avg:>8} {std:>8} {info['since_last']:>7}회 {due:>6} {info['cycle_weight']:>8.3f}"
        )


# ========================
# 가중치 계산
# ========================
def build_weights(freq, recent_nums, zone_analysis, mode="inverse"):
    """
    각 번호(1~45)에 가중치를 계산합니다.
    mode      : "inverse" (역빈도) 또는 "frequency" (빈도)
    적용 순서 : 빈도 가중치 → 최근 5회 불이익 → 구간 주기 가중치
    """
    numbers    = list(range(1, 46))
    freq_vals  = np.array([freq[n] for n in numbers], dtype=float)
    mn, mx     = freq_vals.min(), freq_vals.max()

    if mode == "inverse":
        weights = np.ones(45) if mx == mn else 0.7 + (1.0 - (freq_vals - mn) / (mx - mn)) * 0.6
    else:  # frequency
        weights = np.ones(45) if mx == mn else 0.7 + ((freq_vals - mn) / (mx - mn)) * 0.6

    # 최근 5회 불이익 (5% 감소)
    for i, num in enumerate(numbers):
        if num in recent_nums:
            weights[i] *= 0.95

    # 구간 주기 가중치 반영
    for i, num in enumerate(numbers):
        for (s, e), info in zone_analysis.items():
            if s <= num <= e:
                weights[i] *= info["cycle_weight"]
                break

    return {num: weights[i] for i, num in enumerate(numbers)}


# ========================
# 번호 선택 (구간 패턴 기반)
# ========================
def pick_by_pattern(pattern, weight_dict):
    """
    예측된 구간 패턴(tuple)에 따라 가중치 기반으로 번호를 선택합니다.
    반환: 정렬된 6개 번호 리스트
    """
    selected = []
    for idx, (s, e) in enumerate(ZONES):
        count = pattern[idx]
        if count == 0:
            continue
        nums  = list(range(s, e + 1))
        w     = [weight_dict[n] for n in nums]
        total = sum(w)
        probs = [x / total for x in w] if total > 0 else [1 / len(nums)] * len(nums)
        count = min(count, len(nums))  # 구간 크기 초과 방지
        chosen = np.random.choice(nums, size=count, replace=False, p=probs)
        selected.extend(chosen.tolist())

    selected.sort()
    return selected


# ========================
# 메인
# ========================
def main():
    # ── 파일 경로 ──────────────────────────────
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = input("엑셀/CSV 파일 경로를 입력하세요: ").strip()

    # ── 1. 데이터 로드 ─────────────────────────
    draws = load_lotto_data(filepath)

    # ── 2. 빈도 계산 ───────────────────────────
    freq = calculate_frequency(draws)

    # ── 3. 최근 5회 출현 번호 ──────────────────
    recent_nums = get_recent_numbers(draws, recent_count=5)
    print(f"[정보] 최근 5회 출현 번호: {sorted(recent_nums)}")

    # ── 4. 마르코프 체인 분석 ──────────────────
    transition_probs  = build_markov_chain(draws)
    predicted_patterns = predict_next_patterns(draws, transition_probs, top_n=3)

    # top_n 보장 (데이터 부족 시 반복 채움)
    while len(predicted_patterns) < 3:
        predicted_patterns.append(predicted_patterns[-1])

    # ── 5. 구간 주기 분석 ──────────────────────
    zone_analysis = analyze_zone_cycles(draws)
    print_zone_analysis(zone_analysis)

    # ── 6. 가중치 생성 ─────────────────────────
    w_inv  = build_weights(freq, recent_nums, zone_analysis, mode="inverse")
    w_freq = build_weights(freq, recent_nums, zone_analysis, mode="frequency")

    # ── 7. 5세트 번호 생성 ─────────────────────
    print("\n========================================")
    print("      로또 번호 생성 결과 (v2)")
    print("========================================")

    sets_config = [
        (predicted_patterns[0], w_inv,  "마르코프 1위 + 역빈도"),
        (predicted_patterns[0], w_freq, "마르코프 1위 + 빈도  "),
        (predicted_patterns[1], w_inv,  "마르코프 2위 + 역빈도"),
        (predicted_patterns[1], w_freq, "마르코프 2위 + 빈도  "),
        (predicted_patterns[2], w_inv,  "마르코프 3위 + 역빈도"),
    ]

    for i, (pattern, wdict, label) in enumerate(sets_config, 1):
        numbers = pick_by_pattern(pattern, wdict)
        zone_labels = [f"{s}~{e}" for s, e in ZONES]
        p_str = " | ".join(f"{z}:{c}개" for z, c in zip(zone_labels, pattern))
        print(f"  세트 {i} [{label}]")
        print(f"         패턴: {p_str}")
        print(f"         번호: {numbers}")
        print()

    print("========================================")
    print("  행운을 빕니다!")
    print("========================================\n")


if __name__ == "__main__":
    main()
