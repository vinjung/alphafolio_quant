# KR 퀀트 시스템 - 멀티 에이전트 AI용 추가 개발 계획

## 1. 배경 및 목적

### 사용자의 핵심 질문
| 질문 | 현재 해결 | 개선 필요 |
|------|----------|----------|
| 어떤 종목을 사야 하는가? | ✅ final_score, final_grade | - |
| 지금이 좋은 타이밍인가? | ⚠️ signal (신뢰도 낮음) | ✅ |
| 얼마나 사야 하는가? | ❌ 없음 | ✅ |
| 언제 팔아야 하는가? | ⚠️ support/resistance (신뢰도 낮음) | ✅ |
| 시나리오별 대응은? | ❌ 없음 | ✅ |

### 목표
- 멀티 에이전트 AI에게 **강세/횡보/약세 시나리오별 투자 전략** 제공
- 가짜 정밀도(예: 52,300원 지지) 대신 **신뢰도 있는 정보** 제공
- 사용자가 **구체적으로 행동**할 수 있는 정보 제공

---

## 2. 기존 컬럼 대체/제거

### 제거 대상 (가짜 정밀도)
| 기존 컬럼 | 문제점 | 대체 방안 |
|----------|--------|----------|
| `expected_range_3m_min/max` | 범위 너무 넓음, 의미 없음 | `scenario_returns` |
| `expected_range_1y_min/max` | 범위 너무 넓음, 의미 없음 | `scenario_returns` |
| `support_1/2` | 정확한 가격 예측 불가 | `stop_loss_pct` (ATR 기반) |
| `resistance_1/2` | 정확한 가격 예측 불가 | `take_profit_pct` |
| `signal` | 단순 기술적 신호, 신뢰도 낮음 | `entry_timing_score` + `triggers` |
| `supertrend_value` | SuperTrend 단일 지표 한계 | 제거 |
| `trend` | SuperTrend 추세 | 제거 |

### SQL 제거 쿼리
```sql
ALTER TABLE kr_stock_grade
DROP COLUMN IF EXISTS expected_range_3m_min,
DROP COLUMN IF EXISTS expected_range_3m_max,
DROP COLUMN IF EXISTS expected_range_1y_min,
DROP COLUMN IF EXISTS expected_range_1y_max,
DROP COLUMN IF EXISTS support_1,
DROP COLUMN IF EXISTS support_2,
DROP COLUMN IF EXISTS resistance_1,
DROP COLUMN IF EXISTS resistance_2,
DROP COLUMN IF EXISTS supertrend_value,
DROP COLUMN IF EXISTS trend,
DROP COLUMN IF EXISTS signal;
```

---

## 3. 신규 컬럼 추가

### kr_stock_grade 테이블 추가 컬럼

```sql
-- 1. 진입 타이밍 정보
ALTER TABLE kr_stock_grade ADD COLUMN IF NOT EXISTS entry_timing_score DECIMAL(5,1);
ALTER TABLE kr_stock_grade ADD COLUMN IF NOT EXISTS score_trend_2w DECIMAL(5,2);
ALTER TABLE kr_stock_grade ADD COLUMN IF NOT EXISTS price_position_52w DECIMAL(5,2);

-- 2. 리스크 관리 정보
ALTER TABLE kr_stock_grade ADD COLUMN IF NOT EXISTS stop_loss_pct DECIMAL(5,2);
ALTER TABLE kr_stock_grade ADD COLUMN IF NOT EXISTS take_profit_pct DECIMAL(5,2);
ALTER TABLE kr_stock_grade ADD COLUMN IF NOT EXISTS risk_reward_ratio DECIMAL(4,2);
ALTER TABLE kr_stock_grade ADD COLUMN IF NOT EXISTS position_size_pct DECIMAL(4,2);
ALTER TABLE kr_stock_grade ADD COLUMN IF NOT EXISTS atr_pct DECIMAL(5,2);

-- 3. 시나리오 분석
ALTER TABLE kr_stock_grade ADD COLUMN IF NOT EXISTS scenario_bullish_prob INTEGER;
ALTER TABLE kr_stock_grade ADD COLUMN IF NOT EXISTS scenario_sideways_prob INTEGER;
ALTER TABLE kr_stock_grade ADD COLUMN IF NOT EXISTS scenario_bearish_prob INTEGER;
ALTER TABLE kr_stock_grade ADD COLUMN IF NOT EXISTS scenario_bullish_return VARCHAR(20);
ALTER TABLE kr_stock_grade ADD COLUMN IF NOT EXISTS scenario_sideways_return VARCHAR(20);
ALTER TABLE kr_stock_grade ADD COLUMN IF NOT EXISTS scenario_bearish_return VARCHAR(20);
ALTER TABLE kr_stock_grade ADD COLUMN IF NOT EXISTS scenario_sample_count INTEGER;

-- 4. 행동 트리거 (JSON)
ALTER TABLE kr_stock_grade ADD COLUMN IF NOT EXISTS buy_triggers JSONB;
ALTER TABLE kr_stock_grade ADD COLUMN IF NOT EXISTS sell_triggers JSONB;
ALTER TABLE kr_stock_grade ADD COLUMN IF NOT EXISTS hold_triggers JSONB;
```

### 컬럼 상세 정의

| 컬럼명 | 데이터타입 | 설명 | 계산 방법 |
|--------|-----------|------|----------|
| **진입 타이밍** |
| `entry_timing_score` | DECIMAL(5,1) | 진입 타이밍 점수 (0-100) | final_score × 0.4 + trend × 0.3 + position × 0.2 + confirm × 0.1 |
| `score_trend_2w` | DECIMAL(5,2) | 2주간 final_score 변화 | current - 2weeks_ago |
| `price_position_52w` | DECIMAL(5,2) | 52주 고저 대비 위치 (0-100%) | (현재 - 52저) / (52고 - 52저) × 100 |
| **리스크 관리** |
| `stop_loss_pct` | DECIMAL(5,2) | 손절 기준 % (음수) | -2 × ATR_pct |
| `take_profit_pct` | DECIMAL(5,2) | 익절 기준 % (양수) | 3 × ATR_pct (RR 1.5 기준) |
| `risk_reward_ratio` | DECIMAL(4,2) | 리스크리워드 비율 | take_profit / abs(stop_loss) |
| `position_size_pct` | DECIMAL(4,2) | 권장 포트폴리오 비중 % | 변동성 역가중 기반 |
| `atr_pct` | DECIMAL(5,2) | ATR % (14일) | ATR / close × 100 |
| **시나리오 분석** |
| `scenario_bullish_prob` | INTEGER | 강세 시나리오 확률 % | 백테스트 기반 |
| `scenario_sideways_prob` | INTEGER | 횡보 시나리오 확률 % | 백테스트 기반 |
| `scenario_bearish_prob` | INTEGER | 약세 시나리오 확률 % | 백테스트 기반 |
| `scenario_bullish_return` | VARCHAR(20) | 강세 시 예상 수익률 | 예: "+15~25%" |
| `scenario_sideways_return` | VARCHAR(20) | 횡보 시 예상 수익률 | 예: "-3~+5%" |
| `scenario_bearish_return` | VARCHAR(20) | 약세 시 예상 수익률 | 예: "-10~-18%" |
| `scenario_sample_count` | INTEGER | 분석에 사용된 유사 케이스 수 | 신뢰도 참고용 |
| **행동 트리거** |
| `buy_triggers` | JSONB | 매수 실행 조건 목록 | 규칙 기반 자동 생성 |
| `sell_triggers` | JSONB | 매도 실행 조건 목록 | 규칙 기반 자동 생성 |
| `hold_triggers` | JSONB | 보유 유지 조건 목록 | 규칙 기반 자동 생성 |

---

## 4. 계산 로직 상세

### 4.1 Entry Timing Score (진입 타이밍 점수)

```python
async def calculate_entry_timing_score(self) -> dict:
    """
    진입 타이밍 점수 계산 (0-100)

    공식:
        entry_timing = final_score_level × 0.4
                     + final_score_trend × 0.3
                     + price_position × 0.2
                     + momentum_confirm × 0.1
    """
    # 1. final_score_level: 현재 점수 (0-100)
    score_level = self.final_score

    # 2. final_score_trend: 2주 변화 (-20 ~ +20 → 0-100 정규화)
    score_2w_ago = await self._get_score_n_days_ago(14)
    score_trend = self.final_score - score_2w_ago
    trend_normalized = min(100, max(0, 50 + score_trend * 2.5))

    # 3. price_position: 52주 고저 대비 위치
    # 저점 근처(0-30%): 높은 점수, 고점 근처(70-100%): 낮은 점수
    position = await self._get_52w_position()
    position_score = 100 - position  # 저점일수록 높은 점수

    # 4. momentum_confirm: 모멘텀 점수 > 50 여부
    momentum_confirm = 70 if self.momentum_score > 50 else 30

    # 가중 평균
    entry_timing = (
        score_level * 0.4 +
        trend_normalized * 0.3 +
        position_score * 0.2 +
        momentum_confirm * 0.1
    )

    return {
        'entry_timing_score': round(entry_timing, 1),
        'score_trend_2w': round(score_trend, 2),
        'price_position_52w': round(position, 2)
    }
```

### 4.2 Stop Loss / Take Profit (손절/익절)

```python
async def calculate_stop_take_profit(self) -> dict:
    """
    ATR 기반 손절/익절 계산

    공식:
        ATR% = 14일 ATR / 현재가 × 100
        stop_loss = -2 × ATR%
        take_profit = 3 × ATR% (기본 RR 1.5)
    """
    query = """
    WITH daily_range AS (
        SELECT
            date,
            high - low as tr,
            close
        FROM kr_intraday_total
        WHERE symbol = $1
            AND date <= COALESCE($2, CURRENT_DATE)
        ORDER BY date DESC
        LIMIT 14
    )
    SELECT
        AVG(tr) as atr,
        (SELECT close FROM daily_range LIMIT 1) as current_close
    FROM daily_range
    """

    result = await self.execute_query(query, self.symbol, self.analysis_date)

    atr = result[0]['atr']
    close = result[0]['current_close']
    atr_pct = (atr / close) * 100

    stop_loss_pct = -2 * atr_pct
    take_profit_pct = 3 * atr_pct
    risk_reward = take_profit_pct / abs(stop_loss_pct)

    return {
        'atr_pct': round(atr_pct, 2),
        'stop_loss_pct': round(stop_loss_pct, 2),
        'take_profit_pct': round(take_profit_pct, 2),
        'risk_reward_ratio': round(risk_reward, 2)
    }
```

### 4.3 Position Size (포지션 사이징)

```python
async def calculate_position_size(self) -> float:
    """
    변동성 기반 포지션 사이징

    공식:
        base_allocation = 5%
        volatility_factor = 시장평균변동성 / 종목변동성
        position_size = base_allocation × volatility_factor

    제한:
        - 최소: 1%
        - 최대: 10%
    """
    # 종목 변동성
    stock_vol = self.volatility_annual

    # 시장 평균 변동성 (KOSPI 기준 약 20%)
    market_vol = 20.0

    # 변동성 역가중
    volatility_factor = market_vol / stock_vol if stock_vol > 0 else 1.0

    # 기본 배분 5%
    base_allocation = 5.0
    position_size = base_allocation * volatility_factor

    # 범위 제한
    position_size = min(10.0, max(1.0, position_size))

    return round(position_size, 2)
```

### 4.4 Scenario Probability (시나리오 확률)

```python
async def calculate_scenario_probability(self) -> dict:
    """
    백테스트 기반 시나리오 확률 계산

    과정:
        1. 과거 유사 조건 케이스 검색
           - final_score ±5점
           - 동일 섹터
           - 유사 market_state
        2. 3개월 후 결과 집계
           - 강세: >+10%
           - 횡보: -10% ~ +10%
           - 약세: <-10%
    """
    query = """
    WITH similar_cases AS (
        SELECT
            g.symbol,
            g.date as analysis_date,
            g.final_score,
            g.market_state,
            d.industry,
            -- 3개월 후 수익률
            (future.close - current.close) / current.close * 100 as return_3m
        FROM kr_stock_grade g
        JOIN kr_stock_detail d ON g.symbol = d.symbol
        JOIN kr_intraday_total current ON g.symbol = current.symbol AND g.date = current.date
        JOIN kr_intraday_total future ON g.symbol = future.symbol
            AND future.date = g.date + INTERVAL '63 days'
        WHERE g.final_score BETWEEN $1 - 5 AND $1 + 5
            AND d.industry = $2
            AND g.date < CURRENT_DATE - INTERVAL '63 days'
    )
    SELECT
        COUNT(*) as total_count,
        COUNT(*) FILTER (WHERE return_3m > 10) as bullish_count,
        COUNT(*) FILTER (WHERE return_3m BETWEEN -10 AND 10) as sideways_count,
        COUNT(*) FILTER (WHERE return_3m < -10) as bearish_count,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY return_3m)
            FILTER (WHERE return_3m > 10) as bullish_upper,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY return_3m)
            FILTER (WHERE return_3m > 10) as bullish_lower,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY return_3m)
            FILTER (WHERE return_3m < -10) as bearish_upper,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY return_3m)
            FILTER (WHERE return_3m < -10) as bearish_lower
    FROM similar_cases
    """

    result = await self.execute_query(query, self.final_score, self.industry)

    total = result[0]['total_count'] or 1

    return {
        'scenario_bullish_prob': round(result[0]['bullish_count'] / total * 100),
        'scenario_sideways_prob': round(result[0]['sideways_count'] / total * 100),
        'scenario_bearish_prob': round(result[0]['bearish_count'] / total * 100),
        'scenario_bullish_return': f"+{result[0]['bullish_lower']:.0f}~+{result[0]['bullish_upper']:.0f}%",
        'scenario_sideways_return': "-10~+10%",
        'scenario_bearish_return': f"{result[0]['bearish_upper']:.0f}~{result[0]['bearish_lower']:.0f}%",
        'scenario_sample_count': total
    }
```

### 4.5 Triggers (행동 트리거)

```python
def generate_triggers(self) -> dict:
    """
    규칙 기반 행동 트리거 생성
    """
    current_score = self.final_score

    buy_triggers = [
        f"final_score {current_score + 5:.0f}점 이상 상승 시",
        f"외국인 5일 연속 순매수 전환 시",
        f"섹터 순위 상위 {max(5, self.sector_percentile - 10):.0f}% 진입 시"
    ]

    sell_triggers = [
        f"final_score {current_score - 15:.0f}점 이하 하락 시",
        f"손절선 {self.stop_loss_pct:.1f}% 도달 시",
        f"익절선 +{self.take_profit_pct:.1f}% 도달 시",
        f"기관/외국인 10일 연속 순매도 시"
    ]

    hold_triggers = [
        f"final_score {current_score - 5:.0f}~{current_score + 5:.0f}점 유지 시",
        f"섹터 순위 현재 수준 유지 시",
        f"주요 지표 급변 없을 시"
    ]

    return {
        'buy_triggers': buy_triggers,
        'sell_triggers': sell_triggers,
        'hold_triggers': hold_triggers
    }
```

---

## 5. 최종 데이터 구조 (멀티 에이전트 AI용)

```json
{
  "symbol": "005930",
  "stock_name": "삼성전자",
  "date": "2025-11-27",

  "core_judgment": {
    "final_score": 72.5,
    "final_grade": "매수",
    "value_score": 68.3,
    "quality_score": 78.2,
    "momentum_score": 71.5,
    "growth_score": 72.0
  },

  "entry_timing": {
    "entry_timing_score": 68,
    "interpretation": "양호한 진입 시점",
    "score_trend_2w": 4.2,
    "price_position_52w": 32.5
  },

  "risk_management": {
    "stop_loss_pct": -8.5,
    "take_profit_pct": 17.0,
    "risk_reward_ratio": 2.0,
    "position_size_pct": 4.0,
    "atr_pct": 4.25,
    "var_95": -2.5,
    "cvar_95": -3.8,
    "max_drawdown_1y": -18.5
  },

  "scenario_analysis": {
    "probabilities": {
      "bullish": 45,
      "sideways": 35,
      "bearish": 20
    },
    "expected_returns": {
      "bullish": "+15~25%",
      "sideways": "-3~+5%",
      "bearish": "-10~-18%"
    },
    "sample_count": 127,
    "confidence_note": "과거 유사 패턴 127건 기반 (final_score 70±5, 전자부품 섹터)"
  },

  "action_triggers": {
    "buy": [
      "final_score 78점 이상 상승",
      "외국인 5일 연속 순매수",
      "섹터 순위 상위 10% 진입"
    ],
    "sell": [
      "final_score 58점 이하 하락",
      "손절선 -8.5% 도달",
      "익절선 +17% 도달"
    ],
    "hold": [
      "final_score 68~78점 유지",
      "섹터 순위 현재 수준 유지"
    ]
  },

  "relative_position": {
    "sector_rank": "5/48",
    "sector_percentile": 10.4,
    "industry_rank": 3,
    "industry_percentile": 8.5,
    "market_rank": "89/2,450"
  }
}
```

---

## 6. 구현 우선순위

| 순위 | 항목 | 파일 | 난이도 | 효과 |
|------|------|------|--------|------|
| 1 | ATR 기반 stop_loss/take_profit | kr_additional_metrics.py | 낮음 | 높음 |
| 2 | entry_timing_score | kr_additional_metrics.py | 중간 | 높음 |
| 3 | position_size_pct | kr_additional_metrics.py | 낮음 | 중간 |
| 4 | triggers 자동 생성 | kr_additional_metrics.py | 낮음 | 높음 |
| 5 | scenario_probability | kr_additional_metrics.py | 높음 | 높음 |
| 6 | 기존 컬럼 제거 | kr_main.py, db_async.py | 낮음 | - |

---

## 7. 예상 효과

### Before (기존)
```
expected_range_3m: 45,000 ~ 65,000원
support_1: 52,300원
signal: 매수
```
→ 정보가 있으나 **신뢰도 낮고 행동 지침 없음**

### After (개선)
```
entry_timing_score: 68/100 (양호)
stop_loss: -8.5% (현재가 대비)
take_profit: +17%
risk_reward: 2.0
position_size: 4%

시나리오 (127건 분석 기반):
- 강세 45%: +15~25%
- 횡보 35%: -3~+5%
- 약세 20%: -10~-18%

매수 트리거: final_score 78점 이상 상승 시
매도 트리거: 손절선 -8.5% 도달 시
```
→ **신뢰도 명시 + 구체적 행동 지침 제공**

---

## 8. 파일 변경 목록

| 파일 | 변경 내용 |
|------|----------|
| `kr_additional_metrics.py` | 신규 메서드 6개 추가 |
| `kr_main.py` | grade_data 구조 변경 (기존 컬럼 제거, 신규 추가) |
| `db_async.py` | INSERT 쿼리 컬럼 변경 |
| SQL | 테이블 컬럼 추가/삭제 |

---

*작성일: 2025-11-27*
*버전: 1.0*
