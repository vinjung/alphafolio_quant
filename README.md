# 떡상 - Quant Engine

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white)
![asyncpg](https://img.shields.io/badge/asyncpg-async-blue)
![Railway](https://img.shields.io/badge/Railway-0B0D0E?logo=railway&logoColor=white)

시나리오 기반 주식 투자 전략 서비스 "떡상"의 **멀티팩터 퀀트 분석 엔진**입니다.

---

## 목차

- [이 저장소의 역할](#이-저장소의-역할)
- [프로젝트 구조](#프로젝트-구조)
- [개발환경 및 사용기술](#개발환경-및-사용기술)
- [시스템 아키텍처](#시스템-아키텍처)
- [한국 퀀트 모델](#한국-퀀트-모델)
- [미국 퀀트 모델](#미국-퀀트-모델)
- [핵심 기술 특징](#핵심-기술-특징)
- [라이선스](#라이선스)

---

## 이 저장소의 역할

전체 프로젝트 중 **Quant Engine** 컴포넌트를 담당합니다.

- 한국/미국 주식 약 7,000+ 종목 일괄 분석
- 4대 팩터 기반 투자 등급 산출 (A+ ~ F)
- IC(Information Coefficient) 기반 동적 가중치 시스템
- AI Agent용 트레이딩 지표 생성

## 프로젝트 구조

| 폴더 | 설명 |
|------|------|
| [**alpha/overview/**](https://github.com/vinjung/alphafolio_overview) | 프로젝트 설명 |
| [**alpha_front/client/**](https://github.com/vinjung/alphafolio_client-api) | Frontend (UI/UX) |
| [**alpha_front/api/**](https://github.com/vinjung/alphafolio_client-api) | Frontend <-> Backend API 통신 |
| [**alpha/data/**](https://github.com/vinjung/alphafolio_data) | 데이터 자동 수집 & 지표 계산 |
| [**alpha/chat/**](https://github.com/vinjung/alphafolio_chat) | 주식 투자 전략 전문 LLM |
| [**alpha/quant/**](https://github.com/vinjung/alphafolio_quant) | **📍 멀티팩터 퀀트 분석 엔진 (현재 저장소)** |
| [**alpha/stock_agents/**](https://github.com/vinjung/alphafolio_stock_agent) | 종목 투자 전략 Multi-Agent AI |
| [**alpha/portfolio/**](https://github.com/vinjung/alphafolio_portfolio) | 포트폴리오 생성 & 리밸런싱 엔진 |

---

## 개발환경 및 사용기술

| 구분 | 기술 |
|------|------|
| Language | Python 3.11+ |
| Database | PostgreSQL (asyncpg) |
| Deploy | Railway |
| Scheduler | Cron / Manual Trigger |

### 주요 라이브러리

| 라이브러리 | 용도 |
|------------|------|
| asyncpg | PostgreSQL 비동기 연결 |
| numpy | 수치 계산 (VaR, 변동성 등) |
| python-dotenv | 환경변수 관리 |
| psutil | 성능 모니터링 |

---

## 시스템 아키텍처

### 전체 분석 파이프라인

```mermaid
flowchart TB
    subgraph Input["입력"]
        DATE[분석 날짜]
        SYMBOLS[종목 리스트]
    end

    subgraph Phase1["Phase 1: 공통 데이터"]
        ECON[경제 지표 분석]
        CYCLE[경제 사이클 매칭]
        REGIME[시장 레짐 감지]
    end

    subgraph Phase2["Phase 2: 조건 분석"]
        COND[9가지 조건 분석]
        CLASS[시장 상태 분류]
        WEIGHT[동적 가중치 계산]
    end

    subgraph Phase3["Phase 3: 팩터 계산"]
        VALUE[Value Factor]
        QUALITY[Quality Factor]
        MOMENTUM[Momentum Factor]
        GROWTH[Growth Factor]
    end

    subgraph Phase4["Phase 4: 추가 지표 계산"]
        RISK[리스크 지표]
        AGENT[Agent 지표]
        SCENARIO[시나리오 확률]
    end

    subgraph Output["출력 및 저장"]
        GRADE[투자 등급]
        DB[(stock_grade)]
    end

    DATE --> ECON
    SYMBOLS --> ECON
    ECON --> CYCLE
    CYCLE --> REGIME
    REGIME --> COND
    COND --> CLASS
    CLASS --> WEIGHT
    WEIGHT --> VALUE
    WEIGHT --> QUALITY
    WEIGHT --> MOMENTUM
    WEIGHT --> GROWTH
    VALUE --> RISK
    QUALITY --> RISK
    MOMENTUM --> RISK
    GROWTH --> RISK
    RISK --> AGENT
    AGENT --> SCENARIO
    SCENARIO --> GRADE
    GRADE --> DB
```

### 한국 vs 미국 모델 비교

```mermaid
flowchart LR
    subgraph KR["한국 모델"]
        KR_FACTOR[4 Factor<br/>73개 전략]
        KR_STATE[19개 시장 상태]
        KR_IC[IC 기반 가중치]
    end

    subgraph US["미국 모델"]
        US_FACTOR[4 Factor<br/>+ Interaction]
        US_ENGINE[3대 Engine]
        US_OPT[거래소/섹터 최적화]
    end

    KR_FACTOR --> KR_STATE
    KR_STATE --> KR_IC
    US_FACTOR --> US_ENGINE
    US_ENGINE --> US_OPT
```

---

## 한국 퀀트 모델

### 4대 팩터 (73개 전략)

| 팩터 | 전략 수 | 주요 전략 |
|------|---------|-----------|
| **Value** | 15개 | Magic Formula, EV/EBITDA, 순자산할인, 주주환원율 |
| **Quality** | 17개 | ROE 일관성, 영업마진, 이자보상배율, 현금흐름 |
| **Momentum** | 23개 | 가격 모멘텀, RSI, MACD, 외국인 순매수 |
| **Growth** | 18개 | 매출/EPS/영업이익 성장률, R&D 투자 |

**특징**: 섹터 멀티플라이어 (0.5~1.5) 적용

### 19개 시장 상태 분류

9가지 조건을 분석하여 19개 시장 상태 중 하나로 분류:

```mermaid
flowchart LR
    subgraph Conditions["9가지 조건"]
        C1[거래소]
        C2[시가총액]
        C3[유동성]
        C4[경제사이클]
        C5[시장심리]
        C6[섹터사이클]
        C7[테마]
        C8[변동성]
        C9[수급]
    end

    subgraph States["19개 시장 상태"]
        S1[KOSPI대형-확장과열-공격형]
        S2[KOSPI대형-침체패닉-초방어형]
        S3[KOSDAQ소형-핫섹터-초고위험형]
        S4[...]
    end

    Conditions --> States
```

| 그룹 | 상태 예시 | 특징 |
|------|----------|------|
| 대형주 (6개) | KOSPI대형-확장과열-공격형 | MEGA/LARGE, EXPANSION |
| 중형주 (6개) | KOSPI중형-확장과열-모멘텀형 | MEDIUM |
| 소형주 (4개) | KOSDAQ소형-핫섹터-초고위험형 | SMALL, HOT |
| 특수 (2개) | 전시장-극저유동성-고위험형 | PANIC, LOW liquidity |

### IC 기반 동적 가중치

```mermaid
flowchart LR
    BASE["기본 가중치<br/>V:15% Q:25%<br/>M:20% G:40%"]
    COND["조건별 조정계수"]
    NORM["정규화<br/>(합=100%)"]
    LIMIT["극단값 제한<br/>(5%~50%)"]

    BASE --> COND --> NORM --> LIMIT
```

| 팩터 | 기본 가중치 |
|------|------------|
| Growth | 40% |
| Quality | 25% |
| Momentum | 20% |
| Value | 15% |

### 등급 결정 (6단계)

| 등급 | 조건 |
|------|------|
| 강력 매수 | Final Score >= 임계값 AND 필터 3개+ 통과 AND 최적 모멘텀 |
| 매수 | Final Score >= 임계값 AND 필터 3개+ 통과 |
| 매수 고려 | Final Score >= 임계값 AND 필터 2개+ 통과 |
| 중립 | 매수고려 ~ 매도고려 사이 |
| 매도 고려 | Final Score < 매도고려 임계값 |
| 매도 | Final Score < 매도 임계값 |

---

## 미국 퀀트 모델

### 4대 팩터 + Factor Interaction

```mermaid
flowchart TB
    subgraph Factors["4대 팩터"]
        VL[Value<br/>VL1-VL6]
        QA[Quality<br/>QA1-QA6]
        MO[Momentum<br/>MO1-MO6]
        FG[Growth<br/>FG1-FG6 + NQ1-NQ4]
    end

    subgraph Interaction["Factor Interaction"]
        I1[I1: Growth x Quality]
        I2[I2: Growth x Momentum]
        I3[I3: Quality x Value]
        I4[I4: Momentum x Quality]
        I5[I5: All-Factor Agreement]
    end

    subgraph Score["점수 계산"]
        BASE[Base Score x 70%]
        INT[Interaction Score x 30%]
        TOTAL[Total Score]
    end

    VL --> BASE
    QA --> BASE
    MO --> BASE
    FG --> BASE
    VL --> I1
    QA --> I1
    MO --> I2
    FG --> I2
    I1 --> INT
    I2 --> INT
    I3 --> INT
    I4 --> INT
    I5 --> INT
    BASE --> TOTAL
    INT --> TOTAL
```

### Factor Interaction (비선형 관계 해결)

| Term | 이름 | 가중치 | 계산 방식 |
|------|------|--------|-----------|
| I1 | Growth x Quality | 30% | 기하평균 + 시너지 보너스 |
| I2 | Growth x Momentum | 25% | 조화평균 |
| I3 | Quality x Value | 20% | 산술평균 + 보너스 |
| I4 | Momentum x Quality | 15% | 기하평균 - 패널티 |
| I5 | Conviction Score | 10% | 표준편차 기반 합의도 |

**배경**: Pearson IC (-0.097) vs Spearman IC (+0.174) 갭 해결

### 3대 Engine (Phase 3.4)

```mermaid
flowchart LR
    subgraph Engines["3대 Engine"]
        EVENT[Event Engine<br/>-15 ~ +15점]
        MACRO[Macro Engine<br/>가중치 조정]
        VOL[Volatility Engine<br/>민감도별 조정]
    end

    subgraph EventDetails["Event Engine 상세"]
        E1[실적 발표 임박]
        E2[옵션 센티먼트]
        E3[내부자 거래]
    end

    EVENT --> E1
    EVENT --> E2
    EVENT --> E3
```

<details>
<summary><b>Event Engine 상세</b></summary>

| 이벤트 | Modifier | 설명 |
|--------|----------|------|
| 실적 D-5~D-1 | -5점 | 실적 임박 불확실성 |
| Put/Call > 1.5 | -5점 | Bearish 센티먼트 |
| Put/Call < 0.5 | +3점 | Bullish 센티먼트 |
| 3+ 경영진 매수 | +10점 | Cluster Buying |
| CEO 대량 매도 | -5점 | C-level Selling |

</details>

<details>
<summary><b>Macro Engine 상세</b></summary>

**Rate Sensitivity Matrix**

| 환경 | Fed Rate | CPI | 가중치 조정 |
|------|----------|-----|-------------|
| HIGH_RATE_HIGH_INFLATION | > 4.5% | > 3.0% | Quality +5%, Value +5% |
| HIGH_RATE_LOW_INFLATION | > 4.5% | <= 3.0% | Value +3%, Quality +2% |
| LOW_RATE_HIGH_INFLATION | <= 4.5% | > 3.0% | Value +5% |
| LOW_RATE_LOW_INFLATION | <= 4.5% | <= 3.0% | Growth +4%, Momentum +3% |

**Yield Curve Signal**

| 신호 | 10Y-2Y 스프레드 | 조정 |
|------|-----------------|------|
| INVERTED | < 0% | Quality +3%, Momentum -3% |
| STEEPENING | > 1.0% | Growth +2%, Momentum +2% |

</details>

<details>
<summary><b>Volatility Engine 상세 (Phase 3.4.2)</b></summary>

**Factor별 민감도**

| Factor | Sensitivity | 설명 |
|--------|-------------|------|
| Momentum | 1.0 (100%) | 추세 신뢰도 직결 |
| Growth | 0.7 (70%) | 성장주 특성 |
| Value | 0.3 (30%) | 타이밍 의존도 낮음 |
| Quality | 0.1 (10%) | 펀더멘털 기반 |

**IV Percentile 기반 조정**

| IV Percentile | Modifier |
|---------------|----------|
| >= 90% | -10점 |
| >= 80% | -7점 |
| <= 10% | +5점 |
| <= 20% | +3점 |

</details>

### 시장 레짐 감지

| 레짐 | 설명 | 가중치 배분 |
|------|------|-------------|
| AI_BULL | AI 주도 성장주 랠리 | Momentum 35%, Growth 30% |
| TIGHTENING | 금리 인상/긴축 | Quality 40%, Value 30% |
| RECOVERY | 경기 회복기 | Value 30%, Momentum 30% |
| CRISIS | 위기/급락장 | Quality 50%, Value 25% |
| NEUTRAL | 혼조장/중립 | 균형 배분 (25% 각) |

### 거래소/섹터별 최적화

<details>
<summary><b>NASDAQ 최적화</b></summary>

**NASDAQ 기본 가중치**
- Growth: 45% (IC 0.136 최고)
- Value: 20%
- Quality: 20%
- Momentum: 15% (IC 0.056 최저)

</details>

**Cash Runway 평가**

| Runway | 패널티 | 위험도 |
|--------|--------|--------|
| < 12개월 | -15점 | Critical |
| < 18개월 | -10점 | High Risk |
| < 24개월 | -5점 | Warning |

---

## 핵심 기술 특징

<details>
<summary><b>배치 최적화 (99.96% 쿼리 감소)</b></summary>

**최적화 전**: 36,400 쿼리/배치
**최적화 후**: 15 쿼리/배치

**방법**:
1. 공통 데이터 1회 계산
2. 종목별 데이터 벌크 조회 (4-5 쿼리)
3. 메모리에서 가중치 계산 (DB 쿼리 없음)

</details>

<details>
<summary><b>Agent용 지표 생성</b></summary>

| 지표 | 설명 |
|------|------|
| ATR 기반 손절/익절 | 섹터별 ATR 배수 적용 |
| Entry Timing Score | 진입 타이밍 점수 |
| Position Size | 변동성 기반 포지션 사이즈 |
| Scenario Probability | 상승/횡보/하락 확률 |

**Multi-Tier Stop Loss**

| Tier | 유형 |
|------|------|
| 1 | ATR Stop |
| 2 | Trailing Stop |
| 3 | Time-Based Stop |
| 4 | Score Degradation |

</details>

<details>
<summary><b>리스크 지표</b></summary>

| 지표 | 설명 |
|------|------|
| VaR 95% | Value at Risk |
| CVaR 95% | Conditional VaR |
| MDD | 최대 낙폭 (1년) |
| Beta | 시장 민감도 |
| Sharpe Ratio | 위험조정수익률 |
| Sortino Ratio | 하락 위험조정수익률 |

</details>

---

## 성능 특성

| 항목 | 한국 | 미국 |
|------|------|------|
| 분석 대상 | ~2,748 종목 | ~4,500 종목 |
| 배치 크기 | 20 종목/배치 | 20 종목/배치 |
| DB 연결 풀 | 10~45 connections | 10~45 connections |
| 평균 처리 시간 | ~10-15분 | ~15-20분 |

---

## 라이선스

**All Rights Reserved**

이 프로젝트의 모든 권리는 저작권자에게 있습니다.

- 본 코드의 복제, 배포, 수정, 상업적/비상업적 사용을 금지합니다.
- 채용 검토 목적의 열람만 허용됩니다.
- 무단 사용 시 법적 책임을 물을 수 있습니다.

문의: 저장소 소유자에게 연락해 주세요.
