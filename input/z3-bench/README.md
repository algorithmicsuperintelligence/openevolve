# z3-bench 데이터셋

Z3 SMT 솔버(v4.13.3.0)를 SMT-LIB2 인스턴스 모음에 실행한 벤치마크 결과 데이터셋이다. 각 행은 한 인스턴스에 대한 1회 실행(seed 고정, 동일 파라미터 프로파일)의 결과(만족도, 경과 시간, 솔버 내부 통계, 인스턴스 특성)를 담는다. 솔버 파라미터 튜닝(예: Z3 옵션 자동 튜닝, 휴리스틱 학습)을 위한 학습/평가 코퍼스로 활용 가능하다.

## 구성

| 경로 | 설명 |
|---|---|
| `problems.jsonl` | 한 줄 = 한 실행 결과 (JSON 객체). 50개 행. |
| `problems.csv` | 동일 결과의 평탄화(flatten)된 CSV 버전. 중첩 객체는 일부 컬럼만 노출. |
| `raw-data/*.smt2` | 원본 SMT-LIB2 입력 파일. 파일명 = `<problem_sha256>.smt2`. |
| `raw-data/*__<hash>__seed<N>.meta.jsonl` | 실행별 메타 로그. `meta_path`로 참조됨. |

## 데이터셋 요약(현재 스냅샷)

- 행 수: 50
- Z3 버전: `4.13.3.0` (단일)
- `applied_params_hash`: 1종(`543b29ed...ffcec6`) — 모든 행이 동일 파라미터 프로파일로 실행됨
- `seed`: 0 (단일)
- 결과 분포(`z3_status.result`): `Sat` 31, `Unsat` 19
- 경과 시간(`z3_status.elapsed_ms`): min 221 / max 181,205 / avg ≈ 24,229 ms
- 변수 수(`features.num_variables`): min 3,797 / max 99,809
- `path == "primary"` 47행 / 누락 3행 (5, 19, 49번째 행 — `solver` 필드도 동반 누락; 그 외 모든 필드는 존재)

---

## JSONL 스키마

각 줄은 다음 최상위 필드를 가진 객체이다.

### 최상위 필드

| 필드 | 타입 | 설명 |
|---|---|---|
| `problem_sha256` | string (hex, 64) | SMT2 인스턴스 내용의 SHA-256. 인스턴스 고유 ID. |
| `smt2_filename` | string | SMT2 파일명 (`<problem_sha256>.smt2`). |
| `smt2_path` | string (absolute path) | 호스트 상의 SMT2 파일 절대 경로. `raw-data/` 아래. |
| `meta_path` | string (absolute path) | 실행 메타 로그(`*.meta.jsonl`) 절대 경로. |
| `solver` | string | 사용 솔버명. 현재 데이터는 `"z3"` (3개 행에서 누락). |
| `z3_version` | string | Z3 빌드 버전. 현재 `"4.13.3.0"`. |
| `seed` | int | 실행 시드. 현재 `0`. |
| `path` | string | 실행 경로 라벨. 현재 `"primary"` (3개 행에서 누락). 다중 분기 실행 시 변형(예: ablation) 구분 용도. |
| `applied_params_hash` | string (hex, 64) | `z3_applied_params`의 정규화 해시. 동일 파라미터 프로파일을 빠르게 그룹핑. |
| `cli_params` | object | 실행을 일으킨 상위 도구의 CLI 파라미터(아래 참조). |
| `z3_applied_params` | object | Z3에 실제로 인가된 옵션 키-값(아래 참조). |
| `features` | object | 인스턴스 정적 특성(아래 참조). |
| `z3_statistics` | object | Z3가 실행 후 출력한 내부 통계(아래 참조). |
| `z3_status` | object | 실행 종료 상태(`result`, `elapsed_ms`). |

> 참고: 일부 행에는 `error` 필드가 추가될 수 있으나, 현재 스냅샷에는 없음.

### `cli_params` (도구 측 호출 파라미터)

상위 벤치 러너에서 받은 인자. 문자열로 저장됨에 유의.

| 키 | 타입(원본) | 설명 |
|---|---|---|
| `solver` | str | 사용 솔버 식별자 (`"z3"`). |
| `tech` / `process` | str | 기법/프로세스 라벨 (예: `"sf4lpp"`). 인스턴스 전처리·인코딩 방법론 구분. |
| `effort` | str | 자원 등급 라벨 (`"low"` 등). |
| `seed` | str(int) | 시드. |
| `solver_iter_timeout` | str(float, 초) | 솔버 1회 호출 타임아웃. 현재 `"3600.0"`. |
| `use_reboot` | str(bool) | 솔버 재시작(reboot) 사용 여부. |
| `optimize_m1` / `optimize_m2` | str(bool) | 1차/2차 목적함수 최적화 활성 여부. |
| `m2_offset` | str | 2차 목적의 오프셋 모드 (`"zero"` 등). |
| `num_of_heights` | str(int) | 도메인 특화 파라미터(높이 슬롯 수). |
| `unsat_debug` | str(bool) | UNSAT 디버그 모드 토글. |

### `z3_applied_params` (Z3 옵션 실제 인가값)

Z3에 set-option으로 전달된 키-값. `opt.*`, `sat.*`, `smt.*`, `sls.*`, `parallel.*` 네임스페이스로 구성.

| 키 | 타입 | 설명 |
|---|---|---|
| `opt.enable_core_rotate` | bool | MaxSAT 코어 회전 최적화 활성. |
| `opt.enable_sat` | bool | opt 엔진에서 SAT 백엔드 활성. |
| `opt.enable_sls` | bool | opt 엔진에서 SLS(확률적 지역 탐색) 활성. |
| `opt.maxres.hill_climb` | bool | MaxRes 힐 클라이밍 활성. |
| `opt.maxsat_engine` | string | MaxSAT 엔진 선택(`"wmax"` 등). |
| `opt.priority` | string | 다목적 우선순위 전략 (`"pareto"` 등). |
| `opt.rc2.totalizer` | bool | RC2의 totalizer 인코딩 사용. |
| `parallel.enable` | bool | 병렬 모드 활성. |
| `sat.branching.heuristic` | string | SAT 분기 휴리스틱 (`"vsids"` 등). |
| `sat.pb.solver` | string | PB(Pseudo-Boolean) 솔버 (`"totalizer"` 등). |
| `sat.phase` | string | 위상 선택 정책 (`"caching"` 등). |
| `sat.restart` | string | 재시작 정책 (`"geometric"` 등). |
| `sat.random_seed` | int | SAT 코어 시드. |
| `sat.threads` | int | SAT 스레드 수. |
| `sls.random_seed` | int | SLS 시드. |
| `smt.phase_selection` | int | SMT 위상 선택 모드(번호). |
| `smt.random_seed` | int | SMT 시드. |
| `smt.threads` | int | SMT 스레드 수. |
| `threads` | int | 전역 스레드 수. |

### `features` (인스턴스 정적 특성)

| 키 | 타입 | 설명 |
|---|---|---|
| `num_variables` | int | 선언된 변수 총 개수. |
| `num_bool` | int | Bool 정렬(sort) 변수 수. |
| `num_int` | int | Int 정렬 변수 수. |
| `num_real` | int | Real 정렬 변수 수. |
| `num_hard_constraints` | int | 하드 제약 수(`assert`). |
| `num_soft_constraints` | int | 소프트 제약 수(`assert-soft`). |
| `num_minimize_objectives` | int | `minimize` 목적함수 수. |
| `num_maximize_objectives` | int | `maximize` 목적함수 수. |

### `z3_status` (종료 상태)

| 키 | 타입 | 설명 |
|---|---|---|
| `result` | string | `"Sat"` / `"Unsat"` / `"Unknown"` / 에러 라벨. |
| `elapsed_ms` | int | 벽시계 경과 시간(ms). |

### `z3_statistics` (Z3 내부 통계)

Z3의 `(get-statistics)` 출력을 키-값으로 보존(키에 공백/하이픈 포함). 인스턴스마다 일부 키만 출력될 수 있으므로 누락 가능. 주요 키(현 스냅샷에서 관측된 35개):

- 시간/메모리/자원: `time`(초), `memory`, `max memory`(MB), `rlimit count`, `num allocs`
- 검색 통계: `conflicts`, `decisions`, `restarts`, `propagations`, `binary propagations`, `final checks`, `num checks`, `minimized lits`, `del clause`, `mk clause`, `mk clause binary`, `mk bool var`
- 등식/단순화: `added eqs`, `solve-eqs-elim-vars`, `solve-eqs-steps`
- 산술(arith) 이론: `arith eq adapter`, `arith-bound-propagations-lp`, `arith-conflicts`, `arith-diseq`, `arith-fixed-eqs`, `arith-lower`, `arith-upper`, `arith-make-feasible`, `arith-max-columns`, `arith-max-rows`, `arith-offset-eqs`
- Pseudo-Boolean: `pb conflicts`, `pb predicates`, `pb propagations`, `pb resolves`

타입은 정수 또는 부동소수(특히 `time`, `*memory*`).

---

## CSV 스키마 (`problems.csv`)

JSONL의 평탄화 버전. 컬럼은 다음과 같으며, 중첩 객체에서 일부 핵심 필드만 노출한다.

```
problem_sha256, applied_params_hash, seed, solver, path, z3_version,
result, elapsed_ms,
num_variables, num_bool, num_int, num_real,
num_hard_constraints, num_soft_constraints,
num_minimize_objectives, num_maximize_objectives,
cli_effort, cli_tech, cli_process, cli_solver_iter_timeout,
cli_use_reboot, cli_optimize_m1, cli_optimize_m2, cli_num_of_heights,
z3_conflicts, z3_decisions, z3_propagations, z3_final_checks, z3_num_checks,
z3_max_memory_mb, z3_time_s, z3_rlimit_count,
smt2_filename, smt2_path, meta_path,
error
```

- `result`, `elapsed_ms` ← `z3_status.*`
- `num_*` ← `features.*`
- `cli_*` ← `cli_params.*`(접두사 부여)
- `z3_*` ← `z3_statistics`의 핵심 키(공백/하이픈 → 언더스코어)
- 전체 `z3_applied_params` 및 모든 `z3_statistics` 키가 필요하면 JSONL을 사용할 것.

---

## 사용 예

```python
import json, pandas as pd

rows = [json.loads(l) for l in open("input/z3-bench/problems.jsonl")]

# Sat/Unsat 분포
from collections import Counter
print(Counter(r["z3_status"]["result"] for r in rows))

# 인스턴스 크기 vs 시간
df = pd.DataFrame([{
    "sha": r["problem_sha256"][:8],
    "vars": r["features"]["num_variables"],
    "hard": r["features"]["num_hard_constraints"],
    "soft": r["features"]["num_soft_constraints"],
    "ms":   r["z3_status"]["elapsed_ms"],
    "result": r["z3_status"]["result"],
} for r in rows])
print(df.describe())
```

## 비고

- 모든 행이 동일 `applied_params_hash`/seed 0이므로, 현재 스냅샷은 "단일 파라미터 프로파일에서의 인스턴스 난이도 분포"를 나타낸다. 파라미터 비교 실험을 위해서는 동일 `problem_sha256`에 대해 여러 프로파일을 추가 수집해야 한다.
- 일부 행에서 `path`/`solver` 누락이 있다 → 다운스트림 파서는 결측 허용 처리 권장.
- `z3_statistics` 키 집합은 인스턴스/실행마다 가변. 고정 컬럼 매트릭스를 만들려면 누락은 0/NaN으로 채울 것.
- 큰 인스턴스(`num_variables` ≈ 10⁵, `elapsed_ms` 최대 ≈ 181s)도 포함되어 있으니, 학습 시 시간 컷오프/정규화 고려.
