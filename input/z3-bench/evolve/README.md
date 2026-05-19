# Z3 Parameter Tuning via OpenEvolve — 실행 & 구조

OpenEvolve를 사용해 Z3 SMT 솔버의 파라미터를 진화적으로 탐색한다. 입력은 `input/z3-bench/` 데이터셋(50개 SMT2 인스턴스 + baseline 실행 로그). 목표는 baseline 대비 wall-clock 시간 단축, 단 정답(Sat/Unsat) 보존.

> OpenEvolve 소개, 본 프로젝트의 목적/접근, 스코어링 공식, 설계 결정, 로드맵은 [OPENEVOLVE_INTRO.md](OPENEVOLVE_INTRO.md) 참고.

---

## 1. 디렉토리 구조

```
input/z3-bench/
├── README.md                         # 데이터셋 스키마 설명
├── problems.jsonl                    # baseline 실행 결과 50행
├── problems.csv                      # 평탄화 버전
├── raw-data/                         # 원본 SMT2 + meta jsonl
└── evolve/
    ├── README.md                     # 이 파일 (실행 & 구조)
    ├── OPENEVOLVE_INTRO.md           # 개념/목적/설계
    ├── config.yaml                   # 공유 OpenEvolve config
    ├── run_phase.sh                  # 1/2/3/4 phase 실행 진입점
    ├── build_samples.py              # stage1/stage2 sample 생성
    ├── extract_best.py               # phase N 종료 후 best 추출
    ├── prepare_phase4.py             # phase4 EVOLVE-BLOCK 머터리얼라이즈
    ├── rebaseline_local.py           # 로컬에서 baseline 재측정
    ├── final_verify.py               # P4 best 전수 검증
    ├── shared/
    │   ├── baseline_params.py        # BASELINE 19키, LOCKED 4키
    │   ├── score.py                  # geomean × solved_rate^2
    │   ├── z3_runner.py              # subprocess z3 CLI 호출
    │   ├── evaluator.py              # cascade stage1/stage2
    │   ├── stage1_sample.json        # 5문제 stratified sample (seed=42)
    │   ├── stage2_sample.json        # stage2 problem set
    │   └── phase{1,2,3}_best.json    # 각 phase 종료 후 생성됨
    ├── phase1_opt_sls/
    │   └── initial_program.py        # EVOLVE-BLOCK: OPT_SLS_OVERRIDES (~34키)
    ├── phase2_sat/
    │   └── initial_program.py        # EVOLVE-BLOCK: SAT_OVERRIDES (~121키)
    ├── phase3_smt/
    │   └── initial_program.py        # EVOLVE-BLOCK: SMT_OVERRIDES (~97키)
    └── phase4_unified/
        └── initial_program.py        # EVOLVE-BLOCK: UNIFIED_OVERRIDES (union)
```

## 2. 평가 흐름 (cascade)

```
LLM 변이된 initial_program.py
    ↓
evaluator.py:
    1. get_params() 호출 → dict
    2. LOCKED 위반 체크 → 위반 시 0점 + locked_violated artifact
    3. stage1 (5문제, per-problem 15s timeout):
        for each problem in stage1_sample:
            run_z3(smt2, params, timeout=15s)
            invalid_param 감지 시 즉시 0점 + 어떤 키인지 artifact
        score → cascade_threshold 0.3 통과 시 stage2 진입
    4. stage2 (50문제, per-problem 120s timeout):
        동일 방식, 전수
    5. 최종 metrics + per_problem artifacts (상위 20개) 반환
```

### Stage1 sample (stratified by baseline elapsed_ms, seed=42)

```
ac90ca97ff99      239 ms  Unsat   (fast)
133383a624ef      480 ms  Unsat   (fast)
29efe6d38d7b   12,712 ms  Unsat   (medium)
86468fd861ff   15,671 ms  Sat     (medium)
3854194b901b   66,100 ms  Sat     (slow)
```

5분위 버킷에서 하나씩 → Sat/Unsat + 빠름/느림 골고루.

## 3. Initial program 표준 형태

각 phase의 `initial_program.py`는 동일 패턴:

```python
import pathlib, sys
_SHARED = pathlib.Path(__file__).resolve().parent.parent / "shared"
sys.path.insert(0, str(_SHARED))

from baseline_params import BASELINE

# (phase 2-3-4는 이전 phase best.json 로드)
import json
_PHASE1 = json.loads((_SHARED / "phase1_best.json").read_text()) \
    if (_SHARED / "phase1_best.json").exists() else {}

# EVOLVE-BLOCK-START
PHASE_OVERRIDES = {
    "opt.priority": "pareto",
    "opt.maxsat_engine": "wmax",
    # ...
}
# EVOLVE-BLOCK-END

def get_params():
    p = dict(BASELINE)
    p.update(_PHASE1)            # 누적 (phase 2+)
    p.update(PHASE_OVERRIDES)    # 현재 phase
    return p

def get_phase_overrides():
    """extract_best.py가 사용 — 현재 phase의 dict만 반환."""
    return dict(PHASE_OVERRIDES)
```

evaluator는 phase를 모름 — `get_params()` 결과만 받음. `extract_best.py`는 `get_phase_overrides()`만 호출 → phase별 dict 분리 유지.

## 4. 실행 절차 (Docker)

### Host에서

```bash
export OPENAI_API_KEY="..."           # config.yaml의 api_base에 맞는 키
                                      # (현재 gemini-2.5-flash → Google AI Studio key)
./docker-run.sh dev -s z3evo          # interactive shell
```

`OPENAI_API_KEY`라는 이름은 OpenEvolve가 OpenAI 호환 SDK를 쓰기 때문. 실제 라우팅은 `config.yaml`의 `api_base`가 결정.

### Claude Code 백엔드 사용 시

`config.yaml`에서 `provider: claude_code`로 모델을 정의하면 (`configs/claude_code_example.yaml` 참고) API 키 대신 Claude Code 구독 인증을 쓸 수 있다. Docker 안에서 쓰려면:

1. **Host에서 (1회만)**: long-lived OAuth 토큰 생성
   ```bash
   claude setup-token                # 출력된 토큰 복사
   export CLAUDE_CODE_OAUTH_TOKEN="sk-..."
   ```
   macOS는 OAuth credential을 Keychain에 저장하므로 `~/.claude/` 마운트만으로는 인증 안 됨. 토큰 방식이 필수.

2. **docker-run.sh 실행**: 위 env var가 export 되어 있으면 자동 전달 + `~/.claude/` 마운트 (settings/sessions 공유).
   ```bash
   ./docker-run.sh dev -s z3evo
   ```

3. **Container 안 (1회만)**: `claude` CLI 설치. axion 이미지에는 Node.js/npm 없음 → Anthropic 공식 standalone installer 사용 (Node 번들, 시스템 의존성 없음).
   ```bash
   curl -fsSL https://claude.ai/install.sh | bash
   # 설치 위치: ~/.local/bin/claude
   export PATH="$HOME/.local/bin:$PATH"   # ~/.bashrc에 영구 추가 권장
   claude --version                       # sanity check
   pip install -e ".[claude-code]"        # claude-agent-sdk
   ```
   SDK 탐색 순서: `~/.npm-global/bin/claude` → `/usr/local/bin/claude` → `~/.local/bin/claude` → `~/.claude/local/claude` → PATH. 위 경로 그대로 작동.

   **설치 영속화**: 컨테이너는 `--rm`이라 종료 시 사라지지만 docker-run.sh가 `~/.axion-docker-persist/claude-local/`을 `/root/.local`로 마운트 → 한 번 설치하면 다음 컨테이너에서도 그대로 사용 가능.

   **host 차이**:
   - Linux host: docker-run.sh가 host `claude` 바이너리도 자동 마운트 (`/usr/local/bin/claude` ro) → installer 생략 가능.
   - Mac host: cross-OS 불가 → 위 installer 필수.

   **HOME 분리**: `~/.claude/`가 host에서 마운트되므로 host config 공유됨. 충돌 우려 시 컨테이너 안에서 `CLAUDE_CONFIG_DIR` 등 별도 path 지정.

4. **체크**: 인증 작동 여부
   ```bash
   python -c "from openevolve.llm.claude_code import ClaudeCodeLLM; print('ok')"
   ```

**주의**: Claude Pro/Max 구독은 5시간 윈도우 rate limit 있음. 큰 evolution run은 빠르게 막힘. 작은 iteration으로 검증 먼저.

### Container 안

```bash
cd $SCRIPT_DIR    # rootless: 호스트 경로 그대로 / root: /app

# 1회 셋업
pip install -e ".[dev]"
apt-get install -y z3              # 또는: pip install z3-solver (CLI 동반)
export OPENAI_API_KEY="..."        # 셸 안에서도 export 필요

# (이미 생성됨, 재생성 원할 때만)
python input/z3-bench/evolve/build_samples.py

# 순차 실행 — 각 phase 종료 시 extract_best.py 자동 호출
./input/z3-bench/evolve/run_phase.sh 1
./input/z3-bench/evolve/run_phase.sh 2
./input/z3-bench/evolve/run_phase.sh 3
./input/z3-bench/evolve/run_phase.sh 4
```

### 체크포인트 재개

OpenEvolve가 `phase{N}_*/openevolve_output/checkpoints/checkpoint_K/`에 자동 저장. 재개:

```bash
cd input/z3-bench/evolve/phase2_sat/
python /app/openevolve-run.py \
    initial_program.py \
    ../shared/evaluator.py \
    --config ../config.yaml \
    --checkpoint openevolve_output/checkpoints/checkpoint_50 \
    --iterations 100
```

### Detached 장시간 실행

```bash
./docker-run.sh dev -s z3evo -d
docker exec -it axion-cell-container-dev-$USER-z3evo bash
nohup ./input/z3-bench/evolve/run_phase.sh 1 \
    &> /app/logs/phase1.log &
```

## 5. 환경 변수

| 변수 | 기본 | 용도 |
|---|---|---|
| `OPENAI_API_KEY` | — | LLM API 키 (api_base에 맞는 것) |
| `OPENEVOLVE_MAX_PROBLEMS` | 50 | stage2 문제수 상한 (테스트용 축소) |
| `OPENEVOLVE_STAGE1_TIMEOUT` | 15 | stage1 문제당 초 |
| `OPENEVOLVE_STAGE2_TIMEOUT` | 120 | stage2 문제당 초 |
| `OPENEVOLVE_Z3_BIN` | `z3` | z3 바이너리 경로 |

## 6. 도커 안에서 추가 검증 필요

- `z3 -pmd | less` 출력으로 4.13.3.0의 실제 키 검증 (일부 키명/타입이 마이너 버전마다 다를 수 있음)
- baseline 변이로 stage1 1회 평가 직접 호출 → z3 binary 동작/타임아웃 검증
- LLM API 호출 sanity check (`config.yaml`의 api_base + 키 매칭)
