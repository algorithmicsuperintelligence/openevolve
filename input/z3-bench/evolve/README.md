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

### 4.1 백엔드 선택

| 항목 | OpenAI 호환 (기본) | Claude Code |
|---|---|---|
| 인증 | env var `OPENAI_API_KEY` (Gemini/OpenAI/local 등) | env var `CLAUDE_CODE_OAUTH_TOKEN` (host에서 `claude setup-token`) |
| config.yaml model | `provider` 없음 또는 `openai` | `provider: claude_code` |
| Config 예시 | `configs/default_config.yaml` | `configs/claude_code_example.yaml` |
| Rate limit | 키 발급사 정책 | Pro/Max 5h window — 빠르게 막힘 |
| 재현성 | `temperature`/`seed` 적용 | SDK 미지원, 약함 |
| Docker 자동 셋업 | skip 권장 (`AUTO_INSTALL_CLAUDE=0`) | docker-init-claude.sh가 자동 실행 |

### 4.2 Quickstart — Claude Code 백엔드

```bash
# === Host에서 (1회만) ===
claude setup-token                          # 출력 토큰 복사 (Keychain 우회용 long-lived OAuth)
export CLAUDE_CODE_OAUTH_TOKEN="sk-..."     # ~/.zshrc 등 영구화 권장

./docker-run.sh dev -s z3evo                # 진입 — 첫 실행: claude CLI + SDK 자동 설치 (수 분)
                                            # 두 번째부터: 즉시 셸 (영속 mount로 skip)
```

```bash
# === Container 안 (1회만) ===
pip install -e ".[dev]"                     # OpenEvolve 본체
apt-get install -y z3                       # 또는: pip install z3-solver

python -c "from openevolve.llm.claude_code import ClaudeCodeLLM; print('ok')"  # sanity check

# 데이터 sample (이미 생성됨; 재생성 원할 때만)
python input/z3-bench/evolve/build_samples.py

# Phase 순차 실행 — 각 phase 종료 시 extract_best.py 자동 호출
./input/z3-bench/evolve/run_phase.sh 1
./input/z3-bench/evolve/run_phase.sh 2
./input/z3-bench/evolve/run_phase.sh 3
./input/z3-bench/evolve/run_phase.sh 4
```

### 4.3 Quickstart — OpenAI 호환 백엔드

```bash
# === Host에서 ===
export OPENAI_API_KEY="..."                 # config.yaml의 api_base에 맞는 키
AUTO_INSTALL_CLAUDE=0 ./docker-run.sh dev -s z3evo   # claude 셋업 skip

# === Container 안 ===
pip install -e ".[dev]"
apt-get install -y z3
export OPENAI_API_KEY="..."                 # 셸 안에서도 필요 (-e로 전달됨)
./input/z3-bench/evolve/run_phase.sh 1
```

`OPENAI_API_KEY`라는 이름은 OpenEvolve가 OpenAI 호환 SDK를 쓰기 때문. 실제 라우팅은 `config.yaml`의 `api_base`가 결정 (Gemini, OpenAI, vLLM 등).

### 4.4 docker-run.sh의 Claude Code 셋업 동작

docker-run.sh가 [scripts/docker-init-claude.sh](../../../scripts/docker-init-claude.sh)를 컨테이너 startup 명령으로 실행. **멱등** (이미 설치돼 있으면 skip).

| 단계 | 동작 | Skip 조건 |
|---|---|---|
| 1 | `~/.local/bin`을 PATH 추가 + `~/.bashrc` 영구화 | grep로 중복 차단 |
| 2 | `claude` CLI 설치 (`curl -fsSL https://claude.ai/install.sh \| bash`) | `command -v claude` 성공 |
| 3 | `pip install -e ".[claude-code]"` (= claude-agent-sdk) | `import claude_agent_sdk` 성공 |
| 4 | Auth env var 체크 → 없으면 경고 | — |

**영속화**: `--rm` 컨테이너지만 `~/.axion-docker-persist/claude-local/` → `/root/.local` 마운트로 설치 결과 호스트에 남음. 다음 컨테이너 즉시 사용.

**자동 셋업 끄기**: `AUTO_INSTALL_CLAUDE=0 ./docker-run.sh ...`

**Host OS별 차이**:
- Linux host: host의 `claude` 바이너리도 read-only 자동 마운트 (`/usr/local/bin/claude`) → init script가 installer skip.
- Mac host: cross-OS 불가 → init script가 standalone installer 실행 (Node 번들 포함).

**Mount 요약** (root mode 기준):
- `$HOME/.claude/` → `/root/.claude` (settings/sessions/projects 공유)
- `~/.axion-docker-persist/claude-local/` → `/root/.local` (claude 바이너리 영속)
- env forward: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`, `CLAUDE_CODE_OAUTH_TOKEN`, `CLAUDE_CODE_USE_BEDROCK`, `CLAUDE_CODE_USE_VERTEX`

### 4.5 체크포인트 재개

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

### 4.6 Detached 장시간 실행

```bash
./docker-run.sh dev -s z3evo -d
docker exec -it axion-cell-container-dev-$USER-z3evo bash
nohup ./input/z3-bench/evolve/run_phase.sh 1 \
    &> /app/logs/phase1.log &
```

Detached mode에서도 init script가 background로 한 번 실행됨. `docker logs $CONTAINER_NAME`로 init 출력 확인.

### 4.7 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| 진입 시 `[init-claude] no CLAUDE_CODE_OAUTH_TOKEN ...` 경고 | host에서 `claude setup-token` → token export → 재실행 |
| `claude --version` 안 됨 | `source ~/.bashrc` 또는 `export PATH="$HOME/.local/bin:$PATH"` |
| init이 매번 재설치 | `~/.axion-docker-persist/claude-local/` 마운트 누락 (rootless면 `$HOME` bind에 포함되어 있어야 함). `ls ~/.local/bin/claude` 확인 |
| `pip install` 실패 | cwd 확인 (`pyproject.toml` 있는지) — rootless면 `cd $SCRIPT_DIR` 먼저 |
| Rate limit | Pro/Max 5h window 소진. `OPENEVOLVE_MAX_PROBLEMS` 축소 또는 OpenAI 호환 백엔드 병행 |
| Mac에서 host claude 마운트 시도 | Mac 바이너리는 Linux 컨테이너에서 안 돎 → docker-run.sh가 자동 skip하고 installer로 fallback |

### 4.8 CPU 핀닝 / 코어 격리 (측정 안정성)

z3 wall-clock을 일관되게 재기 위해 컨테이너를 특정 코어에 고정. 호스트 사이드 격리(cgroup v2 isolated partition)와 컨테이너 핀닝을 짝지어 사용한다.

```bash
# === Host (재부팅 없음) ===
sudo ./scripts/host-isolate-cores.sh start    # cores 1-6 격리
sudo ./scripts/host-isolate-cores.sh status   # 상태 확인
# 끝나면
sudo ./scripts/host-isolate-cores.sh stop     # 원복

# === Docker 실행 ===
./docker-run.sh dev -s z3evo --pin            # = --pin 1-6 (default)
./docker-run.sh dev -s z3evo --pin 2-7        # 다른 범위
```

`--pin` 동작:
- `--cpuset-cpus=<list>` 추가 (커널 cpuset 강제)
- `/sys/fs/cgroup/isolated.slice` 존재 시 `--cgroup-parent=/isolated.slice`로 격리된 cgroup에 합류 (없으면 경고만)
- 컨테이너 엔트리포인트를 `taskset -c <list> bash -lc ...`로 래핑

`host-isolate-cores.sh start` 메커니즘:
- `/sys/fs/cgroup/isolated.slice` 생성 → `cpuset.cpus=1-6`, `cpuset.cpus.partition=isolated`
- 커널이 1-6을 system.slice/user.slice 등 다른 모든 cgroup에서 제거 + 스케줄러 load-balancing 끔
- `/proc/irq/*/smp_affinity`를 `0,7-19`로 마스크 → 디바이스 IRQ가 격리 코어에 안 떨어짐
- 상태/IRQ 백업은 `/var/lib/host-isolate-cores/`에 저장됨 (stop이 원복)

| 변수 | 기본 | 용도 |
|---|---|---|
| `ISOLATED_CPUS` | `1-6` | 격리할 CPU 리스트 (`host-isolate-cores.sh`) |
| `CGROUP_NAME` | `isolated.slice` | 격리 cgroup 이름 |
| `ISOLATED_CGROUP_NAME` | `isolated.slice` | `docker-run.sh`가 합류할 cgroup 이름 |

**한계**: 런타임 격리로는 커널 스레드(kthread)와 타이머 틱은 못 막음. 완전 격리가 필요하면 boot param을 추가하고 재부팅:

```
isolcpus=1-6 nohz_full=1-6 rcu_nocbs=1-6
```

(`host-isolate-cores.sh start` 출력 마지막에 같은 안내가 나옴.)

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
