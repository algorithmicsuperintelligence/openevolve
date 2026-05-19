#!/usr/bin/env bash
# Runtime CPU-core isolation via cgroup v2 isolated partition.
#
# Creates /sys/fs/cgroup/<CGROUP_NAME>, sets cpuset.cpus to the isolated list,
# and promotes it to an "isolated" partition root. The kernel then:
#   - excludes those CPUs from every other cgroup (system.slice, user.slice, …)
#   - disables scheduler load balancing on them
# Tasks must be placed into this cgroup explicitly (e.g. docker
# --cgroup-parent=/<CGROUP_NAME>). IRQ affinity is masked to the remaining
# system CPUs so device interrupts don't fire on the isolated cores.
#
# Usage:
#   sudo ./scripts/host-isolate-cores.sh start
#   sudo ./scripts/host-isolate-cores.sh stop
#   sudo ./scripts/host-isolate-cores.sh status
#
# Env overrides:
#   ISOLATED_CPUS=1-6              CPUs to reserve for the docker workload
#   CGROUP_NAME=isolated.slice     Cgroup name under /sys/fs/cgroup

set -euo pipefail

ISOLATED_CPUS="${ISOLATED_CPUS:-1-6}"
CGROUP_NAME="${CGROUP_NAME:-isolated.slice}"
CG_ROOT="/sys/fs/cgroup"
CG_PATH="$CG_ROOT/$CGROUP_NAME"
STATE_DIR="/var/lib/host-isolate-cores"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

need_root() {
    if [[ $EUID -ne 0 ]]; then
        echo -e "${RED}Must run as root (use sudo).${NC}" >&2
        exit 1
    fi
}

check_cgroup_v2() {
    if [[ ! -f "$CG_ROOT/cgroup.controllers" ]]; then
        echo -e "${RED}cgroup v2 not detected at $CG_ROOT${NC}" >&2
        exit 1
    fi
    if ! grep -qw cpuset "$CG_ROOT/cgroup.controllers"; then
        echo -e "${RED}cpuset controller not available in cgroup v2 root${NC}" >&2
        exit 1
    fi
}

enable_cpuset_subtree() {
    if ! grep -qw cpuset "$CG_ROOT/cgroup.subtree_control" 2>/dev/null; then
        echo "+cpuset" > "$CG_ROOT/cgroup.subtree_control"
    fi
}

cpu_list_to_mask() {
    python3 - "$1" <<'PY'
import sys
mask = 0
for p in sys.argv[1].split(','):
    if '-' in p:
        a, b = p.split('-')
        for i in range(int(a), int(b) + 1):
            mask |= 1 << i
    elif p:
        mask |= 1 << int(p)
print(f'{mask:x}')
PY
}

compute_system_cpus() {
    python3 - "$ISOLATED_CPUS" <<'PY'
import sys, os
iso = set()
for p in sys.argv[1].split(','):
    if '-' in p:
        a, b = p.split('-'); iso.update(range(int(a), int(b) + 1))
    elif p:
        iso.add(int(p))
remain = sorted(set(range(os.cpu_count())) - iso)
out, i = [], 0
while i < len(remain):
    j = i
    while j + 1 < len(remain) and remain[j + 1] == remain[j] + 1:
        j += 1
    out.append(str(remain[i]) if i == j else f'{remain[i]}-{remain[j]}')
    i = j + 1
print(','.join(out))
PY
}

start_isolation() {
    need_root
    check_cgroup_v2
    mkdir -p "$STATE_DIR"

    local system_cpus
    system_cpus="$(compute_system_cpus)"
    echo -e "${BLUE}Isolated cores : $ISOLATED_CPUS${NC}"
    echo -e "${BLUE}System cores   : $system_cpus${NC}"
    echo -e "${BLUE}Cgroup         : $CG_PATH${NC}"

    enable_cpuset_subtree
    mkdir -p "$CG_PATH"

    # cpuset.mems must be set before cpuset.cpus on some kernels
    if [[ -f "$CG_PATH/cpuset.mems" ]]; then
        local mems
        mems="$(cat "$CG_ROOT/cpuset.mems.effective" 2>/dev/null || echo 0)"
        echo "$mems" > "$CG_PATH/cpuset.mems" 2>/dev/null || echo 0 > "$CG_PATH/cpuset.mems"
    fi

    echo "$ISOLATED_CPUS" > "$CG_PATH/cpuset.cpus"

    # Newer kernels require an explicit exclusive set before promoting to a
    # partition root. Older ones derive it; the write is harmless either way.
    if [[ -f "$CG_PATH/cpuset.cpus.exclusive" ]]; then
        echo "$ISOLATED_CPUS" > "$CG_PATH/cpuset.cpus.exclusive" 2>/dev/null || true
    fi

    if echo isolated > "$CG_PATH/cpuset.cpus.partition" 2>/dev/null; then
        local pstate
        pstate="$(cat "$CG_PATH/cpuset.cpus.partition")"
        if [[ "$pstate" == "isolated" ]]; then
            echo -e "${GREEN}  partition = isolated (cpus removed from every other cgroup)${NC}"
        else
            echo -e "${YELLOW}  partition state: $pstate -- kernel rejected isolation.${NC}"
            echo -e "${YELLOW}  Hint: 'cat $CG_PATH/cpuset.cpus.partition' shows the reason in brackets.${NC}"
        fi
    else
        echo -e "${YELLOW}  could not write cpuset.cpus.partition; falling back to root partition${NC}"
        echo root > "$CG_PATH/cpuset.cpus.partition" 2>/dev/null || true
    fi

    # IRQ affinity -> system cores only
    local mask
    mask="$(cpu_list_to_mask "$system_cpus")"
    echo -e "${BLUE}IRQ affinity mask: 0x$mask${NC}"
    : > "$STATE_DIR/irq_backup.tsv"
    if [[ -f /proc/irq/default_smp_affinity ]]; then
        printf 'default\t%s\n' "$(cat /proc/irq/default_smp_affinity)" >> "$STATE_DIR/irq_backup.tsv"
        echo "$mask" > /proc/irq/default_smp_affinity 2>/dev/null || true
    fi
    local ok=0 fail=0
    for d in /proc/irq/[0-9]*; do
        [[ -w "$d/smp_affinity" ]] || continue
        local prev
        prev="$(cat "$d/smp_affinity" 2>/dev/null || true)"
        printf '%s\t%s\n' "$d" "$prev" >> "$STATE_DIR/irq_backup.tsv"
        if echo "$mask" > "$d/smp_affinity" 2>/dev/null; then
            ok=$((ok + 1))
        else
            fail=$((fail + 1))
        fi
    done
    echo -e "${GREEN}  IRQs redirected: $ok ok, $fail unmovable (per-CPU IRQs are expected)${NC}"

    cat > "$STATE_DIR/state" <<EOF
ISOLATED_CPUS=$ISOLATED_CPUS
SYSTEM_CPUS=$system_cpus
CGROUP_NAME=$CGROUP_NAME
EOF

    echo
    echo -e "${GREEN}Done.${NC}"
    echo -e "${YELLOW}Run docker with:${NC}"
    echo -e "  ${BLUE}./docker-run.sh --pin $ISOLATED_CPUS${NC}"
    echo
    echo -e "${YELLOW}For full isolation (kernel threads + ticks), reboot with:${NC}"
    echo -e "  isolcpus=$ISOLATED_CPUS nohz_full=$ISOLATED_CPUS rcu_nocbs=$ISOLATED_CPUS"
}

stop_isolation() {
    need_root
    check_cgroup_v2

    if [[ -d "$CG_PATH" ]]; then
        echo member > "$CG_PATH/cpuset.cpus.partition" 2>/dev/null || true
        if [[ -f "$CG_PATH/cgroup.procs" ]]; then
            while read -r pid; do
                [[ -z "$pid" ]] && continue
                echo "$pid" > "$CG_ROOT/cgroup.procs" 2>/dev/null || true
            done < "$CG_PATH/cgroup.procs"
        fi
        find "$CG_PATH" -mindepth 1 -depth -type d -exec rmdir {} \; 2>/dev/null || true
        if ! rmdir "$CG_PATH" 2>/dev/null; then
            echo -e "${YELLOW}  $CG_PATH still has child cgroups (leftover docker containers?)${NC}"
        fi
    fi

    if [[ -f "$STATE_DIR/irq_backup.tsv" ]]; then
        echo -e "${BLUE}Restoring IRQ affinity...${NC}"
        while IFS=$'\t' read -r key prev; do
            if [[ "$key" == "default" ]]; then
                echo "$prev" > /proc/irq/default_smp_affinity 2>/dev/null || true
            else
                [[ -w "$key/smp_affinity" ]] || continue
                echo "$prev" > "$key/smp_affinity" 2>/dev/null || true
            fi
        done < "$STATE_DIR/irq_backup.tsv"
        rm -f "$STATE_DIR/irq_backup.tsv"
    fi
    rm -f "$STATE_DIR/state"
    echo -e "${GREEN}Restored.${NC}"
}

show_status() {
    echo -e "${BLUE}=== host-isolate-cores status ===${NC}"
    if [[ -f "$STATE_DIR/state" ]]; then
        cat "$STATE_DIR/state"
    else
        echo "  (not active)"
    fi
    echo
    if [[ -d "$CG_PATH" ]]; then
        echo "Cgroup: $CG_PATH"
        echo "  cpuset.cpus            = $(cat "$CG_PATH/cpuset.cpus" 2>/dev/null)"
        echo "  cpuset.cpus.effective  = $(cat "$CG_PATH/cpuset.cpus.effective" 2>/dev/null)"
        echo "  cpuset.cpus.partition  = $(cat "$CG_PATH/cpuset.cpus.partition" 2>/dev/null)"
        echo "  tasks                  = $(wc -l < "$CG_PATH/cgroup.procs" 2>/dev/null || echo 0)"
    else
        echo "Cgroup $CG_PATH does not exist."
    fi
    echo
    echo "Kernel cpuset.cpus.isolated:"
    echo "  $(cat "$CG_ROOT/cpuset.cpus.isolated" 2>/dev/null || echo n/a)"
}

case "${1:-}" in
    start)  start_isolation ;;
    stop)   stop_isolation ;;
    status) show_status ;;
    *)
        echo "Usage: $0 {start|stop|status}"
        echo "Env: ISOLATED_CPUS=$ISOLATED_CPUS  CGROUP_NAME=$CGROUP_NAME"
        exit 1
        ;;
esac
