#!/bin/bash

# Docker Container Run Script
# Usage: ./docker-run.sh [dev|prod] [options]
#   dev|prod          : Select development (dev) or production (prod) environment
#   -d, --detached    : Run in detached mode
#   -i, --interactive : Run in interactive mode (default)
#   -s, --suffix      : Add suffix to container name
#   -h, --help        : Show help message

set -e  # Exit on error

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
# Configuration
REGISTRY_URL="192.168.10.12:5050"
IMAGE_NAME="infra/axion-dev-docker"
USERNAME=$(whoami)

# 아키텍처 자동 감지
ARCH=$(uname -m)
case "$ARCH" in
    x86_64)  PLATFORM="linux/amd64" ;;
    aarch64) PLATFORM="linux/arm64" ;;
    arm64)   PLATFORM="linux/arm64" ;;
    *)       echo -e "${RED}Unsupported architecture: $ARCH${NC}"; exit 1 ;;
esac

# Current script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default settings
DETACHED_MODE=false
INTERACTIVE_MODE=true
ENV_TYPE="dev"
IMAGE_TAG="dev-latest"
CONTAINER_SUFFIX=""
PIN_CPUS=""
ISOLATED_CGROUP_NAME="${ISOLATED_CGROUP_NAME:-isolated.slice}"

# Help function
show_help() {
    echo "Usage: $0 [dev|prod] [options]"
    echo ""
    echo "Environment selection (optional, default: dev):"
    echo "  dev                 Use development environment image (dev-latest) [default]"
    echo "  prod                Use production environment image (prod-latest)"
    echo ""
    echo "Options:"
    echo "  -d, --detached      Run in detached mode"
    echo "  -i, --interactive   Run in interactive mode (default)"
    echo "  -s, --suffix TEXT   Add suffix to default container name"
    echo "      --pin [LIST]    Pin container to CPU cores (default: 1-6 if no LIST)."
    echo "                      Adds --cpuset-cpus, joins isolated cgroup if present,"
    echo "                      and wraps the entrypoint with taskset -c LIST."
    echo "                      Pair with: sudo ./scripts/host-isolate-cores.sh start"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                  # Run dev environment in interactive mode (default)"
    echo "  $0 dev              # Run dev environment in interactive mode"
    echo "  $0 prod -d          # Run prod environment in detached mode"
    echo "  $0 -d               # Run dev environment in detached mode"
    echo "  $0 dev -s test      # Container: axion-cell-container-dev-\$USER-test"
    echo "  $0 --pin            # Pin container to cores 1-6 (host isolation recommended)"
    echo "  $0 --pin 2-7        # Pin container to cores 2-7"
    echo ""
}

# Check for help argument
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Check environment type from first argument (optional)
if [[ $# -gt 0 ]]; then
    case $1 in
        dev|DEV)
            ENV_TYPE="dev"
            IMAGE_TAG="dev-latest"
            shift
            ;;
        prod|PROD)
            ENV_TYPE="prod"
            IMAGE_TAG="prod-latest"
            shift
            ;;
        -d|--detached|-i|--interactive|-s|--suffix)
            # If option comes first, use default (dev)
            ;;
        *)
            echo -e "${RED}Error: Invalid argument: $1${NC}"
            echo -e "${YELLOW}Please enter dev, prod, or a valid option.${NC}"
            echo ""
            show_help
            exit 1
            ;;
    esac
fi

# Image name configuration
REGISTRY_IMAGE="$REGISTRY_URL/$IMAGE_NAME:$IMAGE_TAG"
LOCAL_IMAGE="$IMAGE_NAME-$USERNAME:$IMAGE_TAG"
CONTAINER_NAME="axion-cell-container-$ENV_TYPE-$USERNAME"

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--detached)
            DETACHED_MODE=true
            INTERACTIVE_MODE=false
            shift
            ;;
        -i|--interactive)
            INTERACTIVE_MODE=true
            DETACHED_MODE=false
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--suffix)
            if [[ -z "$2" ]] || [[ "$2" == -* ]]; then
                echo -e "${RED}Error: $1 requires a suffix value.${NC}"
                show_help
                exit 1
            fi
            CONTAINER_SUFFIX="$2"
            shift 2
            ;;
        --pin)
            if [[ -n "${2:-}" ]] && [[ "$2" != -* ]]; then
                PIN_CPUS="$2"
                shift 2
            else
                PIN_CPUS="1-6"
                shift
            fi
            ;;
        --pin=*)
            PIN_CPUS="${1#--pin=}"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

if [[ -n "$CONTAINER_SUFFIX" ]]; then
    CONTAINER_NAME="${CONTAINER_NAME}-${CONTAINER_SUFFIX}"
fi

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  AxionCell Docker Container Run${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# Check Docker installation
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed.${NC}"
    exit 1
fi

# Check Docker daemon status and determine if using rootless Docker
ROOTLESS_DOCKER=false
if ! docker info &> /dev/null; then
    # Check if rootless docker is available
    if [[ -S "/run/user/$(id -u)/docker.sock" ]]; then
        export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"
        if docker info &> /dev/null; then
            ROOTLESS_DOCKER=true
            echo -e "${BLUE}Using rootless Docker (DOCKER_HOST=$DOCKER_HOST)${NC}"
        else
            echo -e "${RED}Error: Docker daemon is not running.${NC}"
            echo -e "${YELLOW}Hint: For rootless Docker, run: systemctl --user start docker${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Error: Docker daemon is not running.${NC}"
        echo -e "${YELLOW}Hint: You may need to install rootless Docker:${NC}"
        echo -e "  dockerd-rootless-setuptool.sh install"
        exit 1
    fi
else
    # Check if current user has root privileges for Docker
    if docker info 2>&1 | grep -q "rootless"; then
        ROOTLESS_DOCKER=true
        echo -e "${BLUE}Using rootless Docker${NC}"
    else
        echo -e "${BLUE}Using root Docker${NC}"
    fi
fi

# Pull image from registry
echo -e "${YELLOW}[$ENV_TYPE environment] Registry image: $REGISTRY_IMAGE${NC}"
echo -e "${BLUE}Platform: $PLATFORM${NC}"
echo -e "${BLUE}Pulling image from GitLab Registry...${NC}"
echo ""
if docker pull --platform "$PLATFORM" "$REGISTRY_IMAGE"; then
    echo -e "${GREEN}Successfully pulled image!${NC}"
    echo ""
else
    echo -e "${RED}Error: Failed to pull image.${NC}"
    echo -e "${YELLOW}Hint: You may need to login to GitLab Registry:${NC}"
    echo -e "  docker login $REGISTRY_URL"
    exit 1
fi

# Create local image tag with username
echo -e "${BLUE}Creating local image tag: $LOCAL_IMAGE${NC}"
if docker tag "$REGISTRY_IMAGE" "$LOCAL_IMAGE"; then
    echo -e "${GREEN}Successfully created local image tag!${NC}"
    echo ""
else
    echo -e "${RED}Error: Failed to create local image tag.${NC}"
    exit 1
fi

# Remove existing container if exists
if docker ps -a | grep -q "$CONTAINER_NAME"; then
    echo -e "${YELLOW}Removing existing container...${NC}"
    docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1 || true
fi

# Create result and log directories
mkdir -p "$SCRIPT_DIR/result"
mkdir -p "$SCRIPT_DIR/logs"

# Create persistent directories for Docker credentials and bash history
DOCKER_PERSIST_DIR="$HOME/.axion-docker-persist"
mkdir -p "$DOCKER_PERSIST_DIR"

# Initialize persistent files if they don't exist (to avoid mounting as directories)
touch "$DOCKER_PERSIST_DIR/.bash_history" 2>/dev/null || true

# Claude Code config dir — mounted into container so settings/sessions persist.
# Note: on macOS, OAuth credentials live in Keychain (not in this dir).
# To authenticate from inside the container, run `claude setup-token` on the
# host once and export CLAUDE_CODE_OAUTH_TOKEN (or ANTHROPIC_API_KEY) in your
# shell before invoking this script. The vars are forwarded below.
CLAUDE_CONFIG_DIR="$HOME/.claude"
mkdir -p "$CLAUDE_CONFIG_DIR" 2>/dev/null || true

# Persistent dir for container-side `claude` install (root mode uses --rm, so
# anything written to /root/.local is lost between runs). Mount this so the
# standalone installer's binary at ~/.local/bin/claude survives.
CLAUDE_LOCAL_PERSIST_DIR="$DOCKER_PERSIST_DIR/claude-local"
mkdir -p "$CLAUDE_LOCAL_PERSIST_DIR/bin" 2>/dev/null || true

# Collect Claude/Anthropic env vars to forward. Skip empty ones.
CLAUDE_ENV_OPTS=()
for var in OPENAI_API_KEY ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN \
           ANTHROPIC_BASE_URL CLAUDE_CODE_OAUTH_TOKEN CLAUDE_CODE_USE_BEDROCK \
           CLAUDE_CODE_USE_VERTEX; do
    if [[ -n "${!var}" ]]; then
        CLAUDE_ENV_OPTS+=("-e" "$var=${!var}")
    fi
done

# Container runs as root; Claude CLI refuses --dangerously-skip-permissions
# under uid 0 unless IS_SANDBOX=1 declares the env is already sandboxed.
CLAUDE_ENV_OPTS+=("-e" "IS_SANDBOX=1")

# Try to mount host's `claude` CLI into the container. Only safe when host and
# container share the same OS/arch (Linux x86_64 host, Linux x86_64 container).
# On macOS hosts the Mac binary will NOT run in a Linux container, so we skip
# the mount and rely on the container having `@anthropic-ai/claude-code`
# installed (e.g. `npm install -g @anthropic-ai/claude-code` once inside).
CLAUDE_CLI_MOUNT_OPTS=()
HOST_OS="$(uname -s)"
if [[ "$HOST_OS" == "Linux" ]]; then
    if HOST_CLAUDE_BIN=$(command -v claude 2>/dev/null); then
        # Resolve symlink (npm installs `claude` as a symlink to cli.js).
        HOST_CLAUDE_REAL=$(readlink -f "$HOST_CLAUDE_BIN" 2>/dev/null || echo "$HOST_CLAUDE_BIN")
        CLAUDE_CLI_MOUNT_OPTS+=(
            "-v" "$HOST_CLAUDE_BIN:/usr/local/bin/claude:ro"
        )
        # If symlink target lives elsewhere, mount the real file too.
        if [[ "$HOST_CLAUDE_REAL" != "$HOST_CLAUDE_BIN" ]]; then
            CLAUDE_CLI_MOUNT_OPTS+=("-v" "$HOST_CLAUDE_REAL:$HOST_CLAUDE_REAL:ro")
        fi
        echo -e "${BLUE}Mounting host claude CLI: $HOST_CLAUDE_BIN${NC}"
    fi
fi

# Configure Docker run options based on Docker mode (rootless vs root)
if [ "$ROOTLESS_DOCKER" = true ]; then
    # Rootless Docker: Mount user's home directory as per the guide
    # Container runs as root but binds to user's home directory
    DOCKER_RUN_OPTS=(
        "--name" "$CONTAINER_NAME"
        "--rm"
        "--platform" "$PLATFORM"
        "--cap-add=SYS_PTRACE"
        "--security-opt" "seccomp=unconfined"
        "-v" "$HOME:$HOME"
    )
    if [ -d "/home/share" ]; then
        DOCKER_RUN_OPTS+=("-v" "/home/share:/home/share")
    fi
    DOCKER_RUN_OPTS+=(
        "-v" "$SCRIPT_DIR/logs:$SCRIPT_DIR/logs"
        "-v" "$SCRIPT_DIR/result:$SCRIPT_DIR/result"
        "-v" "$HOME/.ssh:/root/.ssh"
        "-v" "$HOME/.gitconfig:/root/.gitconfig"
        "-w" "$SCRIPT_DIR"
        "-e" "HOME=$HOME"
        "-e" "TZ=Asia/Seoul"
    )
    # Rootless: $HOME bind already exposes ~/.claude. Just forward env vars
    # and (Linux only) the host's claude CLI binary.
    DOCKER_RUN_OPTS+=("${CLAUDE_ENV_OPTS[@]}")
    DOCKER_RUN_OPTS+=("${CLAUDE_CLI_MOUNT_OPTS[@]}")
    # Rootless Docker stores credentials in ~/.config/docker/config.json
    # Mount it to ~/.docker/config.json for container compatibility
    if [[ -f "$HOME/.config/docker/config.json" ]]; then
        DOCKER_RUN_OPTS+=("-v" "$HOME/.config/docker/config.json:$HOME/.docker/config.json:ro")
        DOCKER_RUN_OPTS+=("-v" "$HOME/.config/docker/config.json:/root/.docker/config.json:ro")
        mkdir -p "$HOME/.docker" 2>/dev/null || true
    fi
else
    # Root Docker: Mount project directory and persistent configs
    # Mount to both /root and /home/appuser to support both root and non-root container users
    DOCKER_RUN_OPTS=(
        "--name" "$CONTAINER_NAME"
        "--rm"
        "--platform" "$PLATFORM"
        "--cap-add=SYS_PTRACE"
        "--security-opt" "seccomp=unconfined"
        "-v" "$SCRIPT_DIR:/app"
        "-v" "$SCRIPT_DIR/logs:/app/logs"
        "-v" "$SCRIPT_DIR/result:/app/result"
        "-v" "$HOME/.ssh:/root/.ssh:ro"
        "-v" "$HOME/.ssh:/home/appuser/.ssh:ro"
        "-v" "$HOME/.docker/config.json:/root/.docker/config.json:ro"
        "-v" "$HOME/.docker/config.json:/home/appuser/.docker/config.json:ro"
        "-v" "$DOCKER_PERSIST_DIR/.bash_history:/root/.bash_history"
        "-v" "$DOCKER_PERSIST_DIR/.bash_history:/home/appuser/.bash_history"
        "-v" "$HOME/.gitconfig:/root/.gitconfig"
        "-v" "$HOME/.gitconfig:/home/appuser/.gitconfig"
        "-v" "$CLAUDE_CONFIG_DIR:/root/.claude"
        "-v" "$CLAUDE_CONFIG_DIR:/home/appuser/.claude"
        "-v" "$CLAUDE_LOCAL_PERSIST_DIR:/root/.local"
        "-v" "$CLAUDE_LOCAL_PERSIST_DIR:/home/appuser/.local"
        "-w" "/app"
        "-e" "TZ=Asia/Seoul"
        "-e" "HOST_PROJECT_DIR=$SCRIPT_DIR"
    )
    DOCKER_RUN_OPTS+=("${CLAUDE_ENV_OPTS[@]}")
    DOCKER_RUN_OPTS+=("${CLAUDE_CLI_MOUNT_OPTS[@]}")
fi

# CPU pinning: --cpuset-cpus is the kernel-level pin; --cgroup-parent attaches
# the container to the isolated cgroup created by scripts/host-isolate-cores.sh
# so it can claim CPUs that were carved out of the system slices. taskset
# inside the container is added below as belt-and-suspenders.
JOINED_ISOLATED_CGROUP=false
if [[ -n "$PIN_CPUS" ]]; then
    DOCKER_RUN_OPTS+=("--cpuset-cpus=$PIN_CPUS")
    if [[ -d "/sys/fs/cgroup/$ISOLATED_CGROUP_NAME" ]]; then
        DOCKER_RUN_OPTS+=("--cgroup-parent=/$ISOLATED_CGROUP_NAME")
        JOINED_ISOLATED_CGROUP=true
    else
        echo -e "${YELLOW}Note: /sys/fs/cgroup/$ISOLATED_CGROUP_NAME not found.${NC}"
        echo -e "${YELLOW}      Container will use --cpuset-cpus=$PIN_CPUS only (system tasks still share those cores).${NC}"
        echo -e "${YELLOW}      For host-side isolation, run: sudo ./scripts/host-isolate-cores.sh start${NC}"
    fi
fi

# Add options based on execution mode
if [ "$DETACHED_MODE" = true ]; then
    DOCKER_RUN_OPTS+=("-d")
    echo -e "${YELLOW}Running in detached mode...${NC}"
else
    DOCKER_RUN_OPTS+=("-it")
    echo -e "${YELLOW}Running in interactive mode...${NC}"
fi

# Build container startup command: auto-run claude-init then drop to shell
# (or `tail -f /dev/null` in detached mode). Disable via AUTO_INSTALL_CLAUDE=0.
# The init script lives at ./scripts/docker-init-claude.sh relative to the
# container's working directory (SCRIPT_DIR in rootless mode, /app in root).
INIT_SCRIPT_REL="./scripts/docker-init-claude.sh"
if [ "$DETACHED_MODE" = true ]; then
    CONTAINER_CMD=(
        "bash" "-lc"
        "if [ -x $INIT_SCRIPT_REL ]; then $INIT_SCRIPT_REL || true; fi; tail -f /dev/null"
    )
else
    CONTAINER_CMD=(
        "bash" "-lc"
        "if [ -x $INIT_SCRIPT_REL ]; then $INIT_SCRIPT_REL || true; fi; exec bash"
    )
fi
# Wrap with taskset so the in-container shell (and everything it spawns) is
# explicitly pinned even if the user later loosens cpuset.cpus.
if [[ -n "$PIN_CPUS" ]]; then
    CONTAINER_CMD=("taskset" "-c" "$PIN_CPUS" "${CONTAINER_CMD[@]}")
fi
DOCKER_RUN_OPTS+=("-e" "AUTO_INSTALL_CLAUDE=${AUTO_INSTALL_CLAUDE:-1}")

echo ""
echo -e "${BLUE}Run Configuration:${NC}"
echo -e "  Environment: $ENV_TYPE"
echo -e "  Docker Mode: $([ "$ROOTLESS_DOCKER" = true ] && echo "Rootless" || echo "Root")"
echo -e "  Registry Image: $REGISTRY_IMAGE"
echo -e "  Local Image: $LOCAL_IMAGE"
echo -e "  Container: $CONTAINER_NAME"
if [ "$ROOTLESS_DOCKER" = true ]; then
    echo -e "  Home Directory: $HOME (mounted as \$HOME)"
    echo -e "  Working Directory: $SCRIPT_DIR"
    if [ -d "/home/share" ]; then
        echo -e "  Shared Directory: /home/share"
    fi
else
    echo -e "  Project Directory: $SCRIPT_DIR (→ /app)"
    echo -e "  Result Directory: $SCRIPT_DIR/result"
    echo -e "  Log Directory: $SCRIPT_DIR/logs"
    echo -e "  Persistent Data: $DOCKER_PERSIST_DIR (bash history, gitconfig)"
    echo -e "  Docker Credentials: ~/.docker/config.json (read-only)"
    echo -e "  Claude Code Config: $CLAUDE_CONFIG_DIR (→ /root/.claude, /home/appuser/.claude)"
    echo -e "  Claude Local Bin:   $CLAUDE_LOCAL_PERSIST_DIR (→ /root/.local, /home/appuser/.local)"
fi
if [ ${#CLAUDE_ENV_OPTS[@]} -gt 0 ]; then
    echo -e "  Forwarded env vars:$(printf ' %s' "${CLAUDE_ENV_OPTS[@]}" | sed 's/-e //g' | sed 's/=[^ ]*/=***/g')"
fi
if [[ -n "$PIN_CPUS" ]]; then
    echo -e "  CPU pinning: $PIN_CPUS (taskset + --cpuset-cpus)"
    if [ "$JOINED_ISOLATED_CGROUP" = true ]; then
        echo -e "  Isolated cgroup: /$ISOLATED_CGROUP_NAME (joined)"
    fi
fi
echo ""

# Run Docker container
if docker run "${DOCKER_RUN_OPTS[@]}" "$LOCAL_IMAGE" "${CONTAINER_CMD[@]}"; then
    echo ""
    if [ "$DETACHED_MODE" = true ]; then
        echo -e "${GREEN}================================================${NC}"
        echo -e "${GREEN}  Container is running in background${NC}"
        echo -e "${GREEN}================================================${NC}"
        echo ""
        echo -e "${YELLOW}Useful commands:${NC}"
        echo -e "  View logs:        docker logs -f $CONTAINER_NAME"
        echo -e "  Container status: docker ps"
        echo -e "  Stop container:   docker stop $CONTAINER_NAME"
        echo -e "  Attach to shell:  docker exec -it $CONTAINER_NAME /bin/bash"
        echo ""
    else
        echo -e "${GREEN}Container has exited.${NC}"
    fi

    echo -e "${YELLOW}Result files location: $SCRIPT_DIR/result${NC}"
    echo -e "${YELLOW}Log files location: $SCRIPT_DIR/logs${NC}"
else
    echo ""
    echo -e "${RED}================================================${NC}"
    echo -e "${RED}  Container execution failed!${NC}"
    echo -e "${RED}================================================${NC}"
    exit 1
fi
