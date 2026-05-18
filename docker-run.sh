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
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                  # Run dev environment in interactive mode (default)"
    echo "  $0 dev              # Run dev environment in interactive mode"
    echo "  $0 prod -d          # Run prod environment in detached mode"
    echo "  $0 -d               # Run dev environment in detached mode"
    echo "  $0 dev -s test      # Container: axion-cell-container-dev-\$USER-test"
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
        "-w" "/app"
        "-e" "TZ=Asia/Seoul"
        "-e" "HOST_PROJECT_DIR=$SCRIPT_DIR"
    )
fi

# Add options based on execution mode
if [ "$DETACHED_MODE" = true ]; then
    DOCKER_RUN_OPTS+=("-d")
    echo -e "${YELLOW}Running in detached mode...${NC}"
else
    DOCKER_RUN_OPTS+=("-it")
    echo -e "${YELLOW}Running in interactive mode...${NC}"
fi

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
fi
echo ""

# Run Docker container
if docker run "${DOCKER_RUN_OPTS[@]}" "$LOCAL_IMAGE"; then
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
