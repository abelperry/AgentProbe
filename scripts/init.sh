#!/usr/bin/env bash
# Prepare a machine to run AgentProbe: fetch the offline agent packages, start
# the OpenSandbox server, and print the command to run the bundled example.
#
# Safe to re-run: an existing package is verified against the registry checksum
# and reused, and an already-listening server is left alone.
#
#   ./scripts/init.sh              # Claude Code, glibc images (the common case)
#   ./scripts/init.sh --musl       # also fetch the musl build (Alpine images)
#   ./scripts/init.sh --opencode   # also fetch OpenCode and its Node runtime

set -euo pipefail

# Must match AgentConfig.version in src/agent_probe/config.py.
CC_VERSION="2.1.199"
# Must match OpenCodeAgent._DEFAULT_VERSION and AgentConfig.offline_node_version.
OPENCODE_VERSION="1.1.21"
NODE_VERSION="22.21.1"

REGISTRY="${NPM_REGISTRY:-https://registry.npmjs.org}"
SCOPE="@anthropic-ai"
OFFLINE_DIR="${OFFLINE_PACKAGE_DIR:-$PWD/data/offline_package}"
SANDBOX_CONFIG="${SANDBOX_CONFIG:-.sandbox.toml}"
SANDBOX_LOG=".sandbox.log"
SANDBOX_PID=".sandbox.pid"
ENV_FILE=".agentprobe-env"
EXAMPLE_CONFIG="examples/experiment.yaml"

WANT_MUSL=0
WANT_OPENCODE=0
for arg in "$@"; do
    case "$arg" in
        --musl)     WANT_MUSL=1 ;;
        --opencode) WANT_OPENCODE=1 ;;
        -h|--help)  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)          echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

say()  { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m warn\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror\033[0m %s\n' "$*" >&2; exit 1; }

[ -f "$EXAMPLE_CONFIG" ] || die "run this from the repository root (no $EXAMPLE_CONFIG here)"
command -v curl   >/dev/null || die "curl is required"
command -v uv     >/dev/null || die "uv is required -- https://docs.astral.sh/uv/"
command -v docker >/dev/null || die "docker is required by the sandbox docker runtime"

if command -v sha512sum >/dev/null; then
    sha512() { sha512sum "$1" | cut -d' ' -f1; }
    sha256() { sha256sum "$1" | cut -d' ' -f1; }
else
    sha512() { shasum -a 512 "$1" | cut -d' ' -f1; }
    sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
fi

# A 75 MB download over a flaky link fails often enough that resume and retry
# are not optional. --retry alone does not cover a truncated transfer, and a
# stalled-but-open connection never errors at all, so it also has to be timed
# out explicitly.
curl_opts=(-fL --retry 5 --retry-delay 3 --speed-time 60 --speed-limit 2048 -C -)
curl --help all 2>/dev/null | grep -q -- '--retry-all-errors' && curl_opts+=(--retry-all-errors)
if [ -t 1 ]; then curl_opts+=(--progress-bar); else curl_opts+=(--no-progress-meter); fi

# ---------------------------------------------------------------------------
# 1. Offline agent packages
# ---------------------------------------------------------------------------
# Configs install the CLI from this directory instead of running `npm i -g`
# inside the sandbox: benchmarks that start a container per round hit ECONNRESET
# otherwise, and judge images have no npm to fall back on.
#
# Fetched with curl rather than `npm pack` so the host needs no Node at all. The
# filenames below are exactly what npm pack would produce, which is what the
# in-sandbox installer looks for.
say "offline packages -> $OFFLINE_DIR"
mkdir -p "$OFFLINE_DIR"

fetch_npm() {
    local name="$1" version="$2"                     # e.g. @anthropic-ai/claude-code-linux-x64
    # npm pack names the file after the package with "@" dropped and "/" -> "-",
    # and that is exactly what the in-sandbox installer looks for. Build the
    # basename first: substituting on the full path would eat the directory's
    # own slashes too.
    local file="${name//@/}"
    file="${file//\//-}-$version.tgz"
    local dest="$OFFLINE_DIR/$file"
    local encoded="${name//\//%2f}"

    # Ask the registry for both the checksum and the tarball URL: the published
    # sha512 makes a truncated or corrupted file detectable instead of being
    # silently reused forever, and dist.tarball beats guessing the path.
    local meta want url
    meta="$(curl -fsSL -m 60 "$REGISTRY/$encoded/$version")" \
        || die "cannot reach the npm registry for $name@$version"
    want="$(printf '%s' "$meta" | python3 -c 'import base64, json, sys
dist = json.load(sys.stdin)["dist"]
print(base64.b64decode(dist["integrity"].removeprefix("sha512-")).hex())')"
    url="$(printf '%s' "$meta" | python3 -c 'import json, sys
print(json.load(sys.stdin)["dist"]["tarball"])')"

    if [ -f "$dest" ] && [ "$(sha512 "$dest")" = "$want" ]; then
        say "already present: $(basename "$dest")"
        return
    fi

    say "downloading $(basename "$dest")"
    curl "${curl_opts[@]}" -o "$dest.part" "$url" \
        || die "download failed -- partial data kept at $dest.part, re-run to resume"

    if [ "$(sha512 "$dest.part")" != "$want" ]; then
        rm -f "$dest.part"
        die "checksum mismatch for $name@$version -- discarded, re-run to retry"
    fi
    mv "$dest.part" "$dest"
    say "checksum verified"
}

fetch_node() {
    # OpenCode is a Node program, so an offline sandbox needs the runtime too.
    # The Claude Code CLI does not -- it ships as a self-contained binary.
    local archive="node-v$NODE_VERSION-linux-x64.tar.gz"
    local dest="$OFFLINE_DIR/$archive"
    local base="https://nodejs.org/dist/v$NODE_VERSION"

    local want
    want="$(curl -fsSL -m 60 "$base/SHASUMS256.txt" | awk -v f="$archive" '$2 == f {print $1}')" \
        || die "cannot reach nodejs.org for v$NODE_VERSION"
    [ -n "$want" ] || die "no checksum published for $archive"

    if [ -f "$dest" ] && [ "$(sha256 "$dest")" = "$want" ]; then
        say "already present: $archive"
        return
    fi

    say "downloading $archive"
    curl "${curl_opts[@]}" -o "$dest.part" "$base/$archive" \
        || die "download failed -- partial data kept at $dest.part, re-run to resume"
    if [ "$(sha256 "$dest.part")" != "$want" ]; then
        rm -f "$dest.part"
        die "checksum mismatch for $archive -- discarded, re-run to retry"
    fi
    mv "$dest.part" "$dest"
    say "checksum verified"
}

say "Claude Code $CC_VERSION"
fetch_npm "$SCOPE/claude-code-linux-x64" "$CC_VERSION"
[ "$WANT_MUSL" -eq 1 ] && fetch_npm "$SCOPE/claude-code-linux-x64-musl" "$CC_VERSION"

if [ "$WANT_OPENCODE" -eq 1 ]; then
    say "OpenCode $OPENCODE_VERSION (plus Node $NODE_VERSION)"
    fetch_node
    fetch_npm "opencode-ai" "$OPENCODE_VERSION"
    fetch_npm "opencode-linux-x64" "$OPENCODE_VERSION"
fi

# ---------------------------------------------------------------------------
# 2. OpenSandbox server
# ---------------------------------------------------------------------------
# Anything answering HTTP on the port counts -- a 401 or 404 still means a
# server is there.
sandbox_up() {
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/ 2>/dev/null)" != "000" ]
}

# The offline package directory is bind-mounted into every sandbox, so it has to
# be on the server's allowlist. The server reads its config once, at startup, so
# this is only worth doing on the path where we are about to launch it.
allow_offline_dir() {
    # An empty list means "allow every host path"; leave it alone.
    grep -qE '^[[:space:]]*allowed_host_paths[[:space:]]*=[[:space:]]*\[[[:space:]]*\]' \
        "$SANDBOX_CONFIG" && return 0
    grep -qE '^[[:space:]]*allowed_host_paths[[:space:]]*=' "$SANDBOX_CONFIG" || return 0
    grep -qF "\"$OFFLINE_DIR\"" "$SANDBOX_CONFIG" && return 0

    sed "s|^\([[:space:]]*allowed_host_paths[[:space:]]*=[[:space:]]*\[\)|\1\"$OFFLINE_DIR\", |" \
        "$SANDBOX_CONFIG" > "$SANDBOX_CONFIG.tmp" \
        && mv "$SANDBOX_CONFIG.tmp" "$SANDBOX_CONFIG"
    say "added $OFFLINE_DIR to allowed_host_paths in $SANDBOX_CONFIG"
}

if sandbox_up; then
    say "sandbox server already listening on 127.0.0.1:8080"
else
    docker info >/dev/null 2>&1 || die "the docker daemon is not reachable -- start Docker first"

    uv pip install -q opensandbox-server
    if [ ! -f "$SANDBOX_CONFIG" ]; then
        say "generating $SANDBOX_CONFIG"
        uv run opensandbox-server init-config "$SANDBOX_CONFIG" --example docker >/dev/null
    fi
    allow_offline_dir

    # Without an api_key the server demands an explicit acknowledgment, which
    # would block forever with no TTY attached.
    say "starting the sandbox server (log: $SANDBOX_LOG)"
    OPENSANDBOX_INSECURE_SERVER=YES nohup \
        uv run opensandbox-server --config "$SANDBOX_CONFIG" \
        > "$SANDBOX_LOG" 2>&1 < /dev/null &
    echo $! > "$SANDBOX_PID"

    for _ in $(seq 1 60); do
        sandbox_up && break
        kill -0 "$(cat "$SANDBOX_PID")" 2>/dev/null \
            || die "the sandbox server exited during startup -- see $SANDBOX_LOG"
        sleep 1
    done
    sandbox_up || die "the sandbox server did not come up within 60s -- see $SANDBOX_LOG"
    say "sandbox server up (pid $(cat "$SANDBOX_PID"); stop with: kill \$(cat $SANDBOX_PID))"
fi

# ---------------------------------------------------------------------------
# 3. Hand over
# ---------------------------------------------------------------------------
cat > "$ENV_FILE" <<EOF
# Written by scripts/init.sh -- source this before running an experiment.
export OFFLINE_PACKAGE_DIR="$OFFLINE_DIR"
EOF

cat <<EOF

$(say "ready")

  source $ENV_FILE
  export ZHIPU_API_KEY="your-api-key"
  uv run agentprobe -c $EXAMPLE_CONFIG -l info

That evaluates two bundled zbackendbench tasks end to end. Container images are
pulled on demand, so the first run spends a few minutes on that.
EOF
