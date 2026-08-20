#!/usr/bin/env bash
#
# monitor_claudear.sh - sample the claudear daemon's resource footprint over
# time so a slow leak / runaway spawn can be traced.
#
# Runs on Linux (reads /proc). Meant for the droplet where claudear runs as:
#   /usr/bin/claudear --config claudear.toml start --poll
#
# It appends one CSV row per interval to a central log (default ./memory.log):
#   timestamp,pid,rss_mb,peak_rss_mb,vsz_mb,threads,fds,cpu_pct,
#   descendants,claude_procs,git_procs,sys_mem_used_pct,sys_mem_avail_mb,load1
#
# What each column tells you when it climbs monotonically:
#   rss_mb / peak_rss_mb  -> memory leak (the retained-task / handle growth)
#   threads               -> tokio blocking-pool growth (spawn_blocking piling up)
#   fds                   -> leaked file descriptors / sockets / worktrees
#   descendants/claude/git-> child processes not being reaped
#
# Usage:
#   ./monitor_claudear.sh start      # loop in foreground (nohup it for a droplet)
#   ./monitor_claudear.sh once       # take a single sample and exit
#   ./monitor_claudear.sh status     # show whether a monitor is running
#   ./monitor_claudear.sh stop       # stop a running monitor (via pidfile)
#
# Config via env vars:
#   INTERVAL   seconds between samples          (default 60)
#   LOG        output CSV path                   (default ./memory.log)
#   PIDFILE    monitor pidfile                   (default ./claudear-monitor.pid)
#   MATCH      pgrep pattern for the daemon      (default the start --poll cmd)
#   THRESHOLD_MB  if set, log a WARN line when rss exceeds this
#
set -u

INTERVAL="${INTERVAL:-60}"
LOG="${LOG:-./memory.log}"
PIDFILE="${PIDFILE:-./claudear-monitor.pid}"
MATCH="${MATCH:-claudear.*start}"
THRESHOLD_MB="${THRESHOLD_MB:-}"

HEADER="timestamp,pid,rss_mb,peak_rss_mb,vsz_mb,threads,fds,cpu_pct,descendants,claude_procs,git_procs,sys_mem_used_pct,sys_mem_avail_mb,load1"

now() { date '+%Y-%m-%dT%H:%M:%S%z'; }

# Find the claudear daemon pid. Prefer an exact process-name match, fall back to
# the command-line pattern. Returns empty string if not running.
find_pid() {
  local pid
  pid="$(pgrep -x claudear 2>/dev/null | head -n1)"
  if [ -z "$pid" ]; then
    pid="$(pgrep -f "$MATCH" 2>/dev/null | grep -v "$$" | head -n1)"
  fi
  printf '%s' "$pid"
}

# Recursively list all descendant pids of $1, one per line.
list_descendants() {
  local pid="$1" child
  for child in $(pgrep -P "$pid" 2>/dev/null); do
    printf '%s\n' "$child"
    list_descendants "$child"
  done
}

kb_to_mb() { awk -v k="${1:-0}" 'BEGIN{ printf "%.1f", (k+0)/1024 }'; }

sample() {
  local pid ts
  ts="$(now)"
  pid="$(find_pid)"

  if [ -z "$pid" ] || [ ! -d "/proc/$pid" ]; then
    # Daemon not running right now - record the gap so a crash/restart is visible.
    printf '%s,,,,,,,,,,,,,\n' "$ts" >>"$LOG"
    echo "$ts  claudear not running"
    return
  fi

  local vmrss vmhwm vmsize threads fds cpu
  vmrss="$(awk '/^VmRSS:/{print $2}'  "/proc/$pid/status" 2>/dev/null)"
  vmhwm="$(awk '/^VmHWM:/{print $2}'  "/proc/$pid/status" 2>/dev/null)"
  vmsize="$(awk '/^VmSize:/{print $2}' "/proc/$pid/status" 2>/dev/null)"
  threads="$(awk '/^Threads:/{print $2}' "/proc/$pid/status" 2>/dev/null)"
  fds="$(ls -1 "/proc/$pid/fd" 2>/dev/null | wc -l | tr -d ' ')"
  cpu="$(ps -o %cpu= -p "$pid" 2>/dev/null | tr -d ' ')"

  local rss_mb peak_mb vsz_mb
  rss_mb="$(kb_to_mb "$vmrss")"
  peak_mb="$(kb_to_mb "$vmhwm")"
  vsz_mb="$(kb_to_mb "$vmsize")"

  # Descendant process accounting.
  local desc descendants=0 claude_procs=0 git_procs=0 k comm
  desc="$(list_descendants "$pid")"
  for k in $desc; do
    descendants=$((descendants + 1))
    comm="$(cat "/proc/$k/comm" 2>/dev/null)"
    case "$comm" in
      claude*) claude_procs=$((claude_procs + 1)) ;;
      git*)    git_procs=$((git_procs + 1)) ;;
    esac
  done

  # System memory + load.
  local memtotal memavail used_pct avail_mb load1
  memtotal="$(awk '/^MemTotal:/{print $2}' /proc/meminfo 2>/dev/null)"
  memavail="$(awk '/^MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null)"
  avail_mb="$(kb_to_mb "$memavail")"
  used_pct="$(awk -v t="${memtotal:-0}" -v a="${memavail:-0}" \
    'BEGIN{ if (t>0) printf "%.1f", (t-a)/t*100; else printf "" }')"
  load1="$(awk '{print $1}' /proc/loadavg 2>/dev/null)"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$ts" "$pid" "$rss_mb" "$peak_mb" "$vsz_mb" "$threads" "$fds" "$cpu" \
    "$descendants" "$claude_procs" "$git_procs" "$used_pct" "$avail_mb" "$load1" \
    >>"$LOG"

  echo "$ts  pid=$pid rss=${rss_mb}MB peak=${peak_mb}MB threads=$threads fds=$fds desc=$descendants (claude=$claude_procs git=$git_procs) cpu=${cpu}%"

  if [ -n "$THRESHOLD_MB" ]; then
    awk -v r="$rss_mb" -v t="$THRESHOLD_MB" 'BEGIN{ exit !(r+0 > t+0) }' && {
      echo "$ts,WARN,rss ${rss_mb}MB exceeded threshold ${THRESHOLD_MB}MB,pid=$pid" >>"$LOG"
      echo "$ts  WARN rss ${rss_mb}MB > ${THRESHOLD_MB}MB"
    }
  fi
}

ensure_header() {
  if [ ! -s "$LOG" ]; then
    echo "$HEADER" >>"$LOG"
  fi
}

cmd_start() {
  ensure_header
  echo "$$" >"$PIDFILE"
  trap 'rm -f "$PIDFILE"; echo "monitor stopped"; exit 0' INT TERM
  echo "monitoring claudear every ${INTERVAL}s -> $LOG (monitor pid $$)"
  while true; do
    sample
    sleep "$INTERVAL"
  done
}

cmd_once() {
  ensure_header
  sample
}

cmd_status() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    echo "monitor running (pid $(cat "$PIDFILE"))"
  else
    echo "monitor not running"
  fi
  local dpid
  dpid="$(find_pid)"
  if [ -n "$dpid" ]; then
    echo "claudear daemon pid: $dpid"
  else
    echo "claudear daemon: not running"
  fi
}

cmd_stop() {
  if [ -f "$PIDFILE" ]; then
    local p; p="$(cat "$PIDFILE" 2>/dev/null)"
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
      kill "$p" && echo "stopped monitor (pid $p)"
    else
      echo "no live monitor for pidfile; cleaning up"
    fi
    rm -f "$PIDFILE"
  else
    echo "no pidfile at $PIDFILE"
  fi
}

case "${1:-start}" in
  start)  cmd_start ;;
  once)   cmd_once ;;
  status) cmd_status ;;
  stop)   cmd_stop ;;
  *) echo "usage: $0 {start|once|status|stop}"; exit 2 ;;
esac
