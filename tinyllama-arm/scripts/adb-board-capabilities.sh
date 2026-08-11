#!/usr/bin/env bash
set -euo pipefail

serial="${ADB_SERIAL:?set ADB_SERIAL=<board-ip>:5555}"

adb connect "${serial%:5555}" >/dev/null
adb -s "$serial" get-state >/dev/null

run() {
  local title="$1"
  shift
  printf '\n## %s\n' "$title"
  adb -s "$serial" shell "$@" || true
}

run "identity" id
run "kernel" uname -a
run "os-release" cat /etc/os-release
run "device-tree model" cat /proc/device-tree/model
run "kernel cmdline" cat /proc/cmdline

run "cpu online" cat /sys/devices/system/cpu/online
run "cpu possible" cat /sys/devices/system/cpu/possible
run "cpuinfo" cat /proc/cpuinfo
run "cpu freq current" cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq
run "cpu freq available" cat /sys/devices/system/cpu/cpufreq/policy0/scaling_available_frequencies
run "cpu governor" cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor

run "memory" cat /proc/meminfo
run "free" free -m
run "filesystems" df -h / /tmp /dev /dev/shm /run

run "iree binaries" find /usr/bin -maxdepth 1 -name 'iree*'
run "iree drivers" iree-run-module --list_drivers
run "iree devices" iree-run-module --list_devices
run "iree device details" iree-run-module --dump_devices

run "torq device" ls -l /dev/torq
run "torq sysfs" ls -l /sys/class/misc/torq
run "torq sysfs dev" cat /sys/class/misc/torq/dev
run "synpu sysfs" ls -l /sys/devices/platform/soc/f7600000.synpu
run "syna_npu module" lsmod
run "interrupts" cat /proc/interrupts

run "dma heaps" ls -l /dev/dma_heap
run "libcrypt" find /usr/lib -maxdepth 1 -name 'libcrypt*'
run "readelf" which readelf
run "ldd" which ldd
