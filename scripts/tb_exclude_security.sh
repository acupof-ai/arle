#!/bin/bash
# Scrub terminal-bench security/credential tasks from a dataset dir before a run.
# These tasks (fake-credential grep, bash_history/crontab recon, ssh/hash-crack)
# trip the shared box's host-side HIDS. Deleting them from the cache is NOT
# durable — a registry re-fetch re-adds them — so callers run this against the
# resolved dataset dir at run time. Idempotent; single source of truth for the list.
#   usage: tb_exclude_security.sh <dataset_dir>
set -eu
DIR="${1:?usage: tb_exclude_security.sh <dataset_dir>}"

SECURITY_TASKS=(
  crack-7z-hash crack-7z-hash.easy crack-7z-hash.hard
  password-recovery intrusion-detection git-workflow-hack
  sanitize-git-repo sanitize-git-repo.hard security-vulhub-minio
  qemu-alpine-ssh git-multibranch decommissioning-service-with-sensitive-data
)

removed=0
for t in "${SECURITY_TASKS[@]}"; do
  if [[ -e "$DIR/$t" ]]; then
    rm -rf "${DIR:?}/$t"
    removed=$((removed + 1))
  fi
done
echo "[tb-exclude] scrubbed $removed security task(s) from $DIR"
