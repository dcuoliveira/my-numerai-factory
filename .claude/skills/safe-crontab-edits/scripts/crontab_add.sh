#!/usr/bin/env bash
#
# Append a cron entry without losing the ones already there.
#
#   crontab_add.sh --line "0 3 * * * /bin/bash /abs/path/job.sh" \
#                  [--comment "why this exists and what breaks without it"] \
#                  [--dry-run]
#
# `crontab FILE` and `crontab -` REPLACE the crontab rather than merging into
# it, so the obvious one-liner deletes every other job on the box silently. This
# snapshots first, appends to a copy, refuses to install if any pre-existing
# line would vanish, and prints the diff it actually applied.
#
set -uo pipefail

BACKUP_DIR="${HOME}/.crontab-backups"
LINE=""; COMMENT=""; DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --line)    LINE="$2"; shift 2 ;;
        --comment) COMMENT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown option '$1'" >&2; exit 2 ;;
    esac
done

[ -n "${LINE}" ] || { echo "--line is required" >&2; exit 2; }

# A schedule is five fields then a command. This catches the commonest typo --
# passing only the command, which cron would accept as a malformed line.
if ! printf '%s\n' "${LINE}" | grep -qE '^[^ ]+( +[^ ]+){4} +.+'; then
    echo "refusing: '${LINE}' does not look like '<m> <h> <dom> <mon> <dow> <command>'" >&2
    exit 1
fi

mkdir -p "${BACKUP_DIR}"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="${BACKUP_DIR}/crontab.${STAMP}.bak"

# `crontab -l` exits 1 when there is no crontab yet, which is not an error here.
crontab -l > "${BACKUP}" 2>/dev/null || : > "${BACKUP}"
echo "snapshot -> ${BACKUP} ($(wc -l < "${BACKUP}") lines)"

if grep -Fxq "${LINE}" "${BACKUP}"; then
    echo "that exact line is already installed; nothing to do"
    exit 0
fi

NEW="$(mktemp)"; trap 'rm -f "${NEW}"' EXIT
cp "${BACKUP}" "${NEW}"
# A trailing newline, so the append cannot glue itself onto the last entry.
[ -s "${NEW}" ] && [ -n "$(tail -c1 "${NEW}")" ] && echo >> "${NEW}"
{
    echo
    [ -n "${COMMENT}" ] && printf '# %s\n' "${COMMENT}"
    printf '# added %s by crontab_add.sh; snapshot: %s\n' "$(date -Is)" "${BACKUP}"
    printf '%s\n' "${LINE}"
} >> "${NEW}"

# The check that matters: every line that was there must still be there.
LOST="$(comm -23 <(grep -v '^\s*$' "${BACKUP}" | sort) \
                 <(grep -v '^\s*$' "${NEW}"    | sort))"
if [ -n "${LOST}" ]; then
    echo "refusing to install -- these existing lines would be lost:" >&2
    printf '  %s\n' "${LOST}" >&2
    exit 1
fi

echo "--- diff to be applied ---"
diff "${BACKUP}" "${NEW}"

if [ "${DRY_RUN}" = "1" ]; then
    echo "(--dry-run: nothing installed)"
    exit 0
fi

crontab "${NEW}" || { echo "install failed; crontab unchanged" >&2; exit 1; }

# Verify against what cron actually holds now, not against what we wrote.
if ! diff -q <(crontab -l 2>/dev/null) "${NEW}" >/dev/null; then
    echo "installed crontab does not match the file; restoring the snapshot" >&2
    crontab "${BACKUP}"
    exit 1
fi

echo "installed. restore with:  crontab ${BACKUP}"
