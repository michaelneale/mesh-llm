const YEAR = 365 * 86_400;
const MONTH = 30 * 86_400;
const WEEK = 7 * 86_400;
const DAY = 86_400;
const HOUR = 3_600;
const MINUTE = 60;

/**
 * Format a duration in seconds as a compact human-readable string.
 *
 * Examples: `"20s"`, `"4m"`, `"3h"`, `"12d"`, `"2w 5d"`, `"1mo 10d"`, `"1y 2mo 2d"`.
 * Returns `"-"` when the input is null, undefined, non-finite, or ≤ 0.
 *
 * Units used:
 *   - seconds (s), minutes (m), hours (h), days (d)
 *   - weeks (w) with optional days — for durations under 30 days
 *   - months (mo) with optional days — for durations under 1 year
 *   - years (y) with optional months and days
 *
 * Month length is approximated as 30 days. Year length is approximated as 365 days.
 * These are intentional display approximations — not calendar-accurate arithmetic.
 */
export function formatShortDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds <= 0) return "-";

  const s = Math.floor(seconds);

  if (s >= YEAR) {
    const years = Math.floor(s / YEAR);
    const rem = s % YEAR;
    const months = Math.floor(rem / MONTH);
    const days = Math.floor((rem % MONTH) / DAY);
    const parts = [`${years}y`];
    if (months > 0) parts.push(`${months}mo`);
    if (days > 0) parts.push(`${days}d`);
    return parts.join(" ");
  }

  if (s >= MONTH) {
    const months = Math.floor(s / MONTH);
    const days = Math.floor((s % MONTH) / DAY);
    return days > 0 ? `${months}mo ${days}d` : `${months}mo`;
  }

  if (s >= WEEK) {
    const weeks = Math.floor(s / WEEK);
    const days = Math.floor((s % WEEK) / DAY);
    return days > 0 ? `${weeks}w ${days}d` : `${weeks}w`;
  }

  if (s >= DAY) return `${Math.floor(s / DAY)}d`;
  if (s >= HOUR) return `${Math.floor(s / HOUR)}h`;
  if (s >= MINUTE) return `${Math.floor(s / MINUTE)}m`;
  return `${s}s`;
}
