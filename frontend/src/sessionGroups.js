// Group sidebar sessions (updated_at in epoch seconds) into date buckets.
export function groupSessionsByDate(sessions, now = new Date()) {
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const startOfYesterday = startOfToday - 24 * 60 * 60 * 1000
  const buckets = { Today: [], Yesterday: [], Earlier: [] }
  for (const session of sessions || []) {
    const ts = Number(session.updated_at) * 1000
    const key = ts >= startOfToday ? 'Today' : ts >= startOfYesterday ? 'Yesterday' : 'Earlier'
    buckets[key].push(session)
  }
  return Object.entries(buckets)
    .filter(([, list]) => list.length)
    .map(([label, list]) => ({ label, sessions: list }))
}
