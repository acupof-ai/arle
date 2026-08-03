"""Decode-only per-step ledger. Usage: ledger2.py <db> <label> <anchor_kernel> <per_step>

Picks a window from the LAST `nwin` launches of an anchor kernel that only runs
during decode, then attributes every kernel inside that window. No prefill.
"""
import sqlite3, sys
db, label, anchor, per_step = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
NWIN = 24000
c = sqlite3.connect(db)
cand = c.execute("SELECT s.id, COUNT(*) n FROM StringIds s "
                 "JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.demangledName=s.id "
                 "WHERE s.value LIKE ? GROUP BY s.id ORDER BY n DESC LIMIT 1",
                 (f"%{anchor}%",)).fetchone()
if not cand: sys.exit(f"anchor {anchor} not found in kernel table")
rows = c.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE demangledName=? "
                 "ORDER BY start DESC LIMIT ?", (cand[0], NWIN)).fetchall()
t0, t1 = min(r[0] for r in rows), max(r[1] for r in rows)
steps = len(rows) / per_step
q = """SELECT s.value, COUNT(*) n, SUM(k.end-k.start)/1e6 ms
       FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
       WHERE k.start>=? AND k.end<=? GROUP BY s.value ORDER BY ms DESC"""
res = c.execute(q, (t0, t1)).fetchall()
busy = sum(r[2] for r in res); nk = sum(r[1] for r in res); wall = (t1-t0)/1e6
print(f"== {label} == steps={steps:.0f}  wall/step={wall/steps:.3f} ms  "
      f"busy/step={busy/steps:.3f} ms  launches/step={nk/steps:.0f}  occ={busy/wall:.2f}")
for name, n, ms in res[:20]:
    print(f"  {ms/steps:7.3f} ms {n/steps:7.1f}x  {name[:70]}")
