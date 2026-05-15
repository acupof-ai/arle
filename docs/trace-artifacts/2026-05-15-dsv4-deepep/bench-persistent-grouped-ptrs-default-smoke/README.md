# DSv4 Persistent Grouped Pointer Default Smoke

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20 after grouped expert weight pointer tables moved to load-time caches.
This smoke keeps `ARLE_DSV4_ROUTE_GROUPED_EXPERTS=0`, validating that the
default DeepEP path still loads and returns normal output.

| Case | Result |
| --- | --- |
| `math` | `410` |
| `write_zh` | Normal Chinese release-note style text |
| `decode16` | Normal English continuation, not repeated digits |

Raw outputs:

- `results.json`
- `models.json`
- `server.log.gz`
