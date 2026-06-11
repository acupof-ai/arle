# Serve Ctrl-C wedged behind graceful HTTP drain

## Context

`arle serve --backend metal` received Ctrl-C during a long in-flight generation
but did not exit promptly. This was the rewrite in-process serve path, not the
older REPL cancel path.

## Root Cause

`infer-api::serve_http` used `axum::serve(...).with_graceful_shutdown(ctrl_c)`.
That stops accepting new traffic, but it waits for active handlers to finish.
The active completion handler was blocked in `RequestTicket::collect()`.

The engine-side owner stayed alive because the `ServeHandle` was still held in
the router state while axum waited for the handler. So `ServeHandle::drop()` did
not run, and even if it had run, the normal drop path was drain-to-completion.
Net effect: Ctrl-C requested HTTP graceful shutdown, but no signal reached the
producer loop that was generating tokens.

## Fix

Add a backend-neutral `infer_server::ServeShutdown` token shared by
`serve_http`'s Ctrl-C future and the `ServeHandle` engine loop. HTTP serve
builders now spawn backend handles with this token. When Ctrl-C fires,
`shutdown.request()` runs before axum starts graceful shutdown. The engine loop
observes the token at tick boundaries, drops pending completion channels, and
returns instead of draining in-flight requests to `max_tokens`.

Default `ServeHandle::spawn` / `spawn_with_engine_builder` keep their prior
drain behavior by using an unrequested token. The abort path is opt-in through
the HTTP serve token.

## Evidence

- `cargo test -p infer-server --release shutdown_token_aborts_inflight_without_drain -- --nocapture`
  passed. The test uses a never-ready executor to prove shutdown closes an
  in-flight collector instead of waiting for natural completion.
- `cargo test -p infer-server --release -- --nocapture` passed: 24 tests.
- `cargo check -p infer-api --release --no-default-features --features metal,no-cuda --lib` passed.
- `cargo test -p cli --release --no-default-features --features metal,no-cuda serve::tests -- --nocapture`
  passed: 21 tests.
- `CUDARC_CUDA_VERSION=12060 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib`
  passed on macOS without CUDA compilation.
- Real Metal smoke on `mlx-community/Qwen3.5-9B-MLX-4bit`: started
  `./target/release/arle serve --backend metal --port 8147 --num-slots 1`,
  sent a `/v1/chat/completions` request with `max_tokens=4096`, then sent
  SIGINT to the server process after 1 second. Server exited with status 0 in
  186 ms. The HTTP response was the expected interrupted-request error:
  `engine thread closed before request 0 completed`. No `arle serve` process
  remained afterward; `vm.swapusage` stayed at the residual 553 MiB.

## Rule

Graceful HTTP shutdown is not request cancellation. If a handler is blocked on a
producer-owned channel, Ctrl-C must reach the producer loop before waiting for
handlers to drain.
