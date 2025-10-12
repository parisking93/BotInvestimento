# pipeline_trade.py
# ------------------------------------------------------------
# Pipeline generica a 3 stadi (fetch -> judge -> deliver)
# ------------------------------------------------------------

import asyncio
from typing import (
    Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
)
from dataclasses import dataclass, field  # <--- field aggiunto

Jsonable = Union[dict, list, str, int, float, None]

# ------------------------------
# Utilità
# ------------------------------

async def _maybe_await(fn: Callable, *args, **kwargs):
    """Esegue fn che può essere sync o async."""
    res = fn(*args, **kwargs)
    if asyncio.iscoroutine(res) or isinstance(res, Awaitable):
        return await res
    return res

async def _retry_with_backoff(
    coro_factory: Callable[[], Awaitable[Any]],
    *,
    retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    timeout: Optional[float] = None,
):
    """Retry esponenziale con jitter."""
    attempt = 0
    delay = base_delay
    while True:
        try:
            if timeout is not None:
                return await asyncio.wait_for(coro_factory(), timeout=timeout)
            else:
                return await coro_factory()
        except Exception as e:
            attempt += 1
            if attempt > retries:
                raise
            wait = min(delay, max_delay) + (attempt * 0.05)
            if on_retry:
                on_retry(attempt, e, wait)
            await asyncio.sleep(wait)
            delay = min(delay * 2, max_delay)

def _chunk(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    """Divide una sequenza in blocchi di dimensione `size`."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]

# ------------------------------
# Config
# ------------------------------

@dataclass
class StageConfig:
    concurrency: int = 2
    timeout: Optional[float] = None
    retries: int = 2
    base_backoff: float = 0.5
    max_backoff: float = 8.0

@dataclass
class PipelineConfig:
    batch_size: int = 50
    max_in_flight_batches: int = 6
    fail_fast: bool = False
    # ⚠️ FIX: default_factory per evitare default mutabili
    fetch_cfg: StageConfig = field(default_factory=lambda: StageConfig(concurrency=3, timeout=20, retries=2))
    judge_cfg: StageConfig = field(default_factory=lambda: StageConfig(concurrency=2, timeout=60, retries=2))
    deliver_cfg: StageConfig = field(default_factory=lambda: StageConfig(concurrency=2, timeout=30, retries=1))
    logger: Optional[Callable[[str], None]] = print

# ------------------------------
# Pipeline
# ------------------------------

class TradePipeline:
    """
    Pipeline a 3 stadi:
      1) fetch_batch_fn(batch) -> dati_batch
      2) judge_fn(dati_batch)  -> giudizio_batch
      3) deliver_fn(giudizio_batch) -> None (o esito)
    """

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        self._started = False

        # Code fra stadi
        self._q_fetch_to_judge: asyncio.Queue = asyncio.Queue(
            maxsize=self.cfg.max_in_flight_batches
        )
        self._q_judge_to_deliver: asyncio.Queue = asyncio.Queue(
            maxsize=self.cfg.max_in_flight_batches
        )

    def _log(self, msg: str):
        if self.cfg.logger:
            self.cfg.logger(msg)

    async def _producer(
        self,
        items: Sequence[Any],
        fetch_batch_fn: Callable[[Sequence[Any]], Awaitable[Jsonable]],
    ):
        """Spezza items in batch e pianifica i fetch con backpressure."""
        sem = asyncio.Semaphore(self.cfg.fetch_cfg.concurrency)

        async def do_one(batch_id: int, batch: Sequence[Any]):
            async with sem:
                def _on_retry(attempt, exc, wait):
                    self._log(f"[fetch#{batch_id}] retry {attempt} fra {wait:.2f}s: {exc}")

                async def _call():
                    return await _maybe_await(fetch_batch_fn, batch)

                res = await _retry_with_backoff(
                    _call,
                    retries=self.cfg.fetch_cfg.retries,
                    base_delay=self.cfg.fetch_cfg.base_backoff,
                    max_delay=self.cfg.fetch_cfg.max_backoff,
                    on_retry=_on_retry,
                    timeout=self.cfg.fetch_cfg.timeout,
                )
                await self._q_fetch_to_judge.put((batch_id, batch, res))
                self._log(f"[fetch#{batch_id}] OK — spinto verso judge")

        batch_id = 0
        try:
            for batch in _chunk(items, self.cfg.batch_size):
                if self._stop_event.is_set():
                    break
                # backpressure: se la queue è piena, questa put blocca
                await self._q_fetch_to_judge.put(("__reserve__", None, None))
                _ = await self._q_fetch_to_judge.get()
                self._q_fetch_to_judge.task_done()

                t = asyncio.create_task(do_one(batch_id, batch))
                self._tasks.append(t)
                batch_id += 1

            await asyncio.gather(*[t for t in self._tasks if not t.done()], return_exceptions=False)
        except Exception as e:
            if self.cfg.fail_fast:
                self._log(f"[producer] errore: {e} — stop pipeline")
                self._stop_event.set()
            else:
                self._log(f"[producer] errore non fatale: {e}")
        finally:
            for _ in range(self.cfg.judge_cfg.concurrency):
                await self._q_fetch_to_judge.put((None, None, None))

    async def _stage_worker(
        self,
        name: str,
        in_q: asyncio.Queue,
        out_q: Optional[asyncio.Queue],
        worker_fn: Callable[[Any], Awaitable[Any]],
        cfg: StageConfig,
    ):
        """Worker generico per gli stadi 2/3."""
        while not self._stop_event.is_set():
            item = await in_q.get()
            try:
                if isinstance(item, tuple) and item[0] is None:
                    if out_q is not None:
                        await out_q.put((None, None, None))
                    return

                batch_id, batch_keys, payload = item

                def _on_retry(attempt, exc, wait):
                    self._log(f"[{name}#{batch_id}] retry {attempt} fra {wait:.2f}s: {exc}")

                async def _call():
                    return await _maybe_await(worker_fn, (batch_id, batch_keys, payload))

                res = await _retry_with_backoff(
                    _call,
                    retries=cfg.retries,
                    base_delay=cfg.base_backoff,
                    max_delay=cfg.max_backoff,
                    on_retry=_on_retry,
                    timeout=cfg.timeout,
                )

                if out_q is not None:
                    await out_q.put((batch_id, batch_keys, res))
                self._log(f"[{name}#{batch_id}] OK")
            except Exception as e:
                self._log(f"[{name}] errore su batch: {e}")
                if self.cfg.fail_fast:
                    self._stop_event.set()
                    if out_q is not None:
                        await out_q.put((None, None, None))
                    return
            finally:
                in_q.task_done()

    async def run(
        self,
        items: Sequence[Any],
        *,
        fetch_batch_fn: Callable[[Sequence[Any]], Awaitable[Jsonable]],
        judge_fn: Callable[[Tuple[int, Sequence[Any], Jsonable]], Awaitable[Jsonable]],
        deliver_fn: Callable[[Tuple[int, Sequence[Any], Jsonable]], Awaitable[Any]],
    ):
        """Avvia la pipeline e attende la fine."""
        if self._started:
            raise RuntimeError("Pipeline già avviata")
        self._started = True

        judge_workers = [
            asyncio.create_task(
                self._stage_worker(
                    name="judge",
                    in_q=self._q_fetch_to_judge,
                    out_q=self._q_judge_to_deliver,
                    worker_fn=judge_fn,
                    cfg=self.cfg.judge_cfg,
                )
            )
            for _ in range(self.cfg.judge_cfg.concurrency)
        ]

        deliver_workers = [
            asyncio.create_task(
                self._stage_worker(
                    name="deliver",
                    in_q=self._q_judge_to_deliver,
                    out_q=None,
                    worker_fn=deliver_fn,
                    cfg=self.cfg.deliver_cfg,
                )
            )
            for _ in range(self.cfg.deliver_cfg.concurrency)
        ]

        producer_task = asyncio.create_task(self._producer(items, fetch_batch_fn))

        await producer_task
        await self._q_fetch_to_judge.join()
        await self._q_judge_to_deliver.join()

        for t in judge_workers + deliver_workers:
            if not t.done():
                t.cancel()
        self._log("[pipeline] completata")


    # Pipeline.py — dentro la classe TradePipeline

    async def run_streaming(
        self,
        source_aiter,  # async iterator che YIELDa Sequence[Any] (un batch)
        *,
        judge_fn: Callable[[Tuple[int, Sequence[Any], Jsonable]], Awaitable[Jsonable]],
        deliver_fn: Callable[[Tuple[int, Sequence[Any], Jsonable]], Awaitable[Any]],
    ):
        """
        Variante streaming: prende i batch da una sorgente async e li spinge
        direttamente verso 'judge' (lo stadio fetch è bypassato).
        Spegnimento ordinato: drena le code prima di inviare i sentinel.
        """
        if self._started:
            raise RuntimeError("Pipeline già avviata")
        self._started = True

        # Avvia i worker
        judge_workers = [
            asyncio.create_task(
                self._stage_worker(
                    name="judge",
                    in_q=self._q_fetch_to_judge,
                    out_q=self._q_judge_to_deliver,
                    worker_fn=judge_fn,
                    cfg=self.cfg.judge_cfg,
                )
            ) for _ in range(self.cfg.judge_cfg.concurrency)
        ]
        deliver_workers = [
            asyncio.create_task(
                self._stage_worker(
                    name="deliver",
                    in_q=self._q_judge_to_deliver,
                    out_q=None,
                    worker_fn=deliver_fn,
                    cfg=self.cfg.deliver_cfg,
                )
            ) for _ in range(self.cfg.deliver_cfg.concurrency)
        ]

        # Producer che legge dall'async-iterator
        async def _producer_from_source():
            batch_id = 0
            async for batch in source_aiter:
                await self._q_fetch_to_judge.put((batch_id, list(batch), list(batch)))
                self._log(f"[source#{batch_id}] spinto verso judge (stream)")
                batch_id += 1
            # IMPORTANT: non mettere qui i sentinel! Li mettiamo dopo i join.

        producer_task = asyncio.create_task(_producer_from_source())

        # 1) aspetta che il producer finisca lo stream
        await producer_task

        # 2) drena TUTTO ciò che è già entrato verso judge
        self._log("[pipeline] draining fetch→judge ...")
        await self._q_fetch_to_judge.join()

        # 3) ora è sicuro inviare i sentinel ai worker judge
        for _ in range(self.cfg.judge_cfg.concurrency):
            await self._q_fetch_to_judge.put((None, None, None))

        # 4) aspetta che judge produca e che deliver consumi tutto quello già in coda
        self._log("[pipeline] draining judge→deliver ...")
        await self._q_judge_to_deliver.join()

        # 5) spegni deliver con i suoi sentinel
        for _ in range(self.cfg.deliver_cfg.concurrency):
            await self._q_judge_to_deliver.put((None, None, None))

        # 6) chiusura pulita
        for t in judge_workers + deliver_workers:
            if not t.done():
                t.cancel()
        self._log("[pipeline] completata (streaming, drained)")
