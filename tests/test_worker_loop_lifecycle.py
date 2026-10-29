"""Regression tests for persistent worker event-loop ownership."""

import asyncio
import unittest

from openevolve import process_parallel as worker


class LoopBoundClient:
    """Simulates an asynchronous client retaining a loop-bound resource."""

    def __init__(self):
        self.loop = None
        self.calls = 0

    async def request(self):
        current = asyncio.get_running_loop()

        if self.loop is None:
            self.loop = current
        elif self.loop is not current:
            raise RuntimeError("Event loop is closed")

        await asyncio.sleep(0)
        self.calls += 1
        return self.calls


class WorkerLoopLifecycleTests(unittest.TestCase):

    def tearDown(self):
        worker._close_worker_event_loop()

    def test_reuses_loop_across_twelve_calls(self):
        client = LoopBoundClient()

        for count in range(1, 13):
            self.assertEqual(
                worker._run_worker_async(client.request()),
                count,
            )

        self.assertIsNotNone(client.loop)
        self.assertFalse(client.loop.is_closed())

    def test_generation_and_evaluation_share_loop(self):
        generation_client = LoopBoundClient()
        evaluation_client = LoopBoundClient()

        for _ in range(8):
            worker._run_worker_async(
                generation_client.request()
            )
            worker._run_worker_async(
                evaluation_client.request()
            )

        self.assertIs(
            generation_client.loop,
            evaluation_client.loop,
        )

    def test_exception_propagates_without_closing_loop(self):
        async def fail():
            raise ValueError("deliberate failure")

        with self.assertRaisesRegex(
            ValueError, "deliberate failure"
        ):
            worker._run_worker_async(fail())

        client = LoopBoundClient()
        self.assertEqual(
            worker._run_worker_async(client.request()), 1
        )

    def test_cleanup_closes_loop_and_resets_state(self):
        client = LoopBoundClient()
        worker._run_worker_async(client.request())

        loop = client.loop
        worker._close_worker_event_loop()

        self.assertTrue(loop.is_closed())
        self.assertIsNone(worker._worker_event_loop)

    def test_cleanup_cancels_pending_tasks(self):
        cancelled = []

        async def pending():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)

        async def schedule():
            asyncio.create_task(pending())
            await asyncio.sleep(0)

        worker._run_worker_async(schedule())
        worker._close_worker_event_loop()

        self.assertEqual(cancelled, [True])


if __name__ == "__main__":
    unittest.main()
