"""Exercise the MiniMax configuration through the real SDK's HTTP boundary."""

import json
import os
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import openai

try:
    import httpx
except ModuleNotFoundError:
    import httpx2 as httpx

from openevolve.config import Config, load_config
from openevolve.controller import OpenEvolve
from openevolve.llm.ensemble import LLMEnsemble

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "minimax_config.yaml"


class TestMiniMaxConfig(unittest.IsolatedAsyncioTestCase):
    def load_example(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-for-validation"}):
            return load_config(str(CONFIG_PATH))

    async def capture_request(self, models, messages=None, controller_config=None, **kwargs):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "id": "test-completion",
                    "object": "chat.completion",
                    "created": 0,
                    "model": json.loads(request.content)["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "result"},
                        }
                    ],
                },
            )

        # Replace only the network transport: config loading, ensemble selection,
        # request construction and OpenAI SDK serialization all run normally.
        client_class = openai.AsyncOpenAI

        def create_client(**kwargs):
            return client_class(
                **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
            )

        with ExitStack() as stack:
            stack.enter_context(
                patch("openevolve.llm.openai.openai.AsyncOpenAI", side_effect=create_client)
            )
            if controller_config is not None:
                output_dir = stack.enter_context(TemporaryDirectory())
                stack.enter_context(patch.object(OpenEvolve, "_setup_logging"))
                example = CONFIG_PATH.parents[1] / "examples" / "function_minimization"
                controller = OpenEvolve(
                    str(example / "initial_program.py"),
                    str(example / "evaluator.py"),
                    controller_config,
                    output_dir=output_dir,
                )
                ensemble = (
                    controller.llm_evaluator_ensemble
                    if models is controller_config.llm.evaluator_models
                    else controller.llm_ensemble
                )
                clients = controller.llm_ensemble.models + controller.llm_evaluator_ensemble.models
            else:
                ensemble = LLMEnsemble(models)
                clients = ensemble.models
            try:
                result = await ensemble.generate_with_context(
                    "system", messages or [{"role": "user", "content": "improve"}], **kwargs
                )
                self.assertEqual(result, "result")
            finally:
                for model in clients:
                    await model.client.close()
        self.assertEqual(len(requests), 1)
        return requests[0], json.loads(requests[0].content)

    async def test_example_defaults_reach_both_regional_endpoints(self):
        for base in ("https://api.minimax.io/v1", "https://api.minimaxi.com/v1"):
            config = self.load_example()
            self.assertEqual(
                [model.name for model in config.llm.models],
                ["MiniMax-M3.1-Flash", "MiniMax-M3"],
            )
            self.assertEqual([model.weight for model in config.llm.models], [1.0, 0.0])
            config.llm.update_model_params({"api_base": base}, overwrite=True)
            for models in (config.llm.models, config.llm.evaluator_models):
                with self.subTest(base=base, models=models):
                    request, body = await self.capture_request(models, controller_config=config)
                    self.assertEqual(body["seed"], config.llm.random_seed)
                    self.assertIsInstance(body["seed"], int)
                    self.assertEqual(str(request.url), base + "/chat/completions")
                    self.assertEqual(body["model"], "MiniMax-M3.1-Flash")
                    self.assertEqual(body["max_tokens"], 4096)
                    self.assertEqual(body["temperature"], 0.7)
                    self.assertEqual(body["top_p"], 0.95)
                    self.assertFalse(body["stream"])
                    for field in (
                        "thinking",
                        "reasoning_effort",
                        "reasoning_split",
                        "service_tier",
                    ):
                        self.assertNotIn(field, body)

    async def test_flash_effort_reaches_wire(self):
        for effort in ("low", "medium", "high", "xhigh", "max"):
            with self.subTest(effort=effort):
                config = self.load_example()
                config.llm.update_model_params({"reasoning_effort": effort}, overwrite=True)
                _, body = await self.capture_request(config.llm.models)
                self.assertEqual(body["reasoning_effort"], effort)
        _, body = await self.capture_request(config.llm.models, reasoning_effort="low")
        self.assertEqual(body["reasoning_effort"], "low")

    async def test_m3_can_be_selected_explicitly(self):
        config = self.load_example()
        config.llm.models[0].weight = 0.0
        config.llm.models[1].weight = 1.0
        _, body = await self.capture_request(config.llm.models)
        self.assertEqual(body["model"], "MiniMax-M3")
        self.assertNotIn("reasoning_effort", body)

    async def test_saved_legacy_custom_models_and_content_are_preserved(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                    {"type": "video_url", "video_url": {"url": "mm_file://fixture"}},
                ],
            }
        ]
        for model in ("MiniMax-M2.7", "MiniMax-M2.7-highspeed", "custom-model"):
            for key, value in (("primary_model", model), ("models", [{"name": model}])):
                with self.subTest(model=model, key=key):
                    config = Config.from_dict(
                        {
                            "llm": {
                                "api_base": "https://example.com/v1",
                                "api_key": "test-key-for-validation",
                                key: value,
                            }
                        }
                    )
                    request, body = await self.capture_request(config.llm.models, messages)
                    self.assertEqual(str(request.url), "https://example.com/v1/chat/completions")
                    self.assertEqual(body["model"], model)
                    self.assertEqual(body["messages"][1:], messages)


if __name__ == "__main__":
    unittest.main()
