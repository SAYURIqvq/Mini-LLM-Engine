import time
import asyncio
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .core.model_loader import load_model, load_tokenizer
from .core.request import Request, RequestStatus
from .core.sampler import sample_next_token
from .scheduler import ContinuousBatchScheduler


@dataclass
class SamplingParams:
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 100


class TinyInferEngine:
    def __init__(self, model_path: str, dtype=torch.float16, max_batch_size: int = 8, max_input_tokens: int = 4096):
        print(f"Loading model from {model_path}...")
        t0 = time.time()
        self.model = load_model(model_path, dtype)
        self.tokenizer = load_tokenizer(model_path)
        self.device = next(self.model.parameters()).device
        print(f"Model loaded in {time.time() - t0:.2f}s")

        self.max_input_tokens = max_input_tokens
        self.scheduler = ContinuousBatchScheduler(max_batch_size)
        self._next_request_id = 0
        self._requests: Dict[int, Request] = {}  
        self._finished: Dict[int, Request] = {}  

    def add_request(self, prompt: str, params: Optional[SamplingParams] = None) -> int:
        """Add a request to the queue, return request_id."""
        if params is None:
            params = SamplingParams()

        if not prompt.strip():
            raise ValueError("prompt must not be blank")
        if not 1 <= params.max_tokens <= 4096:
            raise ValueError("max_tokens must be between 1 and 4096")
        if not math.isfinite(params.temperature) or not 0 <= params.temperature <= 10:
            raise ValueError("temperature must be finite and between 0 and 10")
        if not math.isfinite(params.top_p) or not 0 < params.top_p <= 1:
            raise ValueError("top_p must be finite and in (0, 1]")
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if input_ids.shape[-1] == 0 or input_ids.shape[-1] > self.max_input_tokens:
            raise ValueError(f"prompt must contain 1–{self.max_input_tokens} tokens")
        context_limit = getattr(self.model.config, "max_position_embeddings", None)
        if context_limit and input_ids.shape[-1] + params.max_tokens > context_limit:
            raise ValueError("prompt plus max_tokens exceeds the model context window")
        input_ids = input_ids.to(self.device)

        request = Request(
            request_id=self._next_request_id,
            prompt=prompt,
            input_ids=input_ids,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
        )
        self._next_request_id += 1
        self._requests[request.request_id] = request
        self.scheduler.add_request(request)
        return request.request_id

    async def wait_for_result(self, request_id: int) -> Request:
        """Async wait for a request to complete."""
        request = self._requests[request_id]
        try:
            while not request.event.is_set():
                await asyncio.sleep(0.01)
            return self.pop_finished(request_id)
        finally:
            self.discard_request(request_id)

    def discard_request(self, request_id):
        request = self._requests.pop(request_id, None)
        self._finished.pop(request_id, None)
        self.scheduler.waiting = type(self.scheduler.waiting)(
            r for r in self.scheduler.waiting if r.request_id != request_id)
        self.scheduler.running = [r for r in self.scheduler.running if r.request_id != request_id]
        if request is not None:
            request.past_key_values = None
            request.input_ids = None

    def pop_finished(self, request_id):
        result = self._finished.get(request_id)
        if result is not None:
            self.discard_request(request_id)
        return result

    def clear(self):
        for request_id in list(self._requests):
            self.discard_request(request_id)

    def step(self):
        """
        Execute one step:
        1. Ask scheduler for the current batch
        2. Run one forward pass for each request in the batch
        3. Sample next token, check if finished
        """
        batch = self.scheduler.schedule()
        if not batch:
            return

        for request in batch:
            with torch.no_grad():
                if request.past_key_values is None:
                    # prefill prompt
                    outputs = self.model(
                        input_ids=request.input_ids,
                        use_cache=True,
                    )
                else:
                    # decode
                    last_token = torch.tensor(
                        [[request.generated_ids[-1]]],
                        device=self.device,
                    )
                    outputs = self.model(
                        input_ids=last_token,
                        past_key_values=request.past_key_values,
                        use_cache=True,
                    )

            
            logits = outputs.logits[:, -1, :]
            request.past_key_values = outputs.past_key_values
            # token_id is a number, could be decoded by tokenizer, you can regard this as next word
            next_token_id = sample_next_token(logits, request.temperature, request.top_p)
            token_id = next_token_id.item()
            # accumulate the results and later decode together
            request.generated_ids.append(token_id)

            # if out of max token or it's end of sequence
            # self.tokenizer.eos_token_id provided by model's tokenizer
            if token_id == self.tokenizer.eos_token_id or request.num_generated >= request.max_tokens:
                text = self.tokenizer.decode(request.generated_ids, skip_special_tokens=True)
                request.finish(text)
                self._finished[request.request_id] = request

