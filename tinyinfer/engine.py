import time
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
    def __init__(self, model_path: str, dtype=torch.float16, max_batch_size: int = 8):
        print(f"Loading model from {model_path}...")
        t0 = time.time()
        self.model = load_model(model_path, dtype)
        self.tokenizer = load_tokenizer(model_path)
        self.device = next(self.model.parameters()).device
        print(f"Model loaded in {time.time() - t0:.2f}s")

        self.scheduler = ContinuousBatchScheduler(max_batch_size)
        self._next_request_id = 0
        self._requests: Dict[int, Request] = {}  
        self._finished: Dict[int, Request] = {}  

    def add_request(self, prompt: str, params: Optional[SamplingParams] = None) -> int:
        """Add a request to the queue, return request_id."""
        if params is None:
            params = SamplingParams()

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

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
        await request.event.wait()
        self._requests.pop(request_id)
        return self._finished.pop(request_id)

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

