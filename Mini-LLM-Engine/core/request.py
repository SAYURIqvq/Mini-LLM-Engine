import asyncio
from enum import Enum
from typing import List, Optional

import torch


class RequestStatus(Enum):
    WAITING = "waiting"       # queued, waiting to be scheduled
    RUNNING = "running"       # actively generating tokens
    FINISHED = "finished"     # generation complete


class Request:
    def __init__(self, request_id: int, prompt: str, input_ids: torch.Tensor,
                 max_tokens: int = 100, temperature: float = 0.7, top_p: float = 1.0):
        # basic info
        self.request_id = request_id
        self.prompt = prompt
        self.input_ids = input_ids          # tokenized prompt [1, seq_len]
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        # generation state
        self.status = RequestStatus.WAITING
        self.generated_ids: List[int] = []  # tokens generated so far
        self.past_key_values = None         # KV cache for this request
        self.output_text: str = ""          # final decoded text

        # async event to notify client when done
        self.event = asyncio.Event()

    @property
    def num_generated(self) -> int:
        return len(self.generated_ids)

    @property
    def is_finished(self) -> bool:
        return self.status == RequestStatus.FINISHED

    def finish(self, text: str):
        """Mark request as finished and notify waiting client."""
        self.status = RequestStatus.FINISHED
        self.output_text = text
        self.past_key_values = None   # free KV cache memory
        self.event.set()              # wake up the waiting client
