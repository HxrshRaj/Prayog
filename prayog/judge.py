"""AI judges for RLAIF-style preference labelling.

LocalJudge  - a small instruct model run on CPU (used for the reported results).
WatsonxJudge - IBM Granite on watsonx.ai via REST. Needs WATSONX_API_KEY + WATSONX_PROJECT_ID.
               It has NOT been run in this repo (no key); results use LocalJudge only.
"""
from __future__ import annotations

import os

PROMPT = ("You are comparing two assistant replies to the conversation below. Pick the reply that is "
          "more helpful, honest and harmless.\n\nConversation:\n{prompt}\n\nReply A: {a}\n\nReply B: {b}\n\n"
          "Which reply is better? Answer with a single letter, A or B.")


class LocalJudge:
    name = "local"

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()
        self.id_a = self.tok.encode("A", add_special_tokens=False)[0]
        self.id_b = self.tok.encode("B", add_special_tokens=False)[0]

    def _a_minus_b(self, prompt: str, a: str, b: str) -> float:
        msg = [{"role": "user", "content": PROMPT.format(prompt=prompt[-1200:], a=a[:700], b=b[:700])}]
        text = self.tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt")
        with self.torch.no_grad():
            logits = self.model(**enc).logits[0, -1]
        lp = self.torch.log_softmax(logits, -1)
        return (lp[self.id_a] - lp[self.id_b]).item()

    def prefers_first(self, prompt: str, x: str, y: str) -> tuple[bool, bool]:
        """Return (x preferred, position-consistent). Both orderings are scored to cancel position bias."""
        s1 = self._a_minus_b(prompt, x, y)       # x shown as A
        s2 = self._a_minus_b(prompt, y, x)       # x shown as B
        consistent = (s1 > 0) != (s2 > 0)        # same item wins in both orders
        return (s1 - s2) > 0, consistent


class WatsonxJudge:
    """Granite on watsonx.ai as the judge. Same interface as LocalJudge."""
    name = "watsonx"

    def __init__(self, model_id: str = "ibm/granite-3-8b-instruct", region: str = "us-south"):
        import requests
        self.requests = requests
        self.key = os.environ["WATSONX_API_KEY"]
        self.project = os.environ["WATSONX_PROJECT_ID"]
        self.model_id = model_id
        self.base = f"https://{region}.ml.cloud.ibm.com"
        self._token = None

    def _iam(self) -> str:
        r = self.requests.post("https://iam.cloud.ibm.com/identity/token",
                               data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": self.key},
                               timeout=30)
        r.raise_for_status()
        return r.json()["access_token"]

    def _ask(self, prompt: str, a: str, b: str) -> str:
        self._token = self._token or self._iam()
        body = {"model_id": self.model_id, "project_id": self.project, "max_tokens": 2, "temperature": 0,
                "messages": [{"role": "user", "content": PROMPT.format(prompt=prompt[-1200:], a=a[:700], b=b[:700])}]}
        r = self.requests.post(f"{self.base}/ml/v1/text/chat?version=2024-05-01", json=body, timeout=60,
                               headers={"Authorization": f"Bearer {self._token}"})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip().upper()[:1]

    def prefers_first(self, prompt: str, x: str, y: str) -> tuple[bool, bool]:
        x_wins_as_a = self._ask(prompt, x, y) == "A"
        x_wins_as_b = self._ask(prompt, y, x) == "B"
        # inconsistent verdicts = position bias; fall back to the first ordering
        return x_wins_as_a, x_wins_as_a == x_wins_as_b


def make_judge(kind: str = "local"):
    return WatsonxJudge() if kind == "watsonx" else LocalJudge()
