"""Cancel-at-concept-boundary behavior for model_manager + infer."""

from __future__ import annotations

from PIL import Image

from sam3_anime import infer, model_manager


class _FakeProcessor:
    """Records prompts; optionally can be told to set the cancel flag mid-run."""

    def __init__(self, cancel_after: int | None = None, state=None):
        self.calls: list[str] = []
        self._cancel_after = cancel_after
        self._state = state or {}

    def set_image(self, image):
        return self._state

    def set_text_prompt(self, prompt, state):
        self.calls.append(prompt)
        if self._cancel_after is not None and len(self.calls) >= self._cancel_after:
            model_manager.request_cancel()
        return state


def _prompts(n: int) -> list[tuple[str, str]]:
    return [(f"cid_{i}", f"p{i}") for i in range(n)]


def _fake_state():
    # minimal processor state: one full-size mask per prompt is not needed;
    # _state_to_mask just needs something with "masks"
    import numpy as np

    return {"masks": np.ones((4, 4), dtype=np.float32), "scores": np.array([0.9])}


def test_request_cancel_without_generation_reports_nothing_to_cancel():
    assert model_manager.request_cancel() == "SAM3 is not generating — nothing to cancel."
    assert not model_manager.is_cancel_requested()


def test_cancel_flag_clears_when_generation_ends():
    with model_manager.inference_running():
        model_manager.request_cancel()
        assert model_manager.is_cancel_requested()
    assert not model_manager.is_cancel_requested()


def test_generate_masks_cancels_before_remaining_prompts(monkeypatch):
    """Cancel set inside prompt #2: #2 finishes, #3..n are skipped."""
    monkeypatch.setattr(model_manager, "is_loaded", lambda: True)

    class _Proc(_FakeProcessor):
        def set_text_prompt(self, prompt, state):
            self.calls.append(prompt)
            if len(self.calls) == 2:  # cancel during the 2nd concept
                model_manager.request_cancel()
            return state

    proc = _Proc(state=_fake_state())
    monkeypatch.setattr(model_manager, "get_processor", lambda: proc)

    masks, notes, cancelled = infer.generate_masks(Image.new("RGB", (8, 8)), _prompts(4))
    assert cancelled is True
    assert len(proc.calls) == 2  # stopped before prompt 3
    assert set(masks) == {"cid_0", "cid_1"}  # partial result kept
    assert any(n.startswith("Cancelled:") for n in notes)
    assert len([n for n in notes if n.startswith("Cancelled:")]) == 2  # cid_2, cid_3


def test_generate_masks_not_cancelled_runs_all_prompts(monkeypatch):
    monkeypatch.setattr(model_manager, "is_loaded", lambda: True)
    proc = _FakeProcessor(state=_fake_state())
    monkeypatch.setattr(model_manager, "get_processor", lambda: proc)

    masks, notes, cancelled = infer.generate_masks(Image.new("RGB", (8, 8)), _prompts(3))
    assert cancelled is False
    assert len(proc.calls) == 3
    assert set(masks) == {"cid_0", "cid_1", "cid_2"}
    assert not [n for n in notes if n.startswith("Cancelled:")]
