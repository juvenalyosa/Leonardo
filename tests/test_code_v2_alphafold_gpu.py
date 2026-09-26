import code_v2
from alphafold_runtime import AF_ENGINE_AF3_SERVER, AF_ENGINE_COLABFOLD


class _Label:
    def __init__(self):
        self.text, self.visible = "", None

    def setText(self, text):
        self.text = text

    def setVisible(self, visible):
        self.visible = bool(visible)


def _dialog(engine, gpu_info):
    class DummyDialog:
        pass

    for name in ("_af_gpu_info", "_af_local_gpu_available", "_af_update_gpu_warning"):
        setattr(DummyDialog, name, getattr(code_v2.ProtLigInteractDialog, name))
    dlg = DummyDialog()
    dlg._af_current_engine = lambda: engine
    dlg._af_gpu_info_cache = dict(gpu_info)
    dlg.lbl_af_gpu_warning = _Label()
    return dlg


def test_gpu_banner_shown_without_gpu():
    dlg = _dialog(AF_ENGINE_COLABFOLD, {"available": False, "reason": "no NVIDIA GPU detected"})
    msg = dlg._af_update_gpu_warning()
    assert dlg.lbl_af_gpu_warning.visible is True
    assert "GPU required" in dlg.lbl_af_gpu_warning.text and "Google Colab" in msg
    assert dlg._af_local_gpu_available() == (False, "no NVIDIA GPU detected")


def test_gpu_banner_hidden_with_gpu_or_server_engine():
    dlg = _dialog(AF_ENGINE_COLABFOLD, {"available": True, "reason": "nvidia-smi: GPU 0"})
    dlg._af_update_gpu_warning()
    assert dlg.lbl_af_gpu_warning.visible is False
    dlg = _dialog(AF_ENGINE_AF3_SERVER, {"available": False, "reason": "no NVIDIA GPU detected"})
    dlg._af_update_gpu_warning()
    assert dlg.lbl_af_gpu_warning.visible is False


def test_gpu_detection_is_cached_until_refresh(monkeypatch):
    calls = []
    monkeypatch.setattr(code_v2, "af_detect_local_gpu",
                        lambda: calls.append(1) or {"available": False, "reason": "x"})
    dlg = _dialog(AF_ENGINE_COLABFOLD, {})
    del dlg._af_gpu_info_cache
    for _ in range(5):
        dlg._af_gpu_info()
    assert len(calls) == 1
    dlg._af_gpu_info(refresh=True)
    assert len(calls) == 2
