import traceback
from pymol.Qt import QtCore


_Signal = getattr(QtCore, "Signal", None) or getattr(QtCore, "pyqtSignal")
_Slot = getattr(QtCore, "Slot", None) or getattr(QtCore, "pyqtSlot")


class BackgroundTaskSignals(QtCore.QObject):
    success = _Signal(object)
    error = _Signal(str)
    progress = _Signal(object)
    finished = _Signal()


class BackgroundTask(QtCore.QRunnable):
    def __init__(self, task_fn, with_progress=False):
        super().__init__()
        self.task_fn = task_fn
        self.with_progress = bool(with_progress)
        self.signals = BackgroundTaskSignals()

    @_Slot()
    def run(self):
        try:
            if bool(self.with_progress):
                try:
                    result = self.task_fn(self.signals.progress.emit)
                except TypeError:
                    # Fallback for callables that do not accept a progress callback.
                    result = self.task_fn()
            else:
                result = self.task_fn()
        except Exception as exc:
            tb = traceback.format_exc()
            self.signals.error.emit(f"{exc}\n{tb}")
        else:
            self.signals.success.emit(result)
        finally:
            self.signals.finished.emit()


def submit_background_task(
    thread_pool,
    task_fn,
    on_success=None,
    on_error=None,
    on_finished=None,
    on_progress=None,
    with_progress=False,
):
    task = BackgroundTask(task_fn, with_progress=bool(with_progress))
    if callable(on_success):
        task.signals.success.connect(on_success)
    if callable(on_error):
        task.signals.error.connect(on_error)
    if callable(on_finished):
        task.signals.finished.connect(on_finished)
    if callable(on_progress):
        task.signals.progress.connect(on_progress)
    thread_pool.start(task)
    return task
