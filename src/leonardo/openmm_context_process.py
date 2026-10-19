"""
An OpenMM Context that lives in a child process, recycled before Apple's OpenCL leak grows large.

Apple's OpenCL leaks ~50-100 bytes of pageable memory per kernel launch (~1.5 kB per MD step for a
1638-atom ABFE system, ~1.5 GB per ns).  Deleting the Context does not return it; only the end of the
process does.  OpenMMContextProcess keeps the Context in a child (openmm_context_worker.py) and, when the
child's RSS passes ``rss_cap_mb``, saves the full State (positions, velocities, parameters, time),
starts a fresh child from the parent's System (so force parameters edited on the parent are current)
and restores the State.  Callers see an object with the subset of the openmm.Context API that
Leonardo's long runs use.  A recycled child draws new Langevin noise (seed + recycle count): the run
stays a valid trajectory but is not bit-identical to one in-process run.
"""
import atexit
import os
import pickle
import struct
import subprocess
import sys
import weakref

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openmm_context_worker.py")
_LIVE = weakref.WeakSet()


def resolve_worker_python():
    """The interpreter for the child: this one, unless we are inside an entry point such as ``pymol``."""
    exe = str(sys.executable or "")
    base = os.path.basename(exe).lower()
    if exe and base.startswith("python") and os.path.isfile(exe):
        return exe
    for name in ("python3", "python"):
        cand = os.path.join(sys.prefix, "bin", name)
        if os.path.isfile(cand):
            return cand
    return exe or "python3"


def _strip(value, unit_name):
    if hasattr(value, "value_in_unit"):
        import openmm.unit as u

        return value.value_in_unit(getattr(u, unit_name))
    return value


class _PlatformStub:
    def __init__(self, name, properties):
        self._name = str(name)
        self._props = dict(properties or {})

    def getName(self):
        return self._name

    def getPropertyNames(self):
        return list(self._props.keys())

    def getPropertyValue(self, _context, name):
        return self._props[str(name)]


class OpenMMContextProcess:
    def __init__(self, system, integrator, platform_name="", properties=None, rss_cap_mb=1200.0,
                 sync_every=250, log=None):
        import openmm

        self._mm = openmm
        self._system = system
        self._integrator_xml = openmm.XmlSerializer.serialize(integrator)
        try:
            self._seed = int(integrator.getRandomNumberSeed())
        except Exception:
            self._seed = 0
        self._platform_name = str(platform_name or "")
        self._properties = dict(properties or {})
        self._rss_cap_mb = float(rss_cap_mb)
        self._sync_every = int(sync_every)
        self._log = log
        self._proc = None
        self._platform = None
        self._force_index = {}
        self.recycles = 0
        self.child_pids = []
        self._spawn()
        _LIVE.add(self)

    # -- process plumbing -------------------------------------------------------------------------
    def _spawn(self):
        self._proc = subprocess.Popen(
            [resolve_worker_python(), "-I", _WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=dict(os.environ, PYTHONUNBUFFERED="1"),
        )
        self.child_pids.append(int(self._proc.pid))
        ready = self._recv()
        if ready != "ready":
            raise RuntimeError(f"OpenMM context process did not start: {ready!r}")
        integ = self._mm.XmlSerializer.deserialize(self._integrator_xml)
        if self._seed > 0 and self.recycles and hasattr(integ, "setRandomNumberSeed"):
            integ.setRandomNumberSeed(int(self._seed + self.recycles))
        info = self._call("init", self._mm.XmlSerializer.serialize(self._system),
                          self._mm.XmlSerializer.serialize(integ), self._platform_name, self._properties)
        self._platform = _PlatformStub(info.get("platform", ""), info.get("properties", {}))

    def _send(self, obj):
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        self._proc.stdin.write(struct.pack("<Q", len(data)))
        self._proc.stdin.write(data)
        self._proc.stdin.flush()

    def _recv(self):
        head = self._proc.stdout.read(8)
        if len(head) < 8:
            code = self._proc.poll()
            raise RuntimeError(f"OpenMM context process exited (code {code})")
        (n,) = struct.unpack("<Q", head)
        status, value = pickle.loads(self._proc.stdout.read(n))
        if status != "ok":
            raise RuntimeError(f"OpenMM context process error:\n{value}")
        return value

    def _call(self, cmd, *args):
        if self._proc is None:
            raise RuntimeError("OpenMM context process is closed")
        self._send((cmd, args))
        return self._recv()

    def _stop_child(self):
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            self._send_to(proc, ("quit", ()))
            proc.wait(timeout=30)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=10)
            except Exception:
                pass
        for stream in (proc.stdin, proc.stdout):
            try:
                stream.close()
            except Exception:
                pass

    @staticmethod
    def _send_to(proc, obj):
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        proc.stdin.write(struct.pack("<Q", len(data)))
        proc.stdin.write(data)
        proc.stdin.flush()
        head = proc.stdout.read(8)
        if len(head) == 8:
            proc.stdout.read(struct.unpack("<Q", head)[0])

    def child_rss_mb(self):
        try:
            import psutil

            return float(psutil.Process(self._proc.pid).memory_info().rss) / 1048576.0
        except Exception:
            return None

    def recycle(self):
        """Move the run to a fresh child process (the leaked driver memory goes with the old one)."""
        state_xml = self._call("get_state", {"getPositions": True, "getVelocities": True,
                                             "getParameters": True})
        old_rss = self.child_rss_mb()
        self._stop_child()
        self.recycles += 1
        self._spawn()
        self._call("set_state", state_xml)
        if self._log is not None:
            try:
                self._log(f"OpenMM context process recycled (#{self.recycles}, child RSS was "
                          f"{(old_rss or float('nan')):.0f} MB)")
            except Exception:
                pass

    def close(self):
        self._stop_child()

    def __del__(self):
        try:
            self._stop_child()
        except Exception:
            pass

    # -- openmm.Context subset ----------------------------------------------------------------------
    def getPlatform(self):
        return self._platform

    def getSystem(self):
        return self._system

    def setPositions(self, positions):
        import numpy as np

        self._call("set_positions", np.asarray(_strip(positions, "nanometer"), dtype=float))

    def setVelocitiesToTemperature(self, temperature, seed=None):
        self._call("set_velocities_to_temperature", float(_strip(temperature, "kelvin")),
                   None if seed is None else int(seed))

    def setParameter(self, name, value):
        self._call("set_parameter", str(name), float(value))

    def getParameter(self, name):
        return self._call("get_parameter", str(name))

    def getState(self, **kwargs):
        groups = kwargs.get("groups", None)
        if isinstance(groups, set):
            kwargs["groups"] = sum(1 << int(g) for g in groups)
        return self._mm.XmlSerializer.deserialize(self._call("get_state", kwargs))

    def setState(self, state):
        self._call("set_state", self._mm.XmlSerializer.serialize(state))

    def step_synced(self, steps, sync_every=None):
        rss = self.child_rss_mb()
        if rss is not None and rss > self._rss_cap_mb:
            self.recycle()
        return self._call("step", int(steps), int(sync_every or self._sync_every))

    def minimize(self, max_iterations=0, tolerance=10.0):
        self._call("minimize", float(tolerance), int(max_iterations))

    def update_force(self, force):
        """Send the current parameters of ``force`` (a force of the parent's System) to the child."""
        mm = self._mm
        xml = mm.XmlSerializer.serialize(force)
        idx = self._force_index.get(id(force))
        if idx is None or mm.XmlSerializer.serialize(self._system.getForce(idx)) != xml:
            idx = None
            for i in range(self._system.getNumForces()):
                if mm.XmlSerializer.serialize(self._system.getForce(i)) == xml:
                    idx = i
                    break
            if idx is None:
                raise ValueError("force is not part of this context's System")
            self._force_index[id(force)] = idx
        self._call("update_force", int(idx), xml)


@atexit.register
def _close_all():
    for proxy in list(_LIVE):
        try:
            proxy.close()
        except Exception:
            pass
