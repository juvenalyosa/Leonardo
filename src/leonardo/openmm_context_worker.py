"""
Child process that owns one OpenMM Context for openmm_context_process.OpenMMContextProcess.

Run by file path (``python openmm_context_worker.py``): it imports only openmm, numpy and the standard
library, so the installed app and the source tree both work.  Requests and replies are length-prefixed
pickles on stdin / the original stdout; anything OpenMM or the OpenCL driver prints goes to stderr.
The process exits when the parent closes the pipe, so a crashed GUI never leaves a child holding GPU
memory.
"""
import os
import pickle
import struct
import sys
import traceback


def _read(stream):
    head = stream.read(8)
    if len(head) < 8:
        return None
    (n,) = struct.unpack("<Q", head)
    data = stream.read(n)
    if len(data) < n:
        return None
    return pickle.loads(data)


def _write(stream, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("<Q", len(data)))
    stream.write(data)
    stream.flush()


class _Worker:
    def __init__(self):
        import openmm

        self.mm = openmm
        self.context = None
        self.integrator = None
        self.system = None

    def init(self, system_xml, integrator_xml, platform_name, properties):
        mm = self.mm
        self.system = mm.XmlSerializer.deserialize(system_xml)
        self.integrator = mm.XmlSerializer.deserialize(integrator_xml)
        if platform_name:
            platform = mm.Platform.getPlatformByName(str(platform_name))
            if properties:
                self.context = mm.Context(self.system, self.integrator, platform, dict(properties))
            else:
                self.context = mm.Context(self.system, self.integrator, platform)
        else:
            self.context = mm.Context(self.system, self.integrator)
        platform = self.context.getPlatform()
        props = {}
        for name in platform.getPropertyNames():
            try:
                props[str(name)] = str(platform.getPropertyValue(self.context, name))
            except Exception:
                pass
        return {"platform": str(platform.getName()), "properties": props}

    def set_positions(self, positions_nm):
        import numpy as np

        self.context.setPositions(np.asarray(positions_nm, dtype=float))

    def set_velocities_to_temperature(self, temperature_k, seed=None):
        if seed is None:
            self.context.setVelocitiesToTemperature(float(temperature_k))
        else:
            self.context.setVelocitiesToTemperature(float(temperature_k), int(seed))

    def set_parameter(self, name, value):
        self.context.setParameter(str(name), float(value))

    def get_parameter(self, name):
        return float(self.context.getParameter(str(name)))

    def get_state(self, kwargs):
        state = self.context.getState(**dict(kwargs))
        return self.mm.XmlSerializer.serialize(state)

    def set_state(self, state_xml):
        self.context.setState(self.mm.XmlSerializer.deserialize(state_xml))

    def step(self, steps, sync_every):
        total, every = int(steps), max(1, int(sync_every))
        done = 0
        while done < total:
            n = min(every, total - done)
            self.integrator.step(n)
            done += n
            if done < total:
                # a real download drains Apple's OpenCL queue (openmm_dynamics_runtime.openmm_step_synced)
                self.context.getState(getPositions=True)
        return done

    def minimize(self, tolerance, max_iterations):
        self.mm.LocalEnergyMinimizer.minimize(self.context, float(tolerance), int(max_iterations))

    def update_force(self, force_index, force_xml):
        """Copy the per-particle / per-bond parameters of a parent-side force into this context."""
        mm = self.mm
        src = mm.XmlSerializer.deserialize(force_xml)
        dst = self.system.getForce(int(force_index))
        if type(dst) is not type(src):
            raise TypeError(f"force {force_index} is {type(dst).__name__}, update is {type(src).__name__}")
        if isinstance(src, mm.CustomExternalForce):
            for i in range(src.getNumParticles()):
                particle, params = src.getParticleParameters(i)
                dst.setParticleParameters(i, particle, params)
        elif isinstance(src, (mm.CustomBondForce, mm.CustomCentroidBondForce, mm.CustomCompoundBondForce)):
            for i in range(src.getNumBonds()):
                dst.setBondParameters(i, *src.getBondParameters(i))
        elif isinstance(src, mm.HarmonicBondForce):
            for i in range(src.getNumBonds()):
                dst.setBondParameters(i, *src.getBondParameters(i))
        else:
            raise TypeError(f"updating {type(src).__name__} through the context process is not supported")
        dst.updateParametersInContext(self.context)


def main():
    proto_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)                                  # stray prints cannot corrupt the protocol stream
    sys.stdout = sys.stderr
    proto_in = os.fdopen(os.dup(0), "rb")
    try:
        worker = _Worker()
        _write(proto_out, ("ok", "ready"))
    except Exception:
        _write(proto_out, ("err", traceback.format_exc()))
        return 1
    while True:
        req = _read(proto_in)
        if req is None:
            break
        cmd, args = req
        if cmd == "quit":
            _write(proto_out, ("ok", None))
            break
        try:
            out = getattr(worker, cmd)(*args)
            _write(proto_out, ("ok", out))
        except Exception:
            _write(proto_out, ("err", traceback.format_exc()))
    worker.context = None
    worker.integrator = None
    return 0


if __name__ == "__main__":
    sys.exit(main())
