"""Produce a solution-free copy of the straight 3D coupler."""

from pathlib import Path

from ...artifacts import atomic_path
from ...comsol_session import open_model


DIRECTORY = Path(__file__).resolve().parent
SOURCE = DIRECTORY / "lnoi_coupler_3d.mph"
TEMPLATE = DIRECTORY / "lnoi_coupler_3d_ports.mph"
PORTS = ("port1", "port2", "port3", "port4")


def _clear(nodes):
    for tag in list(nodes.tags()):
        if tag in nodes.tags():
            nodes.remove(tag)


def build_template(source=SOURCE, destination=TEMPLATE):
    source = Path(source).resolve(strict=True)
    destination = Path(destination).resolve()
    if source == destination or destination.exists():
        raise ValueError("Требуется новый путь для копии MPH")
    with atomic_path(destination) as temporary:
        with open_model(source, cores=2, label="3D | шаблон") as (_, model):
            java = model.java
            study = java.study("std4")
            if list(study.feature().tags()) != ["bma", "bma1", "bma2", "bma3", "freq"]:
                raise ValueError("Набор шагов исходной 3D-модели изменился")
            physics = java.component("comp2").physics("ewbe")
            for i, tag in enumerate(PORTS):
                port = physics.feature(tag)
                port.set("PortType", "UserDefined")
                port.set("E0", ["0[V/m]", "1[V/m]", "0[V/m]"])
                port.set("beta", "k0mode" + str(i % 2 + 1))
                port.set("PortExcitation", "on" if i < 2 else "off")
            model.parameter("width", "8[um]+g")
            model.parameter("k0mode1", "2*pi*1.8/lbd0")
            model.parameter("k0mode2", "2*pi*1.81/lbd0")
            waves = physics.prop("WaveVector")
            waves.set("k1", ["0", "0", "-k0mode1"])
            waves.set("k2", ["0", "0", "-k0mode2"])
            _clear(java.batch())
            _clear(java.result())
            _clear(java.result().numerical())
            _clear(java.result().dataset())
            _clear(java.sol())
            for tag in ("bma", "bma1", "bma2", "bma3"):
                study.feature().remove(tag)
            study.setGenPlots(False)
            if list(study.feature().tags()) != ["freq"] or list(java.batch().tags()):
                raise RuntimeError("В шаблоне ожидался один Frequency Domain")
            model.save(temporary)
    return destination
