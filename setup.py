from setuptools import setup, find_packages
from pathlib import Path


def _requirements():
    req_path = Path(__file__).with_name("requirements.txt")
    if not req_path.exists():
        return []
    lines = req_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

setup(
    name="visionaid",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=_requirements(),
)
