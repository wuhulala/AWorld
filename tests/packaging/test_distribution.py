"""Build and install real distributions outside the source checkout."""
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile
from email.parser import Parser
import pytest

ROOT = Path(__file__).resolve().parents[2]


def run(*args, cwd=None):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    work = tmp_path_factory.mktemp("distribution")
    output = work / "artifacts"
    for source in (ROOT, ROOT / "aworld-cli"):
        run(sys.executable, "-m", "build", "--no-isolation", "--outdir", str(output), str(source))
    return work, output


def test_wheel_contains_one_kernel_and_explicit_dependencies(artifacts):
    _, output = artifacts
    with zipfile.ZipFile(output / "aworld-1.0.0a1-py3-none-any.whl") as wheel:
        files = set(wheel.namelist())
        assert {"aworld/cli/main.py", "aworld/core/agent/loop.py", "aworld/core/session/models.py"} <= files
        assert not any(name.startswith("aworldv1/") for name in files)
        assert not any("llm_agent" in name or "/runners/" in name or "/events/" in name or "/memory/" in name for name in files)
        metadata = Parser().parsestr(wheel.read("aworld-1.0.0a1.dist-info/METADATA").decode())
        assert metadata["Version"] == "1.0.0a1"
        assert all("extra ==" in value for value in metadata.get_all("Requires-Dist", []))
        scripts = wheel.read("aworld-1.0.0a1.dist-info/entry_points.txt").decode()
        assert "aworld = aworld.cli.main:main" in scripts
        assert "legacy" not in scripts and "aworldv1" not in scripts
    with zipfile.ZipFile(output / "aworld_cli-1.0.0a1-py3-none-any.whl") as wheel:
        files = [name for name in wheel.namelist() if name.endswith(".py")]
        assert set(files) == {"aworld_cli/__init__.py", "aworld_cli/__main__.py", "aworld_cli/entrypoint.py"}
        metadata = Parser().parsestr(wheel.read("aworld_cli-1.0.0a1.dist-info/METADATA").decode())
        assert "aworld==1.0.0a1" in metadata.get_all("Requires-Dist")


def test_normal_install_and_both_commands_work_without_source_or_extras(artifacts):
    work, output = artifacts
    environment = work / "venv"
    run("uv", "venv", "--python", sys.executable, str(environment))
    python = environment / "bin/python"
    run("uv", "pip", "install", "--offline", "--python", str(python),
        str(output / "aworld-1.0.0a1-py3-none-any.whl"), str(output / "aworld_cli-1.0.0a1-py3-none-any.whl"))
    for command in ("aworld", "aworld-cli"):
        results = [json.loads(line) for line in run(str(environment / "bin" / command), "run", "--demo",
            "--task", "hello", "--follow-up", "again", "--json", cwd=work).splitlines()]
        assert [result["status"] for result in results] == ["completed", "completed"]
        assert results[0]["session_id"] == results[1]["session_id"]
    run(str(python), "-I", "-c", "import aworld, importlib.util; assert 'site-packages' in aworld.__file__; assert importlib.util.find_spec('httpx') is None; assert importlib.util.find_spec('aworldv1') is None", cwd=work)


def test_sdist_rebuild_is_self_contained_and_reproducible(artifacts):
    work, output = artifacts
    for name, wheel in (("aworld", "aworld"), ("aworld_cli", "aworld_cli")):
        unpacked = work / (name + "-source")
        with tarfile.open(output / f"{name}-1.0.0a1.tar.gz") as archive:
            archive.extractall(unpacked, filter="data")
        source = next(unpacked.iterdir())
        rebuilt = work / (name + "-rebuilt")
        run(sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(rebuilt), str(source))
        filename = f"{wheel}-1.0.0a1-py3-none-any.whl"
        assert (rebuilt / filename).read_bytes() == (output / filename).read_bytes()


def test_pep660_editable_install_uses_canonical_namespace(artifacts):
    work, _ = artifacts
    environment = work / "editable"
    run("uv", "venv", "--python", sys.executable, str(environment))
    python = environment / "bin/python"
    run("uv", "pip", "install", "--offline", "--python", str(python), "-e", str(ROOT), "-e", str(ROOT / "aworld-cli"))
    run(str(python), "-I", "-c", "import aworld; from aworld.cli.main import main; assert aworld.__version__ == '1.0.0a1'", cwd=work)
    assert json.loads(run(str(environment / "bin/aworld-cli"), "run", "--demo", "--task", "editable", "--json", cwd=work))["status"] == "completed"
