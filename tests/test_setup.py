"""Download integrity and archive-boundary tests for fresh installation."""
import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location('download', Path(__file__).resolve().parents[1] / 'scripts/download.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def test_corrupt_download_is_rejected(tmp_path):
    path = tmp_path / 'asset'
    path.write_bytes(b'incomplete')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        download.verify_checksum(path, hashlib.sha256(b'complete').hexdigest())


def test_archive_traversal_is_rejected_before_extraction(tmp_path):
    archive = tmp_path / 'data.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('good.txt', b'ok')
        z.writestr('../outside.txt', b'bad')
    dest = tmp_path / 'dest'
    dest.mkdir()
    with zipfile.ZipFile(archive) as z, pytest.raises(ValueError, match='Unsafe archive'):
        download.safe_extract(z, dest)
    assert not list(dest.iterdir())
    assert not (tmp_path / 'outside.txt').exists()


def test_ambiguous_checkpoint_archive_is_rejected(tmp_path):
    archive = tmp_path / 'weights.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('one/best_ckpt.pt', b'a')
        z.writestr('two/best_ckpt.pt', b'b')
    with pytest.raises(RuntimeError, match='exactly one'):
        download.extract_member(archive, 'best_ckpt.pt', tmp_path / 'out.pt')
