"""Regression test for the de-mmcv visualizer save path.

``mmcv.imwrite`` created any missing parent directory and raised on failure.
The de-mmcv port replaced it with ``cv2.imwrite``, which instead returns
``False`` silently when the parent directory does not exist, so a requested
visualization would vanish without any error. The fix in
``csrr/visualization/visualizer.py`` creates the parent directory and raises
when the write does not land. This test locks that behavior.
"""
import numpy as np
import pytest

pytest.importorskip("mmengine")
pytest.importorskip("cv2")


def test_visualize_cls_creates_missing_parent_and_writes(tmp_path):
    from csrr.structures import DataSample
    from csrr.visualization.visualizer import UniversalVisualizer

    visualizer = UniversalVisualizer()
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    out_file = tmp_path / "does" / "not" / "exist" / "out.png"
    assert not out_file.parent.exists()

    visualizer.visualize_cls(
        image,
        DataSample(),
        draw_gt=False,
        draw_pred=False,
        out_file=str(out_file),
    )

    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_visualize_cls_raises_when_write_cannot_land(tmp_path, monkeypatch):
    from csrr.structures import DataSample
    from csrr.visualization import visualizer as visualizer_module
    from csrr.visualization.visualizer import UniversalVisualizer

    # Force the underlying write to fail the way cv2 does on an unwritable
    # target; the visualizer must surface it instead of losing the image.
    monkeypatch.setattr(visualizer_module.cv2, "imwrite", lambda *a, **k: False)
    visualizer = UniversalVisualizer()
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(IOError):
        visualizer.visualize_cls(
            image,
            DataSample(),
            draw_gt=False,
            draw_pred=False,
            out_file=str(tmp_path / "out.png"),
        )
