"""Correctness tests against PyTorch. Skipped automatically when no CUDA GPU / Triton is available."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

from triton_kernels import layer_norm, softmax  # noqa: E402


@pytest.mark.parametrize("shape", [(128, 64), (1823, 781), (4096, 1024), (512, 4096)])
def test_softmax(shape):
    x = torch.randn(*shape, device="cuda")
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=1), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", [(128, 64), (1823, 781), (4096, 1024)])
def test_layer_norm(shape):
    x = torch.randn(*shape, device="cuda")
    w, b = torch.randn(shape[1], device="cuda"), torch.randn(shape[1], device="cuda")
    ref = torch.nn.functional.layer_norm(x, (shape[1],), w, b, 1e-5)
    torch.testing.assert_close(layer_norm(x, w, b), ref, atol=1e-4, rtol=1e-4)
