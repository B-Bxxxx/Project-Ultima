import time
import numpy as np
import pytest
import trimesh

from src.stage1_slicer.safe_expr import compile_expression, evaluate_compiled
from src.stage1_slicer.math_strategies import CustomExpressionStrategy


MALICIOUS = [
    "().__class__.__base__.__subclasses__()",
    "np.load('/etc/passwd')",
    "np.save('/tmp/x', x)",
    "__import__('os').system('echo hacked')",
    "(lambda: 1)()",
    "[i for i in range(3)]",
    "{i: i for i in range(3)}",
    "x.__class__",
    "x.__class__.__mro__",
    "np.__dict__",
    "np.core",
    "math.__loader__",
    "open('/etc/passwd')",
    "eval('1+1')",
    "exec('x=1')",
    "'abc'",
    "b'abc'",
    "x[0]",
    "x if y else r",
    "x == y",
    "not x",
    "x @ y",
    "np.sin(x, out=x)",          # keyword argument
    "np.sin(*[x])",              # starargs
    "np.sin",                    # attribute that is not a call or allowed constant
    "sin",                       # bare function name without a call
    "foo(x)",                    # unknown function
    "z + 1",                     # unknown name
    "x" * 600,                   # too long
    "+".join(["x"] * 120),       # too many nodes
    "1e12 * x",                  # constant too large
    "x ** 99",                   # exponent too large
    "x ** y",                    # non-literal exponent
    "9 ** 9 ** 9",               # classic DoS
    "9 ** (9 ** 9)",
    "x ** (2 ** 3)",
    "x // 2",                    # operator not on the whitelist
    "x % 2",
    "(x, y)",                    # tuple
    "x := 1",                    # syntax error / walrus
    "",                          # empty
]


@pytest.mark.parametrize("expr", MALICIOUS)
def test_malicious_or_invalid_expressions_rejected(expr):
    t0 = time.time()
    with pytest.raises(ValueError):
        compile_expression(expr)
    assert time.time() - t0 < 0.5  # rejected quickly, never evaluated


def test_dos_expression_never_hangs_in_strategy():
    mesh = trimesh.creation.box(extents=(10, 10, 10))
    t0 = time.time()
    with pytest.raises(ValueError):
        CustomExpressionStrategy(mesh, {"expression": "9**9**9", "layer_height": 1.0})
    assert time.time() - t0 < 1.0


def test_malicious_expression_rejected_when_stage2_rebuilds_from_metadata():
    # A foreign intermediate JSON must not be able to execute code in Stage 2.
    from src.common.schemas import UniversalSlicedModel, UniversalLayer, SpatialContour, MinimalPrintProfile
    from src.stage2_toolpath.contour_generator import StandardToolpathGenerator
    ring = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 0)]
    layer = UniversalLayer(layer_index=0, z_height=0.0, contours=[
        SpatialContour(points=ring, normals=[(0, 0, 1)] * 4, feature_type="boundary")])
    model = UniversalSlicedModel(layers=[layer], metadata={
        "strategy": "custom_expr", "expression": "().__class__.__base__.__subclasses__()"})
    with pytest.raises(ValueError):
        StandardToolpathGenerator().generate_toolpath(model, MinimalPrintProfile())


@pytest.mark.parametrize("expr, reference", [
    ("5*np.sin(0.1*x)", lambda x, y, r: 5 * np.sin(0.1 * x)),
    ("sin(x) * cos(y)", lambda x, y, r: np.sin(x) * np.cos(y)),
    ("0.01 * r**2", lambda x, y, r: 0.01 * r ** 2),
    ("np.exp(-r / 10.0)", lambda x, y, r: np.exp(-r / 10.0)),
    ("-x / 4 + y * 0.5", lambda x, y, r: -x / 4 + y * 0.5),
    ("np.sqrt(x**2 + y**2) * 0.1", lambda x, y, r: np.sqrt(x ** 2 + y ** 2) * 0.1),
    ("pi * x * 0.01", lambda x, y, r: np.pi * x * 0.01),
    ("np.maximum(x, 0.0) * 0.1", lambda x, y, r: np.maximum(x, 0.0) * 0.1),
    ("0.0", lambda x, y, r: 0.0),
])
def test_valid_expressions_match_independent_numpy_reference(expr, reference):
    code = compile_expression(expr)
    x = np.linspace(-5, 5, 11)
    y = np.linspace(-3, 7, 11)
    r = np.sqrt(x ** 2 + y ** 2)
    np.testing.assert_allclose(evaluate_compiled(code, x, y, r), reference(x, y, r), atol=1e-12)


def test_math_module_functions_work_for_scalars():
    code = compile_expression("math.sqrt(x*x + y*y) * 0.1")
    assert evaluate_compiled(code, 3.0, 4.0, 5.0) == pytest.approx(0.5)


def test_division_by_zero_constant_raises_value_error():
    code = compile_expression("1 / 0")
    with pytest.raises(ValueError):
        evaluate_compiled(code, 0.0, 0.0, 0.0)
