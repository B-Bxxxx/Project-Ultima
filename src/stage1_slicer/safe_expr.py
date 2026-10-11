"""
Safe evaluation of user-supplied f(x, y) expressions (AGENTS.md: "No Blind Eval").

The expression is parsed to an AST and validated against a strict whitelist
BEFORE it is compiled. The compiled code object is then evaluated in a minimal
namespace. Anything outside the whitelist raises ValueError.

Allowed:
  * numbers (int/float, |value| <= MAX_CONSTANT)
  * the variables x, y, r and the constants pi, e
  * + - * / and unary + -
  * `**` ONLY with a small numeric-literal exponent (|n| <= MAX_EXPONENT)
  * calls of a fixed set of functions, either bare (sin(x)) or as np.<fn> /
    math.<fn>; no keyword arguments, no *args
  * np.pi / math.pi / np.e / math.e

Everything else (other attributes, subscripts, lambdas, comprehensions,
strings, dunder names, imports, ...) is rejected.
"""
import ast
import math
from typing import Any, Dict

import numpy as np

MAX_EXPR_LENGTH = 500
MAX_NODES = 100
MAX_CONSTANT = 1e6
MAX_EXPONENT = 10.0
MAX_CALL_ARGS = 3

ALLOWED_NAMES = ("x", "y", "r", "pi", "e")

NP_FUNCS = frozenset({
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2",
    "sinh", "cosh", "tanh", "exp", "log", "log10", "sqrt", "abs", "absolute",
    "floor", "ceil", "sign", "minimum", "maximum", "hypot", "clip",
})
MATH_FUNCS = frozenset({
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "exp", "log", "log10", "sqrt", "fabs",
    "floor", "ceil", "hypot",
})
MODULE_CONSTANTS = frozenset({"pi", "e"})
BARE_FUNCS = NP_FUNCS  # sin(x) is resolved to the numpy function

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _fail(msg: str):
    raise ValueError(f"Unsafe or invalid expression: {msg}")


def _is_small_number(node: ast.AST, limit: float) -> bool:
    """True for a numeric literal (optionally negated) with |value| <= limit."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        node = node.operand
    return (isinstance(node, ast.Constant)
            and type(node.value) in (int, float)
            and abs(node.value) <= limit)


def _check_module_attr(node: ast.Attribute, allowed: frozenset):
    if not (isinstance(node.value, ast.Name) and node.value.id in ("np", "math")):
        _fail("attribute access is only allowed on np / math")
    if node.attr not in allowed:
        _fail(f"'{node.value.id}.{node.attr}' is not an allowed name")


def _validate(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _validate(node.body)
        return

    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            _fail("only int/float literals are allowed")
        if not math.isfinite(node.value) or abs(node.value) > MAX_CONSTANT:
            _fail(f"constant out of range (|c| <= {MAX_CONSTANT:g})")
        return

    if isinstance(node, ast.Name):
        if node.id not in ALLOWED_NAMES:
            _fail(f"unknown name '{node.id}'")
        return

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            _fail(f"operator {type(node.op).__name__} is not allowed")
        if isinstance(node.op, ast.Pow) and not _is_small_number(node.right, MAX_EXPONENT):
            _fail(f"exponent must be a numeric literal with |n| <= {MAX_EXPONENT:g}")
        _validate(node.left)
        _validate(node.right)
        return

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            _fail(f"operator {type(node.op).__name__} is not allowed")
        _validate(node.operand)
        return

    if isinstance(node, ast.Attribute):
        # Only np.pi / math.pi / np.e / math.e as a value (function calls are
        # handled in the Call branch and never reach here through func).
        _check_module_attr(node, MODULE_CONSTANTS)
        return

    if isinstance(node, ast.Call):
        if node.keywords:
            _fail("keyword arguments are not allowed")
        if any(isinstance(a, ast.Starred) for a in node.args):
            _fail("*args are not allowed")
        if not node.args or len(node.args) > MAX_CALL_ARGS:
            _fail(f"functions take 1..{MAX_CALL_ARGS} arguments")
        func = node.func
        if isinstance(func, ast.Name):
            if func.id not in BARE_FUNCS:
                _fail(f"function '{func.id}' is not allowed")
        elif isinstance(func, ast.Attribute):
            allowed = NP_FUNCS if (isinstance(func.value, ast.Name) and func.value.id == "np") else MATH_FUNCS
            _check_module_attr(func, allowed)
        else:
            _fail("only direct function calls are allowed")
        for a in node.args:
            _validate(a)
        return

    _fail(f"{type(node).__name__} is not allowed")


def compile_expression(expr: str):
    """Validate `expr` and return a compiled code object (mode 'eval')."""
    if not isinstance(expr, str):
        _fail("expression must be a string")
    if len(expr) > MAX_EXPR_LENGTH:
        _fail(f"expression longer than {MAX_EXPR_LENGTH} characters")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc.msg}") from exc

    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        _fail(f"expression too complex (more than {MAX_NODES} AST nodes)")

    _validate(tree)
    return compile(tree, "<expression>", "eval")


# The only globals the compiled expression can see. No builtins.
_GLOBALS: Dict[str, Any] = {"__builtins__": {}, "np": np, "math": math}
_GLOBALS.update({name: getattr(np, name) for name in NP_FUNCS})


def evaluate_compiled(code, x, y, r) -> Any:
    local_dict = {"x": x, "y": y, "r": r, "pi": math.pi, "e": math.e}
    try:
        return eval(code, _GLOBALS, local_dict)  # noqa: S307  (validated AST only)
    except ZeroDivisionError as exc:
        raise ValueError("Expression divides by zero") from exc
