import logging
from socket import timeout
import sympy as sp

logger = logging.getLogger(__name__)


def simplify(f, seconds):
    """
    Simplify an expression.
    """
    assert seconds > 0

    @timeout(seconds)
    def _simplify(f):
        try:
            f2 = sp.simplify(f)
            if any(s.is_Dummy for s in f2.free_symbols):
                logger.warning(f"Detected Dummy symbol when simplifying {f} to {f2}")
                return f
            else:
                return f2
        except TimeoutError:
            return f
        except Exception as e:
            logger.warning(f"{type(e).__name__} exception when simplifying {f}")
            return f

    return _simplify(f)


def remove_root_constant_terms(expr, variables, mode):
    """
    Remove root constant terms from a non-constant SymPy expression.
    """
    variables = variables if type(variables) is list else [variables]
    assert mode in ["add", "mul", "pow"]
    if not any(x in variables for x in expr.free_symbols):
        return expr
    if mode == "add" and expr.is_Add or mode == "mul" and expr.is_Mul:
        args = [
            arg
            for arg in expr.args
            if any(x in variables for x in arg.free_symbols) or (arg in [-1])
        ]
        if len(args) == 1:
            expr = args[0]
        elif len(args) < len(expr.args):
            expr = expr.func(*args)
    elif mode == "pow" and expr.is_Pow:
        assert len(expr.args) == 2
        if not any(x in variables for x in expr.args[0].free_symbols):
            return expr.args[1]
        elif not any(x in variables for x in expr.args[1].free_symbols):
            return expr.args[0]
        else:
            return expr
    return expr


def add_multiplicative_constants(expr, multiplicative_placeholder, unary_operators=[]):
    """
    Traverse the tree in post-order fashion and add multiplicative placeholders
    """

    begin = expr

    if not expr.args:
        if type(expr) == sp.core.numbers.NegativeOne:
            return expr
        else:
            return multiplicative_placeholder * expr
    for sub_expr in expr.args:
        expr = expr.subs(
            sub_expr,
            add_multiplicative_constants(
                sub_expr, multiplicative_placeholder, unary_operators=unary_operators
            ),
        )

    if str(type(expr)) in unary_operators:
        expr = multiplicative_placeholder * expr
    return expr


def add_additive_constants(expr, placeholders, unary_operators=[]):
    begin = expr
    if not expr.args:
        if type(expr) == sp.core.numbers.NegativeOne or str(expr) == str(
            placeholders["cm"]
        ):
            return expr
        else:
            return placeholders["ca"] + expr
    for sub_expr in expr.args:
        expr = expr.subs(
            sub_expr,
            add_additive_constants(
                sub_expr, placeholders, unary_operators=unary_operators
            ),
        )

    if str(type(expr)) in unary_operators:
        expr = placeholders["ca"] + expr

    return expr

