"""Calculator tool for mathematical operations.

NOTE: This implementation uses Python's AST module for SAFE expression parsing.
It does NOT use dangerous eval() or similar. The AST is parsed and only
whitelisted operations (arithmetic, math functions) are evaluated manually.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from src.tools.base import BaseTool, ParameterType, ToolExecutionContext, ToolParameter


class CalculatorTool(BaseTool):
    """A safe calculator tool that evaluates mathematical expressions.

    This implementation uses AST parsing for SAFE expression evaluation.
    Only whitelisted operators and functions are allowed - no arbitrary
    code evaluation is possible.

    Supports:
    - Basic arithmetic: +, -, *, /, //, %, **
    - Math functions: sin, cos, tan, sqrt, log, etc.
    - Constants: pi, e
    """

    # Safe operators - only these AST node types are allowed
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # Safe functions - only these function names are allowed
    FUNCTIONS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "floor": math.floor,
        "ceil": math.ceil,
        "factorial": math.factorial,
        "gcd": math.gcd,
        "pow": pow,
    }

    # Safe constants - only these names resolve to values
    CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": math.inf,
    }

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate mathematical expressions safely. "
            "Supports basic arithmetic (+, -, *, /, **, %), "
            "math functions (sin, cos, sqrt, log, etc.), "
            "and constants (pi, e)."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type=ParameterType.STRING,
                description="The mathematical expression to evaluate",
                required=True,
            ),
        ]

    async def run(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> dict[str, Any]:
        """Evaluate the mathematical expression using safe AST parsing."""
        expression = arguments.get("expression", "")

        if not expression:
            return {"error": "No expression provided", "result": None}

        try:
            result = self._safe_ast_evaluate(expression)

            # Handle special float values
            if isinstance(result, float):
                if math.isnan(result):
                    return {"error": "Result is NaN", "result": None}
                if math.isinf(result):
                    return {
                        "result": "infinity" if result > 0 else "-infinity",
                        "expression": expression,
                    }

            return {
                "result": result,
                "expression": expression,
            }

        except ZeroDivisionError:
            return {"error": "Division by zero", "result": None}
        except ValueError as e:
            return {"error": f"Math domain error: {e}", "result": None}
        except Exception as e:
            return {"error": f"Evaluation error: {e}", "result": None}

    # Alias for BaseTool interface
    execute = run

    def _safe_ast_evaluate(self, expression: str) -> float | int:
        """Safely evaluate a mathematical expression using AST parsing.

        This method parses the expression into an Abstract Syntax Tree
        and manually evaluates only whitelisted node types.
        """
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}") from e

        return self._evaluate_ast_node(tree.body)

    def _evaluate_ast_node(self, node: ast.AST) -> float | int:
        """Recursively evaluate an AST node with strict whitelisting."""
        # Numbers - safe, just return the value
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

        # Names (constants like pi, e) - only whitelisted names
        if isinstance(node, ast.Name):
            if node.id in self.CONSTANTS:
                return self.CONSTANTS[node.id]
            raise ValueError(f"Unknown constant: {node.id}")

        # Unary operators (-x, +x) - only whitelisted operators
        if isinstance(node, ast.UnaryOp):
            operand = self._evaluate_ast_node(node.operand)
            op_type = type(node.op)
            if op_type in self.OPERATORS:
                return self.OPERATORS[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type}")

        # Binary operators (x + y, x * y, etc.)
        if isinstance(node, ast.BinOp):
            left = self._evaluate_ast_node(node.left)
            right = self._evaluate_ast_node(node.right)
            op_type = type(node.op)
            if op_type in self.OPERATORS:
                return self.OPERATORS[op_type](left, right)
            raise ValueError(f"Unsupported binary operator: {op_type}")

        # Function calls (sin(x), sqrt(x), etc.)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in self.FUNCTIONS:
                    args = [self._evaluate_ast_node(arg) for arg in node.args]
                    return self.FUNCTIONS[func_name](*args)
                raise ValueError(f"Unknown function: {func_name}")
            raise ValueError("Only simple function calls are supported")

        # Reject all other node types for safety
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")
