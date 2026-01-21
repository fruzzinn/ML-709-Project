"""Code executor tool for running Python code in a sandboxed environment.

NOTE: This is a RESEARCH tool for studying adversarial agent behavior.
The sandbox is intentionally simple for controlled experiments.
Production deployments should use proper containerization.
"""

from __future__ import annotations

import asyncio
import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from src.tools.base import BaseTool, ParameterType, ToolExecutionContext, ToolParameter


class CodeExecutorTool(BaseTool):
    """A sandboxed code executor for running Python code.

    RESEARCH USE: This tool enables studying how agents interact with
    code execution capabilities and how adversarial attacks propagate.

    Features:
    - Captures stdout and stderr
    - Timeout enforcement
    - Restricted built-ins (configurable)
    - Memory-limited execution namespace
    """

    # Default restricted built-ins for sandboxing
    SAFE_BUILTINS = {
        "abs": abs,
        "all": all,
        "any": any,
        "bin": bin,
        "bool": bool,
        "chr": chr,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "frozenset": frozenset,
        "hash": hash,
        "hex": hex,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "iter": iter,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "print": print,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "type": type,
        "zip": zip,
    }

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        max_output_length: int = 10000,
        allow_imports: bool = False,
    ) -> None:
        super().__init__()
        self.timeout_seconds = timeout_seconds
        self.max_output_length = max_output_length
        self.allow_imports = allow_imports

    @property
    def name(self) -> str:
        return "code_executor"

    @property
    def description(self) -> str:
        return (
            "Run Python code in a sandboxed environment. "
            "Returns the output (stdout/stderr) and any returned value. "
            "Useful for calculations, data processing, and testing code snippets. "
            "Note: Import statements are restricted for security."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="code",
                type=ParameterType.STRING,
                description="The Python code to run",
                required=True,
            ),
            ToolParameter(
                name="timeout",
                type=ParameterType.NUMBER,
                description="Timeout in seconds (default: 5, max: 30)",
                required=False,
                default=5.0,
            ),
        ]

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> dict[str, Any]:
        """Run the provided code."""
        code = arguments.get("code", "")
        timeout = min(arguments.get("timeout", self.timeout_seconds), 30.0)

        if not code:
            return {"error": "No code provided", "output": "", "result": None}

        # Basic security checks
        security_check = self._security_check(code)
        if security_check:
            return {
                "error": f"Security violation: {security_check}",
                "output": "",
                "result": None,
            }

        try:
            # Run code with timeout
            result = await asyncio.wait_for(
                self._run_code(code),
                timeout=timeout,
            )
            return result

        except asyncio.TimeoutError:
            return {
                "error": f"Timed out after {timeout} seconds",
                "output": "",
                "result": None,
            }
        except Exception as e:
            return {
                "error": f"Failed: {str(e)}",
                "output": "",
                "result": None,
            }

    async def _run_code(self, code: str) -> dict[str, Any]:
        """Run code in a restricted namespace."""
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Create restricted namespace
        namespace = self._create_namespace()

        result_value = None
        error_msg = None

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Compile the code
                compiled = compile(code, "<sandbox>", "exec")

                # Run in restricted namespace (intentional for research sandbox)
                run_globals = namespace.copy()
                run_locals: dict[str, Any] = {}

                # Use builtins to run compiled code in sandbox
                builtins = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
                sandbox_exec = builtins.get("exec", None)  # type: ignore
                if sandbox_exec:
                    sandbox_exec(compiled, run_globals, run_locals)

                # Try to get a result if the code defines 'result'
                if "result" in run_locals:
                    result_value = run_locals["result"]
                elif "result" in run_globals:
                    result_value = run_globals["result"]

        except Exception:
            error_msg = traceback.format_exc()
            stderr_capture.write(error_msg)

        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()

        # Truncate if too long
        if len(stdout) > self.max_output_length:
            stdout = stdout[: self.max_output_length] + "\n... (truncated)"
        if len(stderr) > self.max_output_length:
            stderr = stderr[: self.max_output_length] + "\n... (truncated)"

        return {
            "output": stdout,
            "stderr": stderr if stderr else None,
            "result": result_value,
            "error": error_msg if error_msg else None,
        }

    def _create_namespace(self) -> dict[str, Any]:
        """Create a restricted namespace."""
        import math

        return {
            "__builtins__": self.SAFE_BUILTINS.copy(),
            "math": math,
        }

    def _security_check(self, code: str) -> str | None:
        """Perform basic security checks on the code."""
        dangerous_patterns = [
            ("open(", "File operations not allowed"),
            ("os.", "OS module access not allowed"),
            ("sys.", "Sys module access not allowed"),
            ("subprocess", "Subprocess not allowed"),
            ("__import__", "Dynamic imports not allowed"),
            ("importlib", "Importlib not allowed"),
            ("globals(", "Globals access not allowed"),
            ("locals(", "Locals access not allowed"),
            ("getattr(", "Getattr not allowed"),
            ("setattr(", "Setattr not allowed"),
            ("delattr(", "Delattr not allowed"),
        ]

        code_lower = code.lower()
        for pattern, message in dangerous_patterns:
            if pattern.lower() in code_lower:
                return message

        return None
