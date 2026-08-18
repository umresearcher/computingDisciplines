import itertools
import pandas as pd

class _Bool:
    def __init__(self, value):
        self.value = bool(value)

    def __bool__(self):
        return self.value

    def __invert__(self):
        return _Bool(not self.value)

    def __and__(self, other):
        return _Bool(self.value and bool(other))

    def __or__(self, other):
        return _Bool(self.value or bool(other))

    def __xor__(self, other):
        return _Bool(self.value != bool(other))

    def __le__(self, other):
        # implication
        return _Bool((not self.value) or bool(other))

    def __eq__(self, other):
        # biconditional
        return _Bool(self.value == bool(other))

def eval_logic(expr_str, env):
    """
    Evaluate a logical expression under
    the given truth assignment.
    """

    if not expr_str.strip():
        return None

    s = expr_str

    # constants
    s = s.replace("T", " True ")
    s = s.replace("F", " False ")

    s = s.replace("∧", " & ")
    s = s.replace("∨", " | ")
    s = s.replace("⊕", " ^ ")
    s = s.replace("¬", " ~ ")
    s = s.replace("→", " <= ")
    s = s.replace("↔", " == ")

    wrapped_env = {
        k: _Bool(v)
        for k, v in env.items()
    }

    try:
        result = eval(
            s,
            {"__builtins__": None},
            wrapped_env
        )

        return bool(result)

    except Exception:
        return None

def extract_variables(expr_str):
    """
    Return variables used in expression.
    """

    return sorted(
        {
            ch
            for ch in expr_str
            if ch in "pqrs"
        }
    )

def generate_truth_table(expr_str):
    """
    Generate all rows of a truth table.
    """

    variables = extract_variables(expr_str)

    rows = []

    for values in itertools.product(
        [True, False],
        repeat=len(variables)
    ):

        env = dict(
            zip(variables, values)
        )

        result = eval_logic(
            expr_str,
            env
        )

        row = {}

        for var in variables:
            row[var] = (
                "T" if env[var]
                else "F"
            )

        row["Result"] = (
            "T"
            if result is True
            else (
                "F"
                if result is False
                else None
            )
        )

        rows.append(row)

    return pd.DataFrame(rows)

def generate_truth_table_for_variables(expr_str, variables):

    rows = []

    for values in itertools.product(
        [True, False],
        repeat=len(variables)
    ):

        env = dict(zip(variables, values))

        result = eval_logic(expr_str, env)

        row = {}

        for var in variables:
            row[var] = "T" if env[var] else "F"

        row["Result"] = (
            "T" if result is True
            else (
                "F"
                if result is False
                else None
            )
        )

        rows.append(row)

    return pd.DataFrame(rows)

def classify_expression(df):
    """
    Tautology / Contradiction / Contingency
    """

    results = df["Result"]

    if all(results == "T"):
        return "Tautology"

    if all(results == "F"):
        return "Contradiction"

    return "Contingency"

def check_equivalence(expr1, expr2):
    """
    Check whether two expressions
    are logically equivalent.
    """

    variables = sorted(
        set(
            extract_variables(expr1)
            +
            extract_variables(expr2)
        )
    )

    for values in itertools.product(
        [True, False],
        repeat=len(variables)
    ):

        env = dict(
            zip(variables, values)
        )

        r1 = eval_logic(expr1, env)
        r2 = eval_logic(expr2, env)

        if r1 != r2:
            return False

    return True

def generate_equivalence_truth_table(expr1, expr2):
    """
    Generate a truth table comparing two logically equivalent
    expressions.
    """

    variables = sorted(
        set(
            extract_variables(expr1)
            +
            extract_variables(expr2)
        )
    )

    rows = []

    for values in itertools.product(
        [True, False],
        repeat=len(variables)
    ):

        env = dict(
            zip(variables, values)
        )

        result1 = eval_logic(
            expr1,
            env
        )

        result2 = eval_logic(
            expr2,
            env
        )

        row = {}

        for var in variables:
            row[var] = (
                "T"
                if env[var]
                else "F"
            )

        row[expr1] = (
            "T"
            if result1 is True
            else (
                "F"
                if result1 is False
                else None
            )
        )

        row[expr2] = (
            "T"
            if result2 is True
            else (
                "F"
                if result2 is False
                else None
            )
        )

        row["Match"] = (
            "✅"
            if result1 == result2
            else "❌"
        )

        rows.append(row)

    return pd.DataFrame(rows)
