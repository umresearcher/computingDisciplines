import pandas as pd
import ast
import random

def parse_set(text):
    items = [
        x.strip()
        for x in text.split(",")
        if x.strip()
    ]
    if len(items) == 0:
        raise ValueError("Set cannot be empty.")
    if len(items) > 100:
        raise ValueError("Maximum size is 100 elements.")
    try:
        items = [int(x) for x in items]
    except ValueError:
        pass
    return sorted(list(set(items)))

ALLOWED_FUNCTIONS = {
    "abs": abs,
    "int": int,
    "float": float,
    "max": max,
    "min": min,
}

def build_relation_function(code_text):
    scope = {}
    exec(
        code_text,
        {"__builtins__": {}, **ALLOWED_FUNCTIONS},
        scope
    )
    return scope["relationDef"]

def validate_relation_code(code_text):

    try:

        tree = ast.parse(code_text)

        funcs = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        ]

        if len(funcs) != 1:
            return False, "Provide exactly one function."

        func = funcs[0]

        if func.name != "relationDef":
            return (
                False,
                "Function must be named relationDef."
            )

        if len(func.args.args) != 2:
            return (
                False,
                "Function must have exactly two parameters."
            )

        forbidden = (
            ast.Import,
            ast.ImportFrom,
            ast.For,
            ast.While,
            ast.With,
            ast.Try,
            ast.ClassDef,
            ast.Lambda,
        )

        for node in ast.walk(tree):

            if isinstance(node, forbidden):
                return (
                    False,
                    f"{type(node).__name__} is not allowed."
                )

            if isinstance(node, ast.Call):

                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "relationDef"
                ):
                    return (
                        False,
                        "Recursion is not allowed."
                    )

        if not any(
            isinstance(node, ast.Return)
            for node in ast.walk(func)
        ):
            return (
                False,
                "Function must contain a return statement."
            )

        return True, "Code is valid."

    except SyntaxError as e:

        return (
            False,
            f"Syntax error: {e}"
        )

def validate_relation_on_sets(
    A,
    B,
    rel_func
):

    for a in A:
        for b in B:

            try:

                result = rel_func(a, b)

            except Exception as e:

                return (
                    False,
                    f"Function failed for "
                    f"({a}, {b}): {e}"
                )

            if not isinstance(result, bool):

                return (
                    False,
                    f"relationDef({a}, {b}) "
                    f"returned "
                    f"{type(result).__name__}. "
                    "Every pair must return "
                    "True or False."
                )

    return True, "Function works correctly on A × B."

def validate_relation(code_text, A, B):

    is_valid, message = (
        validate_relation_code(
            code_text
        )
    )

    if not is_valid:
        return False, message

    rel_func = build_relation_function(
        code_text
    )

    works, message = (
        validate_relation_on_sets(
            A,
            B,
            rel_func
        )
    )

    if not works:
        return False, message

    return True, "Function is valid."

def preview_relation(A, B, rel_func, sample_size=50):
    pairs = [
        (a, b)
        for a in A
        for b in B
    ]
    random.shuffle(pairs)
    rows = []
    for a, b in pairs[:sample_size]:
        try:
            result = rel_func(a, b)
            if result:
                result = "✅   Belongs"
            else:
                result = "❌   Does Not Belong"
        except Exception as e:
            result = f"ERROR: {e}"
        rows.append(
            {
                "a": str(a),
                "b": str(b),
                "Result": result
            }
        )
    return pd.DataFrame(rows)

def build_relation_instance(A, B, rel_func):
    tuples = []
    for a in A:
        for b in B:
            try:
                if rel_func(a, b):
                    tuples.append((a, b))
            except Exception:
                pass
    return tuples

def relation_dataframe(relation):
    df = pd.DataFrame(
        relation,
        columns=["A", "B"]
    )
    df["A"] = df["A"].astype(str)
    df["B"] = df["B"].astype(str)
    return df

def relation_properties(A, relation):
    R = set(relation)
    reflexive = all(
        (a, a) in R
        for a in A
    )
    anti_reflexive = all(
        (a, a) not in R
        for a in A
    )
    symmetric = all(
        (b, a) in R
        for (a, b) in R
    )
    antisymmetric = all(
        a == b or (b, a) not in R
        for (a, b) in R
    )
    transitive = True
    for (a, b) in R:
        for (c, d) in R:
            if b == c and (a, d) not in R:
                transitive = False
                break
        if not transitive:
            break
    return {
        "Reflexive": reflexive,
        "AntiReflexive": anti_reflexive,
        "Symmetric": symmetric,
        "Antisymmetric": antisymmetric,
        "Transitive": transitive,
    }

def reflexive_counterexample(A, relation):

    R = set(relation)

    for a in A:
        if (a, a) not in R:
            return a

    return None

def anti_reflexive_counterexample(A, relation):

    R = set(relation)

    for a in A:
        if (a, a) in R:
            return a

    return None

def symmetric_counterexample(relation):

    R = set(relation)

    for a, b in R:

        if (b, a) not in R:
            return (a, b)

    return None

def antisymmetric_counterexample(relation):

    R = set(relation)

    for a, b in R:

        if a != b and (b, a) in R:
            return (a, b)

    return None

def transitive_counterexample(relation):

    R = set(relation)

    for a, b in R:
        for c, d in R:

            if b == c and (a, d) not in R:

                return (
                    (a, b),
                    (c, d),
                    (a, d)
                )

    return None

def transitive_witness(relation):

    R = set(relation)

    # ------------------------------------------
    # Best case:
    # a != b and b != d
    # Example:
    # (1,2), (2,3) => (1,3)
    # ------------------------------------------

    for a, b in R:
        for c, d in R:

            if (
                b == c
                and (a, d) in R
                and a != b
                and b != d
            ):
                return (
                    (a, b),
                    (c, d),
                    (a, d)
                )

    # ------------------------------------------
    # Next best:
    # at least one inequality
    # ------------------------------------------

    for a, b in R:
        for c, d in R:

            if (
                b == c
                and (a, d) in R
                and (a != b or b != d)
            ):
                return (
                    (a, b),
                    (c, d),
                    (a, d)
                )

    # ------------------------------------------
    # Final fallback:
    # any witness
    # ------------------------------------------

    for a, b in R:
        for c, d in R:

            if (
                b == c
                and (a, d) in R
            ):
                return (
                    (a, b),
                    (c, d),
                    (a, d)
                )

    return None

def adjacency_matrix_dataframe(nodes, relation):

    R = set(relation)

    matrix = []

    for a in nodes:

        row = []

        for b in nodes:

            row.append(
                1 if (a, b) in R else 0
            )

        matrix.append(row)

    return pd.DataFrame(
        matrix,
        index=nodes,
        columns=nodes
    )

def relation_power(
    AB,
    relation_instance,
    k
):
    """
    Returns R^k as a set of ordered pairs.
    """

    if k <= 0:
        return set()

    current = set(relation_instance)

    if k == 1:
        return current

    for _ in range(2, k + 1):

        next_relation = set()

        for a, b in current:

            for c, d in relation_instance:

                if b == c:
                    next_relation.add((a, d))

        current = next_relation

    return current

def find_path_of_length_k(
    AB,
    relation_instance,
    start,
    end,
    k
):
    """
    Returns one path of exactly k hops from
    start to end, or None if no such path exists.

    Preference is given to:
    1. Non-self-loop edges.
    2. Vertices not already on the path.
    """

    adjacency = {}

    for v in AB:
        adjacency[v] = []

    for u, v in relation_instance:
        adjacency[u].append(v)

    def dfs(current, hops_remaining, path):

        if hops_remaining == 0:

            if current == end:
                return path

            return None

        neighbors = sorted(
            adjacency[current],
            key=lambda nxt: (
                nxt == current,  # self-loops last
                nxt in path      # repeated vertices last
            )
        )

        for nxt in neighbors:

            result = dfs(
                nxt,
                hops_remaining - 1,
                path + [nxt]
            )

            if result is not None:
                return result

        return None

    return dfs(
        start,
        k,
        [start]
    )

def relation_power_with_paths(
    AB,
    relation_instance,
    k
):
    """
    Computes R^k and stores one witness path for each
    reachable pair.

    Returns:

        reachable_pairs
            set of (a,b)

        witness_paths
            dict mapping

                (a,b) -> [a,...,b]

            where the list contains exactly k hops.
    """

    if k < 1:
        return set(), {}

    current = {}

    for a, b in relation_instance:
        current[(a, b)] = [a, b]

    if k == 1:
        return set(current.keys()), current

    for _ in range(2, k + 1):

        next_paths = {}

        for (start, mid), path1 in current.items():

            for u, end in relation_instance:

                if mid != u:
                    continue

                candidate_path = (
                    path1 + [end]
                )

                pair = (start, end)

                if pair not in next_paths:

                    next_paths[pair] = candidate_path

                else:

                    old_path = next_paths[pair]

                    old_score = (
                        len(set(old_path)),
                        -sum(
                            1
                            for i in range(
                                len(old_path) - 1
                            )
                            if old_path[i]
                               == old_path[i + 1]
                        )
                    )

                    new_score = (
                        len(set(candidate_path)),
                        -sum(
                            1
                            for i in range(
                                len(candidate_path) - 1
                            )
                            if candidate_path[i]
                               == candidate_path[i + 1]
                        )
                    )

                    if new_score > old_score:
                        next_paths[pair] = (
                            candidate_path
                        )

        current = next_paths

    return (
        set(current.keys()),
        current
    )

