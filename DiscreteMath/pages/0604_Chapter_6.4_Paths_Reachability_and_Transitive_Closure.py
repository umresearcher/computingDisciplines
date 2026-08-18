import streamlit as st
import graphviz
import pandas as pd
from Chapter_6_utils import *

RECOMMENDED_SET_SIZE = 10
MAX_SET_SIZE = 20

st.title(
    "Module 4: Paths, Reachability, and Transitive Closure"
)

st.markdown("""
A relation describes direct connections between elements.

In this module, we study paths, reachability,
graph powers, and transitive closure, which
capture indirect connections.
""")

st.header(
    "Reachability in Directed Graphs"
)

#Initialize to None (if not already defined)
if "validated_AB" not in st.session_state:
    st.session_state.validated_AB = None

if "last_relation_mode" not in st.session_state:
    st.session_state.last_relation_mode = None

if "last_predefined_relation" not in st.session_state:
    st.session_state.last_predefined_relation = None

if "validated_relation_code" not in st.session_state:
    st.session_state.validated_relation_code = None

if "relation_function" not in st.session_state:
    st.session_state.relation_function = None

if "relation_instance" not in st.session_state:
    st.session_state.relation_instance = None

# --------------------------------------------------
# Step 1: Define a Relation on A
# --------------------------------------------------

st.subheader(
    "Step 1: Define a Set and Relation"
)

st.markdown("""
Define a finite set **A** and a binary relation **R**
on **A**.

The relation consists of all pairs (a,b) in A × A
for which relationDef(a,b) returns True.
""")

if "set_ab" not in st.session_state:
    st.session_state.set_ab = "1,2,3,4"

set_ab_text = st.text_input(
    "Enter elements separated by commas",
    key="set_ab"
)

st.caption(
    f"Recommended: {RECOMMENDED_SET_SIZE} or fewer elements. "
    f"Maximum allowed: {MAX_SET_SIZE} elements."
)

def clear_set():
    st.session_state.set_ab = ""
    st.session_state.validated_AB = None
    st.session_state.last_relation_mode = None
    st.session_state.last_predefined_relation = None
    st.session_state.validated_relation_code = None
    st.session_state.relation_function = None
    st.session_state.relation_instance = None

col1, col2 = st.columns(2)

with col1:
    validate_ab = st.button("Validate Set")

with col2:
    st.button(
        "Clear Set",
        on_click=clear_set
    )

if validate_ab:
    try:
        AB = parse_set(set_ab_text)

        if len(AB) > MAX_SET_SIZE:
            st.session_state.validated_AB = None
            raise ValueError(
                f"Maximum allowed set size is {MAX_SET_SIZE} elements."
            )

        st.session_state.validated_AB = AB

        if len(AB) > RECOMMENDED_SET_SIZE:
            st.warning(
                f"Sets larger than {RECOMMENDED_SET_SIZE} elements may "
                "produce graphs and matrices that are difficult to read."
            )

        # Clear everything downstream
        st.session_state.validated_relation_code = None
        st.session_state.relation_function = None
        st.session_state.relation_instance = None       
        st.success(
            f"Set is valid ({len(AB)} elements)"
        )
    except Exception as e:
        st.error(str(e))

if st.session_state.validated_AB is not None:
    AB = st.session_state.validated_AB
    st.markdown(
        f"**A =** "
        f"{{{', '.join(map(str, AB))}}}"
    )

    st.markdown(
        f"**|A| = {len(AB)}**"
    )

# --------------------------------------------------
# Relation Definition
# --------------------------------------------------

if st.session_state.validated_AB is not None:
    st.markdown(
        "#### Define the Relation"
    )
    st.markdown("""
    We now define a binary relation on set A.

    A binary relation is a subset of:

    **A × A**

    Choose a predefined relation or define your own relation
    using the function:

    ```python
    relationDef(a, b)
    ```

    For each pair (a, b) in A × A, the function is evaluated.

    The function should return:

    - `True` if `(a, b)` belongs to the relation.
    - `False` if `(a, b)` does not belong to the relation.""")

    with st.expander("Allowed Python features"): 
        st.markdown(""" You may use: 

        - Variables
        - Arithmetic operations (`+`, `-`, `*`, `/`, `%`)
        - Comparisons (`<`, `<=`, `>`, `>=`, `==`, `!=`)
        - Boolean operators (`and`, `or`, `not`)
        - `if` / `else` statements
        - The functions `abs()`, `min()`, `max()`, `int()`, and `float()`

        The following constructs are not allowed:

        - `import`
        - `for` loops
        - `while` loops
        - File operations
        - Network operations
        - Recursion
        """)

    relation_mode = st.radio(
        "Choose Relation Definition",
        [
            "Predefined Relation",
            "Custom Relation"
        ]
    )

    if relation_mode != st.session_state.last_relation_mode:
        st.session_state.validated_relation_code = None
        st.session_state.relation_instance = None
        st.session_state.last_relation_mode = relation_mode

    PREDEFINED_RELATIONS = {
        "Equality":
    """def relationDef(a, b):
        return a == b
    """,

        "Different":
    """def relationDef(a, b):
        return a != b
    """,

        "Less Than":
    """def relationDef(a, b):
        return a < b
    """,

        "Less Than or Equal":
    """def relationDef(a, b):
        return a <= b
    """,

        "Same Parity":
    """def relationDef(a, b):
        return (a % 2) == (b % 2)
    """,

        "Evenly Divides":
    """def relationDef(a, b):
        if a == 0:
            return False
        return b % a == 0
    """,

        "Immediate Successor":
    """def relationDef(a, b):
        return b == a + 1
    """,

        "Empty Relation":
    """def relationDef(a, b):
        return False
    """
    }

    if relation_mode == "Predefined Relation":

        relation_names = list(PREDEFINED_RELATIONS.keys())

        chosen_relation = st.selectbox(
            "Select a Relation",
            relation_names,
            index=relation_names.index("Immediate Successor")
        )

        if (
            chosen_relation
            != st.session_state.last_predefined_relation
        ):
            st.session_state.relation_instance = None
            st.session_state.last_predefined_relation = (
                chosen_relation
            )

        relation_code = PREDEFINED_RELATIONS[
            chosen_relation
        ]
        st.code(
            relation_code,
            language="python"
        )
        st.session_state.validated_relation_code = relation_code
        st.session_state.relation_function = (
            build_relation_function(relation_code)
        )
    else:
        relation_code = st.text_area(
            "Enter relationDef(a, b)",
            value="""def relationDef(a, b):
        return a == b
    """,
            height=200
        )
        if st.button(
            "Validate Relation Definition"
        ):
            is_valid, message = validate_relation(
                relation_code,
                AB,
                AB
            )
            if is_valid:
                st.session_state.validated_relation_code = relation_code

                st.session_state.relation_function = (
                    build_relation_function(relation_code)
                )
                st.session_state.relation_instance = None
                st.success(message)
            else:
                st.session_state.relation_function = None
                st.error(message)

if (
    st.session_state.validated_AB is not None
    and st.session_state.validated_relation_code is not None
    and st.session_state.relation_function is not None
):

    if st.button(
        "Build Relation"
    ):

        rel_func = (
            st.session_state.relation_function
        )

        relation = build_relation_instance(
            AB,
            AB,
            rel_func
        )

        st.session_state.relation_instance = (
            relation
        )

if (
    st.session_state.validated_AB is not None
    and st.session_state.validated_relation_code is not None
    and st.session_state.relation_function is not None
    and st.session_state.relation_instance is not None
):
    relation = st.session_state.relation_instance
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Possible Pairs",
            len(AB) * len(AB)
        )
    with col2:
        st.metric(
            "|R|",
            len(relation)
        )

if (st.session_state.validated_AB is not None
    and st.session_state.relation_function is not None
    and st.session_state.validated_relation_code is not None
    and st.session_state.relation_instance is not None):

    st.info("""
    Next, we will use the graph and matrix
    representations to study reachability.
    """)

    st.subheader(
        "Step 2: Explore the Relation"
    )
    st.markdown("""
    The relation can be viewed in several equivalent
    representations.

    For a pair `(a,b)`:

    - `(a,b)` is an ordered pair in the relation.
    - There is an arrow from `a` to `b` in the graph.
    - The matrix entry `M[a,b]` is `1`.
    """)

    relation_instance = (
        st.session_state.relation_instance
    )

    st.markdown("### Ordered Pairs")
    ordered_pairs_text = (
        "{" +
        ", ".join(
            f"({a},{b})"
            for a, b in relation_instance
        ) +
        "}"
    )

    st.code(ordered_pairs_text)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "### Directed Graph"
        )

        st.caption(
            "There is an arrow from a to b whenever "
            "(a,b) belongs to the relation."
        )

        try:

            g = graphviz.Digraph(
                format="png"
            )

            g.attr(rankdir="LR")

            g.attr(
                nodesep="0.3"
            )

            g.attr(
                ranksep="0.4"
            )

            g.attr(
                "node",
                shape="circle",
                fontsize="12"
            )

            for n in AB:

                g.node(str(n))

            # Encourage a compact layout
            for i in range(
                len(AB) - 1
            ):
                g.edge(
                    str(AB[i]),
                    str(AB[i + 1]),
                    style="invis"
                )

            for u, v in relation_instance:

                g.edge(
                    str(u),
                    str(v)
                )

            st.graphviz_chart(
                g,
                use_container_width=True
            )

        except Exception:

            st.error(
                "Unable to display graph."
            )

    with col2:

        st.markdown(
            "### Adjacency Matrix"
        )

        st.caption(
            "M[a,b] = 1 iff "
            "(a,b) belongs to the relation."
        )

        matrix_df = (
            adjacency_matrix_dataframe(
                AB,
                relation_instance
            )
        )

        st.dataframe(
            matrix_df.astype(str),
            use_container_width=True
        )

    st.info("""
A path from a to b corresponds to a sequence of edges in the directed graph.

In the next step, we will determine which pairs of 
elements are reachable by paths of a given length.
""")    

if (st.session_state.validated_AB is not None
    and st.session_state.relation_function is not None
    and st.session_state.validated_relation_code is not None
    and st.session_state.relation_instance is not None):
    st.subheader(
        "Step 3: Explore Reachability"
    )

    st.markdown("""
    A pair `(a,b)` in the relation represents a
    direct connection from `a` to `b`.

    Sometimes an element can reach another element indirectly
    by following a sequence of edges in the directed graph.

    Such a sequence of edges is called a path.
    """)

    if len(AB) > 5:
        st.info(
            """
            To keep the examples manageable,
            reachability is currently displayed
            for paths up to 5 hops.

            Longer paths can be explored later
            through graph powers and transitive
            closure.
            """
        )

    k = st.slider(
        "Path Length (Number of Edges)",
        min_value=1,
        max_value=min(len(AB), 5),
        value=2
    )

    st.markdown(
        f"### Reachable by a Path of Length {k}"
    )

    relation_instance = (
        st.session_state.relation_instance
    )

    reachability_relation, witness_paths = (
        relation_power_with_paths(
            AB,
            relation_instance,
            k
        )
    )

    if len(reachability_relation) == 0:

        st.info(
            f"No pairs are reachable by a path of length {k}."
        )

    else:

        best_pair = max(
            witness_paths,
            key=lambda pair: (
                len(set(witness_paths[pair])),
                -sum(
                    1
                    for i in range(
                        len(witness_paths[pair]) - 1
                    )
                    if witness_paths[pair][i]
                    == witness_paths[pair][i + 1]
                )
            )
        )

        best_path = witness_paths[best_pair]

        st.markdown(
            f"**Pairs reachable by a path of length {k}:**"
        )

        pairs_text = (
            "{"
            + ", ".join(
                f"({a},{b})"
                for a, b in sorted(
                    reachability_relation
                )
            )
            + "}"
        )

        st.code(pairs_text)

        st.markdown(
            "### Illustrative Example"
        )

        st.markdown(
            f"""
The pair **{best_pair}** is reachable by a path of length **{k}**.
"""
        )

        path_text = " → ".join(
            str(v)
            for v in best_path
        )

        st.markdown(
            f"""
One path is:

**{path_text}**
"""
        )

        path_pairs = [
            (
                best_path[i],
                best_path[i + 1]
            )
            for i in range(
                len(best_path) - 1
            )
        ]

        relation_pairs_text = ", ".join(
            f"({u},{v})"
            for u, v in path_pairs
        )

        st.markdown(
            f"""
**Corresponding relation pairs:**

{relation_pairs_text}
"""
        )

        st.caption(
            "Each edge in the path corresponds to a pair in the relation."
        )

        st.caption(
            f"This is one illustrative example chosen "
            f"from the {len(reachability_relation)} "
            f"pairs reachable by a path of length {k}."
        )

        st.markdown(
            "### Illustrative Example in the Directed Graph"
        )

        st.caption(
            "The green edges show the path used in the example."
        )

        col1, col2 = st.columns(2)

        with col1:
            path_edges = set(
                (best_path[i], best_path[i + 1])
                for i in range(len(best_path) - 1)
            )

            g = graphviz.Digraph(format="png")

            g.attr(rankdir="LR")

            g.attr(
                "node",
                shape="circle",
                fontsize="12"
            )

            for n in AB:
                g.node(str(n))

            # Encourage a compact layout
            for i in range(
                len(AB) - 1
            ):
                g.edge(
                    str(AB[i]),
                    str(AB[i + 1]),
                    style="invis"
                )

            for u, v in relation_instance:

                if (u, v) in path_edges:

                    g.edge(
                        str(u),
                        str(v),
                        color="green",
                        penwidth="3"
                    )

                else:

                    g.edge(
                        str(u),
                        str(v),
                        color="gray"
                    )

            st.graphviz_chart(
                g,
                use_container_width=True
            )

        with col2:
            st.markdown("")


def superscript(n):
    mapping = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹"
    }

    return "".join(
        mapping[d]
        for d in str(n)
    )

if (st.session_state.validated_AB is not None
    and st.session_state.relation_function is not None
    and st.session_state.validated_relation_code is not None
    and st.session_state.relation_instance is not None):

    st.subheader(
        "Step 4: Relation Powers"
    )

    st.markdown("""
    In Step 3, we explored pairs that are reachable
    by paths of a given length.

    The relation consisting of all pairs reachable by a path of length k is denoted by Rᵏ.
    """)    

    st.markdown("""
    - **R¹** consists of all pairs in the original relation R.
    - **R²** consists of all pairs reachable by a path of length 2.
    - **R³** consists of all pairs reachable by a path of length 3.
    - More generally, **Rᵏ** consists of all pairs reachable by a path of length k.
    """)

    power_k = st.slider(
        "Choose a value for k",
        min_value=1,
        max_value=min(len(AB), 5),
        value=2,
        key="relation_power_slider"
    )

    power_relation, witness_paths = (
        relation_power_with_paths(
            AB,
            relation_instance,
            power_k
        )
    )

    st.markdown(
        f"### Relation R{superscript(power_k)}"
    )

    pairs_text = (
        "{"
        + ", ".join(
            f"({a},{b})"
            for a, b in sorted(power_relation)
        )
        + "}"
    )

    st.code(pairs_text)

    if len(power_relation) > 0:
        st.info(
            f"R{superscript(power_k)} is itself a relation on A.\n\n"
            f"Each pair in R{superscript(power_k)} represents two vertices "
            f"that are reachable by a path of length {power_k} in the original graph."
        )
    else:
        st.info(
            f"R{superscript(power_k)} is itself a relation on A.\n\n"
            f"In this case, R{superscript(power_k)} is the empty relation."
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"### Directed Graph of R{superscript(power_k)}"
        )

        st.caption(
            f"""
        Light gray edges belong only to R.

        Black edges belong only to R{superscript(power_k)}.

        Green edges belong to both R and R{superscript(power_k)}.
        """
        )

        g = graphviz.Digraph(format="png")

        g.attr(rankdir="LR")

        g.attr(
            nodesep="0.3"
        )

        g.attr(
            ranksep="0.4"
        )

        g.attr(
            "node",
            shape="circle",
            fontsize="12"
        )

        for n in AB:
            g.node(str(n))

        # Encourage a compact layout
        for i in range(len(AB) - 1):
            g.edge(
                str(AB[i]),
                str(AB[i + 1]),
                style="invis"
            )

        r_set = set(relation_instance)

        rk_set = set(power_relation)

        all_edges = r_set | rk_set

        for u, v in sorted(all_edges):

            in_r = (u, v) in r_set
            in_rk = (u, v) in rk_set

            if in_r and in_rk:

                g.edge(
                    str(u),
                    str(v),
                    color="darkgreen",
                    penwidth="3"
                )

            elif in_r:

                g.edge(
                    str(u),
                    str(v),
                    color="lightgray"
                )

            else:

                g.edge(
                    str(u),
                    str(v),
                    color="black",
                    penwidth="3"
                )

        st.graphviz_chart(
            g,
            use_container_width=True
        )    

    with col2:
        st.markdown(
            f"### Adjacency Matrix of R{superscript(power_k)}"
        )

        matrix_df = adjacency_matrix_dataframe(
            AB,
            power_relation
        )

        st.dataframe(
            matrix_df.astype(str),
            use_container_width=True
        )

#highlighting a row and a column in matrix
def highlight_row_and_column(df, row_value, col_value):

    styles = pd.DataFrame(
        "",
        index=df.index,
        columns=df.columns
    )

    # Highlight row
    styles.loc[row_value, :] = (
        "background-color: #fff3cd;"
    )

    # Highlight column
    styles.loc[:, col_value] = (
        "background-color: #d1ecf1;"
    )

    # Highlight intersection more strongly
    styles.loc[row_value, col_value] = (
        "background-color: #90ee90;"
        "font-weight: bold;"
    )

    return styles

if (st.session_state.validated_AB is not None
    and st.session_state.relation_function is not None
    and st.session_state.validated_relation_code is not None
    and st.session_state.relation_instance is not None):

    st.subheader(
        "Step 5: Matrix Powers and Reachability"
    )

    st.markdown("""
    In Step 4, we saw that **Rᵏ** contains all pairs
    that are reachable by a path of length **k**.

    We also represented these relations using
    **adjacency matrices**.

    An important question is:

    > How can we compute the adjacency matrices of R²,
    > R³, and Rᵏ directly from M (the adjacency matrix
    > of R)?

    To answer this question, we use a special form
    of matrix multiplication for adjacency matrices.

    The adjacency matrix M contains only 0s and 1s
    and is therefore a Boolean matrix.
    """)

    with st.expander(
        "Boolean Matrix Multiplication"
    ):
        st.markdown("""
    For Boolean matrices:

    - **AND** plays the role of multiplication.
    - **OR** plays the role of addition.

    The Boolean square **M² = M × M** is computed as

    (M²)[a,b]

    =
    (M[a,1] AND M[1,b])
    OR
    (M[a,2] AND M[2,b])
    OR
    ⋯

    If there exists an element x such that

    M[a,x] = 1 and M[x,b] = 1,

    then (M²)[a,b] = 1.

    Otherwise, (M²)[a,b] = 0.
    """)

        st.info("""
    **Interpretation**

    **AND** asks:

    Can we go from a to x and then from x to b?

    **OR** asks:

    Is there at least one intermediate element x
    that makes the path possible?
    """)

    st.markdown("""
    To determine whether **(a,b)** belongs to **R²**:

    - In terms of the **ordered pairs** representation,
    we ask whether there exists an intermediate
    element **x** such that
    **(a,x) ∈ R** and **(x,b) ∈ R**.

    - In terms of the **directed graph** representation,
    we ask whether there exists an intermediate
    vertex **x** such that the path
    **a → x → b** exists.

    - In terms of the **adjacency matrix**
    representation, we ask whether
    **(M²)[a,b] = 1**.
    """)

    st.markdown(
        "### Explore a Single Entry of M²"
    )

    st.markdown("""
    Choose a row vertex **a** and a column vertex **b**.

    We will determine whether **(M²)[a,b]** equals 0 or 1
    using Boolean matrix multiplication.
    """)

    col1, col2 = st.columns(2)

    with col1:
        a = st.selectbox(
            "Row Vertex (a)",
            AB,
            key="a_squared_row"
        )

    with col2:
        b = st.selectbox(
            "Column Vertex (b)",
            AB,
            key="a_squared_col"
        )

    st.markdown(
        "#### Adjacency Matrix M"
    )

    matrix_df = adjacency_matrix_dataframe(
        AB,
        relation_instance
    )

    styled_df = (
        matrix_df.astype(str)
        .style
        .apply(
            lambda _: highlight_row_and_column(
                matrix_df,
                a,
                b
            ),
            axis=None
        )
    )

    st.dataframe(
        styled_df,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        f"The highlighted row corresponds to M[{a},x]. "
        f"The highlighted column corresponds to M[x,{b}]."
    )

    rows = []

    relation_set = set(relation_instance)

    or_values = []

    working_x = []

    for x in AB:

        ax = 1 if (a, x) in relation_set else 0

        xb = 1 if (x, b) in relation_set else 0

        and_value = ax and xb

        if and_value:
            working_x.append(x)

        or_values.append(and_value)

        rows.append(
            {
                "x": x,
                f"M[{a},x]": ax,
                f"M[x,{b}]": xb,
                f"M[{a},x] AND M[x,{b}]": int(and_value)
            }
        )

    boolean_df = pd.DataFrame(rows)

    st.markdown(
        f"#### Computing (M²)[{a},{b}]"
    )

    st.dataframe(
        boolean_df.astype(str),
        hide_index=True,
        use_container_width=True
    )

    or_expression = " OR ".join(
        str(v)
        for v in or_values
    )

    result = int(any(or_values))

    st.markdown(
        "#### Combine Using OR"
    )

    st.code(
        f"{or_expression} = {result}"
    )

    if working_x:
        st.markdown(
            "**Paths of length 2 that work:**"
        )

        for x in working_x:
            st.markdown(
                f"- {a} → {x} → {b}"
            )

    if result == 1:
        st.success(
            f"(M²)[{a},{b}] = 1"
        )

        st.markdown(
            f"""
    There exists an intermediate vertex **x**
    such that

    - ({a},x) belongs to R, and
    - (x,{b}) belongs to R.

    Therefore, ({a},{b}) belongs to **R²**.
    """
        )

    else:
        st.info(
            f"(M²)[{a},{b}] = 0"
        )

        st.markdown(
            f"""
    There is no intermediate vertex **x**
    such that

    - ({a},x) belongs to R, and
    - (x,{b}) belongs to R.

    Therefore, ({a},{b}) does not belong to **R²**.
    """
        )

    st.markdown(
        "### Constructing the Entire Matrix M²"
    )

    st.markdown("""
    Every entry of M² is computed in exactly
    the same way as the example above.

    Applying Boolean matrix multiplication
    to every entry produces the entire matrix M².
    """)

    r2_relation, _ = relation_power_with_paths(
    AB,
    relation_instance,
    2
)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "#### Directed Graph of R²"
        )

        st.caption(
            """
    Light gray edges belong only to R.

    Black edges belong only to R².

    Green edges belong to both R and R².
    """
        )

        g = graphviz.Digraph(
            format="png"
        )

        g.attr(
            rankdir="LR"
        )

        g.attr(
            nodesep="0.3"
        )

        g.attr(
            ranksep="0.4"
        )

        g.attr(
            "node",
            shape="circle",
            fontsize="12"
        )

        for n in AB:
            g.node(str(n))

        # Encourage a compact layout
        for i in range(len(AB) - 1):
            g.edge(
                str(AB[i]),
                str(AB[i + 1]),
                style="invis"
            )

        r_set = set(relation_instance)

        r2_set = set(r2_relation)

        all_edges = r_set | r2_set

        for u, v in all_edges:

            in_r = (u, v) in r_set

            in_r2 = (u, v) in r2_set

            if in_r and in_r2:

                g.edge(
                    str(u),
                    str(v),
                    color="darkgreen",
                    penwidth="3"
                )

            elif in_r:

                g.edge(
                    str(u),
                    str(v),
                    color="lightgray"
                )

            else:

                g.edge(
                    str(u),
                    str(v),
                    color="black",
                    penwidth="3"
                )

        st.graphviz_chart(
            g,
            use_container_width=True
        )

    with col2:

        st.markdown(
            "#### Matrix M²"
        )

        r2_df = adjacency_matrix_dataframe(
            AB,
            r2_relation
        )

        st.dataframe(
            r2_df.astype(str),
            use_container_width=True
        )

    st.success(
        "The matrix M² is identical to the adjacency matrix of R²."
    )

    st.info("""
    More generally:

    M² is the adjacency matrix of R²

    M³ is the adjacency matrix of R³

    ...

    Mᵏ is the adjacency matrix of Rᵏ
    """)

if (st.session_state.validated_AB is not None
    and st.session_state.relation_function is not None
    and st.session_state.validated_relation_code is not None
    and st.session_state.relation_instance is not None):

    st.subheader(
        "Step 6: Higher Matrix Powers"
    )

    st.markdown("""
    We have seen that **Mᵏ** is the adjacency
    matrix of **Rᵏ**.

    Select a value of **k** and compare the graph
    and matrix representations.
    """)

    k_matrix = st.slider(
        "Select k",
        min_value=1,
        max_value=max(2, len(AB) - 1),
        value=min(2, max(2, len(AB) - 1)),
        key="matrix_power_k"
    )

    st.caption(
        "Try different values of k and observe how the graph and matrix change."
    )

    rk_relation, _ = relation_power_with_paths(
        AB,
        relation_instance,
        k_matrix
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            f"### Directed Graph of R{superscript(k_matrix)}"
        )

        st.caption(
            f"""
    Light gray edges belong only to R.

    Black edges belong only to R{superscript(k_matrix)}.

    Green edges belong to both R and R{superscript(k_matrix)}.
    """
        )

        g = graphviz.Digraph(
            format="png"
        )

        g.attr(rankdir="LR")

        g.attr(
            nodesep="0.3"
        )

        g.attr(
            ranksep="0.4"
        )

        g.attr(
            "node",
            shape="circle",
            fontsize="12"
        )

        for n in AB:
            g.node(str(n))

        # Encourage a compact layout
        for i in range(len(AB) - 1):
            g.edge(
                str(AB[i]),
                str(AB[i + 1]),
                style="invis"
            )

        r_set = set(relation_instance)
        rk_set = set(rk_relation)

        all_edges = r_set | rk_set

        for u, v in sorted(all_edges):

            in_r = (u, v) in r_set
            in_rk = (u, v) in rk_set

            if in_r and in_rk:

                g.edge(
                    str(u),
                    str(v),
                    color="darkgreen",
                    penwidth="3"
                )

            elif in_r:

                g.edge(
                    str(u),
                    str(v),
                    color="lightgray"
                )

            else:

                g.edge(
                    str(u),
                    str(v),
                    color="black",
                    penwidth="3"
                )

        st.graphviz_chart(
            g,
            use_container_width=True
        )

    with col2:

        st.markdown(
            f"### Matrix M{superscript(k_matrix)}"
        )

        matrix_df = adjacency_matrix_dataframe(
            AB,
            rk_relation
        )

        st.dataframe(
            matrix_df.astype(str),
            use_container_width=True
        )

    st.success(
        f"M{superscript(k_matrix)} is the adjacency matrix "
        f"of R{superscript(k_matrix)}."
    )

if (st.session_state.validated_AB is not None
    and st.session_state.relation_function is not None
    and st.session_state.validated_relation_code is not None
    and st.session_state.relation_instance is not None):

    st.subheader(
        "Step 7: Transitive Closure"
    )

    st.markdown("""
    In the previous steps, we explored reachability
    using paths of a specific length.

    - R¹ captures paths of length 1.
    - R² captures paths of length 2.
    - R³ captures paths of length 3.
    - ...

    What if we want to know whether one vertex can
    reach another using a path of any positive length?

    To answer this question, we use the transitive closure.
    """)

    st.info("""
The transitive closure of R contains all pairs
(a,b) such that b is reachable from a by a path
of positive length.

The transitive closure is denoted by R⁺.
""")

    max_power = max(1, len(AB) - 1)

    transitive_closure = set()

    for k in range(1, max_power + 1):

        rk_relation, _ = relation_power_with_paths(
            AB,
            relation_instance,
            k
        )

        transitive_closure.update(
            rk_relation
        )

    st.markdown(
        "### Constructing the Transitive Closure"
    )

    st.markdown(
        f"""
For a set with {len(AB)} vertices, we combine:

R¹ ∪ R² ∪ ... ∪ R{superscript(max_power)}

to obtain the transitive closure R⁺.
"""
    )

    with st.expander(
        "Why do we stop at R" + superscript(max_power) + "?"
    ):
        st.markdown(
            f"""
    A path that never repeats a vertex can contain
    at most **{len(AB) - 1} edges**.

    For a set with **{len(AB)} vertices**, any path
    with more than **{len(AB) - 1} edges** must
    repeat at least one vertex.    

    Repeating vertices cannot create any new reachable
    pairs that were not already discovered by shorter paths.

    Therefore, when constructing the transitive closure,
    it is sufficient to combine:

    R¹ ∪ R² ∪ ... ∪ R{superscript(max_power)}

    rather than considering arbitrarily long paths.
    """
        )

    pairs_text = (
        "{"
        + ", ".join(
            f"({a},{b})"
            for a, b in sorted(
                transitive_closure
            )
        )
        + "}"
    )

    st.markdown(
        "### Relation R⁺"
    )

    st.code(
        pairs_text
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### Directed Graph of R⁺"
        )

        st.caption(
            """
    Light gray edges belong only to R.

    Black edges belong only to R⁺.

    Green edges belong to both R and R⁺.
    """
        )

        g = graphviz.Digraph(
            format="png"
        )

        g.attr(rankdir="LR")

        g.attr(
            nodesep="0.3"
        )

        g.attr(
            ranksep="0.4"
        )

        g.attr(
            "node",
            shape="circle",
            fontsize="12"
        )

        for n in AB:
            g.node(str(n))

        # Encourage a compact layout
        for i in range(len(AB) - 1):
            g.edge(
                str(AB[i]),
                str(AB[i + 1]),
                style="invis"
            )

        r_set = set(relation_instance)

        r_plus_set = set(transitive_closure)

        all_edges = r_set | r_plus_set

        for u, v in sorted(all_edges):

            in_r = (u, v) in r_set
            in_r_plus = (u, v) in r_plus_set

            if in_r and in_r_plus:

                g.edge(
                    str(u),
                    str(v),
                    color="darkgreen",
                    penwidth="3"
                )

            elif in_r:

                g.edge(
                    str(u),
                    str(v),
                    color="lightgray"
                )

            else:

                g.edge(
                    str(u),
                    str(v),
                    color="black",
                    penwidth="3"
                )

        st.graphviz_chart(
            g,
            use_container_width=True
        )

    with col2:

        st.markdown(
            "### Adjacency Matrix of R⁺"
        )

        matrix_df = adjacency_matrix_dataframe(
            AB,
            transitive_closure
        )

        st.dataframe(
            matrix_df.astype(str),
            use_container_width=True
        )

    st.success("""
    R⁺ contains exactly the pairs that are reachable
    in the directed graph.

    The directed graph and adjacency matrix above
    provide two equivalent representations of this
    reachability information.
    """)

if (st.session_state.validated_AB is not None
    and st.session_state.relation_function is not None
    and st.session_state.validated_relation_code is not None
    and st.session_state.relation_instance is not None):

    st.markdown(
        "### Knowledge Check"
    )

    with st.expander(
        "Think About These Questions"
    ):

        st.markdown("""
**1. True or False**

The transitive closure of the Immediate Successor
relation on a finite ordered set is the Less Than
relation on the same set.

---

**2. True or False**

If a relation R is already transitive, then the
transitive closure of R is equal to R.

In other words, is R⁺ = R if R is transitive?

---

**3. True or False**

The transitive closure never removes pairs from a
relation.

In other words, is R⁺ ⊇ R?

---

**4. Challenge Question**

Consider the relation

(a == b) or (abs(a - b) == 2)

What relation do you think its transitive closure
becomes on the integers 1,2,3,4,5,... ?
""")

        with st.expander(
            "Show Answers"
        ):

            st.markdown("""
        **1. True**

        The Immediate Successor relation contains pairs
        such as (1,2), (2,3), and (3,4).

        Its transitive closure contains all pairs
        (i,j) where i < j, which is exactly the
        Less Than relation.

        ---

        **2. True**

        If a relation is already transitive, then
        no new pairs need to be added.

        Therefore:

        R⁺ = R

        ---

        **3. True**

        The transitive closure only adds pairs.

        It never removes pairs.

        Therefore:

        R⁺ ⊇ R

        ---

        **4. Challenge Question**

        The transitive closure becomes the Same
        Parity relation.

        Every odd integer becomes reachable from
        every other odd integer.

        Every even integer becomes reachable from
        every other even integer.

        Odd integers do not become related to
        even integers.
        """)

    st.success("""
    Summary

    • Rᵏ contains all pairs reachable by a path of length k.

    • Mᵏ is the adjacency matrix of Rᵏ.

    • R⁺ contains all pairs reachable by a path of positive length.

    • The transitive closure captures reachability in the directed graph.
    """)

if (st.session_state.validated_AB is not None
    and st.session_state.relation_function is not None
    and st.session_state.validated_relation_code is not None
    and st.session_state.relation_instance is not None):
    with st.expander(
        "Applications of Reachability and Transitive Closure"
    ):
        st.markdown("""
    These ideas have many practical applications in
    computer science and everyday life.

    ### Flights Between Cities

    Suppose:

    - Each vertex represents a city.
    - An edge from city A to city B represents a direct flight.

    Then:

    - **R** represents direct flights.
    - **R²** represents flights reachable with exactly one layover.
    - **R³** represents flights reachable with exactly two layovers.
    - **R⁺** represents flights reachable using any number of layovers.

    ### Other Applications

    Reachability and transitive closure also appear in:

    - Course prerequisite planning
    - Software package dependencies
    - Route planning and navigation
    - Social networks
    - Computer network routing

    In each case, the transitive closure identifies
    all locations, people, courses, or systems that
    can be reached through a chain of connections.
    """)


