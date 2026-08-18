import streamlit as st
import graphviz
from Chapter_6_utils import *

RECOMMENDED_SET_SIZE = 10
MAX_SET_SIZE = 20

# --------------------------------------------------
# Module 3: Binary Relations As Graphs and Matrices
# --------------------------------------------------

st.title("Module 3: Binary Relations As Graphs and Matrices")


st.markdown("""
In Module 1, we saw that a binary relation can be represented as
a collection of ordered pairs or as a 2-column table.

In this module, we focus on binary relations on a single set
(A = B) and introduce two additional representations:

* Directed Graphs
* Adjacency Matrices

All three representations describe the same relation:

* Ordered Pairs / Table
* Directed Graph
* Adjacency Matrix
""")

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
# Step 1: Define the Set
# --------------------------------------------------

st.subheader("Step 1: Define the Set")

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
# Step 2: Define the Relation
# --------------------------------------------------

if st.session_state.validated_AB is None:
    st.info(
        "Complete Step 1 by validating the set."
    )
else:
    st.subheader("Step 2: Define the Relation")
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
        chosen_relation = st.selectbox(
            "Select a Relation",
            list(PREDEFINED_RELATIONS.keys())
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

# --------------------------------------------------
# Step 3: Build the Relation
# --------------------------------------------------

if (
    st.session_state.validated_AB is not None
    and st.session_state.relation_function is not None
    and st.session_state.validated_relation_code is not None
):
    st.subheader("Step 3: Build the Relation")

    st.markdown("""
    The relation is obtained by evaluating `relationDef(a, b)`
    for every pair in **A × A**.
    """)

    if st.button("Build the Relation"):
        rel_func = st.session_state.relation_function
        relation = build_relation_instance(
            AB,
            AB,
            rel_func
        )
        st.session_state.relation_instance = relation

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Possible Pairs in A × A",
                len(AB) * len(AB)
            )

        with col2:
            st.metric(
                "Relation Size (|R|)",
                len(relation)
            )

if (st.session_state.validated_AB is not None
    and st.session_state.relation_function is not None
    and st.session_state.validated_relation_code is not None
    and st.session_state.relation_instance is not None):

    st.subheader(
        "Step 4: View the Relation in Different Representations"
    )
    relation_instance = st.session_state.relation_instance
    st.markdown("""
    The four views below all represent the same relation.

    For a pair `(a, b)`, the following statements are equivalent:

    - `(a, b)` appears in the ordered-pair representation.
    - `(a, b)` appears in the relation table.
    - There is an arrow from `a` to `b` in the directed graph.
    - The matrix entry `M[a,b]` is `1`.
    """)

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.markdown("### Relation as Ordered Pairs")
        if len(relation_instance) == 0:
            st.write("∅")
        else:
            pairs_text = "\n".join(
                f"({a}, {b})"
                for a, b in relation_instance
            )

            st.text_area(
                "",
                value=pairs_text,
                height=250,
                disabled=True
            )

    with row1_col2:
        st.markdown("### Relation as a Table")
        df = relation_dataframe(
            relation_instance
        )
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True
        )

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("### Relation as a Directed Graph")
        st.caption(
            "There is an arrow from a to b whenever "
            "(a, b) belongs to the relation."
        )
        try:
            g = graphviz.Digraph(format="png")
            g.attr(rankdir="LR")

            g.attr(nodesep="0.3")
            g.attr(ranksep="0.4")
            g.attr(
                "node",
                shape="circle",
                fontsize="12"
            )

            for n in AB:
                g.node(str(n))

            # Invisible edges to encourage a compact layout
            for i in range(len(AB)-1):
                g.edge(
                    str(AB[i]),
                    str(AB[i+1]),
                    style="invis"
                )

            for u, v in relation_instance:
                g.edge(str(u), str(v))
            st.graphviz_chart(
                g,
                use_container_width=True
            )
        except Exception:
            st.error("Unable to display graph.")

    with row2_col2:
        st.markdown("### Relation as an Adjacency Matrix")

        st.caption(
            "Rows correspond to a and columns correspond to b. "
            "M[a,b] = 1 iff (a,b) belongs to the relation."
        )

        matrix_df = adjacency_matrix_dataframe(
            AB,
            relation_instance
        )

        st.dataframe(
            matrix_df,
            use_container_width=True
        )

# --------------------------------------------------
# Step 5: Explore Relation Properties
# --------------------------------------------------

if (
    st.session_state.relation_instance is not None
    and st.session_state.validated_AB is not None
    and st.session_state.relation_function is not None
):

    st.subheader("Step 5: Explore Relation Properties")

    st.markdown("""
    A relation can be analyzed using several important properties.
    """)

    with st.expander(
        "Review: Relation Properties",
        expanded=False
    ):
        st.markdown("""
    - Reflexive: For every x ∈ A, (x, x) ∈ R.
    - Anti-reflexive: For every x ∈ A, (x, x) ∉ R.
    - Symmetric: For every pair (x, y) ∈ R, (y, x) must also belong to R.
    - Antisymmetric: For every pair (x, y) ∈ R where x ≠ y, (y, x) must not belong to R.
    (Equivalently, if (x, y) ∈ R and (y, x) ∈ R, then x = y.)    
    - Transitive: If (x, y) ∈ R and (y, z) ∈ R, then (x, z) must also belong to R.
    """)
    
    props = relation_properties(
        AB,
        relation_instance
    )

    st.markdown("### Properties of the Current Relation")
    st.markdown(
        f"**Reflexive:** "
        f"{'✅' if props['Reflexive'] else '❌'}"
    )
    st.markdown(
        f"**Anti-reflexive:** "
        f"{'✅' if props['AntiReflexive'] else '❌'}"
    )
    st.markdown(
        f"**Symmetric:** "
        f"{'✅' if props['Symmetric'] else '❌'}"
    )
    st.markdown(
        f"**Anti-symmetric:** "
        f"{'✅' if props['Antisymmetric'] else '❌'}"
    )
    st.markdown(
        f"**Transitive:** "
        f"{'✅' if props['Transitive'] else '❌'}"
    )

    selected_property = st.radio(
        "Select a property to explore in detail",
        [
            "Reflexive",
            "Anti-reflexive",
            "Symmetric",
            "Antisymmetric",
            "Transitive"
        ]
    )

    if selected_property == "Reflexive":
        st.markdown("""
        ### Reflexive

        A relation is reflexive if:

        For every element `a ∈ A`, the pair `(a,a)`
        belongs to the relation.

        #### Graph Interpretation

        Every node has a self-loop.

        #### Matrix Interpretation

        For every element `a ∈ A`,
        the matrix entry `M[a,a]` is `1`.        
        """)

        if props["Reflexive"]:
            st.success(
                "✅ The relation is reflexive."
            )
        else:
            st.error(
                "❌ The relation is not reflexive."
            )

        counterexample = reflexive_counterexample(
            AB,
            relation_instance
        )

        if counterexample is not None:
            st.markdown(
                f"""
**Counterexample**

The pair **({counterexample}, {counterexample})**
does not belong to the relation.
"""
            )
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    "#### Graph Representation"
                )
                g = graphviz.Digraph(format="png")
                g.attr(rankdir="LR")
                # Add all nodes
                for n in AB:
                    if n == counterexample:
                        g.node(
                            str(n),
                            color="red",
                            penwidth="3"
                        )
                    else:
                        g.node(str(n))
                # Invisible edges for layout
                for i in range(len(AB) - 1):
                    g.edge(
                        str(AB[i]),
                        str(AB[i + 1]),
                        style="invis"
                    )
                # Relation edges
                for u, v in relation_instance:
                    g.edge(str(u), str(v))
                st.graphviz_chart(
                    g,
                    use_container_width=True
                )
                st.caption(
                    f"Node {counterexample} is highlighted "
                    "because it is missing a self-loop."
                )
            with col2:
                st.markdown(
                    "#### Matrix Representation"
                )
                matrix_df = adjacency_matrix_dataframe(
                    AB,
                    relation_instance
                )
                def highlight_diagonal(cell):
                    return (
                        "background-color: #ffdddd;"
                    )
                styled_matrix = (
                    matrix_df.style
                    .apply(
                        lambda row: [
                            (
                                "background-color: #ffdddd;"
                                if (
                                    row.name == counterexample
                                    and col == counterexample
                                )
                                else ""
                            )
                            for col in matrix_df.columns
                        ],
                        axis=1
                    )
                )
                st.dataframe(
                    styled_matrix,
                    use_container_width=True
                )
                st.caption(
                    f"M[{counterexample},{counterexample}] = 0, "
                    "so the relation is not reflexive."
                )
        else:
            st.info(
                "Every element a in A satisfies "
                "(a,a) ∈ R."
            )
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    "#### Graph Representation"
                )
                g = graphviz.Digraph(format="png")
                g.attr(rankdir="LR")

                for n in AB:
                    g.node(str(n))

                for i in range(len(AB) - 1):
                    g.edge(
                        str(AB[i]),
                        str(AB[i + 1]),
                        style="invis"
                    )

                for u, v in relation_instance:

                    if u == v:
                        g.edge(
                            str(u),
                            str(v),
                            color="green",
                            penwidth="3"
                        )
                    else:
                        g.edge(
                            str(u),
                            str(v)
                        )

                st.graphviz_chart(
                    g,
                    use_container_width=True
                )

                st.caption(
                    "All self-loops are present."
                )
            with col2:
                st.markdown(
                    "#### Matrix Representation"
                )

                matrix_df = adjacency_matrix_dataframe(
                    AB,
                    relation_instance
                )

                styled_matrix = (
                    matrix_df.style
                    .apply(
                        lambda row: [
                            (
                                "background-color: #ddffdd;"
                                if row.name == col
                                else ""
                            )
                            for col in matrix_df.columns
                        ],
                        axis=1
                    )
                )

                st.dataframe(
                    styled_matrix,
                    use_container_width=True
                )

                st.caption(
                    "For every element a ∈ A, M[a,a] = 1, "
                    "so the relation is reflexive."
                )

    if selected_property == "Anti-reflexive":

        st.markdown("""
        ### Anti-reflexive

        A relation is anti-reflexive if:

        For every element `a ∈ A`, the pair `(a,a)`
        does not belong to the relation.

        #### Graph Interpretation

        No node has a self-loop.

        #### Matrix Interpretation

        For every element `a ∈ A`,
        the matrix entry `M[a,a]` is `0`.
        """)

        if props["AntiReflexive"]:
            st.success(
                "✅ The relation is anti-reflexive."
            )
        else:
            st.error(
                "❌ The relation is not anti-reflexive."
            )

        counterexample = anti_reflexive_counterexample(
            AB,
            relation_instance
        )
        if counterexample is not None:

            st.markdown(
                f"""
    **Counterexample**

    The pair **({counterexample}, {counterexample})**
    belongs to the relation.
    """
            )

            col1, col2 = st.columns(2)
            with col1:

                st.markdown(
                    "#### Graph Representation"
                )

                g = graphviz.Digraph(format="png")

                g.attr(rankdir="LR")

                for n in AB:

                    if n == counterexample:

                        g.node(
                            str(n),
                            color="red",
                            penwidth="3"
                        )

                    else:

                        g.node(str(n))

                for i in range(len(AB) - 1):
                    g.edge(
                        str(AB[i]),
                        str(AB[i + 1]),
                        style="invis"
                    )

                for u, v in relation_instance:

                    if (
                        u == counterexample
                        and
                        v == counterexample
                    ):

                        g.edge(
                            str(u),
                            str(v),
                            color="red",
                            penwidth="3"
                        )

                    else:

                        g.edge(
                            str(u),
                            str(v)
                        )

                st.graphviz_chart(
                    g,
                    use_container_width=True
                )

                st.caption(
                    f"Node {counterexample} has a self-loop."
                )

            with col2:

                st.markdown(
                    "#### Matrix Representation"
                )

                matrix_df = adjacency_matrix_dataframe(
                    AB,
                    relation_instance
                )

                styled_matrix = (
                    matrix_df.style
                    .apply(
                        lambda row: [
                            (
                                "background-color: #ffdddd;"
                                if (
                                    row.name == counterexample
                                    and col == counterexample
                                )
                                else ""
                            )
                            for col in matrix_df.columns
                        ],
                        axis=1
                    )
                )

                st.dataframe(
                    styled_matrix,
                    use_container_width=True
                )

                st.caption(
                    f"M[{counterexample},{counterexample}] = 1, "
                    "so the relation is not anti-reflexive."
                )
        else:

            st.info(
                "✅ No element is related to itself."
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "#### Graph Representation"
                )

                g = graphviz.Digraph(format="png")

                g.attr(rankdir="LR")

                for n in AB:

                    g.node(
                        str(n),
                        color="green",
                        penwidth="3"
                    )

                for i in range(len(AB) - 1):
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

                st.caption(
                    "No node has a self-loop."
                )

            with col2:

                st.markdown(
                    "#### Matrix Representation"
                )

                matrix_df = adjacency_matrix_dataframe(
                    AB,
                    relation_instance
                )

                styled_matrix = (
                    matrix_df.style
                    .apply(
                        lambda row: [
                            (
                                "background-color: #ddffdd;"
                                if row.name == col
                                else ""
                            )
                            for col in matrix_df.columns
                        ],
                        axis=1
                    )
                )

                st.dataframe(
                    styled_matrix,
                    use_container_width=True
                )

                st.caption(
                    "For every element a ∈ A, M[a,a] = 0, "
                    "so the relation is anti-reflexive."
                )

    if selected_property == "Symmetric":
        st.markdown("""
        ### Symmetric

        A relation is symmetric if:

        Whenever `(a,b)` belongs to the relation,
        `(b,a)` also belongs to the relation.

        #### Graph Interpretation

        Every arrow from a to b is matched by
        an arrow from b to a.

        #### Matrix Interpretation

        Whenever M[a,b] = 1,
        M[b,a] must also equal 1.
        """)

        if props["Symmetric"]:
            st.success(
                "✅ The relation is symmetric."
            )
        else:
            st.error(
                "❌ The relation is not symmetric."
            )

        counterexample = symmetric_counterexample(
            relation_instance
        )

        if counterexample is not None:
            a, b = counterexample
            st.markdown(
                f"""
        **Counterexample**

        The pair **({a}, {b})** belongs to the relation,
        but **({b}, {a})** does not.
        """
            )
            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "#### Graph Representation"
                )

                g = graphviz.Digraph(format="png")

                g.attr(rankdir="LR")

                # Add all nodes
                for n in AB:
                    g.node(str(n))

                # Invisible edges for layout
                for i in range(len(AB) - 1):
                    g.edge(
                        str(AB[i]),
                        str(AB[i + 1]),
                        style="invis"
                    )

                # Relation edges
                for u, v in relation_instance:

                    if (u, v) == (a, b):
                        g.edge(
                            str(u),
                            str(v),
                            color="red",
                            penwidth="3"
                        )
                    else:
                        g.edge(
                            str(u),
                            str(v)
                        )

                # Missing reverse edge
                g.edge(
                    str(b),
                    str(a),
                    color="red",
                    style="dashed"
                )

                st.graphviz_chart(
                    g,
                    use_container_width=True
                )

                st.caption(
                    f"({a}, {b}) is present, but "
                    f"({b}, {a}) is missing."
                )

            with col2:

                st.markdown(
                    "#### Matrix Representation"
                )

                matrix_df = adjacency_matrix_dataframe(
                    AB,
                    relation_instance
                )

                styled_matrix = (
                    matrix_df.style
                    .apply(
                        lambda row: [
                            (
                                "background-color: #ffdddd;"
                                if (
                                    (row.name == a and col == b)
                                    or
                                    (row.name == b and col == a)
                                )
                                else ""
                            )
                            for col in matrix_df.columns
                        ],
                        axis=1
                    )
                )

                st.dataframe(
                    styled_matrix,
                    use_container_width=True
                )

                st.caption(
                    f"M[{a},{b}] = 1 but "
                    f"M[{b},{a}] = 0."
                )

        else:

            st.info(
                "Whenever (a,b) belongs to the relation, "
                "(b,a) also belongs to the relation."
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "#### Graph Representation"
                )

                g = graphviz.Digraph(format="png")

                g.attr(rankdir="LR")

                # Add all nodes
                for n in AB:
                    g.node(str(n))

                # Invisible edges for layout
                for i in range(len(AB) - 1):
                    g.edge(
                        str(AB[i]),
                        str(AB[i + 1]),
                        style="invis"
                    )

                R = set(relation_instance)

                for u, v in relation_instance:

                    if (v, u) in R:
                        g.edge(
                            str(u),
                            str(v),
                            color="green",
                            penwidth="3"
                        )
                    else:
                        g.edge(
                            str(u),
                            str(v)
                        )

                st.graphviz_chart(
                    g,
                    use_container_width=True
                )

                st.caption(
                    "Every edge has a corresponding "
                    "edge in the opposite direction."
                )

            with col2:

                st.markdown(
                    "#### Matrix Representation"
                )

                matrix_df = adjacency_matrix_dataframe(
                    AB,
                    relation_instance
                )

                R = set(relation_instance)

                styled_matrix = (
                    matrix_df.style
                    .apply(
                        lambda row: [
                            (
                                "background-color: #ddffdd;"
                                if (
                                    (row.name, col) in R
                                    and
                                    (col, row.name) in R
                                )
                                else ""
                            )
                            for col in matrix_df.columns
                        ],
                        axis=1
                    )
                )

                st.dataframe(
                    styled_matrix,
                    use_container_width=True
                )

                st.caption(
                    "Whenever M[a,b] = 1, "
                    "M[b,a] is also 1."
                )

    if selected_property == "Antisymmetric":
        st.markdown("""
        ### Antisymmetric

        A relation is antisymmetric if:

        For every pair `(a,b)` that belongs to the relation,
        where `a ≠ b`,

        `(b,a)` must not belong to the relation.

        #### Graph Interpretation

        Distinct nodes must not have arrows in both directions.

        #### Matrix Interpretation

        For distinct elements `a` and `b`,
        `M[a,b]` and `M[b,a]` cannot both be `1`.
        """)

        if props["Antisymmetric"]:
            st.success(
                "✅ The relation is antisymmetric."
            )
        else:
            st.error(
                "❌ The relation is not antisymmetric."
            )

        counterexample = antisymmetric_counterexample(
            relation_instance
        )

        if counterexample is not None:
            a, b = counterexample
            st.markdown(
                f"""
    **Counterexample**

    Both **({a}, {b})** and **({b}, {a})**
    belong to the relation, even though
    **{a} ≠ {b}**.
    """
            )
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    "#### Graph Representation"
                )
                g = graphviz.Digraph(format="png")
                g.attr(rankdir="LR")
                for n in AB:
                    g.node(str(n))

                for i in range(len(AB) - 1):
                    g.edge(
                        str(AB[i]),
                        str(AB[i + 1]),
                        style="invis"
                    )
                for u, v in relation_instance:
                    if (
                        (u, v) == (a, b)
                        or
                        (u, v) == (b, a)
                    ):
                        g.edge(
                            str(u),
                            str(v),
                            color="red",
                            penwidth="3"
                        )
                    else:
                        g.edge(
                            str(u),
                            str(v)
                        )
                st.graphviz_chart(
                    g,
                    use_container_width=True
                )
                st.caption(
                    f"Both ({a}, {b}) and ({b}, {a}) "
                    "are present."
                )

            with col2:

                st.markdown(
                    "#### Matrix Representation"
                )

                matrix_df = adjacency_matrix_dataframe(
                    AB,
                    relation_instance
                )

                styled_matrix = (
                    matrix_df.style
                    .apply(
                        lambda row: [
                            (
                                "background-color: #ffdddd;"
                                if (
                                    (
                                        row.name == a
                                        and col == b
                                    )
                                    or
                                    (
                                        row.name == b
                                        and col == a
                                    )
                                )
                                else ""
                            )
                            for col in matrix_df.columns
                        ],
                        axis=1
                    )
                )

                st.dataframe(
                    styled_matrix,
                    use_container_width=True
                )

                st.caption(
                    f"M[{a},{b}] = 1 and "
                    f"M[{b},{a}] = 1."
                )

        else:

            st.info(
                "No distinct pair of elements has arrows "
                "in both directions."
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "#### Graph Representation"
                )

                g = graphviz.Digraph(format="png")

                g.attr(rankdir="LR")

                for n in AB:
                    g.node(str(n))

                # Invisible edges for layout
                for i in range(len(AB) - 1):
                    g.edge(
                        str(AB[i]),
                        str(AB[i + 1]),
                        style="invis"
                    )

                for u, v in relation_instance:

                    if u != v:

                        g.edge(
                            str(u),
                            str(v),
                            color="green",
                            penwidth="3"
                        )

                    else:

                        g.edge(
                            str(u),
                            str(v)
                        )

                st.graphviz_chart(
                    g,
                    use_container_width=True
                )

                st.caption(
                    "No distinct pair of nodes has arrows "
                    "in both directions."
                )

            with col2:

                st.markdown(
                    "#### Matrix Representation"
                )

                matrix_df = adjacency_matrix_dataframe(
                    AB,
                    relation_instance
                )

                styled_matrix = (
                    matrix_df.style
                    .apply(
                        lambda row: [
                            (
                                "background-color: #ddffdd;"
                                if (
                                    row.name != col
                                    and
                                    matrix_df.loc[
                                        row.name,
                                        col
                                    ] == 1
                                )
                                else ""
                            )
                            for col in matrix_df.columns
                        ],
                        axis=1
                    )
                )

                st.dataframe(
                    styled_matrix,
                    use_container_width=True
                )

                st.caption(
                    "No distinct elements a and b have "
                    "both M[a,b] = 1 and M[b,a] = 1."
                )

    if selected_property == "Transitive":

        st.markdown("""
        ### Transitive

        A relation is transitive if:

        Whenever `(a,b)` and `(b,c)` belong to the relation,

        `(a,c)` must also belong to the relation.

        #### Graph Interpretation

        Whenever there is a path

        `a → b → c`

        there must also be a direct arrow

        `a → c`.

        #### Matrix Interpretation

        Whenever `M[a,b] = 1` and `M[b,c] = 1`,

        `M[a,c]` must also equal `1`.
        """)

        if props["Transitive"]:
            st.success(
                "✅ The relation is transitive."
            )
        else:
            st.error(
                "❌ The relation is not transitive."
            )

        counterexample = transitive_counterexample(
            relation_instance
        )

        if counterexample is not None:

            pair1, pair2, missing_pair = (
                counterexample
            )

            st.markdown(
                f"""
            **Counterexample**

            The pairs **{pair1}** and **{pair2}**
            belong to the relation.

            However, **{missing_pair}**
            does not belong to the relation.
            """
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "#### Graph Representation"
                )

                g = graphviz.Digraph(format="png")

                g.attr(rankdir="LR")

                # Add all nodes
                for n in AB:
                    g.node(str(n))

                # Invisible edges for layout
                for i in range(len(AB) - 1):
                    g.edge(
                        str(AB[i]),
                        str(AB[i + 1]),
                        style="invis"
                    )

                # Draw all relation edges normally
                for u, v in relation_instance:

                    g.edge(
                        str(u),
                        str(v)
                    )

                # Highlight the two edges involved
                # in the transitivity violation
                g.edge(
                    str(pair1[0]),
                    str(pair1[1]),
                    color="red",
                    penwidth="3"
                )

                g.edge(
                    str(pair2[0]),
                    str(pair2[1]),
                    color="red",
                    penwidth="3"
                )

                # Show the missing edge
                g.edge(
                    str(missing_pair[0]),
                    str(missing_pair[1]),
                    color="red",
                    style="dashed",
                    penwidth="3"
                )

                st.graphviz_chart(
                    g,
                    use_container_width=True
                )

                st.caption(
                    f"{pair1} and {pair2} are present, "
                    f"but {missing_pair} is missing."
                )

            with col2:

                st.markdown(
                    "#### Matrix Representation"
                )

                matrix_df = adjacency_matrix_dataframe(
                    AB,
                    relation_instance
                )

                styled_matrix = (
                    matrix_df.style
                    .apply(
                        lambda row: [
                            (
                                "background-color: #ffdddd;"
                                if (
                                    (
                                        row.name == pair1[0]
                                        and col == pair1[1]
                                    )
                                    or
                                    (
                                        row.name == pair2[0]
                                        and col == pair2[1]
                                    )
                                    or
                                    (
                                        row.name == missing_pair[0]
                                        and col == missing_pair[1]
                                    )
                                )
                                else ""
                            )
                            for col in matrix_df.columns
                        ],
                        axis=1
                    )
                )

                st.dataframe(
                    styled_matrix,
                    use_container_width=True
                )

                st.caption(
                    f"M[{pair1[0]},{pair1[1]}] = 1 and "
                    f"M[{pair2[0]},{pair2[1]}] = 1, "
                    f"but M[{missing_pair[0]},{missing_pair[1]}] = 0."
                )

        else:

            witness = transitive_witness(
                relation_instance
            )

            if witness is not None:

                pair1, pair2, implied_pair = witness

                st.markdown(
                    f"""
                **Illustrative Example**

                The pairs **{pair1}** and **{pair2}**
                belong to the relation.

                Since **{implied_pair}** also belongs
                to the relation, this example satisfies
                transitivity.

                This example is shown for illustration
                purposes only.

                Determining whether a relation is
                transitive requires checking **every**
                pair `(a,b)` in `R` and `(b,c)` in `R`.
                """
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown(
                        "#### Graph Representation of the Example"
                    )

                    g = graphviz.Digraph(format="png")

                    g.attr(rankdir="LR")

                    for n in AB:
                        g.node(str(n))

                    for i in range(len(AB) - 1):
                        g.edge(
                            str(AB[i]),
                            str(AB[i + 1]),
                            style="invis"
                        )

                    highlight_edges = {
                        pair1,
                        pair2,
                        implied_pair
                    }

                    for u, v in relation_instance:
                        if (u, v) in highlight_edges:
                            g.edge(
                                str(u),
                                str(v),
                                color="green",
                                penwidth="3"
                            )
                        else:
                            g.edge(
                                str(u),
                                str(v)
                            )

                    st.graphviz_chart(
                        g,
                        use_container_width=True
                    )

                    st.caption(
                        f"{pair1} and {pair2} imply "
                        f"{implied_pair}, which is present."
                    )

                with col2:

                    st.markdown(
                        "#### Matrix Representation of the Example"
                    )

                    matrix_df = adjacency_matrix_dataframe(
                        AB,
                        relation_instance
                    )

                    styled_matrix = (
                        matrix_df.style
                        .apply(
                            lambda row: [
                                (
                                    "background-color: #ddffdd;"
                                    if (
                                        (
                                            row.name == pair1[0]
                                            and col == pair1[1]
                                        )
                                        or
                                        (
                                            row.name == pair2[0]
                                            and col == pair2[1]
                                        )
                                        or
                                        (
                                            row.name == implied_pair[0]
                                            and col == implied_pair[1]
                                        )
                                    )
                                    else ""
                                )
                                for col in matrix_df.columns
                            ],
                            axis=1
                        )
                    )

                    st.dataframe(
                        styled_matrix,
                        use_container_width=True
                    )

                    st.caption(
                        f"M[{pair1[0]},{pair1[1]}] = 1, "
                        f"M[{pair2[0]},{pair2[1]}] = 1, "
                        f"and M[{implied_pair[0]},{implied_pair[1]}] = 1."
                    )

            else:

                st.markdown("""
                **No Example Available**

                The relation is transitive, but it does not
                contain a nontrivial example that can be used
                to illustrate the property.
                """)


