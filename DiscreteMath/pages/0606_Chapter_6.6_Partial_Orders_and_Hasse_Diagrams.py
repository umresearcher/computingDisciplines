import streamlit as st
import graphviz
import pandas as pd
from Chapter_6_utils import *

RECOMMENDED_SET_SIZE = 10
MAX_SET_SIZE = 20

st.title(
    "Module 6: Partial Orders and Hasse Diagrams"
)

st.markdown("""
In Module 5, we studied equivalence relations,
which are reflexive, symmetric, and transitive.

In this module, we study partial orders,
which are reflexive, antisymmetric, and
transitive.

Partial orders allow us to compare elements
and describe precedence relationships.
""")

st.header(
    "Partially Ordered Sets"
)

st.markdown("""
A relation R on a set A is a **partial order**
if it is:

- Reflexive
- Antisymmetric
- Transitive

A set together with a partial order is called a
**partially ordered set** (or **poset**).
""")

col1, col2 = st.columns(2)

with col1:
    st.info("""
Equivalence Relations

• Reflexive

• Symmetric

• Transitive
""")

with col2:
    st.info("""
Partial Orders

• Reflexive

• Antisymmetric

• Transitive
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
            index=relation_names.index("Less Than or Equal")
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
    and st.session_state.relation_function is not None
    and st.session_state.validated_relation_code is not None
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
    and st.session_state.relation_function is not None
    and st.session_state.validated_relation_code is not None
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

if (st.session_state.validated_AB is not None
    and st.session_state.relation_function is not None
    and st.session_state.validated_relation_code is not None
    and st.session_state.relation_instance is not None):

    st.subheader(
        "Step 3: Is This a Partial Order?"
    )

    relation = (
        st.session_state.relation_instance
    )

    props = relation_properties(
        AB,
        relation
    )

    reflexive = props["Reflexive"]

    antisymmetric = props["Antisymmetric"]

    transitive = props["Transitive"]

    st.markdown("""
    A partial order is a relation that is:

    - Reflexive
    - Antisymmetric
    - Transitive
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Reflexive",
            "✅" if reflexive else "❌"
        )

    with col2:
        st.metric(
            "Antisymmetric",
            "✅" if antisymmetric else "❌"
        )

    with col3:
        st.metric(
            "Transitive",
            "✅" if transitive else "❌"
        )

    if (
        reflexive
        and antisymmetric
        and transitive
    ):

        st.success(
            "The relation is a partial order."
        )

    else:

        st.error(
            "The relation is not a partial order."
        )

if (st.session_state.validated_AB is not None
    and st.session_state.relation_function is not None
    and st.session_state.validated_relation_code is not None
    and st.session_state.relation_instance is not None):
    st.subheader(
        "Step 4: Partial Orders and Total Orders"
    )

    st.markdown("""
    Two elements a and b are **comparable** if:

    - aRb, or
    - bRa

    Otherwise, they are **incomparable**.

    A partial order is a **total order** if every
    pair of distinct elements is comparable.
    """)

    if reflexive and antisymmetric and transitive:
        is_total_order = True
        for a in AB:
            for b in AB:

                if a != b:

                    if (
                        (a, b) not in relation
                        and
                        (b, a) not in relation
                    ):

                        is_total_order = False
                        break

            if not is_total_order:
                break

        if is_total_order:

            st.success("""
    The relation is a total order.

    Every pair of distinct elements is comparable.
    """)

        else:

            st.info("""
    The relation is a partial order, but not a
    total order.

    Some pairs of elements are incomparable.
    """)

        if not is_total_order:

            incomparable_pairs = []

            for i in range(len(AB)):
                for j in range(i + 1, len(AB)):

                    a = AB[i]
                    b = AB[j]

                    if (
                        (a, b) not in relation
                        and
                        (b, a) not in relation
                    ):
                        incomparable_pairs.append(
                            (a, b)
                        )

            st.markdown(
                "### Incomparable Pairs"
            )

            for a, b in incomparable_pairs:
                st.markdown(
                    f"""
            - ({a},{b}) and ({b},{a}) are not in R.
            """
                )

if (st.session_state.validated_AB is not None
    and st.session_state.relation_function is not None
    and st.session_state.validated_relation_code is not None
    and st.session_state.relation_instance is not None):
    st.subheader(
        "Step 5: Hasse Diagrams"
    )

    st.markdown("""
    A Hasse diagram is a simplified representation
    of a partial order.

    To construct a Hasse diagram:

    - Remove all self-loops.
    - Remove edges implied by transitivity.
    - Position vertices so that if aRb,
    then a appears lower in the diagram
    than b.
    
    In other words, include an edge from a to b only if:

    - a ≠ b,
    - aRb, and
        - there is no element c such that aRc and cRb.
    """)

    if reflexive and antisymmetric and transitive:
        st.markdown(
            "### Hasse Diagram"
        )

        st.caption(
            "Compare the directed graph (without self-loops) "
            "with the corresponding Hasse diagram."
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "#### Directed Graph (Without Self-Loops)"
            )

            st.caption(
                """
        Black edges remain in the Hasse diagram.

        Red dashed edges are removed because they
        are implied by transitivity.
        """
            )

            g = graphviz.Digraph(
                format="png"
            )

            g.attr(
                rankdir="LR"
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

            R = set(relation)

            hasse_edges = set()

            for a, b in R:

                # Remove self-loops
                if a == b:
                    continue

                implied = False

                for c in AB:

                    if c != a and c != b:

                        if (
                            (a, c) in R
                            and
                            (c, b) in R
                        ):
                            implied = True
                            break

                if not implied:

                    hasse_edges.add(
                        (a, b)
                    )

            for a, b in R:

                # Do not show self-loops
                if a == b:
                    continue

                if (a, b) in hasse_edges:

                    g.edge(
                        str(a),
                        str(b),
                        color="black",
                        penwidth="2"
                    )

                else:

                    g.edge(
                        str(a),
                        str(b),
                        color="red",
                        style="dashed"
                    )

            st.graphviz_chart(
                g,
                use_container_width=True
            )

        with col2:

            st.markdown(
                "#### Hasse Diagram"
            )

            hasse = graphviz.Digraph(
                format="png"
            )

            # Bottom-to-top
            hasse.attr(
                rankdir="BT"
            )

            # Compact layout
            hasse.attr(
                ranksep="0.25"
            )

            hasse.attr(
                nodesep="0.15"
            )

            hasse.attr(
                "node",
                shape="circle",
                width="0.5",
                height="0.5",
                fixedsize="true",
                fontsize="12"
            )

            for n in AB:
                hasse.node(str(n))

            R = set(relation)

            hasse_edges = []

            for a, b in R:

                # Remove self-loops
                if a == b:
                    continue

                # Is (a,b) implied by transitivity?
                implied = False

                for c in AB:

                    if c != a and c != b:

                        if (
                            (a, c) in R
                            and
                            (c, b) in R
                        ):
                            implied = True
                            break

                if not implied:

                    hasse_edges.append(
                        (a, b)
                    )

            for a, b in hasse_edges:

                hasse.edge(
                    str(a),
                    str(b),
                    dir="none",
                    penwidth="2"
                )

            st.graphviz_chart(
                hasse
            )

        st.info("""
        The Hasse diagram removes self-loops and
        transitive edges.

        Only the essential ordering relationships
        are shown.

        If a appears below b and the vertices are
        connected by an edge, then aRb belongs to
        the relation.

        The direction is determined by the vertical
        position of the vertices, so arrowheads are
        not needed.
        """)

    else:
        st.info(
            "A Hasse diagram is defined only for partial orders."
        )

if (st.session_state.validated_AB is not None
    and st.session_state.relation_function is not None
    and st.session_state.validated_relation_code is not None
    and st.session_state.relation_instance is not None):
    st.subheader(
        "Step 6: Minimal and Maximal Elements"
    )

    st.markdown("""
    An element x is **minimal** if there is no
    distinct element y such that yRx.

    An element x is **maximal** if there is no
    distinct element y such that xRy.
    """)

    minimal_elements = []

    for x in AB:
        is_minimal = True
        for y in AB:
            if y != x and (y, x) in relation:
                is_minimal = False
                break

        if is_minimal:
            minimal_elements.append(x)

    maximal_elements = []

    for x in AB:
        is_maximal = True
        for y in AB:

            if y != x and (x, y) in relation:

                is_maximal = False
                break

        if is_maximal:
            maximal_elements.append(x)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### Minimal Elements"
        )

        st.code(
            "{"
            + ", ".join(map(str, minimal_elements))
            + "}"
        )

    with col2:

        st.markdown(
            "### Maximal Elements"
        )

        st.code(
            "{"
            + ", ".join(map(str, maximal_elements))
            + "}"
        )

    if reflexive and antisymmetric and transitive:

        st.info("""
        A partial order on a finite set always has
        at least one minimal element and at least
        one maximal element.

        If the partial order is also a total order,
        then it has exactly one minimal element and
        exactly one maximal element.
        """)

        st.info("""
        In a Hasse diagram:

        • Minimal elements appear at the bottom.

        • Maximal elements appear at the top.
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

    Every partial order is reflexive.

    ---

    **2. True or False**

    Every partial order is symmetric.

    ---

    **3. True or False**

    Every total order is also a partial order.

    ---

    **4. True or False**

    Every pair of distinct elements is comparable
    in a total order.

    ---

    **5. True or False**

    A partial order may have multiple minimal
    elements.

    ---

    **6. True or False**

    A finite partial order always has at least
    one minimal element and at least one maximal
    element.

    ---

    **7. True or False**

    A Hasse diagram includes all edges of the
    corresponding directed graph.

    ---

    **8. Challenge Question**

    Consider the relation "Evenly Divides" on

    {1, 2, 3, 6}.

    Which elements are minimal?
    Which elements are maximal?
    """)

        with st.expander(
            "Show Answers"
        ):

            st.markdown("""
        **1. True**

        A partial order must be:

        - Reflexive
        - Antisymmetric
        - Transitive

        ---

        **2. False**

        A partial order is antisymmetric, not symmetric.

        For example, ≤ is a partial order but is not
        symmetric.

        ---

        **3. True**

        Every total order is a partial order in which
        every pair of distinct elements is comparable.

        ---

        **4. True**

        This is the defining property of a total order.

        Every pair of distinct elements must be
        comparable.

        ---

        **5. True**

        A partial order may have multiple minimal
        elements.

        For example, under the "Evenly Divides"
        relation on {2,3,6}, both 2 and 3 are minimal.

        ---

        **6. True**

        A partial order on a finite set always has
        at least one minimal element and at least
        one maximal element.
        """)

            with st.expander(
                "Why does a finite partial order always have a minimal and maximal element?"
            ):
                st.markdown("""
            Suppose a finite partial order had no minimal
            element.

            Then every element would have another distinct
            element below it.

            Following these elements downward would produce
            an infinite chain of distinct elements, which is
            impossible because the set is finite.

            Therefore, a finite partial order must contain
            at least one minimal element.

            A similar argument shows that it must also contain
            at least one maximal element.
            """)        

            st.markdown("""
        ---

        **7. False**

        A Hasse diagram removes:

        - self-loops
        - edges implied by transitivity

        Only the essential ordering relationships
        are shown.

        ---

        **8. Challenge Question**

        For the relation "Evenly Divides" on

        {1, 2, 3, 6}

        Minimal elements:

        {1}

        Maximal elements:

        {6}

        Since 1 evenly divides every element,
        nothing is below 1.

        Since no element is above 6,
        it is maximal.
        """)

            st.success("""
            Summary

            • A partial order is reflexive,
            antisymmetric, and transitive.

            • A total order is a partial order in
            which every pair of distinct elements
            is comparable.

            • A Hasse diagram removes self-loops and
            transitive edges.

            • In a Hasse diagram, minimal elements
            appear at the bottom and maximal
            elements appear at the top.
            """)

if (st.session_state.validated_AB is not None
    and st.session_state.relation_function is not None
    and st.session_state.validated_relation_code is not None
    and st.session_state.relation_instance is not None):
    with st.expander(
        "Additional Material: Strict Partial and Strict Total Orders"
    ):

        st.markdown("""
    The textbook defines a **partial order** as a
    relation that is:

    - Reflexive
    - Antisymmetric
    - Transitive

    A **strict partial order** is a relation that is:

    - Anti-reflexive
    - Transitive

    Examples:

    - ≤ is a partial order.
    - < is a strict partial order.

    A **total order** is a partial order in which
    every pair of distinct elements is comparable.

    Similarly, a **strict total order** is a strict
    partial order in which every pair of distinct
    elements is comparable.

    Examples:

    - ≤ is a total order on the integers.
    - < is a strict total order on the integers.

    Many dependency relationships in computing,
    such as course prerequisites and task scheduling,
    are naturally modeled using strict partial orders
    because self-loops are not meaningful (an item
    cannot depend on itself).
    """)

if (st.session_state.validated_AB is not None
    and st.session_state.relation_function is not None
    and st.session_state.validated_relation_code is not None
    and st.session_state.relation_instance is not None):
    with st.expander(
        "Applications of Partial Orders"
    ):
        st.markdown("""
        ### Course Prerequisites

        A course prerequisite relation is a strict
        partial order.

        For example:

        MTH 230 → CSC 379

        means that MTH 230 must be completed before
        CSC 379.

        ### Software Dependencies

        If package A depends on package B (often
        shown as B → A), then package B must be
        installed first.

        ### Task Scheduling

        Some tasks must be completed before others.

        For example:

        Design → Implementation → Testing

        defines a dependency relationship.

        ### Topological Sorting

        Many dependency relationships in computing
        are partial orders (or strict partial orders).

        A practical requirement is to find a valid
        total ordering that satisfies all
        dependencies in the given partial order.

        For example:

        - In what order should courses be taken?
        - In what order should tasks be scheduled?
        - In what order should software components
        be built?

        The problem of finding such a valid total
        ordering is called **topological sorting**.

        A topological sort produces a total ordering
        in which every dependency in the given
        partial order is respected.
        """)


