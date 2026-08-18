import streamlit as st
import graphviz
import pandas as pd
from Chapter_6_utils import *

RECOMMENDED_SET_SIZE = 10
MAX_SET_SIZE = 20

st.title(
    "Module 5: Equivalence Relations and Equivalence Classes"
)

st.markdown("""
In Module 3, we studied the properties of
binary relations, including reflexivity,
symmetry, and transitivity.

In this module, we explore **equivalence
relations**, which satisfy all three of
these properties simultaneously.

We then study **equivalence classes** and see how
the equivalence classes of an equivalence relation
partition a set into groups of related elements.
""")

st.header(
    "Equivalence Relations"
)

st.markdown("""
An **equivalence relation** is a relation
that is:

- Reflexive
- Symmetric
- Transitive

Examples include:

- Equality
- Same Parity
- Congruence Modulo n

Equivalence relations help us group
elements that should be considered
equivalent in some meaningful way.
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
            index=relation_names.index("Same Parity")
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

    st.info("""
    The graph and adjacency matrix provide two
    different views of the same relation.

    In the next step, we will determine whether
    the relation is an equivalence relation and,
    if so, explore its equivalence classes.
    """)

if (st.session_state.validated_AB is not None
    and st.session_state.relation_function is not None
    and st.session_state.validated_relation_code is not None
    and st.session_state.relation_instance is not None):

    st.subheader(
        "Step 3: Is This an Equivalence Relation?"
    )

    st.markdown("""
An equivalence relation is a relation that is:

- Reflexive
- Symmetric
- Transitive

We now determine whether the relation satisfies
all three properties.
""")

    relation = (
        st.session_state.relation_instance
    )

    props = relation_properties(
        AB,
        relation
    )

    reflexive = props["Reflexive"]
    symmetric = props["Symmetric"]
    transitive = props["Transitive"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Reflexive",
            "✅" if reflexive else "❌"
        )

    with col2:
        st.metric(
            "Symmetric",
            "✅" if symmetric else "❌"
        )

    with col3:
        st.metric(
            "Transitive",
            "✅" if transitive else "❌"
        )

    if (
        reflexive
        and symmetric
        and transitive
    ):

        st.success("""
The relation is an equivalence relation because it is reflexive, symmetric, and transitive.
""")

    else:

        st.error("""
The relation is not an equivalence relation because it does not satisfy all three required properties.
""")

if (st.session_state.validated_AB is not None
    and st.session_state.relation_function is not None
    and st.session_state.validated_relation_code is not None
    and st.session_state.relation_instance is not None):
    if reflexive and symmetric and transitive:
        st.subheader(
            "Step 4: Explore Equivalence Classes"
        )
        st.markdown("""
An equivalence class contains all elements that
are related to a given element.

For an element **a**, the equivalence class of a
is denoted by:

[a]

and consists of all elements related to a.
""")

        chosen_element = st.selectbox(
            "Choose an Element",
            AB,
            key="equivalence_class_element"
        )

        equivalence_class = sorted(
            [
                b
                for b in AB
                if (chosen_element, b)
                   in relation
            ]
        )

        class_text = (
            "{"
            + ", ".join(
                map(str, equivalence_class)
            )
            + "}"
        )

        st.markdown(
            f"""
### Equivalence Class

**[{chosen_element}] = {class_text}**
"""
        )

        st.info(
            f"""
The equivalence class [{chosen_element}]
contains all elements related to
{chosen_element}.
"""
        )

        st.markdown(
            "### Equivalence Class in the Graph"
        )

        st.caption(
            "Vertices belonging to the selected "
            "equivalence class are highlighted "
            "in green."
        )

        st.info("""
        For a vertex a:

        • There is an edge from a to every vertex in its
        equivalence class (including a itself).

        • There is no edge from a to any vertex outside
        its equivalence class.

        Thus, each equivalence class forms its own
        connected component in the graph.
        """)

        col1, col2 = st.columns(2)

        with col1:

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

            class_set = set(
                equivalence_class
            )

            for n in AB:

                if n in class_set:

                    g.node(
                        str(n),
                        style="filled",
                        fillcolor="lightgreen"
                    )

                else:

                    g.node(
                        str(n)
                    )

            # Encourage a compact layout
            for i in range(
                len(AB) - 1
            ):
                g.edge(
                    str(AB[i]),
                    str(AB[i + 1]),
                    style="invis"
                )

            for u, v in relation:

                if (
                    u in class_set
                    and
                    v in class_set
                ):

                    g.edge(
                        str(u),
                        str(v),
                        color="darkgreen",
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
            pass

if (st.session_state.validated_AB is not None
    and st.session_state.relation_function is not None
    and st.session_state.validated_relation_code is not None
    and st.session_state.relation_instance is not None):
    if reflexive and symmetric and transitive:
        st.subheader(
            "Step 5: Partitioning the Set"
        )

        st.markdown("""
        An important theorem states:

        **The equivalence classes of an equivalence
        relation form a partition of the set.**

        This means that **every element belongs to
        exactly one equivalence class.**

        In other words:

        - Different equivalence classes do not overlap.
        - Together, the equivalence classes contain
        every element of the set.
        """)

        remaining = set(AB)
        partition = []
        relation_set = set(relation)
        while remaining:
            a = next(iter(remaining))
            eq_class = {
                b
                for b in AB
                if (a, b) in relation_set
            }
            partition.append(
                sorted(eq_class)
            )
            remaining -= eq_class

        st.markdown(
            "### Equivalence Classes"
        )

        for i, eq_class in enumerate(
            partition,
            start=1
        ):
            class_text = (
                "{"
                + ", ".join(
                    map(str, eq_class)
                )
                + "}"
            )
            representative = eq_class[0]

            st.markdown(
                f"Equivalence Class [{representative}] = {class_text}"
            )
        st.markdown(
            "### Partition of A"
        )

        partition_text = (
            "{ "
            + ", ".join(
                "{" + ", ".join(
                    map(str, cls)
                ) + "}"
                for cls in partition
            )
            + " }"
        )

        st.code(partition_text)

        st.success("""
    The equivalence classes form a partition of A.

    Every element belongs to exactly one
    equivalence class.
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

        Different equivalence classes may overlap.

        ---

        **2. True or False**

        Equality is an equivalence relation.

        ---

        **3. True or False**

        The Same Parity relation partitions the integers
        into two equivalence classes.

        ---

        **4. Challenge Question**

        If an equivalence relation on a set has only
        one equivalence class, what does its graph
        look like?
        """)

            with st.expander(
                "Show Answers"
            ):

                st.markdown("""
            **1. False**

            The equivalence classes of an equivalence
            relation form a partition of the set.

            Every element belongs to exactly one
            equivalence class, and different equivalence
            classes do not overlap.

            ---

            **2. True**

            Equality is an equivalence relation.

            - Every element is equal to itself (reflexive).
            - If a = b, then b = a (symmetric).
            - If a = b and b = c, then a = c (transitive).

            ---

            **3. True**

            Same Parity is an equivalence relation.

            - Every integer has the same parity as itself.
            - If a has the same parity as b, then b has
                the same parity as a.
            - If a has the same parity as b and b has
                the same parity as c, then a has the same
                parity as c.

            Its equivalence classes are:

            [Even] = {all even integers}

            [Odd] = {all odd integers}

            Every integer belongs to exactly one of
            these two equivalence classes.

            ---

            **4. Challenge Question**

            If an equivalence relation has only one
            equivalence class, then every element is
            related to every other element.

            Therefore, for every pair of elements
            (a, b), the pair (a, b) belongs to R.

            The graph contains an edge from every
            vertex to every vertex, including
            self-loops.
            """)

        st.success("""
        Summary

        • An equivalence relation is reflexive,
        symmetric, and transitive.

        • An equivalence class contains all elements
        related to a given element.

        • The equivalence classes of an equivalence
        relation form a partition of the set.

        • Every element belongs to exactly one
        equivalence class.
        """)

        with st.expander(
            "Applications of Equivalence Relations"
        ):
            st.markdown("""
        These ideas appear whenever we group objects
        that should be treated as equivalent.

        ### Same Remainder Modulo n

        Two integers are considered equivalent if they
        leave the same remainder when divided by n.

        ### Natural Language Processing

        Different phrases such as:

        - USA
        - United States
        - America
        - United States of America

        may be treated as equivalent because they refer
        to the same real-world entity.

        ### Classification

        Objects may be grouped into categories based on
        shared properties. Each category forms an
        equivalence class.

        ### Finite State Automata

        States that behave identically can be placed in
        the same equivalence class and combined into a
        single state.

        ### Data Deduplication

        Multiple records that refer to the same real-world
        entity may be grouped together and treated as
        equivalent.
        """)
