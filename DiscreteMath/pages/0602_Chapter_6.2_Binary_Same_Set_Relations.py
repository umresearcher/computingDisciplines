import streamlit as st
from Chapter_6_utils import *

#to keep display reasonable
MAX_DISPLAY_ROWS = 100

# --------------------------------------------------
# Module 2: Binary Relations on the Same Set
# --------------------------------------------------

st.title("Module 2: Binary Relations on the Same Set")

st.markdown("""
In Module 1, we studied binary relations between two sets **A** and **B**.

In this module, we focus on binary relations where:

**A = B**

These relations allow us to study important properties such as
**reflexivity**, **anti-reflexivity**, **symmetry**, **anti-symmetry**, and **transitivity**.
""")

st.info(
    "For this module, Set B must be the same as Set A."
)

#Initialize to None (if not already defined)
if "validated_A" not in st.session_state:
    st.session_state.validated_A = None

if "validated_B" not in st.session_state:
    st.session_state.validated_B = None

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

if "set_a" not in st.session_state:
    st.session_state.set_a = "1,2,3,4"

set_a_text = st.text_input(
    "Enter elements separated by commas",
    key="set_a"
)

def clear_set():
    st.session_state.set_a = ""
    st.session_state.validated_A = None
    st.session_state.validated_B = None
    st.session_state.last_relation_mode = None
    st.session_state.last_predefined_relation = None
    st.session_state.validated_relation_code = None
    st.session_state.relation_function = None
    st.session_state.relation_instance = None

col1, col2 = st.columns(2)

with col1:
    validate_a = st.button("Validate Set")

with col2:
    st.button(
        "Clear Set",
        on_click=clear_set
    )

if validate_a:
    try:
        A = parse_set(set_a_text)
        st.session_state.validated_A = A
        st.session_state.validated_B = A
        # Clear everything downstream
        st.session_state.validated_relation_code = None
        st.session_state.relation_function = None
        st.session_state.relation_instance = None       
        st.success(
            f"Set is valid ({len(A)} elements)"
        )
    except Exception as e:
        st.error(str(e))

if st.session_state.validated_A is not None:
    A = st.session_state.validated_A
    st.markdown(
        f"**A = B =** "
        f"{{{', '.join(map(str, A))}}}"
    )
    st.markdown(
        f"**|A| = |B| = {len(A)}**"
    )
    #Assign  Set B to be the same as Set A
    B = st.session_state.validated_B
    st.caption("Set B is automatically assigned to Set A.")

# --------------------------------------------------
# Step 2: Define the Relation
# --------------------------------------------------

if st.session_state.validated_A is None:
    st.info(
        "Complete Step 1 by validating the set."
    )
else:
    st.subheader("Step 2: Define the Relation")
    st.markdown("""
    We now define a binary relation on the same set.

    Since **A = B**, the relation will be a subset of:

    **A × A**

    Choose a predefined relation or define your own relation
    using the function:

    ```python
    relationDef(a, b)
    ```

    For each pair (a, b) in A × A, the function is evaluated.

    The function should return:

    - `True` if `(a, b)` belongs to the relation.
    - `False` if `(a, b)` does not belong to the relation.

    In this module, we will use the resulting relation to study important properties such as:

    - Reflexivity
    - Symmetry
    - Antisymmetry
    - Transitivity
    """)

    with st.expander("Allowed Python features"): 
        st.markdown("""
    If you define your own relation, you may use:

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
                A,
                A
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
# Step 3: Preview Relation Behavior
# --------------------------------------------------

if (
    st.session_state.validated_A is not None
    and st.session_state.relation_function is not None
):
    A = st.session_state.validated_A
    B = A
    rel_func = st.session_state.relation_function

    st.subheader("Step 3: Preview Relation Behavior")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"**Set A** = "
            f"{{{', '.join(map(str, A))}}}"
        )
    with col2:
        st.markdown(
            "**Set B = Set A**"
        )
    st.caption(
        f"Since A = B, the relation is a subset of "
        f"A × A. There are {len(A) * len(A)} possible pairs."
    )
    preview_rows = st.selectbox(
        "Number of sample pairs",
        [5, 10, 20, 50],
        index=1
    )
    preview_df = preview_relation(
        A,
        B,
        rel_func,
        sample_size=preview_rows
    )
    st.dataframe(
        preview_df,
        hide_index=True,
        width="stretch"
    )
    st.info(
        "The relation function is evaluated on sample "
        "pairs from A × A. True means the pair belongs "
        "to the relation."
    )

# --------------------------------------------------
# Step 4: Build the Relation
# --------------------------------------------------

if (
    st.session_state.validated_A is not None and
    st.session_state.validated_relation_code is not None
    and st.session_state.relation_function is not None
):
    st.subheader("Step 4: Build the Relation")
    st.markdown("""
The relation instance is obtained by evaluating every pair in **A × A**.

All pairs for which `relationDef(a, b)` returns `True`
are included in the relation.
""")
    if st.button("Build the Relation"):
        relation = build_relation_instance(
            A,
            B,
            rel_func
        )
        st.session_state.relation_instance = relation

#Display relation instance
if (
    st.session_state.validated_A is not None and
    st.session_state.validated_relation_code is not None
    and st.session_state.relation_function is not None
    and st.session_state.relation_instance is not None
):
    relation = st.session_state.relation_instance
    #st.subheader("Step 5: View the Relation")
    st.metric(
        "Relation Size (|R|)",
        len(relation)
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "### Relation as a Set of Ordered Pairs"
        )
        if len(relation) == 0:
            relation_text = "∅"
        else:
            relation_text = (
                "{"
                + ", ".join(
                    f"({a}, {b})"
                    for a, b in relation
                )
                + "}"
            )

        st.code(relation_text)
    with col2:
        st.markdown(
            "### Relation as a Table"
        )
        df = relation_dataframe(
            relation
        )
        st.dataframe(
            df,
            hide_index=True,
            width="stretch"
        )

# --------------------------------------------------
# Step 5: Investigate Relation Properties
# --------------------------------------------------

if (
    st.session_state.validated_A is not None and
    st.session_state.validated_relation_code is not None
    and st.session_state.relation_function is not None
    and st.session_state.relation_instance is not None
):
    st.subheader("Step 5: Investigate Relation Properties")
    with st.expander(
        "Property Definitions",
        expanded=False
    ):
        st.markdown("""
    ### Reflexive
    For every x ∈ A, (x, x) ∈ R.

    ### Anti-reflexive
    For every x ∈ A, (x, x) ∉ R.

    ### Symmetric
    For every pair (x, y) ∈ R,
    (y, x) must also belong to R.

    ### Anti-symmetric
    For every pair (x, y) ∈ R where x ≠ y,
    (y, x) must not belong to R.

    Equivalently, if (x, y) ∈ R and
    (y, x) ∈ R, then x = y.    
    
    ### Transitive
    If (x, y) ∈ R and (y, z) ∈ R,
    then (x, z) must also belong to R.
    """)

    st.info(
        "These properties are evaluated for the current "
        "relation and the current set A. The same relation "
        "may have different properties on different sets. "
        "For example, ≤ is symmetric on {1}, but not on "
        "{1,2}."
    )    
    A = st.session_state.validated_A
    relation = (
        st.session_state.relation_instance
    )
    props = relation_properties(
        A,
        relation
    )
    st.markdown(
        f"**Current Set:** A = B = "
        f"{{{', '.join(map(str, A))}}}"
    )
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

    chosen_property = st.selectbox(
        "Select a Property",
        [
            "Reflexive",
            "Anti-reflexive",
            "Symmetric",
            "Anti-symmetric",
            "Transitive"
        ]
    )

    if chosen_property == "Reflexive":
        st.markdown("""
    ### Reflexive

    For every x ∈ A, (x, x) ∈ R.
    """)

        R = set(relation)
        rows = []
        counterexample = None
        for a in A:
            pair_present = (a, a) in R
            rows.append(
                {
                    "Element x": str(a),
                    "Expected": f"({a}, {a}) ∈ R",
                    "Result":
                        "✅ Present"
                        if pair_present
                        else "❌ Missing"
                }
            )
            if not pair_present and counterexample is None:
                counterexample = a

        reflexive_df = pd.DataFrame(rows)
        display_rows = min(
            MAX_DISPLAY_ROWS,
            len(reflexive_df)
        )

        st.caption(
            f"Displaying {display_rows} "
            f"out of {len(reflexive_df)} checks."
        )

        st.dataframe(
            reflexive_df.head(MAX_DISPLAY_ROWS),
            hide_index=True,
            width="stretch"
        )

        if counterexample is None:
            st.success(
                "The relation is reflexive."
            )

        else:
            st.error(
                "The relation is not reflexive."
            )
            st.markdown(
                f"""
    **Counterexample**

    ({counterexample}, {counterexample})
    is not in the relation.
    """
            )

    elif chosen_property == "Anti-reflexive":
        st.markdown("""
    ### Anti-reflexive

    For every x ∈ A, (x, x) ∉ R.
    """)

        R = set(relation)
        rows = []
        counterexample = None
        for a in A:
            pair_absent = (a, a) not in R
            rows.append(
                {
                    "Element x": str(a),
                    "Expected": f"({a}, {a}) ∉ R",
                    "Result":
                        "✅ Absent"
                        if pair_absent
                        else "❌ Present"
                }
            )
            if not pair_absent and counterexample is None:
                counterexample = a

        anti_reflexive_df = pd.DataFrame(rows)

        display_rows = min(
            MAX_DISPLAY_ROWS,
            len(anti_reflexive_df)
        )

        st.caption(
            f"Displaying {display_rows} "
            f"out of {len(anti_reflexive_df)} checks."
        )

        st.dataframe(
            anti_reflexive_df.head(MAX_DISPLAY_ROWS),
            hide_index=True,
            width="stretch"
        )

        if counterexample is None:
            st.success(
                "The relation is anti-reflexive."
            )

        else:
            st.error(
                "The relation is not anti-reflexive."
            )
            st.markdown(
                f"""
    **Counterexample**

    ({counterexample}, {counterexample})
    belongs to the relation.
    """
            )

    elif chosen_property == "Symmetric":

        st.markdown("""
    ### Symmetric

    For every pair (a, b) ∈ R, the pair (b, a) must also belong to R.
    """)

        R = set(relation)

        rows = []

        counterexample = None

        for a, b in sorted(R):

            reverse_present = (b, a) in R

            rows.append(
                {
                    "Pair in R": f"({a}, {b})",
                    "Expected":
                        f"({b}, {a}) ∈ R",
                    "Result":
                        "✅ Present"
                        if reverse_present
                        else "❌ Missing"
                }
            )

            if (
                not reverse_present
                and counterexample is None
            ):
                counterexample = (a, b)

        symmetric_df = pd.DataFrame(rows)

        display_rows = min(
            MAX_DISPLAY_ROWS,
            len(symmetric_df)
        )

        st.caption(
            f"Displaying {display_rows} "
            f"out of {len(symmetric_df)} checks."
        )

        st.dataframe(
            symmetric_df.head(MAX_DISPLAY_ROWS),
            hide_index=True,
            width="stretch"
        )

        if counterexample is None:

            st.success(
                "The relation is symmetric."
            )

        else:

            st.error(
                "The relation is not symmetric."
            )

            a, b = counterexample

            st.markdown(
                f"""
    **Counterexample**

    ({a}, {b}) belongs to the relation.

    However,

    ({b}, {a}) does not belong to the relation.
    """
            )

    elif chosen_property == "Anti-symmetric":

        st.markdown("""
    ### Anti-symmetric

    For every pair (a, b) ∈ R where a ≠ b,
    the pair (b, a) must not belong to R.
    """)

        R = set(relation)

        rows = []

        counterexample = None

        for a, b in sorted(R):

            if a == b:

                rows.append(
                    {
                        "Pair in R": f"({a}, {b})",
                        "Expected":
                            "a = b (allowed)",
                        "Result":
                            "✅ Allowed"
                    }
                )

            else:

                reverse_present = (b, a) in R

                rows.append(
                    {
                        "Pair in R": f"({a}, {b})",
                        "Expected":
                            f"({b}, {a}) ∉ R",
                        "Result":
                            "✅ Satisfied"
                            if not reverse_present
                            else "❌ Violated"
                    }
                )

                if (
                    reverse_present
                    and counterexample is None
                ):
                    counterexample = (a, b)

        antisymmetric_df = pd.DataFrame(rows)

        display_rows = min(
            MAX_DISPLAY_ROWS,
            len(antisymmetric_df)
        )

        st.caption(
            f"Displaying {display_rows} "
            f"out of {len(antisymmetric_df)} checks."
        )

        st.dataframe(
            antisymmetric_df.head(MAX_DISPLAY_ROWS),
            hide_index=True,
            width="stretch"
        )

        if counterexample is None:

            st.success(
                "The relation is anti-symmetric."
            )

        else:

            st.error(
                "The relation is not anti-symmetric."
            )

            a, b = counterexample

            st.markdown(
                f"""
    **Counterexample**

    ({a}, {b}) belongs to the relation.

    ({b}, {a}) also belongs to the relation.

    Since {a} ≠ {b}, this violates the
    definition of an anti-symmetric relation.
    """
            )

    elif chosen_property == "Transitive":

        st.markdown("""
    ### Transitive

    A relation **R** on set **A** is **transitive** if:

    Whenever (a, b) ∈ R and (b, c) ∈ R,
    then (a, c) must also belong to R.
    """)

        R = set(relation)

        rows = []

        counterexample = None

        for a, b in sorted(R):
            for c, d in sorted(R):

                if b == c:

                    expected_pair = (a, d)

                    present = expected_pair in R

                    rows.append(
                        {
                            "Pair 1": f"({a}, {b})",
                            "Pair 2": f"({c}, {d})",
                            "Expected":
                                f"({a}, {d}) ∈ R",
                            "Result":
                                "✅ Present"
                                if present
                                else "❌ Missing"
                        }
                    )

                    if (
                        not present
                        and counterexample is None
                    ):
                        counterexample = (
                            (a, b),
                            (c, d),
                            (a, d)
                        )

        transitive_df = pd.DataFrame(rows)

        display_rows = min(
            MAX_DISPLAY_ROWS,
            len(transitive_df)
        )

        st.caption(
            f"Displaying {display_rows} "
            f"out of {len(transitive_df)} checks."
        )

        st.dataframe(
            transitive_df.head(MAX_DISPLAY_ROWS),
            hide_index=True,
            width="stretch"
        )

        if counterexample is None:

            st.success(
                "The relation is transitive."
            )

        else:

            st.error(
                "The relation is not transitive."
            )

            pair1, pair2, missing_pair = (
                counterexample
            )

            st.markdown(
                f"""
    **Counterexample**

    {pair1} belongs to the relation.

    {pair2} belongs to the relation.

    Therefore, the pair

    {missing_pair}

    must belong to the relation for transitivity.

    However, {missing_pair} is not in the relation.
    """
            )

