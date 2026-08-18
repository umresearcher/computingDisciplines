import streamlit as st
import pandas as pd
from Chapter_6_utils import *

st.title("Module 1: Binary Relations ↔ Tables")

st.markdown("""
A binary relation between two sets **A** and **B**
can be represented as:

- A set of ordered pairs
- A two-column table
""")

if "validated_A" not in st.session_state:
    st.session_state.validated_A = None

if "last_set_b_mode" not in st.session_state:
    st.session_state.last_set_b_mode = None

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
# Step 1: Define Set A and Set B
# --------------------------------------------------

st.subheader("Step 1: Define Set A and Set B")

if "set_a" not in st.session_state:
    st.session_state.set_a = "1,2,3,4"

set_a_text = st.text_input(
    "Enter elements separated by commas",
    key="set_a"
)

def clear_set_a():
    st.session_state.set_a = ""
    st.session_state.validated_A = None
    st.session_state.validated_relation_code = None
    st.session_state.relation_function = None
    st.session_state.relation_instance = None
    st.session_state.last_predefined_relation = None

col1, col2 = st.columns(2)

with col1:
    validate_a = st.button("Validate Set A")

with col2:
    st.button(
        "Clear Set A",
        on_click=clear_set_a
    )

if validate_a:
    try:
        A = parse_set(set_a_text)
        st.session_state.validated_A = A
        st.session_state.validated_relation_code = None
        st.session_state.relation_function = None
        st.session_state.relation_instance = None
    except Exception as e:
        st.error(str(e))

if st.session_state.validated_A is not None:
    A = st.session_state.validated_A
    st.success(f"Set A is valid ({len(A)} elements)")
    st.markdown(
        f"**A =** {{{', '.join(map(str, A))}}}"
    )
    st.markdown(
        f"**|A| = {len(A)}**"
    )

# --------------------------------------------------
# Define Set B
# --------------------------------------------------

st.subheader("Define Set B")
set_b_mode = st.radio(
    "Select Set B",
    [
        "Use Set A",
        "Define a Different Set B"
    ]
)

if (
    set_b_mode
    != st.session_state.last_set_b_mode
):
    st.session_state.validated_B = None
    st.session_state.validated_relation_code = None
    st.session_state.relation_function = None
    st.session_state.relation_instance = None
    st.session_state.last_set_b_mode = set_b_mode

if set_b_mode == "Use Set A":
    if st.session_state.validated_A is not None:
        B = st.session_state.validated_A
        st.session_state.validated_B = B
        st.markdown(
            f"**B =** "
            f"{{{', '.join(map(str, B))}}}"
        )
        st.markdown(
            f"**|B| = {len(B)}**"
        )
        st.caption(
            "Set B is the same as Set A."
        )

else:
    if "set_b" not in st.session_state:
        st.session_state.set_b = ""
    set_b_text = st.text_input(
        "Enter elements of Set B",
        key="set_b"
    )

    def clear_set_b():
        st.session_state.set_b = ""
        st.session_state.validated_B = None
        st.session_state.validated_relation_code = None
        st.session_state.relation_function = None
        st.session_state.relation_instance = None
        st.session_state.last_predefined_relation = None

    col1, col2 = st.columns(2)

    with col1:
        validate_b = st.button(
            "Validate Set B"
        )

    with col2:
        st.button(
            "Clear Set B",
            on_click=clear_set_b
        )

    if validate_b:
        try:
            B = parse_set(set_b_text)
            st.session_state.validated_B = B
            st.session_state.validated_relation_code = None
            st.session_state.relation_function = None
            st.session_state.relation_instance = None

        except Exception as e:
            st.error(str(e))

    if st.session_state.validated_B is not None:
        B = st.session_state.validated_B

        st.success(
            f"Set B is valid ({len(B)} elements)"
        )

        st.markdown(
            f"**B =** "
            f"{{{', '.join(map(str, B))}}}"
        )

        st.markdown(
            f"**|B| = {len(B)}**"
        )

if set_b_mode == "Use Set A":
    if st.session_state.validated_A is None:
        st.info(
            "Complete Step 1 by validating Set A."
        )
else:
    if (
        st.session_state.validated_A is None
        or st.session_state.validated_B is None
    ):
        st.info(
            "Complete Step 1 by validating Set A and Set B."
        )

# --------------------------------------------------
# Step 2: Define Relation
# --------------------------------------------------

if (
    st.session_state.validated_A is not None
    and st.session_state.validated_B is not None
):
    st.subheader("Step 2: Define the Relation")
    st.markdown("""
A binary relation between Set **A** and Set **B**
is a set of ordered pairs.

A pair **(a, b)** belongs to the relation if:

```python
relationDef(a, b)
```

returns True.

For each pair `(a, b)` in `A × B`, the function is evaluated.

Choose a predefined relation or define your own relation using relationDef(a, b).

The function must return:

- `True` if `(a, b)` belongs to the relation.
- `False` if `(a, b)` does not belong to the relation.
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
        "Less Than":
    """def relationDef(a, b):
        return a < b
    """,

        "Greater Than":
    """def relationDef(a, b):
        return a > b
    """,

        "Equality":
    """def relationDef(a, b):
        return a == b
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

        "Square Less Than":
    """def relationDef(a, b):
        x = a * a
        return x < b
    """
    }

    if relation_mode == "Predefined Relation":
        chosen_relation = st.selectbox(
            "Select a Relation",
            list(PREDEFINED_RELATIONS.keys())
        )
        relation_code = PREDEFINED_RELATIONS[
            chosen_relation
        ]
        st.code(
            relation_code,
            language="python"
        )
    else:
        relation_code = st.text_area(
            "Enter relationDef(a, b)",
            value="""def relationDef(a, b):
        return a < b
    """,
            height=200
        )

    if relation_mode == "Predefined Relation":

        if (
            chosen_relation
            != st.session_state.last_predefined_relation
        ):
            st.session_state.relation_instance = None

            st.session_state.last_predefined_relation = (
                chosen_relation
            )

        rel_func = build_relation_function(
            relation_code
        )

        works, message = (
            validate_relation_on_sets(
                A,
                B,
                rel_func
            )
        )

        if works:

            st.session_state.validated_relation_code = (
                relation_code
            )

            st.session_state.relation_function = (
                rel_func
            )

        else:

            st.session_state.validated_relation_code = None
            st.session_state.relation_function = None
            st.session_state.relation_instance = None

            st.error(message)

    else:

        if st.button(
            "Validate Relation Definition"
        ):

            is_valid, message = validate_relation(
                relation_code,
                A,
                B
            )

            if is_valid:
                st.session_state.validated_relation_code = (
                    relation_code
                )
                st.session_state.relation_function = (
                    build_relation_function(
                        relation_code
                    )
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
    and st.session_state.validated_B is not None
    and st.session_state.relation_function is not None
):
    A = st.session_state.validated_A
    B = st.session_state.validated_B
    rel_func = st.session_state.relation_function
    st.subheader("Step 3: Preview Relation Behavior")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"**A** = "
            f"{{{', '.join(map(str, A))}}}"
        )

    with col2:
        st.markdown(
            f"**B** = "
            f"{{{', '.join(map(str, B))}}}"
        )

    st.caption(
        f"|A × B| = {len(A) * len(B)} possible pairs"
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
        "pairs from A × B. True means the pair belongs "
        "to the relation."
    )

    # ----------------------------------------------
    # Build Relation Instance
    # ----------------------------------------------

    if st.button("Build Relation Instance"):
        relation = build_relation_instance(
            A,
            B,
            rel_func
        )
        st.session_state.relation_instance = relation

# --------------------------------------------------
# Step 4: Display Relation Instance
# --------------------------------------------------

if (
    st.session_state.validated_A is not None
    and st.session_state.validated_B is not None
    and st.session_state.relation_function is not None
    and st.session_state.relation_instance is not None
):
    relation = st.session_state.relation_instance
    st.subheader("Step 4: Display Relation Instance")
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
    st.info("""
    The order of rows in the table does not matter.

    A relation is a set of ordered pairs, so each
    ordered pair appears at most once.
    """)
