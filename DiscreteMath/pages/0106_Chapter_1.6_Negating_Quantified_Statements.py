import streamlit as st
import pandas as pd

st.title("Module 6: Negating Quantified Statements")

st.markdown("""
In Module 5, we introduced universal and existential quantifiers.

In this module, we learn how to negate quantified statements.
""")

col1, col2, col3 = st.columns([3,4,4])

with col1:
    st.markdown("**Predicate and Domain**")
    st.markdown("""
    Consider

    ```text
    P(x) : x submitted the assignment.
    Domain: set of students
    ```
    """)

with col2:
    st.markdown("""
    ```text
    ∀x P(x)
    ```

    means

    ```text
    Every student submitted the assignment.
    ```
    """)

with col3:
    st.markdown("""
```text
∃x P(x)
```

means

```text
There exists a student who submitted the assignment.
```
""")

### Common Mistake 
st.warning(""" When negating a quantified statement, simply adding NOT is often incorrect. """)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Negating Universal Quantifiers")

    st.markdown("""
    The negation of

    ```text
    Every student submitted the assignment.
    ```

    is not

    ```text
    Every student did not submit the assignment.
    ```
    """)

    ### Correct Negation

    st.success("""
    The correct negation is
    """)

    st.markdown("""
    ```text
    There exists a student who did not submit the assignment.
    ```
    """)

    st.markdown("""
    ```text
    ¬∀x P(x) ≡ ∃x ¬P(x)
    ```
    """)

with col2:
    st.subheader("Negating Existential Quantifiers")

    st.markdown("""
    The negation of

    ```text
    There exists a student who submitted the assignment.
    ```

    is not

    ```text
    There exists a student who did not submit the assignment.
    ```
    """)

    ### Correct Negation
    st.success("""
    The correct negation is
    """)

    st.markdown("""
    ```text
    No student submitted the assignment.
    ```
    """)

    st.markdown("""
    ```text
    ¬∃x P(x) ≡ ∀x ¬P(x)
    ```
    """)

st.subheader("Explore Negating Quantified Statements")

col1, col2, col3 = st.columns([3,4,4])

with col1:
    st.markdown("""
```text
Domain: {Alice, Bob, Charlie}
```
    """)
    st.markdown("Choose who all submitted the assignment")
    alice = st.checkbox(
        "Alice submitted",
        value=True
    )
    bob = st.checkbox(
        "Bob submitted",
        value=True
    )
    charlie = st.checkbox(
        "Charlie submitted",
        value=False
    )
    values = [alice, bob, charlie]

with col2:   
    forall_p = all(values)
    exists_not_p = any(not v for v in values)

    results_df = pd.DataFrame(
        {
            "Statement": [
                "∀x P(x)",
                "¬∀x P(x)",
                "∃x ¬P(x)",
            ],
            "Truth Value": [
                "T" if forall_p else "F",
                "T" if not forall_p else "F",
                "T" if exists_not_p else "F",
            ],
        }
    )

    st.markdown("**Universal Negation**")

    st.dataframe(
        results_df,
        hide_index=True,
        width="content"
    )
    st.info("""
    Notice that for the examples explored here,\n\n
    ¬∀x P(x) and ∃x ¬P(x) have the same truth value.
    """)

with col3:
    exists_p = any(values) 
    forall_not_p = all(not v for v in values) 

    results_df = pd.DataFrame( 
        { 
            "Statement": [ 
                "∃x P(x)", 
                "¬∃x P(x)", 
                "∀x ¬P(x)", 
            ],
            "Truth Value": [ 
                "T" if exists_p else "F", 
                "T" if not exists_p else "F", 
                "T" if forall_not_p else "F", 
            ], 
        } 
    ) 
    st.markdown("**Existential Negation**")
    st.dataframe( results_df, hide_index=True, width="content" )
    st.info("""
    Notice that for the examples explored here,\n
    ¬∃x P(x) and ∀x ¬P(x) have the same truth value.
    """)

st.subheader("Summary of Quantifier Negation")

summary_df = pd.DataFrame(
    {
        "Statement": [
            "¬∀x P(x)",
            "¬∃x P(x)"
        ],
        "Equivalent Statement": [
            "∃x ¬P(x)",
            "∀x ¬P(x)"
        ]
    }
)

st.dataframe(
    summary_df,
    hide_index=True,
    width="content"
)

with st.expander("Connection to DeMorgan's Laws"):

    st.markdown("""
For propositions, DeMorgan's Laws state

```text
¬(p ∧ q) ≡ ¬p ∨ ¬q
¬(p ∨ q) ≡ ¬p ∧ ¬q
```

For quantified statements, similar laws hold:

```text
¬∀x P(x) ≡ ∃x ¬P(x)
¬∃x P(x) ≡ ∀x ¬P(x)
```

In both cases, negation moves inward and the operator changes:

```text
∧ ↔ ∨
∀ ↔ ∃
```

""")
