import streamlit as st
import pandas as pd

st.title("Module 7: Nested Quantifiers")

st.markdown("""
In Module 6, we learned how to negate quantified statements.

In this module, we study nested quantifiers and see how the order of quantifiers affects meaning.
""")

st.markdown("""
A nested quantifier occurs when a quantified statement contains more than one quantifier.
""")

st.subheader("Example")

col1, col2 = st.columns(2)

with col1:

    col3, col4 = st.columns([1,2])

    with col3:
        st.markdown("""""")

        st.markdown("""
        **Predicate**
        """)

    with col4:
        st.markdown("""
        ```text
        Enrolled(student, course)
        ```

        which is True when the student is enrolled in the course.""")

    col3, col4 = st.columns([1,2])

    with col3:
        st.markdown("""""")
        st.markdown("""**Domain of students**""")

    with col4:
        st.markdown("""
        ```text
        {Alice, Bob, Charlie}
        ```
        """)


    col3, col4 = st.columns([1,2])

    with col3:
        st.markdown("""""")
        st.markdown("""**Domain of courses**""")

    with col4:
        st.markdown("""
        ```text
        {CSC 379, MTH 328}
        ```
        """)

with col2:
    st.markdown("""
    The table contains the combinations of values for which the predicate is True.
    """)

    enrolled_df = pd.DataFrame( { 
        "Student": [ "Alice", "Alice", "Bob", "Charlie" ], 
        "Course": [ "CSC 379", "MTH 328", "CSC 379", "MTH 328" ] } )

    st.markdown("""**Enrolled**""")
    st.dataframe( enrolled_df, hide_index=True, width="content" )

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
```text
∀student ∃course
Enrolled(student, course)
```

means

```text
Every student is enrolled
in at least one course.
```
""")

with col2:
    st.markdown("""
    ```text
    ∃course ∀student
    Enrolled(student, course)
    ```
    means

    ```text
    There exists a course
    in which every student is enrolled.
    ```
    """)

st.markdown("""For the given table:""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ```text
    ∀student ∃course Enrolled(student, course)
    ```

    is **True** because every student appears in at least one row.
    """)

with col2:
    st.markdown("""
    ```text
    ∃course ∀student Enrolled(student, course)
    ```

    is **False** because no single course contains every student.
    """)

st.warning(""" These two statements are not equivalent. """)

st.subheader("Common Nested Quantified Statements")

nested_df = pd.DataFrame(
    {
        "Quantified Statement": [
            "∀student ∀course Enrolled(student, course)",
            "∃student ∃course Enrolled(student, course)",
            "∀student ∃course Enrolled(student, course)",
            "∃course ∀student Enrolled(student, course)",
            "∀course ∃student Enrolled(student, course)",
            "∃student ∀course Enrolled(student, course)",
        ],
        "Meaning": [
            "Every student is enrolled in every course.",
            "There exists a student enrolled in a course.",
            "Every student is enrolled in at least one course.",
            "There exists a course in which every student is enrolled.",
            "Every course has at least one enrolled student.",
            "There exists a student enrolled in every course.",
        ],
        "Quantifier Order": [
            "Can be swapped. Equivalent to ∀course ∀student ...",
            "Can be swapped. Equivalent to ∃course ∃student ...",
            "May change meaning. May not be equivalent to ∃course ∀student ...",
            "May change meaning. May not be equivalent to ∀student ∃course ...",
            "May change meaning. May not be equivalent to ∃student ∀course ...",
            "May change meaning. May not be equivalent to ∀course ∃student ...",
        ],
    }
)

st.dataframe(
    nested_df,
    hide_index=True,
    width="stretch",
)

st.subheader("Explore Nested Quantifiers")

st.markdown("""
Select the rows that belong to the relation

```text
Enrolled(student, course)
```
""")

col1, col2, col3 = st.columns(3) 

with col1: 
    alice_379 = st.checkbox( "Enrolled(Alice, CSC 379)", value=True ) 
    alice_328 = st.checkbox( "Enrolled(Alice, MTH 328)", value=True ) 

with col2:
    bob_379 = st.checkbox( "Enrolled(Bob, CSC 379)", value=True ) 
    bob_328 = st.checkbox( "Enrolled(Bob, MTH 328)", value=False ) 

with col3: 
    charlie_379 = st.checkbox( "Enrolled(Charlie, CSC 379)", value=False ) 
    charlie_328 = st.checkbox( "Enrolled(Charlie, MTH 328)", value=True )

enrolled = {
    ("Alice", "CSC 379"): alice_379,
    ("Alice", "MTH 328"): alice_328,
    ("Bob", "CSC 379"): bob_379,
    ("Bob", "MTH 328"): bob_328,
    ("Charlie", "CSC 379"): charlie_379,
    ("Charlie", "MTH 328"): charlie_328,
}

students = ["Alice", "Bob", "Charlie"]
courses = ["CSC 379", "MTH 328"]

forall_exists = all(
    any(
        enrolled[(student, course)]
        for course in courses
    )
    for student in students
)

exists_forall = any(
    all(
        enrolled[(student, course)]
        for student in students
    )
    for course in courses
)

results_df = pd.DataFrame(
    {
        "Statement": [
            "∀student ∃course Enrolled(student, course)",
            "∃course ∀student Enrolled(student, course)",
        ],
        "Truth Value": [
            "T" if forall_exists else "F",
            "T" if exists_forall else "F",
        ],
    }
)

forall_exists2 = all(
    any(
        enrolled[(student, course)]
        for student in students
    )
    for course in courses
)

exists_forall2 = any(
    all(
        enrolled[(student, course)]
        for course in courses
    )
    for student in students
)

results_df2 = pd.DataFrame(
    {
        "Statement": [
            "∀course ∃student Enrolled(student, course)",
            "∃student ∀course Enrolled(student, course)",
        ],
        "Truth Value": [
            "T" if forall_exists2 else "F",
            "T" if exists_forall2 else "F",
        ],
    }
)

forall_forall = all(
    all(
        enrolled[(student, course)]
        for student in students
    )
    for course in courses
)

exists_exists = any(
    any(
        enrolled[(student, course)]
        for course in courses
    )
    for student in students
)

results_df3 = pd.DataFrame(
    {
        "Statement": [
            "∀course ∀student Enrolled(student, course)",
            "∃student ∃course Enrolled(student, course)",
        ],
        "Truth Value": [
            "T" if forall_forall else "F",
            "T" if exists_exists else "F",
        ],
    }
)


col1, col2, col3 = st.columns(3)

with col1:
    st.dataframe(
        results_df,
        hide_index=True,
        width="stretch",
    )

with col2:
    st.dataframe(
        results_df2,
        hide_index=True,
        width="stretch",
    )

with col3:
    st.dataframe(
        results_df3,
        hide_index=True,
        width="stretch",
    )

st.info("""
Experiment with the rows.

Notice that when different quantifiers are nested,
changing the order of the quantifiers can change
the meaning and truth value of a statement.
""")

st.subheader("Summary")

summary_df = pd.DataFrame(
    {
        "Statement": [
            "∀x ∀y P(x,y)",
            "∃x ∃y P(x,y)",
            "∀x ∃y P(x,y)",
            "∃y ∀x P(x,y)",
        ],
        "Observation": [
            "Order may be swapped",
            "Order may be swapped",
            "Order may matter",
            "Order may matter",
        ],
    }
)

st.dataframe(
    summary_df,
    hide_index=True,
    width="content",
)

st.warning("""
Quantifiers of the same type can be reordered.

When different quantifiers are nested, the order of the quantifiers may change the meaning of the statement.
""")

with st.expander("Additional Observation"):

    st.markdown("""
Consider a binary relation

```text
R(x,y)
```

Then

```text
∃y ∀x R(x,y) implies ∀x ∃y R(x,y)
```

Why is this true?

If there exists a value of y for which R(x,y) is true for every x, 
then for every x there certainly exists a y (namely the same y) 
for which R(x,y) is true.

However, the converse is not always true. In other words,

```text
∀x ∃y R(x,y) does not imply ∃y ∀x R(x,y) 
```

In our example,

```text
∃course ∀student Enrolled(student, course) implies ∀student ∃course Enrolled(student, course)
∀student ∃course Enrolled(student, course) does not imply ∃course ∀student Enrolled(student, course)
```

By the same reasoning,

```text
∃student ∀course Enrolled(student, course) implies ∀course ∃student Enrolled(student, course)
∀course ∃student Enrolled(student, course) does not imply ∃student ∀course Enrolled(student, course)
```
""")

