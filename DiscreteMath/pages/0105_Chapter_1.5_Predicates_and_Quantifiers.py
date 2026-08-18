import streamlit as st
import pandas as pd

st.title("Module 5: Predicates and Quantifiers")

st.markdown("""
In Module 4, every proposition already had a truth value.

In this module, we study predicates, whose truth value depends on one or more variables.
""")

st.subheader("Predicates and Propositions")

col1, col2 = st.columns(2)

with col1:

    st.success("""
**Proposition**

```text
17 is even
```

Truth Value:

```text
False
```
""")

with col2:
    st.info("""
**Predicate**

```text
x is even
```

Truth Value:

```text
Depends on x 
```
""")

st.markdown("---")

## Programming Connection  
st.subheader("Programming Connection") 

st.markdown(""" Suppose 
```python 
x = 4
```

Then

```python 
x % 2 == 0
```

evaluates to True.

However,

```python 
x % 2 == 0
```

is not a proposition until x has a value. """)

st.markdown("---")

## Interactive Example
st.subheader("Explore a Predicate")

col1, col2 = st.columns([1,1])

with col1:
    x = st.slider( 
        "Choose a value for x", 
        min_value=1, 
        max_value=10, 
        value=4 ) 

with col2:
    is_even = (x % 2 == 0) 
    st.code(
        f"""P(x): x is even

P({x}) = {"True" if is_even else "False"}""",
        language="text"
    )

st.markdown("---")

## Database Connection
st.subheader("Database Connection") 
students_df = pd.DataFrame( 
    { "Student": ["Alice", "Bob", "Charlie"], 
     "GPA": ["3.9", "2.4", "3.7"] } ) 

col1, col2 = st.columns([1,2])

with col1:
    st.markdown("")
    st.markdown("**StudentGPAs**")
    st.dataframe(
        students_df,
        hide_index=True,
        width="content"
    )

with col2:

    st.markdown("""
    Consider the predicate

    ```text
    GPA(x) > 3.5
    ```

    This is not yet a proposition because x has not been specified.

    However,

    ```text
    Alice's GPA > 3.5
    ```

    becomes

    ```text
    3.9 > 3.5
    ```

    which is a proposition with truth value True. """)

st.markdown("---")

## Predicates with Multiple Variables

st.subheader("Predicates with Multiple Variables") 
enrolled_df = pd.DataFrame( 
    { "Student": ["Alice", "Bob"], 
     "Course": ["CSC 379", "MTH 328"] } ) 

col1, col2 = st.columns([1,2])

with col1:
    st.markdown("**Enrolled**")
    st.dataframe( enrolled_df, hide_index=True, width="stretch" )

    st.markdown("""
    The table can be viewed as the predicate

    ```text
    Enrolled(student, course)
    ```
    
    with two variables:

    ```text
    student
    course
    ```
    
    """)

with col2:
    st.markdown("""
    ```text
    Enrolled(Alice, CSC 379)
    ```

    is True because the row appears in the table.

    Similarly,

    ```text
    Enrolled(Bob, MTH 328)
    ```

    is True. 

    However, 

    ```text
    Enrolled(Alice, MTH 328)
    ```

    is False. 
    """)

st.info("The table stores all combinations of values for which the predicate is True.")

st.markdown("---")

st.subheader("Domains")

st.markdown("""
A domain is the set of values that a variable can take.
""")

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
**Example**: Consider the predicate and domain shown below.
    """)

    st.markdown("""
**Predicate**

```text
P(x): x is even
```
""")

    st.markdown("""
**Domain**

```text
{1, 2, 3, 4, 5}
```
""")

with col2:
    st.markdown("""
The truth value of the predicate for each element of the domain is shown below.
    """)

    domain = [1, 2, 3, 4, 5] 
    domain_df = pd.DataFrame( 
        { "x": [str(x) for x in domain],
        "P(x): x is even": [ "T" if x % 2 == 0 else "F" for x in domain ] } ) 
    st.dataframe( domain_df, hide_index=True, width="content" )

st.subheader("Universal Quantifier")

st.markdown("""
The symbol

```text
∀
```

is a **universal quantifier**. It means "for all." """)

st.markdown(""" 
```text 
∀x P(x)
```

means

```text 
For every value of x in the domain,
P(x) is True.
```
""")

st.markdown("""
The quantified statement

```text 
∀x P(x) 
```

is a proposition because it has a truth value.
""")

st.markdown(""" 
Consider:
""")

col1, col2 = st.columns(2) 
with col1: 
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""**Predicate**""")
        st.markdown(""" 
    ```text
    P(x): x is even
    ```
    """)

    with col4:
        st.markdown("""**Domain**""")
        st.markdown(""" 
    ```text
    {1,2,3,4,5}
    ```
    """)

    st.info("""
For the domain {1,2,3,4,5},

```text
∀x P(x)
```

means

```text
P(1) ∧ P(2) ∧ P(3) ∧ P(4) ∧ P(5)
```
""")

with col2:
    st.markdown("")

    st.dataframe( domain_df, hide_index=True, width="content" ) 
    st.error(""" ∀x P(x) is False because P(1), P(3), and P(5) are False. """)

st.subheader("Existential Quantifier")

st.markdown("""
The symbol

```text
∃
```

is an **existential quantifier**. It means "there exists." """)

st.markdown(""" 
```text 
∃x P(x)
```

means

```text 
There exists a value of x in the domain
for which P(x) is True.
```
""")

st.markdown("""
The quantified statement

```text 
∃x P(x) 
```

is a proposition because it has a truth value.
""")
st.markdown(""" 
Consider:
""")

col1, col2 = st.columns(2) 
with col1: 
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""**Predicate**""")
        st.markdown(""" 
    ```text
    P(x): x is even
    ```
    """)

    with col4:
        st.markdown("""**Domain**""")
        st.markdown(""" 
    ```text
    {1,2,3,4,5}
    ```
    """)

    st.info("""
For the domain {1,2,3,4,5},

```text
∃x P(x)
```

means

```text
P(1) ∨ P(2) ∨ P(3) ∨ P(4) ∨ P(5)
```
""")

with col2:
    st.markdown("")

    st.dataframe( domain_df, hide_index=True, width="content" ) 
    st.success(""" ∃x P(x) is True because P(2) and P(4) are True. """)


