import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="SheetPilot AI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 SheetPilot AI")
st.subheader("Smart Excel Transformation Engine")

st.write(
    "Upload your raw Excel data, upload your target template, "
    "and transform your data into the required structure."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Raw Data")
    raw_file = st.file_uploader(
        "Choose raw Excel or CSV file",
        type=["xlsx", "csv"],
        key="raw"
    )

with col2:
    st.subheader("2. Upload Target Template")
    template_file = st.file_uploader(
        "Choose target Excel template",
        type=["xlsx", "csv"],
        key="template"
    )

command = st.text_area(
    "3. Enter Your Command",
    placeholder=(
        "Example: Remove duplicates, standardize dates, "
        "and arrange columns according to my template."
    )
)

if raw_file is not None:
    try:
        if raw_file.name.lower().endswith(".csv"):
            raw_df = pd.read_csv(raw_file, dtype=str)
        else:
            raw_df = pd.read_excel(raw_file, dtype=str)

        st.subheader("Raw Data Preview")
        st.dataframe(raw_df.head(100), use_container_width=True)
        st.caption(f"Rows: {len(raw_df)} | Columns: {len(raw_df.columns)}")

        if st.button("Inspect Template and Transform"):
            if template_file is None:
                st.warning("Please upload your target template.")
            elif not command.strip():
                st.warning("Please enter your transformation command.")
            else:
                if template_file.name.lower().endswith(".csv"):
                    template_df = pd.read_csv(
                        template_file, nrows=0
                    )
                else:
                    template_df = pd.read_excel(
                        template_file, nrows=0
                    )

                st.subheader("Target Template Columns")
                st.write(list(template_df.columns))

                st.info(
                    "Uploads are working. The next development step "
                    "is implementing column mapping, command execution, "
                    "validation, and Excel export."
                )

    except Exception as e:
        st.error(f"Unable to read the file: {e}")
