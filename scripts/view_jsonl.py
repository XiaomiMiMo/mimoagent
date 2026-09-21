#!/usr/bin/env python3
"""Streamlit app to view test.jsonl data."""

import json
from pathlib import Path

import streamlit as st


def count_entries(file_path: Path) -> int:
    """Count number of entries in JSONL file."""
    with open(file_path) as f:
        return sum(1 for line in f if line.strip())


def load_entry_at_index(file_path: Path, idx: int) -> dict:
    """Load a specific entry by index from JSONL file."""
    with open(file_path) as f:
        current_idx = 0
        for line in f:
            if line.strip():
                if current_idx == idx:
                    return json.loads(line)
                current_idx += 1
    raise IndexError(f"Index {idx} out of range")


def display_entry(entry: dict):
    """Display a single entry."""
    st.header(f"Entry ID: {entry.get('id', 'N/A')}")

    # Display basic metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Event", entry.get("event", "N/A"))
    with col2:
        st.metric("Code", entry.get("code", "N/A"))
    with col3:
        st.metric("Delay (ms)", entry.get("delay", "N/A"))
    with col4:
        st.metric("Cost", entry.get("cost", "N/A"))

    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Basic Info", "📥 Request", "📤 Response", "🔍 Raw Data"])

    with tab1:
        st.subheader("Basic Information")
        basic_info = {
            "ID": entry.get("id", "N/A"),
            "Distinct ID": entry.get("distinctid", "N/A"),
            "Event": entry.get("event", "N/A"),
            "Component ID": entry.get("compid", "N/A"),
            "Timestamp": entry.get("ts", "N/A"),
            "User Label": entry.get("user_label", "N/A"),
            "Response Code": entry.get("code", "N/A"),
            "Delay (ms)": entry.get("delay", "N/A"),
            "Cost": entry.get("cost", "N/A"),
        }
        for key, value in basic_info.items():
            st.text(f"{key}: {value}")

    with tab2:
        st.subheader("Request Data")
        request_str = entry.get("request", "")
        if request_str:
            try:
                request_json = json.loads(request_str)
                st.json(request_json)
            except json.JSONDecodeError:
                st.text_area("Request (Raw)", request_str, height=400, key="request")
        else:
            st.info("No request data")

    with tab3:
        st.subheader("Response Data")
        response_str = entry.get("response", "")
        if response_str:
            try:
                response_json = json.loads(response_str)
                st.json(response_json)
            except json.JSONDecodeError:
                st.text_area("Response (Raw)", response_str, height=400, key="response")
        else:
            st.info("No response data")

    with tab4:
        st.subheader("Raw JSON Data")
        st.json(entry)


def main():
    st.set_page_config(page_title="Test.jsonl Viewer", layout="wide")
    st.title("🔍 Test.jsonl Viewer")

    # File path input
    file_path = st.text_input("JSONL File Path:", value="", placeholder="/path/to/test.jsonl")
    if not file_path:
        st.info("Enter a JSONL file path to start")
        return

    if not Path(file_path).exists():
        st.error(f"File not found: {file_path}")
        return

    # Count entries
    if "total_count" not in st.session_state or st.session_state.get("file_path") != file_path:
        with st.spinner("Counting entries..."):
            st.session_state.total_count = count_entries(Path(file_path))
            st.session_state.file_path = file_path
            st.session_state.current_idx = 0

    total = st.session_state.total_count

    st.success(f"Loaded {total} entries")

    # Navigation controls
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        if st.button("⬅️ Previous", disabled=(st.session_state.current_idx == 0)):
            st.session_state.current_idx -= 1
            st.rerun()

    with col2:
        # Slider for navigation
        idx = st.slider(
            "Select entry:", min_value=0, max_value=total - 1, value=st.session_state.current_idx, key="nav_slider"
        )
        if idx != st.session_state.current_idx:
            st.session_state.current_idx = idx
            st.rerun()

    with col3:
        if st.button("Next ➡️", disabled=(st.session_state.current_idx >= total - 1)):
            st.session_state.current_idx += 1
            st.rerun()

    # Progress bar
    progress = (st.session_state.current_idx + 1) / total
    st.progress(progress, text=f"Entry {st.session_state.current_idx + 1} / {total}")

    # Load and display current entry
    st.divider()
    current_entry = load_entry_at_index(Path(file_path), st.session_state.current_idx)
    display_entry(current_entry)

    # Quick jump section
    with st.sidebar:
        st.header("Quick Jump")

        # Jump to specific index
        jump_idx = st.number_input(
            "Jump to index:", min_value=0, max_value=total - 1, value=st.session_state.current_idx, step=1
        )
        if st.button("Go to index"):
            st.session_state.current_idx = jump_idx
            st.rerun()

        st.divider()
        st.info(f"💡 Total: {total} entries\n\n📊 Current: {st.session_state.current_idx + 1}")

        # Current entry info
        st.divider()
        st.subheader("Current Entry")
        st.write(f"**ID:** {current_entry.get('id', 'N/A')}")
        st.write(f"**Event:** {current_entry.get('event', 'N/A')}")
        st.write(f"**Code:** {current_entry.get('code', 'N/A')}")
        st.write(f"**User Label:** {current_entry.get('user_label', 'N/A')}")


if __name__ == "__main__":
    main()
