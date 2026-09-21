import argparse
import json
import random
import re
from pathlib import Path

import streamlit as st

# Page configuration
st.set_page_config(page_title="Rollout Trajectory Viewer", page_icon="🚀", layout="wide")


def parse_conversation_string(conv_str: str) -> list[dict[str, str]]:
    """
    Parse a conversation string into system / user / assistant messages.
    Format: <|im_start|>role\ncontent<|im_end|>
    """
    messages = []
    # Match the <|im_start|>role\ncontent<|im_end|> pattern
    pattern = r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>"
    matches = re.findall(pattern, conv_str, re.DOTALL)

    for role, content in matches:
        messages.append({"role": role, "content": content.strip()})

    return messages


def extract_thought_and_tools(assistant_message: str) -> tuple[str, list[str]]:
    """
    Extract the thought and the tool calls from an assistant message
    """
    thought = ""
    tools = []

    # Thought: a <think>...</think> block or a THOUGHT: prefix
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, assistant_message, re.DOTALL)
    if think_match:
        thought = think_match.group(1).strip()
    else:
        # Look for a THOUGHT: prefix
        thought_pattern = r"THOUGHT:\s*(.*?)(?=\n\s*<tool|\n\s*$|$)"
        thought_match = re.search(thought_pattern, assistant_message, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

    # Tool calls
    tool_pattern = r'<tool name="([^"]*)">(.*?)</tool>'
    tool_matches = re.findall(tool_pattern, assistant_message, re.DOTALL)

    for tool_name, tool_content in tool_matches:
        tools.append(f"Tool: {tool_name}\n{tool_content.strip()}")

    return thought, tools


def calculate_stats(rollout_prefix: str, rollout_data: dict) -> dict[str, float]:
    """Statistics for one data item."""
    if not rollout_data or "outputs" not in rollout_data:
        return {"passrate": 0.0, "thought_ratio": 0.0, "total_samples": 0}

    total_thought_chars = 0
    total_all_chars = 0
    total_trajectories = 0
    passed_trajectories = 0

    # Prefix messages
    prefix_messages = parse_conversation_string(rollout_prefix)
    for msg in prefix_messages:
        if msg["role"] == "assistant":
            thought, _ = extract_thought_and_tools(msg["content"])
            total_thought_chars += len(thought)
            total_all_chars += len(msg["content"])

    # Each trajectory
    for reward_score, trajectory in rollout_data["outputs"]:
        total_trajectories += 1
        # A positive reward counts as a pass
        if reward_score > 0:
            passed_trajectories += 1

        traj_messages = parse_conversation_string(trajectory)
        for msg in traj_messages:
            if msg["role"] == "assistant":
                thought, _ = extract_thought_and_tools(msg["content"])
                total_thought_chars += len(thought)
                total_all_chars += len(msg["content"])

    # Pass rate
    passrate = (passed_trajectories / total_trajectories) if total_trajectories > 0 else 0.0
    thought_ratio = (total_thought_chars / total_all_chars * 100) if total_all_chars > 0 else 0.0

    return {"passrate": passrate, "thought_ratio": thought_ratio, "total_samples": total_trajectories}


def calculate_overall_stats(data: list[tuple[str, dict]]) -> dict[str, float]:
    """Aggregate statistics over all samples."""
    total_passrate = 0.0
    valid_samples = 0

    for rollout_prefix, rollout_data in data:
        if rollout_data and "outputs" in rollout_data:
            stats = calculate_stats(rollout_prefix, rollout_data)
            total_passrate += stats["passrate"]
            valid_samples += 1

    avg_passrate = (total_passrate / valid_samples) if valid_samples > 0 else 0.0

    return {"avg_passrate": avg_passrate, "valid_samples": valid_samples}


def load_json_file(file_path: str) -> list[tuple[str, dict]] | None:
    """Load a JSON file and return a list of (prefix, rollout_data)."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read().strip()
            data_obj = json.loads(content)
            # A dict becomes one list item per key/value pair
            if isinstance(data_obj, dict):
                return [(key, value) for key, value in data_obj.items()]
            else:
                st.error("Unsupported data format: expected a dict")
                return None
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"JSON parse error: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def render_message(role: str, content: str, expanded: bool = True):
    """Render one message."""
    if role == "system":
        with st.expander("🔧 System message", expanded=True):
            st.text(content)
    elif role == "user":
        with st.expander("👤 User message", expanded=False):
            st.text(content)
    elif role == "assistant":
        thought, tools = extract_thought_and_tools(content)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("🤖 **Assistant message**")

        if thought:
            with st.expander("💭 Thought", expanded=True):
                st.info(thought)

        if tools:
            with st.expander(f"🔧 Tool calls ({len(tools)})", expanded=False):
                for i, tool in enumerate(tools):
                    st.code(tool, language="xml")
                    if i < len(tools) - 1:
                        st.divider()


def render_rollout_prefix(rollout_prefix: str):
    """Render the ROLLOUT_PREFIX section."""
    st.markdown("### 🎯 ROLLOUT PREFIX")
    messages = parse_conversation_string(rollout_prefix)

    for msg in messages:
        render_message(msg["role"], msg["content"])

    st.markdown("---")


def render_rollout_trajectory(trajectory_str: str, title: str = "Trajectory"):
    """Render one ROLLOUT_TRAJECTORY."""
    st.markdown(f"#### {title}")
    messages = parse_conversation_string("<|im_end|>\n<|im_start|>assistant\n" + trajectory_str)

    if not messages:
        st.warning("⚠️ Could not parse the trajectory content")
        return

    for msg in messages:
        render_message(msg["role"], msg["content"])


def main():
    st.title("🚀 Rollout Trajectory Viewer")

    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Path to the input JSON file")
    args, _ = parser.parse_known_args()

    # Sidebar
    st.sidebar.header("📁 Input file")
    input_file = st.sidebar.text_input(
        "Input JSON file path",
        value=args.input_file if args.input_file else "",
        placeholder="e.g. /path/to/rollout_data.json",
    )

    if not input_file or not Path(input_file).exists():
        st.info("📂 Enter a valid JSON file path to start")
        return

    # Reset the state when the file changes
    if "last_input_file" not in st.session_state or st.session_state.last_input_file != input_file:
        st.session_state.last_input_file = input_file
        st.session_state.overall_stats = None
        st.session_state.current_page = 1
        st.session_state.left_traj_index = 0
        st.session_state.right_traj_index = 1

    # Load the data
    data = load_json_file(input_file)
    if data is None:
        return

    st.success(f"✅ Loaded {len(data)} items")

    # Aggregate statistics (computed once)
    if st.session_state.overall_stats is None:
        with st.spinner("Computing aggregate statistics..."):
            st.session_state.overall_stats = calculate_overall_stats(data)

    overall_stats = st.session_state.overall_stats

    # Initialize the session state
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    if "left_traj_index" not in st.session_state:
        st.session_state.left_traj_index = 0
    if "right_traj_index" not in st.session_state:
        st.session_state.right_traj_index = 1

    # Paging controls
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        if st.button("⬅️ Previous page"):
            if st.session_state.current_page > 1:
                st.session_state.current_page -= 1
                st.session_state.left_traj_index = 0
                st.session_state.right_traj_index = 1
                st.rerun()

    with col2:
        if st.button("➡️ Next page"):
            if st.session_state.current_page < len(data):
                st.session_state.current_page += 1
                st.session_state.left_traj_index = 0
                st.session_state.right_traj_index = 1
                st.rerun()

    with col3:
        if st.button("🎲 Random page"):
            st.session_state.current_page = random.randint(1, len(data))
            st.session_state.left_traj_index = 0
            st.session_state.right_traj_index = 1
            st.rerun()

    # Current page data (prefix, rollout_data)
    rollout_prefix, rollout_data = data[st.session_state.current_page - 1]

    # Statistics for the current page
    current_stats = calculate_stats(rollout_prefix, rollout_data)

    # Show statistics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Page", f"{st.session_state.get('current_page', 1)}")
    with col2:
        st.metric("Pages", len(data))
    with col3:
        st.metric("Trajectories", current_stats["total_samples"])
    with col4:
        st.metric("Overall pass rate", f"{overall_stats['avg_passrate']:.3f}")
    with col5:
        st.metric("Page pass rate", f"{current_stats['passrate']:.3f}")
    with col6:
        st.metric("Thought ratio", f"{current_stats['thought_ratio']:.1f}%")

    # Validate the data
    if not rollout_data or "outputs" not in rollout_data:
        st.error("❌ No valid rollout data in the current item")
        return

    # Render ROLLOUT_PREFIX
    render_rollout_prefix(rollout_prefix)

    # Render the ROLLOUT_TRAJECTORY area
    st.markdown("### 🎯 ROLLOUT TRAJECTORIES")

    trajectories = rollout_data["outputs"]
    if not trajectories:
        st.warning("⚠️ The current item has no trajectory output")
        return

    # Navigation buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        if st.button("⬅️ Left: previous"):
            if st.session_state.left_traj_index > 0:
                st.session_state.left_traj_index -= 1
                st.rerun()

    with col2:
        if st.button("➡️ Left: next"):
            if st.session_state.left_traj_index < len(trajectories) - 1:
                st.session_state.left_traj_index += 1
                st.rerun()

    with col3:
        if st.button("⬅️ Right: previous"):
            if st.session_state.right_traj_index > 0:
                st.session_state.right_traj_index -= 1
                st.rerun()

    with col4:
        if st.button("➡️ Right: next"):
            if st.session_state.right_traj_index < len(trajectories) - 1:
                st.session_state.right_traj_index += 1
                st.rerun()

    # Current indices
    st.info(
        f"Left: trajectory {st.session_state.left_traj_index + 1}/{len(trajectories)} | Right: trajectory {st.session_state.right_traj_index + 1}/{len(trajectories)}"
    )

    # Two columns of trajectories
    left_col, right_col = st.columns(2)

    with left_col:
        if st.session_state.left_traj_index < len(trajectories):
            left_reward, left_trajectory = trajectories[st.session_state.left_traj_index]
            st.markdown(f"**🏆 Reward: {left_reward}**")
            render_rollout_trajectory(left_trajectory, f"Left trajectory {st.session_state.left_traj_index + 1}")
        else:
            st.warning("⚠️ Left trajectory index out of range")

    with right_col:
        if st.session_state.right_traj_index < len(trajectories):
            right_reward, right_trajectory = trajectories[st.session_state.right_traj_index]
            st.markdown(f"**🏆 Reward: {right_reward}**")
            render_rollout_trajectory(right_trajectory, f"Right trajectory {st.session_state.right_traj_index + 1}")
        else:
            st.warning("⚠️ Right trajectory index out of range")


if __name__ == "__main__":
    main()
