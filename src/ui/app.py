import streamlit as st
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.llm_integration import LLMClient
from src.clarification_agent import ClarificationAgent
from src.prompt_builder import PromptBuilder
from src.features.context_manager import scan_directory, read_key_files
from src.features.experiment_planner import generate_experiment_prompt_snippet
from src.features.idea_generator import generate_idea_and_prompt, generate_idea_questions, generate_raw_idea
from src.features.file_interface import read_project_file, get_file_metadata
import asyncio
import threading

def run_async(coro):
    """
    Safely execute async functions in a dedicated thread.
    This prevents conflicts with Streamlit's internal event loop.
    """
    result = []
    exception = []

    def target():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result.append(loop.run_until_complete(coro))
            loop.close()
        except Exception as e:
            exception.append(e)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()

    if exception:
        raise exception[0]
    return result[0]

# AI Page Config
st.set_page_config(
    page_title="AI Prompt Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .stTextArea textarea { font-family: 'IBM Plex Mono', monospace; }
    .main .block-container { padding-top: 2rem; }
    .stButton button { width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# State Initialization
if "intention" not in st.session_state: st.session_state.intention = ""
if "generated_prompt" not in st.session_state: st.session_state.generated_prompt = ""
if "current_questions" not in st.session_state: st.session_state.current_questions = []
if "qa_history" not in st.session_state: st.session_state.qa_history = []
if "clarification_status" not in st.session_state: st.session_state.clarification_status = "IDLE"
if "estimated_turns" not in st.session_state: st.session_state.estimated_turns = 0
if "project_context_str" not in st.session_state: st.session_state.project_context_str = ""
if "generated_idea" not in st.session_state: st.session_state.generated_idea = ""
if "idea_qa_history" not in st.session_state: st.session_state.idea_qa_history = []
if "idea_questions" not in st.session_state: st.session_state.idea_questions = []
if "idea_clarification_status" not in st.session_state: st.session_state.idea_clarification_status = "IDLE"
if "selected_files" not in st.session_state: st.session_state.selected_files = {}
if "discovered_files" not in st.session_state: st.session_state.discovered_files = []

def reset_state():
    st.session_state.generated_prompt = ""
    st.session_state.current_questions = []
    st.session_state.qa_history = []
    st.session_state.clarification_status = "IDLE"
    st.session_state.estimated_turns = 0

# Sidebar logic
try:
    temp_client = LLMClient()
    available_models = temp_client.list_available_models()
except Exception:
    available_models = ["gpt-5.2", "gemini-3", "claude-4.5", "o3-mini", "o1", "claude-3.5-sonnet", "gemini-2.0-flash"]

with st.sidebar:
    st.title("🤖 Configuration")
    selected_model = st.selectbox("Select Model", available_models, index=0)
    api_key = st.text_input("API Key (Optional)", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.divider()
    creativity_mode = st.toggle("🧠 Creativity Mode", help="Skip questions and use agent creativity.")
    
    st.divider()
    st.info("Target: CS Researchers\nHigh-quality, context-aware prompt generation.")

# Initialize Core Services
try:
    client = LLMClient(default_model=selected_model)
    clarifier = ClarificationAgent(client)
    builder = PromptBuilder(client)
except Exception as e:
    st.error(f"Error initializing services: {e}")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["🆕 New Project", "🛠 Evolve Project", "📄 Paper to Code"])

def skip_question_callback(key):
    st.session_state[key] = "[User skipped this question]"

# --- Tab 1: New Project ---
with tab1:
    st.header("Start a New Project")
    user_intention = st.text_area("What do you want to build?", height=100, placeholder="e.g., A distributed key-value store in Go...")
    
    if st.button("Analyze & Start", key="btn_new"):
        reset_state()
        if user_intention:
            st.session_state.intention = user_intention
            if creativity_mode:
                st.session_state.clarification_status = "READY"
            else:
                with st.spinner("Analyzing requirements..."):
                    result = run_async(clarifier.analyze_status(user_intention))
                    st.session_state.clarification_status = result["status"]
                    st.session_state.current_questions = result["questions"]
                    st.session_state.estimated_turns = result.get("estimated_turns_remaining", 1)
            st.rerun()

    # Clarification Loop
    if st.session_state.clarification_status == "REFINING":
        st.subheader("Clarification Questions")
        st.info(f"⏳ Estimated turns remaining: **{st.session_state.estimated_turns}**")
        
        for i, q in enumerate(st.session_state.current_questions):
            q_key = f"q_input_{i}_{len(st.session_state.qa_history)}"
            
            col1, col2 = st.columns([5, 1])
            with col1:
                if q_key not in st.session_state:
                    st.session_state[q_key] = ""
                st.text_input(f"Q{i+1}: {q}", key=q_key)
            with col2:
                st.write(" ") # Padding
                st.write(" ")
                st.button("Skip", key=f"skip_{q_key}", on_click=skip_question_callback, args=(q_key,))
        
        if st.button("Evaluate All & Proceed", type="primary"):
            # Commit current answers to history
            for i, q in enumerate(st.session_state.current_questions):
                q_key = f"q_input_{i}_{len(st.session_state.qa_history)}"
                ans = st.session_state.get(q_key, "")
                st.session_state.qa_history.append({"q": q, "a": ans})
            
            with st.spinner("Re-evaluating technical clarity..."):
                result = run_async(clarifier.analyze_status(st.session_state.intention, st.session_state.qa_history))
                st.session_state.clarification_status = result["status"]
                st.session_state.current_questions = result["questions"]
                st.session_state.estimated_turns = result.get("estimated_turns_remaining", 1)
            st.rerun()

    # Generation
    if st.session_state.clarification_status == "READY":
        st.success("✅ Requirements clear!")
        if st.session_state.qa_history:
            with st.expander("Review Q&A History"):
                for item in st.session_state.qa_history:
                    st.markdown(f"**Q:** {item['q']}\n\n**A:** {item['a']}")
        
        mode_mapping = {
            "One-Shot (Production)": "one-shot",
            "Iterative (Evolutionary)": "iterative",
            "CoT (Research & Planning)": "chain-of-thought"
        }
        mode_label = st.radio("Development Strategy", list(mode_mapping.keys()), index=1, horizontal=True)
        
        if st.button("Build Final Prompt"):
            with st.spinner("🧠 AI is architecting your prompt..."):
                questions = [item['q'] for item in st.session_state.qa_history]
                answers = [item['a'] for item in st.session_state.qa_history]
                final_prompt, disc_paths = run_async(builder.build_prompt(
                    st.session_state.intention, answers, questions, mode=mode_mapping[mode_label]
                ))
                st.session_state.generated_prompt = final_prompt
                st.session_state.discovered_files = disc_paths

        if st.session_state.generated_prompt:
            if st.session_state.discovered_files:
                with st.expander("👀 View Autonomously Read Files"):
                    for f in st.session_state.discovered_files:
                        st.text(f"• {f}")
            st.code(st.session_state.generated_prompt, language="markdown")

# --- Tab 2: Evolve Project ---
with tab2:
    st.header("Project Evolution")
    
    # Context Scanning (Global for both modes)
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        project_path = st.text_input("Project Root Path", value=os.getcwd(), key="scan_path")
    with col_p2:
        st.write(" ")
        st.write(" ")
        if st.button("Scan Project"):
            with st.spinner("Scanning..."):
                st.session_state.project_context_str = scan_directory(project_path)
                key_files = read_key_files(project_path)
                for k, v in key_files.items():
                    st.session_state.project_context_str += f"\n\n--- {k} ---\n{v}"
                st.success("Scanned!")

    if st.session_state.project_context_str:
        col_ctx1, col_ctx2 = st.columns([1, 1])
        with col_ctx1:
            with st.expander("📂 Project Structure"):
                st.code(st.session_state.project_context_str)
        
        with col_ctx2:
            with st.expander("📄 File Explorer & Context Injector"):
                auto_discover = st.toggle("🤖 Autonomous Context Discovery", help="Agent will automatically pick and read relevant files.")
                
                st.divider()
                file_to_read = st.text_input("Enter relative path to read", placeholder="e.g., src/main.py")
                if st.button("Read & Add to Context"):
                    if file_to_read:
                        content = read_project_file(project_path, file_to_read)
                        if not content.startswith("Error:"):
                            st.session_state.selected_files[file_to_read] = content
                            st.success(f"Added {file_to_read} to context!")
                        else:
                            st.error(content)
                
                if st.session_state.selected_files:
                    st.write("**Included Files:**")
                    for f in list(st.session_state.selected_files.keys()):
                        col_f1, col_f2 = st.columns([4, 1])
                        col_f1.text(f"• {f}")
                        if col_f2.button("🗑", key=f"del_{f}"):
                            del st.session_state.selected_files[f]
                            st.rerun()

        # Augment project_context_str with selected file contents for the prompt
        augmented_context = st.session_state.project_context_str
        if st.session_state.selected_files:
            augmented_context += "\n\n### SELECTED FILE CONTENTS\n"
            for f, c in st.session_state.selected_files.items():
                augmented_context += f"\n--- FILE: {f} ---\n{c}\n"

        st.divider()
        
        # --- MODE SWITCHER ---
        evolution_mode = st.segmented_control(
            "Select Evolution Branch", 
            ["🔬 Experimentation Lab", "🏭 Feature Factory"],
            default="🔬 Experimentation Lab"
        )

        if evolution_mode == "🔬 Experimentation Lab":
            st.subheader("Scientific Research & Verification")
            st.info("Focus: Hypotheses, benchmarks, ablation studies, and robustness.")
            
            # --- Experimentation Lab UI ---
            exp_choice_tab = st.radio("Tooling", ["✨ AI Researcher", "🛠 Manual Design"], horizontal=True)
            
            if exp_choice_tab == "✨ AI Researcher":
                if st.button("✨ Brainstorm Next Experiment"):
                    with st.spinner("Analyzing project metrics..."):
                        idea = run_async(generate_raw_idea(client, st.session_state.project_context_str, "conduct experiment"))
                        st.session_state.generated_idea = idea
                        st.session_state.idea_clarification_status = "IDLE"
                
                if st.session_state.generated_idea:
                    st.session_state.generated_idea = st.text_area("**Research Proposal (Edit as needed):**", value=st.session_state.generated_idea, height=100)
                    if st.button("🔍 Design Experimental Protocol"):
                        if creativity_mode:
                            st.session_state.idea_clarification_status = "READY"
                        else:
                            with st.spinner("Defining variables and controls..."):
                                questions = run_async(generate_idea_questions(client, st.session_state.project_context_str, st.session_state.generated_idea, "conduct experiment"))
                                st.session_state.idea_questions = questions
                                st.session_state.idea_clarification_status = "REFINING"
            
            else: # Manual Design
                exp_int = st.text_input("What do you want to verify?", placeholder="e.g., Performance impact of RCU locks...")
                exp_tp = st.selectbox("Methodology", ["ablation", "hyperparameter_search", "robustness_test", "custom"])
                exp_par = st.text_input("Parameters/Variables", placeholder="e.g., thread_count=[1, 2, 4, 8]")
                
                if st.button("🏗 Architect Experiment"):
                    st.session_state.generated_idea = f"Manual {exp_tp}: {exp_int} (Params: {exp_par})"
                    if creativity_mode:
                        st.session_state.idea_clarification_status = "READY"
                    else:
                        with st.spinner("Synthesizing research questions..."):
                            questions = run_async(generate_idea_questions(client, st.session_state.project_context_str, st.session_state.generated_idea, "conduct experiment"))
                            st.session_state.idea_questions = questions
                            st.session_state.idea_clarification_status = "REFINING"

        else: # 🏭 Feature Factory
            st.subheader("High-Performance Feature Engineering")
            st.info("Focus: Architectural expansion, new capabilities, and integration.")
            
            # --- Feature Factory UI ---
            feat_choice_tab = st.radio("Tooling", ["✨ AI Architect", "✍️ Manual Specification"], horizontal=True)
            
            if feat_choice_tab == "✨ AI Architect":
                if st.button("✨ Brainstorm New Feature"):
                    with st.spinner("Scanning for architectural opportunities..."):
                        idea = run_async(generate_raw_idea(client, st.session_state.project_context_str, "new features"))
                        st.session_state.generated_idea = idea
                        st.session_state.idea_clarification_status = "IDLE"
                
                if st.session_state.generated_idea:
                    st.session_state.generated_idea = st.text_area("**Feature Proposal (Edit as needed):**", value=st.session_state.generated_idea, height=100)
                    if st.button("📐 Design System Architecture"):
                        if creativity_mode:
                            st.session_state.idea_clarification_status = "READY"
                        else:
                            with st.spinner("Mapping data flows and interfaces..."):
                                questions = run_async(generate_idea_questions(client, st.session_state.project_context_str, st.session_state.generated_idea, "new features"))
                                st.session_state.idea_questions = questions
                                st.session_state.idea_clarification_status = "REFINING"
            
            else: # Manual Specification
                custom_feat = st.text_area("Specify Feature", placeholder="Describe the new functionality you want to add...")
                if st.button("📐 Architect Custom Feature"):
                    st.session_state.generated_idea = custom_feat
                    if creativity_mode:
                        st.session_state.idea_clarification_status = "READY"
                    else:
                        with st.spinner("Analyzing architectural impact..."):
                            questions = run_async(generate_idea_questions(client, st.session_state.project_context_str, custom_feat, "new features"))
                            st.session_state.idea_questions = questions
                            st.session_state.idea_clarification_status = "REFINING"

        # --- SHARED CLARIFICATION LOOP (Universal UI) ---
        if st.session_state.idea_clarification_status == "REFINING":
            st.write("---")
            st.markdown("### 🗣 Technical Clarification")
            current_choice = "conduct experiment" if evolution_mode == "🔬 Experimentation Lab" else "new features"
            
            for i, q in enumerate(st.session_state.idea_questions):
                q_key = f"evolve_q_input_{i}"
                col_q, col_s = st.columns([5, 1])
                with col_q:
                    st.text_input(f"**{i+1}.** {q}", key=q_key)
                with col_s:
                    st.write(" ")
                    st.write(" ")
                    st.button("Skip", key=f"skip_evolve_{i}", on_click=skip_question_callback, args=(q_key,))
            
            if st.button("🚀 Generate Final Implementation Prompt", type="primary"):
                # Collect answers
                st.session_state.idea_qa_history = []
                for i, q in enumerate(st.session_state.idea_questions):
                    ans = st.session_state.get(f"evolve_q_input_{i}", "")
                    st.session_state.idea_qa_history.append({"q": q, "a": ans})

                with st.spinner("Combining context and intelligence..."):
                    final_prompt, disc_paths = run_async(generate_idea_and_prompt(
                        client, builder, augmented_context, 
                        current_choice, st.session_state.generated_idea, st.session_state.idea_qa_history,
                        root_path=project_path, auto_discover=auto_discover
                    ))
                    st.session_state.generated_prompt = final_prompt
                    st.session_state.discovered_files = disc_paths
                    st.session_state.idea_clarification_status = "READY"
                    st.rerun()

        if st.session_state.generated_prompt and st.session_state.get("idea_clarification_status") == "READY":
            st.divider()
            st.success("✨ Evolution Prompt Finalized!")
            if st.session_state.discovered_files:
                with st.expander("👀 View Autonomously Read Files"):
                    for f in st.session_state.discovered_files:
                        st.text(f"• {f}")
            st.code(st.session_state.generated_prompt, language="markdown")
            if st.button("🗑 Reset Evolution"):
                st.session_state.generated_idea = ""
                st.session_state.generated_prompt = ""
                st.session_state.discovered_files = []
                st.session_state.idea_clarification_status = "IDLE"
                st.session_state.idea_qa_history = []
                st.rerun()

# --- Tab 3: Paper to Code ---
with tab3:
    st.header("Paper Implementation")
    paper_content = st.text_area("Paste Abstract/Methodology", height=300)
    if st.button("Generate Implementation Plan"):
        if paper_content:
            with st.spinner("📄 Analyzing paper..."):
                final_prompt, disc_paths = run_async(builder.build_prompt(
                    intention="Implement method from paper", answers=[], questions=[], mode="iterative",
                    project_context=f"### Paper Content\n{paper_content}"
                ))
                st.session_state.generated_prompt = final_prompt
                st.session_state.discovered_files = disc_paths
    
    if st.session_state.generated_prompt and not st.session_state.intention: # Simple check for tab 3 context
        if st.session_state.discovered_files:
            with st.expander("👀 View Autonomously Read Files"):
                for f in st.session_state.discovered_files:
                    st.text(f"• {f}")
        st.code(st.session_state.generated_prompt, language="markdown")
    