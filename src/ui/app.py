import streamlit as st
import os
import sys
import json
import asyncio
import threading

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.llm_integration import LLMClient
from src.clarification_agent import ClarificationAgent
from src.prompt_builder import PromptBuilder
from src.features.context_manager import scan_directory, read_key_files
from src.features.experiment_planner import generate_experiment_prompt_snippet
from src.features.idea_generator import generate_idea_and_prompt, generate_idea_questions, generate_raw_idea
from src.features.file_interface import read_project_file, get_file_metadata
from src.features.research_journal import ResearchJournal, ResearchEntry
from src.features.academic_exporter import AcademicExporter
from src.features.prompt_refiner import PromptRefiner
from src.security_engine import SecurityEngine, SecurityState
from src.features.pdf_parser import extract_text_from_pdf

# Initialize Journal
journal = ResearchJournal()

def run_async(coro):
    """
    Safely execute async functions in a dedicated thread.
    Enhanced with timeout and better error propagation.
    """
    result = []
    exception = []

    def target():
        try:
            # Set a new event loop for this thread to avoid Streamlit's event loop conflicts
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Add a global task timeout of 180s to prevent orphan threads
            task = loop.create_task(coro)
            res = loop.run_until_complete(asyncio.wait_for(task, timeout=180.0))
            result.append(res)
            
            # Clean up the loop
            loop.close()
        except asyncio.TimeoutError:
            exception.append(TimeoutError("The operation took too long and was terminated."))
        except Exception as e:
            import traceback
            # Log the full traceback for cloud debugging
            print(f"Error in run_async thread: {e}")
            traceback.print_exc()
            exception.append(e)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join()

    if exception:
        # Reset stuck status if an error occurs
        st.session_state.clarification_status = "IDLE"
        st.session_state.idea_clarification_status = "IDLE"
        # Provide a more user-friendly error in the UI
        st.error(f"Background Task Error: {exception[0]}")
        raise exception[0]
    
    if not result:
        st.error("Background task completed but returned no result.")
        return None
        
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

# State Initialization (Fixed & Unified)
if "intention" not in st.session_state: st.session_state.intention = ""
if "generated_prompt" not in st.session_state: st.session_state.generated_prompt = ""
if "second_prompt" not in st.session_state: st.session_state.second_prompt = ""
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
if "paper_text" not in st.session_state: st.session_state.paper_text = ""

def reset_state():
    st.session_state.generated_prompt = ""
    st.session_state.second_prompt = ""
    st.session_state.current_questions = []
    st.session_state.qa_history = []
    st.session_state.clarification_status = "IDLE"
    st.session_state.idea_clarification_status = "IDLE"
    st.session_state.estimated_turns = 0
    st.session_state.discovered_files = []

# Sidebar logic
try:
    temp_client = LLMClient()
    available_models = temp_client.list_available_models()
except Exception:
    available_models = ["gpt-5.2", "gemini-3", "claude-4.5", "o3-mini", "o1", "claude-3.5-sonnet", "gemini-2.0-flash"]

with st.sidebar:
    st.title("🤖 Configuration")
    selected_model = st.selectbox("Select Primary Model", available_models, index=0)
    
    consensus_mode = st.toggle("🤝 Consensus Mode", help="Generate prompts from two models for comparison.")
    second_model = None
    if consensus_mode:
        second_model = st.selectbox("Select Secondary Model", available_models, index=min(1, len(available_models)-1))

    api_key = st.text_input("API Key (Optional)", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.divider()
    creativity_mode = st.toggle("🧠 Creativity Mode", help="Self-answer architectural questions.")
    
    st.divider()
    if st.button("🧹 Clear All States"):
        st.session_state.clear()
        st.rerun()
    
    st.info("Target: CS Researchers\nHigh-quality, context-aware prompt generation.")

# Initialize Core Services
try:
    client = LLMClient(default_model=selected_model)
    clarifier = ClarificationAgent(client)
    builder = PromptBuilder(client)
    refiner = PromptRefiner(client)
    security = SecurityEngine()
except Exception as e:
    st.error(f"Error initializing services: {e}")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🆕 New Project", "🛠 Evolve Project", "📄 Paper to Code", "🔬 Research Hub"])

def skip_question_callback(key):
    st.session_state[key] = "[User skipped this question]"

# --- Tab 1: New Project ---
with tab1:
    st.header("Start a New Project")
    user_intention = st.text_area("What do you want to build?", height=100, placeholder="e.g., A distributed key-value store in Go...", key="new_project_intent")
    
    if st.button("Analyze & Start", key="btn_new"):
        reset_state()
        if user_intention:
            st.session_state.intention = user_intention
            with st.status("🧠 Agent is architecting your project...", expanded=True) as status:
                st.write("Analyzing requirements...")
                result = run_async(clarifier.analyze_status(user_intention))
                
                if creativity_mode and result["status"] == "REFINING":
                    st.write("Creativity Mode: Self-answering architectural questions...")
                    self_qa = run_async(clarifier.self_answer_questions(user_intention, result["questions"]))
                    st.session_state.qa_history = self_qa
                    st.session_state.clarification_status = "READY"
                    status.update(label="✅ Project Architecture Scoped!", state="complete", expanded=False)
                else:
                    st.session_state.clarification_status = result["status"]
                    st.session_state.current_questions = result["questions"]
                    st.session_state.estimated_turns = result.get("estimated_turns_remaining", 1)
                    if result["status"] == "READY":
                        status.update(label="✅ Requirements Clear!", state="complete", expanded=False)
                    else:
                        status.update(label="🗣 Technical Clarification Required", state="complete", expanded=True)
            st.rerun()

    # Clarification Loop
    if st.session_state.clarification_status == "REFINING":
        st.subheader("Clarification Questions")
        st.info(f"⏳ Estimated turns remaining: **{st.session_state.estimated_turns}**")
        
        for i, q in enumerate(st.session_state.current_questions):
            q_key = f"q_input_{i}_{len(st.session_state.qa_history)}"
            col1, col2 = st.columns([5, 1])
            with col1:
                st.text_input(f"Q{i+1}: {q}", key=q_key)
            with col2:
                st.write(" ") # Padding
                st.write(" ")
                st.button("Skip", key=f"skip_{q_key}", on_click=skip_question_callback, args=(q_key,))
        
        if st.button("Evaluate All & Proceed", type="primary"):
            for i, q in enumerate(st.session_state.current_questions):
                q_key = f"q_input_{i}_{len(st.session_state.qa_history)}"
                ans = st.session_state.get(q_key, "")
                st.session_state.qa_history.append({"q": q, "a": ans})
            
            with st.status("Re-evaluating technical clarity...", expanded=True):
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
            with st.status("🧠 AI is architecting your prompt...", expanded=True) as status:
                questions = [item['q'] for item in st.session_state.qa_history]
                answers = [item['a'] for item in st.session_state.qa_history]
                
                # Fetch insights for the journal
                tree = scan_directory(os.getcwd())
                st.write("Investigating context...")
                insights = run_async(builder.discovery_agent.investigate_and_analyze(os.getcwd(), st.session_state.intention, tree))
                
                st.write("Generating primary prompt...")
                final_prompt, disc_paths = run_async(builder.build_prompt(
                    st.session_state.intention, answers, questions, mode=mode_mapping[mode_label]
                ))
                
                second_p = ""
                if consensus_mode and second_model:
                    st.write(f"Generating consensus with {second_model}...")
                    second_client = LLMClient(default_model=second_model)
                    second_builder = PromptBuilder(second_client)
                    second_p, _ = run_async(second_builder.build_prompt(
                        st.session_state.intention, answers, questions, mode=mode_mapping[mode_label]
                    ))

                st.session_state.generated_prompt = final_prompt
                st.session_state.second_prompt = second_p
                st.session_state.discovered_files = disc_paths
                
                # Save to Journal
                entry = ResearchEntry(
                    intention=st.session_state.intention,
                    mode=mode_mapping[mode_label],
                    insights=insights,
                    final_prompt=final_prompt,
                    tags=["new-project", f"model:{selected_model}"]
                )
                if second_p:
                    entry.tags.append(f"consensus:{second_model}")
                    entry.metrics["second_prompt"] = second_p
                journal.add_entry(entry)
                status.update(label="✨ Prompt Built!", state="complete")

        if st.session_state.generated_prompt:
            if st.session_state.second_prompt:
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.subheader(f"Model: {selected_model}")
                    st.code(st.session_state.generated_prompt, language="markdown")
                with col_p2:
                    st.subheader(f"Model: {second_model}")
                    st.code(st.session_state.second_prompt, language="markdown")
            else:
                st.code(st.session_state.generated_prompt, language="markdown")
            
            # Interactive Refinement
            st.divider()
            st.subheader("💬 Interactive Refinement")
            refine_input = st.chat_input("Suggest a change (e.g., 'Make it more modular')", key="chat_tab1")
            if refine_input:
                with st.status("Refining prompt...", expanded=True):
                    new_p = run_async(refiner.refine_prompt(st.session_state.generated_prompt, refine_input))
                    st.session_state.generated_prompt = new_p
                st.rerun()

# --- Tab 2: Evolve Project ---
with tab2:
    st.header("Project Evolution")
    
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        project_path = st.text_input("Project Root Path", value=os.getcwd(), key="scan_path")
    with col_p2:
        st.write(" ")
        st.write(" ")
        if st.button("Scan Project"):
            with st.status("Scanning project structure...", expanded=True):
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
            with st.expander("📄 Context Injector"):
                auto_discover = st.toggle("🤖 Autonomous Context Discovery", value=True)
                st.divider()
                file_to_read = st.text_input("Enter relative path to read", placeholder="e.g., src/main.py")
                if st.button("Read & Add"):
                    if file_to_read:
                        content = read_project_file(project_path, file_to_read)
                        if not content.startswith("Error:"):
                            st.session_state.selected_files[file_to_read] = content
                            st.success(f"Added {file_to_read}")
                        else:
                            st.error(content)
                if st.session_state.selected_files:
                    for f in list(st.session_state.selected_files.keys()):
                        if st.button(f"🗑 {f}", key=f"del_{f}"):
                            del st.session_state.selected_files[f]
                            st.rerun()

        # Augment context
        augmented_context = st.session_state.project_context_str
        if st.session_state.selected_files:
            augmented_context += "\n\n### SELECTED FILE CONTENTS\n"
            for f, c in st.session_state.selected_files.items():
                augmented_context += f"\n--- FILE: {f} ---\n{c}\n"

        st.divider()
        evolution_mode = st.segmented_control("Branch", ["🔬 Experimentation Lab", "🏭 Feature Factory"], default="🔬 Experimentation Lab")
        current_choice = "conduct experiment" if evolution_mode == "🔬 Experimentation Lab" else "new features"

        # Evolution Flows
        if evolution_mode == "🔬 Experimentation Lab":
            choice_tab = st.radio("Tool", ["✨ AI Researcher", "🛠 Manual"], horizontal=True)
            if choice_tab == "✨ AI Researcher":
                if st.button("✨ Brainstorm"):
                    with st.status("Thinking..."):
                        st.session_state.generated_idea = run_async(generate_raw_idea(client, augmented_context, current_choice))
                if st.session_state.generated_idea:
                    st.session_state.generated_idea = st.text_area("Proposal", value=st.session_state.generated_idea)
                    if st.button("🔍 Design Protocol"):
                        with st.status("Designing...", expanded=True) as status:
                            qs = run_async(generate_idea_questions(client, augmented_context, st.session_state.generated_idea, current_choice))
                            if creativity_mode:
                                self_qa = run_async(clarifier.self_answer_questions(st.session_state.generated_idea, qs))
                                st.session_state.idea_qa_history = self_qa
                                st.session_state.idea_clarification_status = "READY_AUTO"
                            else:
                                st.session_state.idea_questions = qs
                                st.session_state.idea_clarification_status = "REFINING"
                        st.rerun()
            else:
                exp_int = st.text_input("Intention", placeholder="e.g. Test RCU locks")
                if st.button("🏗 Architect"):
                    st.session_state.generated_idea = exp_int
                    with st.status("Architecting..."):
                        qs = run_async(generate_idea_questions(client, augmented_context, exp_int, current_choice))
                        if creativity_mode:
                            st.session_state.idea_qa_history = run_async(clarifier.self_answer_questions(exp_int, qs))
                            st.session_state.idea_clarification_status = "READY_AUTO"
                        else:
                            st.session_state.idea_questions = qs
                            st.session_state.idea_clarification_status = "REFINING"
                    st.rerun()
        else: # Feature Factory
            choice_tab = st.radio("Tool", ["✨ AI Architect", "✍️ Manual"], horizontal=True)
            if choice_tab == "✨ AI Architect":
                if st.button("✨ Brainstorm Feature"):
                    with st.status("Thinking..."):
                        st.session_state.generated_idea = run_async(generate_raw_idea(client, augmented_context, current_choice))
                if st.session_state.generated_idea:
                    st.session_state.generated_idea = st.text_area("Feature", value=st.session_state.generated_idea)
                    if st.button("📐 Design Architecture"):
                        with st.status("Designing..."):
                            qs = run_async(generate_idea_questions(client, augmented_context, st.session_state.generated_idea, current_choice))
                            if creativity_mode:
                                self_qa = run_async(clarifier.self_answer_questions(st.session_state.generated_idea, qs))
                                st.session_state.idea_qa_history = self_qa
                                st.session_state.idea_clarification_status = "READY_AUTO"
                            else:
                                st.session_state.idea_questions = qs
                                st.session_state.idea_clarification_status = "REFINING"
                        st.rerun()
            else:
                feat_int = st.text_area("Specify Feature")
                if st.button("📐 Architect Custom"):
                    st.session_state.generated_idea = feat_int
                    with st.status("Architecting..."):
                        qs = run_async(generate_idea_questions(client, augmented_context, feat_int, current_choice))
                        if creativity_mode:
                            st.session_state.idea_qa_history = run_async(clarifier.self_answer_questions(feat_int, qs))
                            st.session_state.idea_clarification_status = "READY_AUTO"
                        else:
                            st.session_state.idea_questions = qs
                            st.session_state.idea_clarification_status = "REFINING"
                    st.rerun()

        # Shared Clarification/Generation Logic
        if st.session_state.idea_clarification_status == "READY_AUTO":
            with st.status("🚀 Architecting Evolution...", expanded=True) as status:
                st.write("Combining context...")
                final_p, d_paths = run_async(generate_idea_and_prompt(
                    client, builder, augmented_context, current_choice, 
                    st.session_state.generated_idea, st.session_state.idea_qa_history,
                    root_path=project_path, auto_discover=auto_discover
                ))
                st.write("Generating insights...")
                ins = run_async(builder.discovery_agent.investigate_and_analyze(project_path, st.session_state.generated_idea, augmented_context))
                st.session_state.generated_prompt = final_p
                st.session_state.discovered_files = d_paths
                st.session_state.idea_clarification_status = "READY"
                journal.add_entry(ResearchEntry(intention=st.session_state.generated_idea, mode=current_choice, insights=ins, final_prompt=final_p, tags=["evolution", current_choice, "auto"]))
                status.update(label="✅ Finalized!", state="complete")
            st.rerun()

        if st.session_state.idea_clarification_status == "REFINING":
            st.subheader("🗣 Technical Clarification")
            for i, q in enumerate(st.session_state.idea_questions):
                q_key = f"evolve_q_{i}"
                st.text_input(q, key=q_key)
            if st.button("🚀 Generate Final Implementation Prompt"):
                st.session_state.idea_qa_history = [{"q": q, "a": st.session_state.get(f"evolve_q_{i}", "")} for i, q in enumerate(st.session_state.idea_questions)]
                st.session_state.idea_clarification_status = "READY_AUTO"
                st.rerun()

        if st.session_state.generated_prompt and st.session_state.idea_clarification_status == "READY":
            st.code(st.session_state.generated_prompt, language="markdown")
            if st.button("🗑 Reset Evolution"):
                reset_state()
                st.session_state.generated_idea = ""
                st.rerun()

# --- Tab 3: Paper Implementation ---
with tab3:
    st.header("Paper Implementation")
    uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_pdf and st.button("Parse PDF"):
        with st.status("Parsing..."):
            st.session_state.paper_text = extract_text_from_pdf(uploaded_pdf)
    paper_input = st.text_area("Paper Content", value=st.session_state.paper_text, height=300)
    if st.button("Generate Plan"):
        if paper_input:
            with st.status("Analyzing..."):
                final_p, _ = run_async(builder.build_prompt("Implement paper", [], [], mode="iterative", project_context=f"### Paper\n{paper_input}"))
                st.session_state.generated_prompt = final_p
    if st.session_state.generated_prompt:
        st.code(st.session_state.generated_prompt, language="markdown")

# --- Tab 4: Research Hub ---
with tab4:
    st.header("Research Journal")
    entries = journal.get_entries()
    for e in reversed(entries):
        with st.expander(f"📌 {e.timestamp} | {e.intention[:50]}..."):
            st.markdown(f"**Insights**: {e.insights}")
            st.code(e.final_prompt[:1000], language="markdown")
            if st.button("🛡 Privacy Audit", key=f"sec_{e.entry_id}"):
                with st.status("Auditing..."):
                    res = run_async(security.process_content(e.final_prompt))
                    if res.threat_level == "LOW": st.success("Pass")
                    else: st.error(f"Threat: {res.threat_level}")
            if st.button("📄 Export LaTeX", key=f"tex_{e.entry_id}"):
                st.text_area("LaTeX", AcademicExporter.to_latex_methodology(e.insights or "", e.intention))
    if st.button("🗑 Clear Journal"):
        if os.path.exists(journal.storage_path): os.remove(journal.storage_path)
        journal._ensure_storage()
        st.rerun()
