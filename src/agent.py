"""
LegisYukti Agent - LangGraph-based RAG System for Indian Law

This module implements a sophisticated legal research agent using LangGraph state machines.
The agent orchestrates the complete RAG pipeline: query routing → legal research → response generation → quality auditing.

Key Components:
- ChatState: TypedDict managing conversation state and metadata
- verify_citations_node: Post-generation audit for legal accuracy
- retrieve_legal_context: Tool for searching legal documents
- generate_response_node: Final answer synthesis with statutory grounding
- call_tools_and_save_context: Tool execution and context management
- chat_node: Intelligent routing and law classification
- evaluate_response_node: Quality scoring and feedback generation
"""

import re
import time
from loguru import logger
from .engine import get_retriever
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from .config import NO_CONTEXT_MSG, llm, fast_llm
from .utils import clean_feedback, prune_legal_context
from typing import Annotated, Optional, TypedDict, Union
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from .prompts import get_qa_prompt, get_auditor_prompt, get_chat_persona_prompt, get_followup_classifier_prompt, get_router_prompt

# Core state management for the legal conversation graph
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # Conversation history with automatic merging
    context: list[str]  # Retrieved legal context snippets
    evaluation_score: int  # Quality score (0-10) from auditor
    evaluation_feedback: str  # Detailed feedback on response quality
    retry_count: int  # Number of regeneration attempts
    law_filter: Optional[str]  # Targeted legal act filter for search
    intent: str  # Query classification: "LEGAL" or "CHAT"
    is_followup: bool  # Whether this is a follow-up question


def verify_citations_node(state: ChatState):
    """
    Citation Verification Auditor - Ensures legal responses are grounded in actual statutes.

    This critical quality control node audits AI-generated legal advice against retrieved context.
    It prevents hallucinations by cross-referencing citations with verified legal sources.

    Process:
    1. Extract the proposed legal advice from conversation
    2. Prune retrieved context to manageable size (6000 chars)
    3. Use fast LLM to audit for accuracy and proper citations
    4. Return either verified response or corrected version

    Args:
        state: Current conversation state with messages and context

    Returns:
        Updated state with verified/corrected legal response
    """
    logger.info("🛡️ Auditor: Starting citation verification process...")

    proposed_advice = state["messages"][-1].content
    raw_context = "\n\n".join(state.get("context", []))
    context_used = prune_legal_context(raw_context, max_chars=6000)

    start_time = time.time()
    try:
        chain = get_auditor_prompt() | fast_llm
        audit_result = chain.invoke({
            "advice": proposed_advice,
            "context": context_used
        })
        duration = time.time() - start_time

        audit_text = audit_result.content.strip()

        # VERDICT-BASED RESPONSE HANDLING
        # Auditor uses emoji prefixes to indicate verification status
        if audit_text.startswith("✅"):
            # Scenario A: Advice is perfect - use original response
            final_content = proposed_advice
            logger.success(f"✅ Auditor: Citations verified in {duration:.2f}s.")
        elif audit_text.startswith("🚨"):
            # Scenario B: Corrections needed - use auditor's refined version
            final_content = audit_text.replace("🚨", "").replace("REFINED ADVICE:", "").strip()
            logger.warning(f"🚨 Auditor: Hallucination corrected in {duration:.2f}s.")
        else:
            # Fallback: Auditor didn't use expected format
            final_content = audit_text
            logger.info("🛡️ Auditor: Processed without explicit verdict symbol.")

        # Remove any remaining meta-commentary from the response
        final_content = re.sub(r"^(The provided advice is sound|I have corrected).*?\n+", "", final_content, flags=re.IGNORECASE).strip()

        return {"messages": [AIMessage(content=final_content)]}

    except Exception as e:
        logger.warning(f"⚠️ Auditor bypassed due to API error: {str(e)}")
        # On audit failure, pass through original advice rather than crash
        return {"messages": [state["messages"][-1]]}

@tool
def retrieve_legal_context(query: str, law_filter: Optional[Union[str, list[str]]] = None):
    """
    High-density legal document search tool with intelligent filtering and fallback logic.

    This tool performs semantic search across the Indian legal knowledge base using
    hybrid retrieval (dense + sparse) with contextual compression and reranking.

    Features:
    - Multi-query expansion for comprehensive search
    - Payload filtering by legal act (e.g., "BNS", "CPC")
    - Dynamic thresholding based on query complexity
    - Succession law redirection (ISA over HSA)
    - Civil vs Criminal domain sanitization
    - Global fallback for failed filtered searches

    Args:
        query: User's legal question
        law_filter: Optional filter for specific legal acts

    Returns:
        Formatted string of verified legal references with citations
    """
    logger.info(f"🔍 AI is investigating: '{query}' | Filter: {law_filter}")

    active_filter = None

    # INTELLIGENT FILTERING LOGIC
    # Optimize search strategy based on filter complexity
    if isinstance(law_filter, list):
        if len(law_filter) > 2:
            # Skip filtered search for complex multi-act queries to save time
            active_filter = None
            logger.info("🚀 High complexity detected: Using Global Search for speed.")
        elif len(law_filter) > 0:
            active_filter = law_filter[0]
    else:
        # Single string filter or None
        active_filter = law_filter

    # CLEAN CONSOLIDATED QUERIES
    # Remove internal consolidation markers from multi-query expansion
    search_query = query.split(" | ")[0] if " | " in query else query

    start_time = time.time()
    docs = get_retriever(fast_llm, law_name_filter=active_filter).invoke(search_query)

    # GLOBAL FALLBACK: If filtered search returns insufficient results
    if not docs:
        logger.warning("⚠️ High-density filter returned insufficient context. Re-executing Global Search.")
        docs = get_retriever(fast_llm, law_name_filter=None).invoke(search_query)

    duration = time.time() - start_time
    logger.info(f"⏱️ Retrieval completed in {duration:.2f}s. Evaluating {len(docs)} candidates...")

    verified_references = []
    for i, doc in enumerate(docs):
        score = doc.metadata.get("relevance_score", 0.0)
        law_ref = doc.metadata.get("law_name", "Unknown Law").lower()
        section = doc.metadata.get("section", "N/A")

        # SUCCESSION LAW REDIRECTION BOOST
        # Prioritize Indian Succession Act over Hindu Succession Act for inheritance queries
        if any(x in query.lower() for x in ["inheritance", "succession", "will", "flat"]):
            if "indian succession" in law_ref:
                score += 0.05

        # DYNAMIC THRESHOLDING BASED ON LEGAL DOMAIN
        # Adjust strictness based on whether the law deals with civil/family matters
        is_civil_or_cyber = any(x in law_ref for x in ["marriage", "civil", "contract", "property", "it act", "succession", "wage", "technology", "cyber", "ndps", "constitution", "cpc"])

        if active_filter is None:  # Global Search - be more permissive
            threshold = 0.08
        else:  # Filtered Search - be more surgical
            threshold = 0.12 if is_civil_or_cyber else 0.22  # Civil laws need precision

        if score >= threshold:
            # Truncate content to prevent token overflow while preserving key information
            truncated_content = doc.page_content[:700].strip() + "..."
            verified_references.append(f"--- VERIFIED REFERENCE: {law_ref.upper()} (Section {section}) ---\n{truncated_content}\n")

    # FORCE-ADD HIGH-SCORING CANDIDATES IF CONTEXT IS STARVED
    if len(verified_references) < 3 and active_filter is None:
        logger.warning("⚠️ Context starved. Force-adding top 3 candidates.")
        for doc in docs[:3]:
            law_ref = doc.metadata.get("law_name", "Unknown Law").upper()
            verified_references.append(f"--- VERIFIED REFERENCE: {law_ref} ---\n{doc.page_content[:500]}...")


    logger.success(f"⚖️ Found {len(verified_references)} high-confidence snippets.")
    return "\n\n".join(verified_references)

def generate_response_node(state: ChatState):
    """
    Final Legal Response Synthesis - Combines retrieved context with statutory expertise.

    This node generates the final legal advice by synthesizing verified context with
    specialized legal knowledge. It handles both legal queries and general conversation,
    with special logic for follow-up questions and temporal legal considerations.

    Process:
    1. Determine query intent (legal vs general chat)
    2. For chat: Use persona-guided response
    3. For legal: Synthesize advice with statutory grounding
    4. Add temporal hints for pre-2024 incidents
    5. Include source citations and disclaimers

    Args:
        state: Conversation state with context and metadata

    Returns:
        State update with final synthesized response
    """
    intent = state.get("intent", "LEGAL")
    is_followup = state.get("is_followup", False)

    # HANDLE GENERAL CONVERSATION WITH LEGAL PERSONA
    if intent == "CHAT":
        logger.info("👋 Pivot: Handling small talk with Persona Guardrail.")
        last_msg = state["messages"][-1].content

        # Use specialized persona prompt to keep AI focused on legal expertise
        chat_prompt = get_chat_persona_prompt()
        response_content = fast_llm.invoke(
            chat_prompt.format(user_input=last_msg)
        ).content
        return {
            "messages": [AIMessage(content=response_content)],
            "evaluation_score": 10,  # Auto-pass for non-legal chat
            "evaluation_feedback": "Professional pivot delivered."
        }

    # LEGAL RESPONSE SYNTHESIS
    logger.info("📝 Final Answer: Synthesizing legal guidance...")

    # Count references for logging
    raw_context = state.get("context", [])
    # Prune context to stay within token limits while preserving legal integrity
    full_context = prune_legal_context(raw_context, max_chars=6000)

    # Extract question for logging and temporal analysis
    try:
        question = [m.content for m in state["messages"] if isinstance(m, HumanMessage)][-1]
        # Temporal guardrail: Detect pre-2024 incidents requiring IPC/CrPC references
        is_old_law = any(yr in question for yr in ["2020", "2021", "2022", "2023"])
    except (IndexError, AttributeError, StopIteration):
        question = "the user's request"
        is_old_law = False

    logger.info(f"📊 CONTEXT READY: {len(full_context)} chars. Question: '{question[:50]}...'")

    if NO_CONTEXT_MSG in full_context or not full_context.strip():
        logger.warning(f"⚠️ No context found for: '{question[:50]}...'")
        return {"messages": [AIMessage(content="I'm sorry, I couldn't find a high-confidence reference.")]}

    logger.debug(f"📚 Synthesizing from {len(full_context)} verified snippets")

    # ENHANCED QUESTION WITH TEMPORAL AND FOLLOWUP CONTEXT
    temporal_hint = " (Note: Incident is pre-July 2024, prioritize IPC/CrPC)" if is_old_law else ""
    followup_instruction = "\n(Instruction: This is a follow-up question. Use the provided context and previous discussion to clarify or expand.)" if is_followup else ""

    chain = get_qa_prompt() | llm
    start_time = time.time()
    response = chain.invoke({
        "context": full_context,
        "question": f"{question}{temporal_hint}{followup_instruction}"
    })

    # EMERGENCY MAPPING: Handle cases where LLM refuses context
    if any(phrase in response.content for phrase in ["haven't provided", "specific legal query", "I cannot"]):
        logger.warning("🚨 Advisor tried to refuse context. Triggering emergency mapping.")
        # Direct prompt bypass to force context usage
        emergency_prompt = f"Using this context: {full_context}\n\nAnswer this specific legal question: {question}"
        response = llm.invoke(emergency_prompt)

    # SOURCE CITATION ASSEMBLY
    # Extract unique source names from verified reference markers
    sources = set()
    source_matches = re.findall(r"VERIFIED REFERENCE:\s*([^(|\n]+)", full_context)
    for s in source_matches:
        sources.add(s.strip())

    # Format source footer if sources found
    source_footer = "\n\n**Verified Sources Analyzed:**\n" + "\n".join([f"• {s}" for s in sorted(sources)]) if sources else ""

    # MANDATORY LEGAL DISCLAIMER
    disclaimer = (
        "\n\n---\n"
        "**Disclaimer:** *This response is generated by an AI research assistant based on "
        "available legal documents in public domain. It does not "
        "constitute formal legal advice. Laws are subject to specific judicial interpretations. "
        "Consult with a qualified legal professional before taking action.*"
    )

    # Assemble final response with sources and disclaimer
    response.content += source_footer
    response.content += disclaimer

    logger.success(f"⚖️ Final response generated in {time.time() - start_time:.2f}s")

    return {"messages": [response]}

def call_tools_and_save_context(state: ChatState):
    """Simple Tool Node: Runs the tool and saves context."""
    last_msg = state["messages"][-1]
    new_contexts = []
    tool_msgs = []

    # Check if the LLM actually called the tool
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        for tool_call in last_msg.tool_calls:
            # Manually trigger your @tool
            result_str = retrieve_legal_context.invoke(tool_call["args"])
            
            # Save the text for the final_answer node
            new_contexts.append(result_str)
            
            # Required: Add a ToolMessage so the graph knows the tool finished
            tool_msgs.append(ToolMessage(
                tool_call_id=tool_call["id"],
                content="Context stored."
            ))

    # Return the state update
    return {
        "messages": tool_msgs,
        "context": new_contexts
    }

def chat_node(state: ChatState):
    """Identifies the correct law for filtering and enforces tool-call efficiency."""
    logger.info("🧠 Agent: Analyzing conversation state...")

    # 1. LAW ROUTING LOGIC: Determine the law filter based on the latest user message
    user_query = state["messages"][-1].content.lower()
    law_filter = None
    routing_confidence = "LOW"

    # Define the 17-PDF Reference Menu for Guided Classification
    LEGAL_LIBRARY = [
        "BHARATIYA NYAYA SANHITA BNS 2023",
        "BHARATIYA NAGARIK SURAKSHA SANHITA BNSS 2023",
        "BHARATIYA SAKSHYA ADHINIYAM BSA 2023",
        "CODE OF CIVIL PROCEDURE CPC 1908",
        "INDIAN SUCCESSION ACT 1925",
        "THE HINDU MARRIAGE ACT 1955 ",
        "SPECIAL MARRIAGE ACT 1954",
        "THE INDIAN CONTRACT ACT 1872",
        "TRANSFER OF PROPERTY ACT 1882",
        "REGISTRATION ACT 1908",
        "NEGOTIABLE INSTRUMENTS ACT 1881",
        "CODE ON WAGES 2019",
        "CONSUMER PROTECTION ACT 2019",
        "INFORMATION TECHNOLOGY ACT 2000",
        "NARCOTIC DRUGS AND PYSCHOTROPIC SUBSTANCES ACT 1985",
        "POCSO ACT 2012",
        "CONSTITUTION OF INDIA FUNDAMENTAL RIGHTS"
    ]

    # Routing Map with hardcoded abbreviations to prevent hallucinations
    routing_map = {
        "BHARATIYA NYAYA SANHITA BNS 2023": ["bns", "theft", "murder", "assault", "rape", "crime", "punishment", "cheating"],
        "BHARATIYA NAGARIK SURAKSHA SANHITA BNSS 2023": ["bnss", "fir", "arrest", "bail", "summons", "investigation", "police"],
        "BHARATIYA SAKSHYA ADHINIYAM BSA 2023": ["bsa", "evidence", "witness", "testimony"],
        "CODE OF CIVIL PROCEDURE CPC 1908": ["cpc", "plaint", "written statement", "summons", "jurisdiction", "stay of suit", "injunction"],
        "INDIAN SUCCESSION ACT 1925": ["inheritance", "succession", "will", "flat", "share", "probate", "intestate"],
        "THE HINDU MARRIAGE ACT 1955 ": ["divorce", "alimony", "maintenance", "hindu marriage", "custody"],
        "SPECIAL MARRIAGE ACT 1954": ["court marriage", "inter-religion marriage", "civil marriage"],
        "THE INDIAN CONTRACT ACT 1872": ["agreement", "breach", "contract", "consideration"],
        "TRANSFER OF PROPERTY ACT 1882": ["sale", "gift", "mortgage", "lease", "possession", "tenant"],
        "REGISTRATION ACT 1908": ["stamp duty", "registrar", "registered", "notary", "deed"],
        "NEGOTIABLE INSTRUMENTS ACT 1881": ["cheque", "dishonour", "138", "ni act"],
        "CODE ON WAGES 2019": ["salary", "wage", "firing", "dues", "termination"],
        "CONSUMER PROTECTION ACT 2019": ["defective", "service deficiency", "consumer court"],
        "INFORMATION TECHNOLOGY ACT 2000": ["cyber", "whatsapp", "hacking", "online fraud", "it act"],
        "NARCOTIC DRUGS AND PYSCHOTROPIC SUBSTANCES ACT 1985": ["drugs", "narcotics", "trafficking", "ganja", "cannabis", "ndps"],
        "POCSO ACT 2012": ["child", "sexual offense", "minor", "pocso"],
        "CONSTITUTION OF INDIA FUNDAMENTAL RIGHTS": ["fundamental rights", "article", "writ petition", "supreme court", "constitution"]
    }

    detected_laws = []
    for law, keywords in routing_map.items():
        if any(k in user_query for k in keywords):
            detected_laws.append(law)
            routing_confidence = "HIGH"
    
    # 🎯 GENERALIZED PROCEDURAL BRIDGE
    civil_laws = ["INDIAN SUCCESSION ACT 1925", "THE HINDU MARRIAGE ACT 1955 ", "SPECIAL MARRIAGE ACT 1954", "THE INDIAN CONTRACT ACT 1872", "TRANSFER OF PROPERTY ACT 1882", "REGISTRATION ACT 1908", "CODE ON WAGES 2019", "CONSUMER PROTECTION ACT 2019"]
    criminal_laws = ["BHARATIYA NYAYA SANHITA BNS 2023", "BHARATIYA SAKSHYA ADHINIYAM BSA 2023", "POCSO ACT 2012", "NARCOTIC DRUGS AND PYSCHOTROPIC SUBSTANCES ACT 1985"]

    if any(law in detected_laws for law in civil_laws):
        if "CODE OF CIVIL PROCEDURE CPC 1908" not in detected_laws:
            detected_laws.append("CODE OF CIVIL PROCEDURE CPC 1908")
            logger.info("🛠️ Auto-injecting CPC for procedural support.")
            
    if any(law in detected_laws for law in criminal_laws):
        if "BHARATIYA NAGARIK SURAKSHA SANHITA BNSS 2023" not in detected_laws:
            detected_laws.append("BHARATIYA NAGARIK SURAKSHA SANHITA BNSS 2023")
            logger.info("🛠️ Auto-injecting BNSS for criminal procedure.")
    
    law_filter = detected_laws if detected_laws else None

    # REFINED LLM FALLBACK: Restricts AI to your 17 PDFs
    if not law_filter:
        library_menu = "\n".join([f"- {l}" for l in LEGAL_LIBRARY])
        intent_prompt = f"""Identify the most relevant Indian Law from this list for: '{user_query}'.
        
        ### PERMITTED ACTS:
        {library_menu}

        Return only the full Act Name from the list or 'GENERAL'. No explanations."""
        llm_guess = fast_llm.invoke(intent_prompt).content.strip().upper()
        
        # Validating LLM selection against our library
        for valid_law in LEGAL_LIBRARY:
            if valid_law in llm_guess:
                law_filter = [valid_law]
                routing_confidence = "MEDIUM"
                break
    
    # 2. DISPATCHER PREPARATION
    system_msg = f"""
    You are a Senior Legal Dispatcher. Target Law: {law_filter if law_filter else 'General Indian Law'}.
    - BNSS refers to 'BHARATIYA NAGARIK SURAKSHA SANHITA 2023'.
    - BNS refers to 'BHARATIYA NYAYA SANHITA 2023'.
    - Consolidate investigation into ONE tool call.
    - If query involves Money/Work/Property, DO NOT prioritize 'BNS' unless criminal intent exists.
    """
    
    messages = [{"role": "system", "content": system_msg}] + state["messages"]
    llm_with_tools = llm.bind_tools([retrieve_legal_context])
    response = llm_with_tools.invoke(messages)
    
    # 3. THROTTLING & CONSOLIDATION
    if response.tool_calls:
        if len(response.tool_calls) > 1:
            logger.warning(f"⚠️ Throttling: Consolidating {len(response.tool_calls)} calls to 1.")
            queries = [tc['args'].get('query', '') for tc in response.tool_calls]
            response.tool_calls = [response.tool_calls[0]]
            response.tool_calls[0]['args']['query'] = " | ".join(queries)

        for tc in response.tool_calls:
            if tc['name'] == 'retrieve_legal_context':
                tc['args']['law_filter'] = law_filter
                tc['args']['is_strict'] = (routing_confidence == "HIGH")
    else:
        logger.info("💬 Agent decision: Direct response (no tool needed)")
        
    return {"messages": [response], "law_filter": law_filter}

def route_after_agent(state: ChatState):
    """
    Determines if the LLM needs to perform a search or is ready to answer.
    """
    last_message = state["messages"][-1]
    
    # If the LLM decided to call a tool (e.g., search_statutes)
    if last_message.tool_calls:
        logger.info(f"🛠️ Agent: Tool calls detected ({len(last_message.tool_calls)}). Moving to Tools.")
        return "call_tools"
    
    # If no tool calls, it means the agent thinks it has enough info (or is finishing)
    logger.info("🏁 Agent: No tool calls. Proceeding to Final Synthesis.")
    return "finalize"

def evaluate_response_node(state: ChatState):
    intent = state.get("intent", "LEGAL")
    
    # Auto-pass for General Chat
    if intent == "CHAT":
        return {"evaluation_score": 10, "evaluation_feedback": "General greeting/small talk is accurate and professional."}
    
    logger.info("📊 Evaluator: Calculating production quality metrics...")
    
    final_response = state["messages"][-1].content
    raw_context = state.get("context", [])
    context_for_eval = prune_legal_context(raw_context, max_chars=1500)

    # 🚨 PRE-PROCESS: Strip headers so the Evaluator doesn't get confused by "VERIFIED"
    clean_response = re.sub(r"^\*\*VERIFIED\*\*.*?\n+", "", final_response, flags=re.IGNORECASE).strip()

    eval_prompt = f"""
        You are a Legal Quality Auditor. 
        Compare the RESPONSE against the VERIFIED CONTEXT.

        ### OUTPUT FORMAT (STRICT):
        SCORE: <number>
        REASON: <concise summary>

        ### SCORING RULES:
        1. **CONTRACT/PROPERTY:** Must cite TPA or Contract Act and CPC Order 39.
        2. **HALLUCINATION:** Deduct 3 points if it cites the 'Succession Act' for a Contract case.
        3. **PROCEDURE:** Must include a roadmap with CPC or BNSS.

        RESPONSE: {clean_response}
        CONTEXT: {context_for_eval}
    """
    try:
        # Use a slightly higher max_tokens to ensure it doesn't cut off
        eval_result = fast_llm.invoke(eval_prompt).content
        
        # 🎯 FIX: Robust Regex that finds the first number associated with 'score'
        score_match = re.search(r"(?:SCORE|Score|score)[:\s]*(\d+)", eval_result)
        
        if score_match:
            score = int(score_match.group(1))
        else:
            # Fallback: Search for any standalone digit 1-10 in the first line
            first_line = eval_result.split('\n')[0]
            digit_match = re.search(r"(\d+)", first_line)
            score = int(digit_match.group(1)) if digit_match else 5

        # 🎯 FIX: Cap the score
        score = max(0, min(score, 10))
        
        logger.info(f"📈 Evaluation Complete | Score: {score}/10")

        final_feedback = clean_feedback(eval_result)
        
        return {
            "messages": state["messages"],
            "evaluation_score": score,
            "evaluation_feedback": final_feedback
        }
    except Exception as e:
        logger.error(f"❌ Evaluator Error: {str(e)}")
        # If the evaluator fails, assume the Advisor did okay to break the loop
        return {"messages": state["messages"], "evaluation_score": 8}
    
def route_after_evaluation(state: ChatState):
    """
    Decides whether to END the conversation or LOOP back for a better answer.
    """
    score = state.get("evaluation_score", 10)
    retries = state.get("retry_count", 0)

    # THRESHOLD: If score < 6 and we haven't retried more than once
    if score < 6 and retries < 1:
        logger.warning(f"🔄 Low Score ({score}). Looping back for improvement...")
        return "retry"
    
    return "end"
    
def retry_prep_node(state: ChatState):
    """Adds a hint to the conversation so the agent knows WHY it's retrying."""
    feedback = state.get("evaluation_feedback", "Improve accuracy.")
    retry_msg = AIMessage(content=f"Self-Correction: Previous attempt was low quality ({feedback}). Refining search and synthesis...")
    
    return {
        "messages": [retry_msg],
        "retry_count": state.get("retry_count", 0) + 1
    }

def router_node(state: ChatState):
    messages = state["messages"]
    last_msg = messages[-1].content
    
    # Check for Chat vs Legal
    intent_chain = get_router_prompt() | fast_llm
    intent_decision = intent_chain.invoke({"query": last_msg}).content.strip().upper()
    intent = "LEGAL" if "LEGAL" in intent_decision else "CHAT"

    # Check for Follow-up vs New Topic (only if Legal)
    is_followup = False
    if intent == "LEGAL" and len(messages) > 2:
        prev_advice = messages[-2].content
        followup_chain = get_followup_classifier_prompt() | fast_llm
        f_decision = followup_chain.invoke({
            "prev_advice": prev_advice[:500], 
            "current_msg": last_msg
        }).content.strip().upper()
        is_followup = "FOLLOW_UP" in f_decision

    logger.info(f"🚦 Router: Intent={intent} | Follow-up={is_followup}")
    return {"intent": intent, "is_followup": is_followup}

def route_after_router(state: ChatState):
    """
    The Gatekeeper: Skips RAG for small talk or recognized follow-ups.
    """
    intent = state.get("intent", "LEGAL")
    is_followup = state.get("is_followup", False)
    
    # If it's small talk (CHAT), go straight to the answer node
    if intent == "CHAT":
        logger.info("🚦 Router: General Chat detected. Skipping to Response.")
        return "finalize"
        
    # If it's a follow-up we already have context for, skip the Search step
    if is_followup:
        logger.info("🚦 Router: Follow-up detected. Using existing context.")
        return "finalize"
    
    # Otherwise, start the standard Legal Agent flow
    return "continue"

def create_graph(checkpointer):
    """Logs the graph assembly and checkpointer status."""
    logger.info("🕸️  Constructing LangGraph State Machine...")

    workflow = StateGraph(ChatState)
    workflow.add_node("router", router_node)
    workflow.add_node("agent", chat_node)    
    workflow.add_node("tools", call_tools_and_save_context)
    workflow.add_node("final_answer", generate_response_node)
    workflow.add_node("auditor", verify_citations_node)
    workflow.add_node("evaluator", evaluate_response_node)
    workflow.add_node("retry_prep", retry_prep_node)
    
    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "continue": "agent",
            "finalize": "final_answer"
        }
    )

    workflow.add_conditional_edges(
    "agent", 
    route_after_agent, 
    {
        "call_tools": "tools",
        "finalize": "final_answer"
    })
    workflow.add_edge("tools", "final_answer")
    workflow.add_edge("final_answer", "auditor")
    workflow.add_edge("auditor", "evaluator")

    workflow.add_conditional_edges("evaluator", route_after_evaluation, {
        "retry": "retry_prep",
        "end": END
    })
    
    workflow.add_edge("retry_prep", "agent")

    logger.success("✅ LangGraph compiled with Neon checkpointing enabled.")
    return workflow.compile(checkpointer=checkpointer)