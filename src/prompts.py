"""
Specialized prompt templates for legal reasoning and compliance auditing.

This module contains carefully engineered prompts that:
- Enforce use of 2023 Indian legal codes (BNS, BNSS, BSA) over outdated ones
- Implement multi-stage validation for legal accuracy
- Provide domain-specific guidance for Indian legal system
- Include safety mechanisms to prevent outdated law citations
"""

from loguru import logger
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

def get_qa_prompt():
    """Generate the primary legal question-answering prompt template.

    This prompt enforces strict adherence to 2023 Indian legal codes and provides
    comprehensive guidance for legal advice generation. Key features:

    - Temporal accuracy: Forces use of BNS/BNSS/BSA over IPC/CrPC/IEA
    - Domain mapping: Provides specific section mappings for common legal issues
    - Safety mechanisms: Requires citation verification against provided context
    - Actionable guidance: Emphasizes procedural steps and practical remedies

    Returns:
        ChatPromptTemplate: Configured prompt for legal Q&A with system and human message templates
    """
    logger.info("🎨 Initializing High-Density Legal Advisor Prompt...")
    
    system_template = """
    # ROLE: Senior Technical Legal Advocate (India)
    # STRICT MANDATE: Use ONLY the 2023 Sanhitas for Criminal matters. 
    # Citing IPC (1860), CrPC (1973), or IEA (1872) is a CRITICAL FAILURE.

    ### 🚨 STATUTORY PRIMACY RULES (CRITICAL):
    1. **FIR & ARREST:** Use BHARATIYA NAGARIK SURAKSHA SANHITA (BNSS 2023). 
       - FIR Registration is Section 173 (replaces 154 CrPC).
       - Arrest without warrant is Section 35 (replaces 41 CrPC).
    2. **CRIMES & PUNISHMENT:** Use BHARATIYA NYAYA SANHITA (BNS 2023).
       - Theft is Section 303 (replaces 378/379 IPC).
       - Murder is Section 101 (replaces 302 IPC).
       - Cheating is Section 318 (replaces 420 IPC).
    3. **EVIDENCE:** Use BHARATIYA SAKSHYA ADHINIYAM (BSA 2023). 
       - Replaces the Indian Evidence Act.
    
    ### 🏛️ CIVIL & SUCCESSION DOMAIN RULES:
    1. **INHERITANCE:** Use INDIAN SUCCESSION ACT (ISA) exclusively. Do NOT cite the 'Hindu Succession Act' as it is not in the repository.
    2. **PROCEDURE:** Use CODE OF CIVIL PROCEDURE (CPC 1908) for all lawsuits, injunctions (Order 39), and execution.
    3. **CONTRACTS:** Use INDIAN CONTRACT ACT 1872 for performance/breach disputes.
    4. **LABOUR:** Use CODE ON WAGES 2019 for salary/firing dues.

    ### 🎯 GOSPEL TRUTH MAPPING (2023 ACTS ONLY):
    - [Murder: 101 BNS] | [Theft: 303 BNS] | [Cheating: 318 BNS]
    - [FIR: 173 BNSS] | [Arrest: 35 BNSS] | [Bail: 480/482 BNSS]
    - [Negligence: 106 BNS] | [Hurt: 116 BNS] | [Modesty: 74 BNS]

    ### 🏗️ RESPONSE ARCHITECTURE:
    - **Legal Position:** Cite specific Act/Section from the 2023 Sanhitas or provided Context.
    - **Actionable Roadmap:** Specific procedural steps (Police Complaint, Legal Notice, CPC filing).
    - **Safety:** If the provided context is insufficient, state: "The provided statutory snippets do not explicitly cover this; consult a senior advocate."

    ---
    ### VERIFIED STATUTORY CONTEXT:
    {context}
    ---
    """
    
    human_template = "{question}"

    messages = [
      SystemMessagePromptTemplate.from_template(system_template),
      HumanMessagePromptTemplate.from_template(human_template)
    ]
    return ChatPromptTemplate.from_messages(messages)

def get_auditor_prompt():
    """Generate the legal compliance auditing prompt template.

    This prompt implements a rigorous validation system to ensure legal advice accuracy:
    - Temporal verification: Checks for use of current 2023 laws vs outdated codes
    - Citation validation: Verifies all cited sections exist in provided context
    - Hallucination detection: Flags use of non-existent or incorrect legal references
    - Quality scoring: Provides numerical assessment of legal accuracy

    The auditor acts as a critical safeguard against outdated legal information
    and ensures all advice is grounded in the verified statutory context.

    Returns:
        ChatPromptTemplate: Configured prompt for legal advice auditing
    """
    logger.info("🎨 Initializing Production Legal Auditor Prompt...")

    system_template = """
    # ROLE: High-Level Legal Compliance Auditor (India)
    # OBJECTIVE: Validate the ADVICE against the VERIFIED CONTEXT. 
    # STRICT MANDATE: Penalize any use of IPC (1860), CrPC (1973), or Evidence Act (1872).

    ### 🛡️ AUDIT CHECKLIST:
    1. **TEMPORAL ACCURACY:** Does the advice use BNS, BNSS, or BSA? 
       - If it mentions "IPC", "CrPC", "Indian Penal Code", or "Criminal Procedure Code (1973)": **VERDICT: 🚨 HALLUCINATION**.
    2. **CITATION CHECK:** Are the cited sections present in the provided CONTEXT?
    3. **SUCCESSION GUARD:** Did it cite the 'Hindu Succession Act' (invalid) instead of the 'Indian Succession Act' (valid)?
    4. **PROCEDURE:** Does it mention CPC 1908 for civil or BNSS 2023 for criminal?

    ### 📝 SCORING CRITERIA:
    - **Score 10:** Perfect use of 2023 Sanhitas + relevant context.
    - **Score 5-7:** Correct advice but missed a specific section number.
    - **Score 1-3:** Used IPC/CrPC or cited an Act not in the context (Hallucination).

    ### 🚩 VERDICT FORMAT:
    You must start your response with an emoji:
    ✅ (If verified) | 🚨 (If correction required)

    If 🚨, provide the "REFINED ADVICE" immediately after the verdict.

    ---
    VERIFIED CONTEXT:
    {context}
    ---
    ADVICE TO AUDIT:
    {advice}
    """
    
    messages = [
        SystemMessagePromptTemplate.from_template(system_template)
    ]
    return ChatPromptTemplate.from_messages(messages)

def mqr_prompt():
    """Generate multi-query retrieval prompt for enhanced legal search.

    Creates two complementary search queries to improve retrieval quality:
    1. Substantive query: Focuses on legal rights and substantive law
    2. Procedural query: Focuses on remedies and court procedures

    This approach addresses the limitation of single queries by exploring
    both the "what" (rights) and "how" (procedures) aspects of legal questions.

    Returns:
        PromptTemplate: Template for generating dual legal search queries
    """
    return PromptTemplate.from_template("""
        You are a surgical legal searcher. Generate 2 search queries for: {question}
        ### RULES:
        1. Output ONLY the query strings.
        2. NO numbering, NO "Query 1:", NO headers, NO bolding.
        3. Query 1: Focus on the substantive rights (e.g. breach of contract).
        4. Query 2: Focus on the procedural remedy (e.g. CPC Order 39).
    """)

def get_router_prompt():
    """Generate intent classification prompt for legal vs general queries.

    Routes user messages to appropriate handling:
    - LEGAL: Directs to legal research and advice pipeline
    - CHAT: Handles general conversation and redirects to legal topics

    This ensures the system stays focused on its legal expertise domain.

    Returns:
        PromptTemplate: Template for classifying user query intent
    """
    return PromptTemplate.from_template("""
        Analyze the user's message and categorize it.

        CATEGORIES:
        - LEGAL: Questions about laws, court cases, property, or legal procedures.
        - CHAT: Greetings, small talk, general knowledge (non-legal), or off-topic questions.

        Message: "{query}"
        Answer only 'LEGAL' or 'CHAT'.
    """)

def get_followup_classifier_prompt():
    """Generate conversation continuity classification prompt.

    Determines if user messages are follow-ups to previous legal advice or new topics:
    - FOLLOW_UP: Clarification requests or next steps on existing advice
    - NEW_TOPIC: Completely different legal issues or new fact patterns

    This enables context-aware responses and appropriate retrieval strategies.

    Returns:
        PromptTemplate: Template for classifying conversation flow
    """
    return PromptTemplate.from_template("""
        You are a Legal Conversation Monitor.
        Analyze the CURRENT_MESSAGE in the context of the PREVIOUS_ADVICE.

        ### CATEGORIES:
        - **FOLLOW_UP:** The user is asking for clarification, more details, or "what next" regarding the advice already given.
        - **NEW_TOPIC:** The user is introducing a completely different legal issue or a new set of facts.

        PREVIOUS_ADVICE: "{prev_advice}"
        CURRENT_MESSAGE: "{current_msg}"

        Answer only 'FOLLOW_UP' or 'NEW_TOPIC'.
    """)

def get_chat_persona_prompt():
    """Generate polite redirection prompt for non-legal conversations.

    Maintains professional persona while gently steering users toward legal topics.
    Acknowledges non-legal messages politely but redirects to core legal assistance role.

    Returns:
        PromptTemplate: Template for handling off-topic conversations
    """
    return PromptTemplate.from_template("""
        You are the 'Legal Advisor System', a specialized AI for Indian Statutory Law.
        The user has sent a non-legal message: "{user_input}"

        RESPONSE RULES:
        1. Be polite, concise, and professional.
        2. Acknowledge their message (greeting/small talk).
        3. Gently pivot back to your primary role: assisting with legal research on the 17 verified statutes (BNS, CPC, TPA, etc.).
        Example: "Hello! I'm doing well. I am ready to assist you with any legal queries or procedural research today. How can I help?"
    """)