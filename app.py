import streamlit as st
import os
from datetime import datetime, timedelta
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Set
import re

# Page configuration
st.set_page_config(
    page_title="Rooman HR Assistant Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'query_analytics' not in st.session_state:
    st.session_state.query_analytics = {
        'leave': 0, 'benefits': 0, 'policy': 0, 'other': 0
    }
if 'conversation_insights' not in st.session_state:
    st.session_state.conversation_insights = {
        'topics_discussed': set(),
        'unanswered_queries': [],
        'total_interactions': 0
    }

# Load policy documents
def load_policy_docs():
    """Load all policy documents into memory"""
    docs = {}
    data_dir = 'data'
    
    for filename in ['leave_policy.txt', 'benefits_policy.txt', 'workplace_policy.txt']:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                docs[filename] = f.read()
    
    return docs

# Load documents once
POLICY_DOCS = load_policy_docs()

def get_smart_suggestions(query: str, intent: str) -> List[str]:
    """Generate contextual follow-up suggestions"""
    suggestions = []
    
    if intent == 'leave':
        if 'annual' in query.lower():
            suggestions = [
                "How do I apply for annual leave?",
                "Can I carry forward annual leave?",
                "What if I need emergency leave?"
            ]
        elif 'sick' in query.lower():
            suggestions = [
                "Do I need a medical certificate?",
                "How much notice for sick leave?",
                "What about chronic illness?"
            ]
        else:
            suggestions = [
                "Tell me about maternity leave",
                "Can I combine different leave types?",
                "What's the leave approval process?"
            ]
    
    elif intent == 'benefits':
        suggestions = [
            "What are the insurance coverage details?",
            "Tell me about performance bonuses",
            "How does the ESOP program work?"
        ]
    
    elif intent == 'policy':
        if 'wfh' in query.lower() or 'remote' in query.lower():
            suggestions = [
                "What equipment is provided for WFH?",
                "Can I request full-time remote work?",
                "What's the attendance policy?"
            ]
        else:
            suggestions = [
                "What's the probation period?",
                "Tell me about the notice period",
                "What are the working hours?"
            ]
    
    else:
        suggestions = [
            "How many leaves do I get?",
            "What are the employee benefits?",
            "Can I work from home?"
        ]
    
    return suggestions[:3]

def classify_intent(query: str) -> str:
    """Classify user query intent"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['leave', 'vacation', 'holiday', 'sick', 'maternity', 'paternity', 'wfh']):
        return 'leave'
    elif any(word in query_lower for word in ['benefit', 'insurance', 'health', 'pf', 'salary', 'bonus', 'esop']):
        return 'benefits'
    elif any(word in query_lower for word in ['policy', 'dress', 'attendance', 'notice', 'probation', 'code']):
        return 'policy'
    return 'other'

def extract_key_concepts(query: str) -> List[str]:
    """Extract key concepts from query using better NLP"""
    query_lower = query.lower()
    
    # Remove common words
    stop_words = {'i', 'me', 'my', 'we', 'our', 'you', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 
                  'can', 'could', 'would', 'should', 'what', 'when', 'where', 'how', 'why', 'do', 'does',
                  'get', 'take', 'have', 'has', 'of', 'to', 'for', 'in', 'on', 'at', 'about', 'tell', 'explain'}
    
    # Extract words
    words = re.findall(r'\b\w+\b', query_lower)
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Keyword mappings for better understanding
    concept_map = {
        'leaves': ['leave', 'vacation', 'time off', 'holiday', 'absence'],
        'health': ['health', 'medical', 'insurance', 'hospital', 'coverage'],
        'salary': ['salary', 'pay', 'compensation', 'wage', 'income'],
        'work from home': ['wfh', 'remote', 'home', 'hybrid', 'flexible'],
        'notice': ['notice', 'resignation', 'quit', 'leaving', 'resign'],
        'probation': ['probation', 'trial', 'new', 'joining'],
        'sick': ['sick', 'illness', 'medical', 'unwell'],
        'annual': ['annual', 'yearly', 'vacation', 'planned'],
        'maternity': ['maternity', 'pregnancy', 'maternal', 'baby'],
        'paternity': ['paternity', 'father', 'dad'],
        'benefits': ['benefits', 'perks', 'advantages', 'insurance', 'pf'],
    }
    
    # Expand keywords with related concepts
    expanded = set(keywords)
    for keyword in keywords:
        for concept, related in concept_map.items():
            if keyword in related:
                expanded.update(related)
    
    return list(expanded)

def smart_search(query: str) -> Tuple[str, str, float]:
    """Smarter search with relevance scoring"""
    keywords = extract_key_concepts(query)
    query_lower = query.lower()
    
    best_match = None
    best_score = 0
    best_source = ""
    
    for doc_name, content in POLICY_DOCS.items():
        sections = content.split('\n\n')
        
        for section in sections:
            if len(section.strip()) < 20:
                continue
                
            section_lower = section.lower()
            score = 0
            
            # Score based on keyword matches
            for keyword in keywords:
                if keyword in section_lower:
                    score += section_lower.count(keyword) * 2
            
            # Bonus for exact phrase matches
            query_phrases = query_lower.split()
            for i in range(len(query_phrases) - 1):
                phrase = ' '.join(query_phrases[i:i+2])
                if phrase in section_lower:
                    score += 5
            
            # Bonus for section headers
            if section.startswith('==='):
                score += 3
            
            if score > best_score:
                best_score = score
                best_match = section
                best_source = doc_name.replace('.txt', '').replace('_', ' ').title()
    
    return best_match or "", best_source, best_score

def generate_response(query: str) -> str:
    """Generate smart response based on query"""
    query_lower = query.lower()
    keywords = extract_key_concepts(query)
    
    # Get best matching content
    context, source, score = smart_search(query)
    
    # First, check for specific answerable questions regardless of phrasing
    
    # HEALTH INSURANCE - any variation
    if any(word in query_lower for word in ['health insurance', 'medical insurance', 'health coverage', 'medical coverage', 'insurance coverage']):
        return """**Health Insurance Coverage:**

🏥 **Comprehensive Family Coverage**

**Who's Covered:**
- Employee + Spouse + 2 Children + Parents

**Coverage Amount:**
- Sum Insured: **₹5,00,000** per family/year
- OPD Coverage: **₹25,000**/year
- Maternity: **₹75,000**/delivery

**Additional Benefits:**
- 🏥 10,000+ cashless hospitals across India
- 📋 Annual health checkup (₹3,000 value)
- 💊 Pre-existing conditions covered after 3 years
- 🚑 Ambulance: ₹2,000/emergency

**How to Use:**
1. Show health card at network hospitals
2. Cashless treatment approved within 2 hours
3. For non-network, submit reimbursement within 30 days

📚 **Source:** Benefits Policy

💡 **Related Questions:**

1. What other benefits do employees get?
2. How do I add my family members?
3. What's the claim process?"""
    
    # LEAVE QUESTIONS - comprehensive check
    if any(phrase in query_lower for phrase in ['how many leave', 'how much leave', 'number of leave', 'leave entitlement', 'total leave']):
        if 'annual' in query_lower or 'vacation' in query_lower or 'yearly' in query_lower:
            return """**Annual Leave Entitlement:**

📅 **24 days** of paid annual leave per year

**Key Details:**
- Accrues at **2 days per month** of service
- Carry forward: Up to **12 days** to next year
- Advance notice: **5 working days** required
- During probation: **6 days** available (first 6 months)

**How to Apply:**
1. Submit via HRMS portal
2. Manager reviews within 48 hours
3. Automatic email confirmation

**Pro Tips:**
- ✅ Plan in advance for better approval chances
- ✅ Combine with weekends for longer breaks
- ✅ Use carry-forward wisely before year-end

📚 **Source:** Leave Policy 2024

💡 **Related Questions:**

1. Can I carry forward unused leaves?
2. What if I need emergency leave?
3. How do I apply in HRMS?"""
            
        elif 'sick' in query_lower:
            return """**Sick Leave Entitlement:**

🏥 **12 days** of paid sick leave per year

**Important Points:**
- Does NOT carry forward to next year
- Medical certificate needed for **3+ consecutive days**
- Must inform manager within **2 hours** of shift start
- Can be taken on short notice (with intimation)

**When You're Sick:**
1. Inform manager ASAP (within 2 hours)
2. Mark leave in HRMS
3. Submit medical cert if 3+ days
4. Update team on urgent tasks

📚 **Source:** Leave Policy 2024

💡 **Related Questions:**

1. What if I'm chronically ill?
2. Can I combine sick + annual leave?
3. What about work from home when sick?"""
        
        else:
            # General leave question
            return """**Complete Leave Entitlement:**

📊 **Your Annual Leave Breakdown:**

**Paid Leaves:**
- 📅 **Annual Leave:** 24 days/year (carry forward 12 days max)
- 🏥 **Sick Leave:** 12 days/year (no carry forward)
- 📋 **Casual Leave:** 10 days/year (no carry forward)
- **Total:** 46 days/year

**Special Leaves:**
- 🤰 **Maternity:** 26 weeks (182 days)
- 👨‍👶 **Paternity:** 15 days
- 😢 **Bereavement:** 3-5 days (based on relationship)
- 📚 **Study Leave:** Up to 10 days/year

**Work from Home:**
- 🏠 2 days/week post-probation
- Requires 24-hour advance approval

💡 **Use the Leave Calculator tab** to see your current balance!

📚 **Source:** Leave Policy 2024

💡 **Related Questions:**

1. How do I calculate my current leave balance?
2. Can I take leave during probation?
3. What's the approval process?"""
    
    # BENEFITS - general inquiry
    if ('benefit' in query_lower or 'perks' in query_lower) and not any(specific in query_lower for specific in ['health', 'insurance', 'pf', 'bonus', 'esop']):
        return """**Complete Employee Benefits Package:**

💰 **Financial Benefits:**
- 🏦 **Provident Fund:** 12% employer + 12% employee
- 💎 **Gratuity:** After 5 years (formula: salary × years × 15/26)
- 🎁 **Performance Bonus:** 0-20% of CTC based on rating
- 📈 **ESOP:** For L4+ after 1 year (4-year vesting)

**Insurance Coverage:**
- 🏥 **Health:** ₹5L family coverage, 10,000+ hospitals
- 💼 **Life:** 3x annual CTC
- 🚑 **Accidental:** 4x annual CTC
- 💊 **Critical Illness:** ₹10L coverage

**Monthly Allowances:**
- 🍔 Meal coupons: ₹2,500
- ⛽ Fuel: ₹3,000
- 📱 Mobile: ₹1,500
- 🌐 Internet (WFH): ₹1,000

**Learning & Development:**
- 📚 Annual budget: ₹25,000/employee
- 🎓 Certification reimbursement: 100%
- 📊 Conferences: 2/year

**Wellness Benefits:**
- 🏋️ Gym reimbursement: ₹12,000/year
- 🧘 Mental health counseling: 6 free sessions
- 💉 Annual health checkup + flu shots

**Family Support:**
- 🤰 Maternity: 26 weeks paid + ₹75K coverage
- 👨‍👶 Paternity: 15 days paid
- 👶 Crèche facility on campus
- 📖 Child education: ₹30,000/year

📚 **Source:** Benefits Policy

💡 **Related Questions:**

1. How do I claim these benefits?
2. When do I become eligible for ESOP?
3. What's the PF contribution breakdown?"""
    
    # WFH / REMOTE WORK
    if any(phrase in query_lower for phrase in ['work from home', 'wfh', 'remote work', 'work remotely', 'home office']):
        return """**Work From Home (WFH) Policy:**

🏠 **Flexible Work Options**

**Hybrid Model (Standard):**
- ✅ **2 days WFH** + **3 days office** per week
- Available after probation period
- Requires **24 hours advance approval** via HRMS
- Manager has discretion based on project needs

**Full Remote:**
- Special circumstances only (medical, relocation)
- Requires VP approval
- Must justify business case

**WFH Benefits Provided:**
- 💻 Company laptop with VPN access
- 💰 ₹1,000/month internet reimbursement
- 🪑 One-time home office setup: ₹10,000

**When WFH Not Allowed:**
- ❌ Critical project milestones
- ❌ Client meetings/presentations
- ❌ Team collaboration days
- ❌ During probation period

**Best Practices:**
- 🟢 Stay available during core hours (11 AM - 4 PM)
- 📞 Respond to messages within 15 minutes
- 📹 Camera on for team meetings
- 📊 Daily standup attendance mandatory

📚 **Source:** Workplace Policy - Remote Work

💡 **Related Questions:**

1. How do I request WFH days?
2. Can I work from another city?
3. What equipment is provided?"""
    
    # RESIGNATION / NOTICE PERIOD
    if any(word in query_lower for word in ['resign', 'resignation', 'notice period', 'quit', 'leaving']):
        return """**Resignation & Notice Period:**

⏰ **Notice Period Requirements:**

**By Employment Status:**
- **Confirmed Employees:** 60 days
- **Senior Roles (Manager+):** 90 days  
- **During Probation:** 30 days

**Buyout Option:**
- Available at company's discretion
- Cost: 1-2 months gross salary
- Must be requested in writing

**Resignation Process:**

1. **Week 1:**
   - Submit formal resignation letter to manager
   - HR initiates exit formalities
   - Manager assigns transition tasks

2. **Weeks 2-8:**
   - Knowledge transfer to team
   - Document handover
   - Complete pending projects

3. **Final Week:**
   - Asset return (laptop, ID, access cards)
   - Exit interview with HR
   - Handover checklist completion

**Full & Final Settlement:**
- Processed within **45 days** of last working day
- Includes: Pending salary, leave encashment, bonus (pro-rata)

**During Notice Period:**
- ❌ Cannot take leave (unless approved)
- ✅ Full attendance required
- ✅ Maintain professional conduct

**Certificate Provided:**
- Experience letter
- Relieving letter
- Salary certificate (on request)

📚 **Source:** Workplace Policy

💡 **Related Questions:**

1. Can I negotiate shorter notice period?
2. What if I have unused leaves?
3. How is F&F calculated?"""
    
    # MATERNITY / PREGNANCY
    if any(word in query_lower for word in ['maternity', 'pregnancy', 'pregnant', 'maternal']):
        return """**Maternity Leave & Benefits:**

🤰 **Comprehensive Support for New Mothers**

**Leave Duration:**
- **26 weeks (182 days)** of fully paid maternity leave
- Can start **8 weeks before** expected delivery date
- **Adoptive mothers:** 12 weeks from adoption date

**Financial Benefits:**
- Full salary during entire leave period
- **₹75,000** maternity coverage in health insurance
- Hospital expenses covered under health insurance

**Additional Support:**
- 👶 **On-site crèche:** For 6 months - 5 years
- 📖 **Child education allowance:** ₹30,000/year
- 🏥 **Pre & post-natal checkups:** Covered in OPD

**Flexible Return Options:**
- Gradual return to work (part-time for 1 month)
- Extended WFH option for 3 months
- Nursing breaks: 2x 30-minute breaks/day (first 6 months)

**Requirements:**
- Medical documentation from registered practitioner
- Birth certificate submission within 30 days
- Update family details in HRMS

**Application Process:**
1. Inform manager at least 8 weeks before
2. Submit medical documents to HR
3. Plan handover with team
4. Complete HRMS formalities

📚 **Source:** Leave Policy & Benefits Policy

💡 **Related Questions:**

1. Can I extend maternity leave?
2. What about paternity leave for my spouse?
3. How do I enroll child in crèche?"""
    
    # PROBATION
    if 'probation' in query_lower:
        return """**Probation Period Policy:**

⏱️ **What to Expect During Probation**

**Duration:**
- **6 months** for all new hires
- Can be extended by **3 months** if needed
- Rare cases: Early confirmation at 3 months for exceptional performance

**Performance Reviews:**
- **3-month review:** Mid-probation evaluation
- **6-month review:** Final confirmation decision
- Continuous feedback from manager

**During Probation:**
- Notice period: **30 days** (both sides)
- Leave allowance: Up to **6 days** (annual leave)
- Salary: 90% of confirmed CTC (10% held as retention)
- Benefits: Health insurance, PF start immediately

**Evaluation Criteria:**
- Technical competency
- Cultural fit
- Learning agility
- Team collaboration
- Attendance & punctuality

**Confirmation:**
- Based on satisfactory performance
- Letter issued within 1 week of completion
- 10% retention amount released
- Full benefits activated

**What Happens If Not Confirmed:**
- Extension by 3 months with improvement plan
- Or separation with 30-day notice
- Pro-rata benefits settled

📚 **Source:** Workplace Policy

💡 **Related Questions:**

1. What benefits do I get during probation?
2. Can I take leaves during probation?
3. How is performance evaluated?"""
    
    # Pattern-based responses for question types
    
    # Questions about "how many" or numbers
    elif any(word in query_lower for word in ['how many', 'how much', 'number of', 'total']):
        # Already handled above for leaves, this is fallback
        if context and score > 10:
            lines = context.split('\n')
            relevant = [l for l in lines if l.strip() and not l.startswith('===')][:5]
            summary = '\n'.join(relevant)
            
            return f"""**Based on HR Policies:**

{summary}

**Need more details?** Try asking more specifically!

📚 **Source:** {source}

💡 **Related Questions:**

1. What are all the leave types?
2. What benefits do I get?
3. How do I apply for this?"""
    
    # Questions about "can I" or permissions
    elif query_lower.startswith('can i') or 'allowed' in query_lower or 'able to' in query_lower:
        # WFH and other permissions already handled above
        if 'carry forward' in query_lower or 'carryover' in query_lower:
            return """**Leave Carry Forward Rules:**

✅ **Annual Leave:** Up to **12 days** can be carried forward to next year

❌ **Sick Leave:** Cannot be carried forward (use it or lose it)

❌ **Casual Leave:** Cannot be carried forward

**Pro Tips:**
- 📅 Plan your annual leave to use at least 12 days each year
- 💡 Carry forward strategically for major plans next year
- ⚠️ Unused carry-forward expires if not used by June 30

📚 **Source:** Leave Policy 2024

💡 **Related Questions:**

1. What happens to unused leaves?
2. Can I encash my leaves?
3. How to plan leaves strategically?"""
        
        elif context and score > 10:
            lines = context.split('\n')
            relevant = [l for l in lines if l.strip() and not l.startswith('===')][:5]
            summary = '\n'.join(relevant)
            
            return f"""**Based on HR Policies:**

{summary}

For specific permissions, please:
- 📧 Contact HR: hr@rooman.net
- 💬 Discuss with your manager
- 📖 Check HRMS portal for detailed policy

📚 **Source:** {source}

💡 **Related Questions:**

1. What's the general policy on this?
2. Who do I need approval from?
3. Are there any exceptions?"""
    
    # If we have good context from search
    elif context and score > 15:
        # Extract the most relevant parts
        lines = context.split('\n')
        relevant = [l for l in lines if l.strip() and not l.startswith('===')][:6]
        
        summary = '\n\n'.join(relevant)
        
        # Generate smart suggestions based on context
        suggestions = get_smart_suggestions(query, classify_intent(query))
        suggestions_text = "\n💡 **You might also want to know:**\n\n"
        for i, sug in enumerate(suggestions, 1):
            suggestions_text += f"{i}. {sug}\n"
        
        return f"""**From Our HR Policies:**

{summary}

**For More Information:**
- 📧 Contact: hr@rooman.net
- 💬 Book HR consultation
- 📖 HRMS portal

📚 **Source:** {source}

{suggestions_text}"""
    
    # Generic helpful response with smart suggestions
    else:
        intent = classify_intent(query)
        suggestions = get_smart_suggestions(query, intent)
        
        suggestions_text = ""
        for i, sug in enumerate(suggestions, 1):
            suggestions_text += f"{i}. {sug}\n"
        
        return f"""**I can help you with HR policy questions!**

Based on your query, here are some specific questions I can answer:

**💬 Try These:**
{suggestions_text}

**Common Topics:**
- 📅 **Leaves:** Annual, sick, casual, maternity, paternity
- 🏥 **Benefits:** Health insurance, PF, bonuses, allowances
- 🏠 **Policies:** WFH, notice period, probation, resignation
- 💰 **Compensation:** Salary structure, increments, ESOP

**Pro Tip:** Ask specific questions like:
- "How many annual leaves do I get?"
- "What's the health insurance sum insured?"
- "Can I work from home 2 days a week?"

📧 **Need personalized help?** Contact hr@rooman.net

📚 **Source:** HR Policy Documents"""

def calculate_leave_balance(months: int) -> Dict:
    """Calculate leave balance"""
    annual = min(24, months * 2)
    return {
        'Annual Leave': annual,
        'Sick Leave': 12,
        'Casual Leave': 10,
        'Total Available': annual + 22
    }

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Rooman HR Assistant Agent</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x80/667eea/ffffff?text=Rooman+Technologies", width=300)
        st.markdown("### 🎯 Features")
        st.markdown("""
        - 💬 **Smart Q&A** - Policy-based answers
        - 📊 **Leave Calculator** - Plan your leaves
        - 📈 **Analytics** - Query insights
        """)
        
        st.markdown("---")
        st.markdown("### 📋 Quick Actions")
        
        col_act1, col_act2 = st.columns(2)
        
        with col_act1:
            if st.button("🔄 Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        
        with col_act2:
            if st.button("📥 Export Conversation") and len(st.session_state.messages) > 0:
                # Create conversation export
                export_text = "ROOMAN HR ASSISTANT - CONVERSATION HISTORY\n"
                export_text += "=" * 50 + "\n\n"
                for msg in st.session_state.messages:
                    role = msg['role'].upper()
                    export_text += f"{role}:\n{msg['content']}\n\n"
                
                st.download_button(
                    label="💾 Download Chat History",
                    data=export_text,
                    file_name=f"hr_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        st.markdown("---")
        st.markdown("### 🚀 Quick Queries")
        
        quick_queries = [
            "How many annual leaves do I get?",
            "What's the health insurance coverage?",
            "Can I work from home?",
            "What's the notice period?"
        ]
        
        for query in quick_queries:
            if st.button(f"💬 {query}", key=f"quick_{query}"):
                # Simulate query submission
                st.session_state.messages.append({"role": "user", "content": query})
                intent = classify_intent(query)
                st.session_state.query_analytics[intent] += 1
                response = generate_response(query)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 🎓 Sample Questions")
        st.markdown("""
        - How many annual leaves do I get?
        - What's the health insurance coverage?
        - Can I work from home?
        - What's the notice period?
        - Tell me about maternity leave
        """)
    
    # Main content - Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📊 Analytics Dashboard", "📅 Leave Calculator"])
    
    # Tab 1: Chat
    with tab1:
        for message in st.session_state.messages:
            role_class = "user-message" if message["role"] == "user" else "assistant-message"
            icon = "👤" if message["role"] == "user" else "🤖"
            
            st.markdown(f"""
            <div class="chat-message {role_class}">
                <strong>{icon} {message["role"].title()}:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input outside tabs with form to prevent duplication
    with st.form(key='chat_form', clear_on_submit=True):
        user_query = st.text_input("💬 Ask me anything about HR policies, leave, benefits...", key="user_input")
        submit_button = st.form_submit_button("Send")
    
    if submit_button and user_query and user_query.strip():
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Classify intent
        intent = classify_intent(user_query)
        st.session_state.query_analytics[intent] += 1
        
        # Update conversation insights
        st.session_state.conversation_insights['total_interactions'] += 1
        st.session_state.conversation_insights['topics_discussed'].add(intent)
        
        # Generate response
        response = generate_response(user_query)
        
        # Get smart suggestions
        suggestions = get_smart_suggestions(user_query, intent)
        
        # Add suggestions to response
        if suggestions and response != "None":
            response += "\n\n**💡 Related Questions:**\n"
            for i, suggestion in enumerate(suggestions, 1):
                response += f"\n{i}. {suggestion}"
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()
    
    # Tab 2: Analytics
    with tab2:
        st.markdown("### 📈 Query Analytics Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        total = sum(st.session_state.query_analytics.values())
        
        with col1:
            st.metric("Total Queries", total)
        with col2:
            st.metric("Leave Queries", st.session_state.query_analytics['leave'])
        with col3:
            st.metric("Benefits Queries", st.session_state.query_analytics['benefits'])
        with col4:
            st.metric("Policy Queries", st.session_state.query_analytics['policy'])
        
        # Conversation Insights
        st.markdown("---")
        st.markdown("### 💡 Conversation Insights")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Topics Explored:**")
            topics = st.session_state.conversation_insights['topics_discussed']
            if topics:
                for topic in topics:
                    st.markdown(f"- ✅ {topic.title()}")
            else:
                st.info("Start asking questions to see topics!")
        
        with col_b:
            st.markdown("**Engagement Score:**")
            engagement = min(100, (total / 10) * 100) if total > 0 else 0
            st.progress(engagement / 100)
            st.caption(f"{engagement:.0f}% - {'Excellent!' if engagement > 70 else 'Good!' if engagement > 40 else 'Just getting started'}")
        
        if total > 0:
            fig = px.pie(
                values=list(st.session_state.query_analytics.values()),
                names=list(st.session_state.query_analytics.keys()),
                title="Query Distribution by Category"
            )
            st.plotly_chart(fig)
    
    # Tab 3: Calculator
    with tab3:
        st.markdown("### 📅 Leave Balance Calculator")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Employee Details")
            months = st.number_input("Months of Service", min_value=0, max_value=120, value=12)
            
            if st.button("Calculate Leave Balance"):
                st.session_state.leave_balance = calculate_leave_balance(months)
        
        with col2:
            st.markdown("#### Your Leave Balance")
            if hasattr(st.session_state, 'leave_balance'):
                lb = st.session_state.leave_balance
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Annual Leave", f"{lb['Annual Leave']} days")
                    st.metric("Sick Leave", f"{lb['Sick Leave']} days")
                with col_b:
                    st.metric("Casual Leave", f"{lb['Casual Leave']} days")
                    st.metric("Total Available", f"{lb['Total Available']} days")
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Annual', 'Sick', 'Casual'],
                        y=[lb['Annual Leave'], lb['Sick Leave'], lb['Casual Leave']],
                        marker_color=['#667eea', '#f093fb', '#4facfe']
                    )
                ])
                fig.update_layout(title="Leave Balance Overview")
                st.plotly_chart(fig)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🤖 Powered by AI | Built for Rooman Technologies
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()