"""
Beautiful Streamlit App - Complete Dutch Legal Analysis Results
Elegant, simplistic UI with comprehensive information display
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import anthropic
from document_processor import DocumentProcessor
import os
import time

# Page config
st.set_page_config(
    page_title="Dutch Legal Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Dutch Legal Document Analysis System - Real Results from AI Analysis"
    }
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 1rem;
    }
    
    /* Custom card styling */
    .custom-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Risk badge styling */
    .risk-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        color: white;
        margin: 5px 0;
    }
    
    /* Custom table styling */
    .analysis-table {
        border-collapse: collapse;
        width: 100%;
        margin-top: 20px;
    }
    
    .analysis-table th {
        background-color: #2c3e50;
        color: white;
        padding: 12px;
        text-align: left;
        font-weight: 600;
    }
    
    .analysis-table td {
        padding: 10px;
        border: 1px solid #e0e0e0;
        vertical-align: top;
    }
    
    /* Evidence box styling */
    .evidence-box {
        background-color: #f0f8ff;
        border-left: 4px solid #3498db;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    /* Recommendation box styling */
    .recommendation-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)  # Cache for 60 seconds only
def load_and_process_batch_results():
    """Load and properly process all batch results"""
    all_results = []
    
    # Load all JSON result files
    result_files = list(Path("results").glob("*.json"))
    
    for file in result_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Handle different file formats
                if isinstance(data, list):
                    # Direct analysis results - ensure all items have required fields
                    for item in data:
                        if isinstance(item, dict) and 'document' in item:
                            # Handle old format (May 29 files) that don't have check_id or risk_level
                            if 'check_id' not in item and 'check' in item:
                                # Old format: convert check number to check_id
                                item['check_id'] = f"check_{item['check']}"
                            
                            # Ensure we have a risk_level
                            if 'risk_level' not in item:
                                # Try to determine from risk field (old format)
                                if 'risk' in item:
                                    item['risk_level'] = item['risk'].lower()
                                # Or from compliance field
                                elif 'compliance' in item:
                                    compliance = item['compliance'].lower()
                                    if 'compliant' in compliance:
                                        item['risk_level'] = 'ok'
                                    elif 'non-compliant' in compliance or 'risk' in compliance:
                                        item['risk_level'] = 'risk'
                                    else:
                                        item['risk_level'] = 'missing'
                                else:
                                    item['risk_level'] = 'unknown'
                            
                            # Only add if we have required fields
                            if 'check_id' in item:
                                all_results.append(item)
                elif isinstance(data, dict) and 'results' in data:
                    # Batch format with results key
                    all_results.extend(data['results'])
                else:
                    # Original batch format
                    for item in data:
                        if isinstance(item, dict) and 'result' in item:
                            # Parse batch API format
                            custom_id = item.get("custom_id", "")
                            if "_check_" in custom_id:
                                parts = custom_id.split("_check_")
                                doc_part = parts[0]
                                check_num = parts[1]
                                check_id = f"check_{check_num}"
                                
                                # Parse document name
                                doc_name = doc_part.replace("_", " ") + ".pdf"
                                if "1745836913159 Share Purchase Agreement" in doc_name:
                                    doc_name = "1745836913159_Share Purchase Agreement.pdf"
                                elif "1745836913159 Converteerbare leningovereenkomst" in doc_name:
                                    doc_name = "1745836913159_Converteerbare leningovereenkomst.pdf"
                                elif "1745836913158 Subordination Agreement" in doc_name:
                                    doc_name = "1745836913158_Subordination Agreement.pdf"
                                else:
                                    doc_name = doc_part + ".pdf"
                                
                                # Extract analysis
                                if item.get("result", {}).get("type") == "succeeded":
                                    content = item["result"]["message"]["content"][0]["text"]
                                    if "```json" in content:
                                        content = content.split("```json")[1].split("```")[0]
                                    
                                    analysis = json.loads(content.strip())
                                    analysis.update({
                                        'document': doc_name,
                                        'check_id': check_id
                                    })
                                    all_results.append(analysis)
        except Exception as e:
            print(f"Error loading file {file}: {str(e)}")
            continue
    
    if not all_results:
        return None, None
    
    # Load playbook for reference
    with open("playbook_dutch_law.json", 'r') as f:
        playbook = json.load(f)
    check_lookup = {check['id']: check for check in playbook}
    
    # Process and group results - DEDUPLICATE by document+check_id
    processed_results = []
    documents = {}
    seen_combinations = {}  # Track document+check_id combinations
    
    # Sort results by timestamp if available (newest first)
    def get_timestamp(result):
        # Try to extract timestamp from various fields
        if 'timestamp' in result:
            return result['timestamp']
        if 'created_at' in result:
            return result['created_at']
        return ''  # Default to empty string for sorting
    
    all_results.sort(key=get_timestamp, reverse=True)
    
    for result in all_results:
        # Ensure we have required fields
        if 'document' in result and 'check_id' in result:
            # Create unique key for this document-check combination
            combo_key = f"{result['document']}_{result['check_id']}"
            
            # Skip if we've already seen this combination (keep first/newest)
            if combo_key in seen_combinations:
                continue
            
            seen_combinations[combo_key] = True
            
            # Add check details if missing
            check = check_lookup.get(result['check_id'], {})
            if not result.get('question'):
                result['question'] = check.get('question', '')
            if not result.get('category'):
                result['category'] = check.get('category', '')
            if not result.get('rule'):
                result['rule'] = check.get('rule', '')
            
            processed_results.append(result)
            
            # Group by document
            doc_name = result['document']
            if doc_name not in documents:
                documents[doc_name] = []
            documents[doc_name].append(result)
    
    return processed_results, documents

def get_risk_color(risk_level):
    """Get color based on risk level"""
    colors = {
        'ok': '#28a745',
        'risk': '#dc3545',
        'missing': '#ffc107',
        'error': '#6c757d'
    }
    return colors.get(risk_level.lower(), '#6c757d')

def get_risk_emoji(risk_level):
    """Get emoji based on risk level"""
    emojis = {
        'ok': '✅',
        'risk': '🚨',
        'missing': '⚠️',
        'error': '❌'
    }
    return emojis.get(risk_level.lower(), '❓')

def create_overview_table(documents, checks):
    """Create beautiful overview table for all checks"""
    # Dynamic column width based on number of checks
    num_checks = len(checks)
    col_width = 75.0 / num_checks if num_checks > 0 else 12.5
    
    # Create the table HTML with dynamic headers
    html = """
    <table class="analysis-table">
        <thead>
            <tr>
                <th style="width: 25%;">Document</th>
    """
    
    # Add headers for each check
    for check in checks:
        # Make headers unique by showing check number with category
        check_num = check['id'].split('_')[1]
        html += f'<th style="width: {col_width}%;">Check {check_num}: {check["category"]}<br/><small>{check["question"][:30]}...</small></th>'
    
    html += """
            </tr>
        </thead>
        <tbody>
    """
    
    for doc_name in sorted(documents.keys()):
        doc_results = documents[doc_name]
        html += f'<tr><td style="font-weight: bold;">{doc_name}</td>'
        
        # Add cells for each check
        for check in checks:
            check_id = check['id']
            check_result = next((r for r in doc_results if r.get('check_id') == check_id), None)
            
            if check_result:
                risk_color = get_risk_color(check_result.get('risk_level', 'unknown'))
                risk_emoji = get_risk_emoji(check_result.get('risk_level', 'unknown'))
                risk_text = check_result.get('risk_level', 'unknown').upper()
                
                # Create cell content
                answer_preview = check_result.get('answer', '')[:100] + '...' if len(check_result.get('answer', '')) > 100 else check_result.get('answer', '')
                
                cell_content = f"""
                <div style="border: 2px solid {risk_color}; border-radius: 8px; padding: 8px;">
                    <div style="background-color: {risk_color}; color: white; padding: 4px 8px; 
                                border-radius: 15px; display: inline-block; font-size: 11px; margin-bottom: 5px;">
                        {risk_emoji} {risk_text}
                    </div>
                    <div style="font-size: 12px; margin-top: 5px;">
                        {answer_preview}
                    </div>
                </div>
                """
                html += f'<td>{cell_content}</td>'
            else:
                html += '<td style="text-align: center; color: #999;">N/A</td>'
        
        html += '</tr>'
    
    html += '</tbody></table>'
    return html

def display_document_details(doc_name, doc_results):
    """Display detailed analysis for a single document - beautiful version"""
    st.markdown(f"# 📄 {doc_name}")
    
    # Document summary metrics
    risk_counts = {}
    for result in doc_results:
        risk = result.get('risk_level', 'unknown')
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    
    # Display metrics in a beautiful card
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("✅ Compliant", risk_counts.get('ok', 0))
    with col2:
        st.metric("🚨 Risk", risk_counts.get('risk', 0))
    with col3:
        st.metric("⚠️ Missing", risk_counts.get('missing', 0))
    with col4:
        st.metric("❌ Error", risk_counts.get('error', 0))
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display each check result in expandable sections
    sorted_results = sorted(doc_results, key=lambda x: x['check_id'])
    
    for result in sorted_results:
        check_num = result['check_id'].split('_')[1]
        risk_level = result.get('risk_level', 'unknown')
        risk_color = get_risk_color(risk_level)
        risk_emoji = get_risk_emoji(risk_level)
        
        # Create expander with styled header
        with st.expander(f"{risk_emoji} Check {check_num}: {result['category']} - {result['question']}", expanded=False):
            # Risk status badge
            st.markdown(
                f'<div class="risk-badge" style="background-color: {risk_color};">'
                f'{risk_level.upper()}</div>',
                unsafe_allow_html=True
            )
            
            # Answer section
            st.markdown("### 📝 Answer")
            st.write(result.get('answer', 'No answer provided'))
            
            # Legal Rationale
            if result.get('rationale'):
                st.markdown("### ⚖️ Legal Rationale")
                st.info(result['rationale'])
            
            # Relevant Law
            if result.get('relevant_law') and result['relevant_law'] != 'N/A':
                st.markdown("### 📚 Relevant Dutch Law")
                st.code(result['relevant_law'], language='text')
            
            # Evidence
            if result.get('evidence') and isinstance(result['evidence'], list):
                st.markdown("### 🔍 Evidence from Document")
                for i, evidence in enumerate(result['evidence'], 1):
                    st.markdown(f'<div class="evidence-box">', unsafe_allow_html=True)
                    
                    # Evidence header with location
                    location = evidence.get('location', 'Unknown')
                    st.markdown(f"**Evidence {i} - {location}**")
                    
                    # The quoted text
                    quoted_text = evidence.get('text', '')
                    if quoted_text:
                        # Show in a formatted quote box
                        st.markdown(f"""
                        <div style="background-color: #f0f0f0; padding: 15px; border-left: 4px solid #3498db;
                                    margin: 10px 0; font-style: italic; border-radius: 5px;">
                            "{quoted_text}"
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Page and article details
                    if "Page" in location:
                        st.caption(f"📄 **Source:** {location}")
                    
                    # Relevance explanation
                    if evidence.get('relevance'):
                        st.caption(f"💡 **Why this matters:** {evidence['relevance']}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Recommendations
            if result.get('recommendations'):
                st.markdown("### 💡 Recommendations")
                st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                st.warning(result['recommendations'])
                st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Header with gradient
    st.markdown("""
    <h1 style='text-align: center; background: linear-gradient(120deg, #2c3e50, #3498db); 
               color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
        🏛️ Dutch Legal Document Analysis Dashboard
    </h1>
    """, unsafe_allow_html=True)
    
    # Load results
    all_results, documents = load_and_process_batch_results()
    
    if not all_results:
        st.error("❌ No batch results found. Please run the batch retrieval first.")
        return
    
    # Sidebar with elegant styling
    with st.sidebar:
        st.markdown("## 📊 Analysis Overview")
        
        # Summary metrics
        st.metric("📑 Total Analyses", len(all_results))
        st.metric("📁 Documents Analyzed", len(documents))
        # Count actual checks
        with open("playbook_dutch_law.json", 'r', encoding='utf-8') as f:
            num_checks = len(json.load(f))
        st.metric("✅ Compliance Checks", f"{num_checks} per document")
        
        # Risk distribution chart
        st.markdown("### 🎯 Overall Risk Distribution")
        risk_counts = {}
        for result in all_results:
            risk = result.get('risk_level', 'unknown')
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        # Create donut chart
        if risk_counts:
            fig = go.Figure(data=[go.Pie(
                labels=list(risk_counts.keys()),
                values=list(risk_counts.values()),
                hole=.3,
                marker_colors=['#28a745', '#dc3545', '#ffc107', '#6c757d']
            )])
            fig.update_layout(
                showlegend=True,
                height=300,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 🗂️ Navigate")
        selected_view = st.radio(
            "Select View:",
            ["📊 Overview Dashboard", "📄 Document Details", "⚙️ Manage Checks", "📁 Manage Documents"],
            label_visibility="collapsed"
        )
        
        if selected_view == "📄 Document Details":
            selected_doc = st.selectbox(
                "Select Document:",
                list(documents.keys()),
                format_func=lambda x: x.replace(".pdf", "")
            )
        else:
            selected_doc = None
    
    # Main content area
    if selected_view == "📊 Overview Dashboard":
        # Overview dashboard
        st.markdown("## 📋 Compliance Analysis Matrix")
        st.markdown("*Comprehensive overview of all documents and compliance checks*")
        
        # Load current checks for table headers
        with open("playbook_dutch_law.json", 'r', encoding='utf-8') as f:
            current_checks = json.load(f)
        
        # Display the overview table
        table_html = create_overview_table(documents, current_checks)
        st.markdown(table_html, unsafe_allow_html=True)
        
        # Risk summary by document
        st.markdown("## 📈 Risk Summary by Document")
        
        summary_data = []
        for doc_name, doc_results in documents.items():
            risk_summary = {'Document': doc_name.replace('.pdf', '')}
            for result in doc_results:
                risk = result.get('risk_level', 'unknown')
                risk_summary[risk.upper()] = risk_summary.get(risk.upper(), 0) + 1
            summary_data.append(risk_summary)
        
        summary_df = pd.DataFrame(summary_data).fillna(0)
        
        # Create stacked bar chart
        fig = go.Figure()
        for risk_type in ['OK', 'RISK', 'MISSING', 'ERROR']:
            if risk_type in summary_df.columns:
                fig.add_trace(go.Bar(
                    name=risk_type,
                    x=summary_df['Document'],
                    y=summary_df[risk_type],
                    marker_color={'OK': '#28a745', 'RISK': '#dc3545', 
                                 'MISSING': '#ffc107', 'ERROR': '#6c757d'}[risk_type]
                ))
        
        fig.update_layout(
            barmode='stack',
            xaxis_tickangle=-45,
            height=400,
            margin=dict(b=100)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # High risk items
        st.markdown("## 🚨 High Risk Findings")
        high_risk_items = [r for r in all_results if r.get('risk_level') == 'risk']
        
        if high_risk_items:
            for item in high_risk_items:
                with st.expander(f"⚠️ {item['document']} - {item['question'][:80]}..."):
                    st.markdown(f"**Answer:** {item.get('answer', 'N/A')}")
                    if item.get('rationale'):
                        st.markdown(f"**Rationale:** {item['rationale']}")
                    if item.get('recommendations'):
                        st.warning(f"**Action Required:** {item['recommendations']}")
        else:
            st.success("✅ No high-risk items found!")
        
        # Export options
        st.markdown("---")
        st.markdown("## 📥 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV export
            df = pd.DataFrame(all_results)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Download CSV",
                data=csv,
                file_name="legal_analysis_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # JSON export
            json_str = json.dumps(all_results, indent=2)
            st.download_button(
                label="📄 Download JSON",
                data=json_str,
                file_name="legal_analysis_results.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            # Excel export placeholder
            st.button("📑 Generate Report", use_container_width=True, disabled=True,
                     help="Excel report generation coming soon")
    
    elif selected_view == "📄 Document Details":
        # Document details view
        if selected_doc and selected_doc in documents:
            display_document_details(selected_doc, documents[selected_doc])
    
    elif selected_view == "⚙️ Manage Checks":
        # Check Management View
        st.markdown("## ⚙️ Manage Compliance Checks")
        st.markdown("*Add, edit or remove compliance checks and re-analyze documents*")
        
        # Load current playbook
        playbook_file = "playbook_dutch_law.json"
        with open(playbook_file, 'r', encoding='utf-8') as f:
            checks = json.load(f)
        
        # Tabs for different actions
        tab1, tab2, tab3, tab4 = st.tabs(["📋 View Checks", "➕ Add Check", "✏️ Edit Check", "🔄 Re-Analyze"])
        
        with tab1:
            st.markdown("### Current Compliance Checks")
            for check in checks:
                with st.expander(f"{check['id']}: {check['category']} - {check['question'][:60]}..."):
                    st.markdown(f"**Question:** {check['question']}")
                    st.markdown(f"**Category:** {check['category']}")
                    st.markdown(f"**Legal Rule:** {check['rule']}")
                    if check.get('created_at'):
                        st.caption(f"Created: {check['created_at']}")
        
        with tab2:
            st.markdown("### Add New Compliance Check")
            with st.form("add_check_form"):
                new_question = st.text_area("Question", placeholder="What specific aspect should be checked?")
                new_rule = st.text_area("Legal Rule", placeholder="Reference to Dutch law and compliance requirements")
                new_category = st.selectbox("Category",
                    ["Financial", "Contract Terms", "Corporate", "Legal", "Employment", "IP",
                     "Environmental", "Data Protection", "Tax", "Warranties", "Dispute Resolution"])
                
                if st.form_submit_button("Add Check"):
                    if new_question and new_rule and new_category:
                        # Generate new ID
                        existing_ids = [int(c['id'].split('_')[1]) for c in checks if c['id'].startswith('check_')]
                        new_id_num = max(existing_ids) + 1 if existing_ids else 1
                        new_id = f"check_{new_id_num}"
                        
                        new_check = {
                            "id": new_id,
                            "question": new_question,
                            "rule": new_rule,
                            "category": new_category,
                            "created_at": datetime.now().isoformat()
                        }
                        
                        checks.append(new_check)
                        
                        # Save updated playbook
                        with open(playbook_file, 'w', encoding='utf-8') as f:
                            json.dump(checks, f, indent=2, ensure_ascii=False)
                        
                        st.success(f"✅ Added new check: {new_id}")
                        st.rerun()
                    else:
                        st.error("Please fill in all fields")
        
        with tab3:
            st.markdown("### Edit Existing Check")
            
            check_to_edit = st.selectbox("Select check to edit",
                options=[f"{c['id']}: {c['question'][:60]}..." for c in checks])
            
            if check_to_edit:
                check_id = check_to_edit.split(":")[0]
                current_check = next(c for c in checks if c['id'] == check_id)
                
                with st.form("edit_check_form"):
                    edited_question = st.text_area("Question", value=current_check['question'])
                    edited_rule = st.text_area("Legal Rule", value=current_check['rule'])
                    edited_category = st.selectbox("Category",
                        ["Financial", "Contract Terms", "Corporate", "Legal", "Employment", "IP",
                         "Environmental", "Data Protection", "Tax", "Warranties", "Dispute Resolution"],
                        index=["Financial", "Contract Terms", "Corporate", "Legal", "Employment", "IP",
                               "Environmental", "Data Protection", "Tax", "Warranties", "Dispute Resolution"].index(current_check['category']))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.form_submit_button("Update Check", type="primary"):
                            # Update the check
                            for check in checks:
                                if check['id'] == check_id:
                                    check['question'] = edited_question
                                    check['rule'] = edited_rule
                                    check['category'] = edited_category
                                    check['updated_at'] = datetime.now().isoformat()
                                    break
                            
                            # Save updated playbook
                            with open(playbook_file, 'w', encoding='utf-8') as f:
                                json.dump(checks, f, indent=2, ensure_ascii=False)
                            
                            st.success(f"✅ Updated {check_id}")
                            st.rerun()
                    
                    with col2:
                        if st.form_submit_button("Delete Check", type="secondary"):
                            if st.checkbox("Confirm deletion"):
                                checks = [c for c in checks if c['id'] != check_id]
                                
                                # Save updated playbook
                                with open(playbook_file, 'w', encoding='utf-8') as f:
                                    json.dump(checks, f, indent=2, ensure_ascii=False)
                                
                                st.success(f"✅ Deleted {check_id}")
                                st.rerun()
        
        with tab4:
            st.markdown("### Re-Analyze Documents with Updated Checks")
            st.info("This will analyze all documents using the current set of compliance checks.")
            
            # Show which documents are available
            doc_files = [f for f in os.listdir("documents") if f.endswith(('.pdf', '.docx'))]
            st.markdown(f"**Available documents:** {len(doc_files)}")
            
            # Show current number of checks
            st.markdown(f"**Current compliance checks:** {len(checks)}")
            
            if st.button("🔄 Start Re-Analysis", type="primary"):
                with st.spinner("Analyzing documents... This may take a few minutes"):
                    # Initialize API client
                    API_KEY = "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"
                    MODEL = "claude-3-5-sonnet-20241022"  # Latest Sonnet 3.5 with enhanced capabilities
                    client = anthropic.Anthropic(api_key=API_KEY)
                    processor = DocumentProcessor()
                    
                    # Import the analyze function
                    from analyze_new_documents import analyze_single_document
                    
                    all_results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, doc_file in enumerate(doc_files):
                        doc_path = os.path.join("documents", doc_file)
                        status_text.text(f"Analyzing {doc_file}...")
                        
                        results = analyze_single_document(client, processor, doc_path, checks)
                        all_results.extend(results)
                        
                        progress_bar.progress((i + 1) / len(doc_files))
                    
                    # Save results
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = f"results/reanalysis_{timestamp}.json"
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(all_results, f, indent=2, ensure_ascii=False)
                    
                    status_text.text("Analysis complete!")
                    st.success(f"✅ Re-analysis complete! Results saved to {output_file}")
                    
                    # Clear cache to show new results
                    st.cache_data.clear()
                    
                    # Display summary
                    risk_summary = {'ok': 0, 'risk': 0, 'missing': 0, 'error': 0}
                    for r in all_results:
                        risk_summary[r.get('risk_level', 'error')] += 1
                    
                    st.markdown("### Analysis Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("✅ OK", risk_summary['ok'])
                    with col2:
                        st.metric("🚨 Risk", risk_summary['risk'])
                    with col3:
                        st.metric("⚠️ Missing", risk_summary['missing'])
                    with col4:
                        st.metric("❌ Error", risk_summary['error'])
    
    elif selected_view == "📁 Manage Documents":
        # Document Management View
        st.markdown("## 📁 Document Management")
        st.markdown("*Add documents to the 'documents' folder and analyze them here*")
        
        # Get all documents in folder
        all_docs = [f for f in os.listdir("documents") if f.endswith(('.pdf', '.docx', '.PDF', '.DOCX'))]
        
        # Check which are already analyzed
        analyzed_docs = set()
        new_docs = []
        
        # Load existing results to determine analyzed docs
        result_files = list(Path("results").glob("*.json"))
        for file in result_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if 'document' in item:
                                analyzed_docs.add(item['document'])
            except:
                pass
        
        # Determine new documents
        new_docs = [d for d in all_docs if d not in analyzed_docs]
        
        # Document statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Total Documents", len(all_docs))
        with col2:
            st.metric("✅ Already Analyzed", len(analyzed_docs))
        with col3:
            st.metric("🆕 New Documents", len(new_docs))
        
        # Tabs for different views
        tab1, tab2 = st.tabs(["📋 All Documents", "🆕 New Documents"])
        
        with tab1:
            st.markdown("### All Documents in Folder")
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Update List", use_container_width=True):
                    st.rerun()
            
            # Display all documents
            for i, doc in enumerate(sorted(all_docs), 1):
                status = "✅ Analyzed" if doc in analyzed_docs else "🆕 New"
                st.write(f"{i}. **{doc}** - {status}")
        
        with tab2:
            st.markdown("### New Documents Ready for Analysis")
            
            if new_docs:
                st.info(f"Found {len(new_docs)} new document(s) ready for analysis")
                
                # List new documents
                for i, doc in enumerate(sorted(new_docs), 1):
                    st.write(f"{i}. **{doc}**")
                
                # Analyze button
                if st.button("🚀 Analyze New Documents", type="primary", use_container_width=True):
                    with st.spinner(f"Analyzing {len(new_docs)} new documents... This may take a few minutes"):
                        # Initialize API client
                        API_KEY = "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"
                        MODEL = "claude-3-5-sonnet-20241022"  # Latest Sonnet 3.5 with enhanced capabilities
                        client = anthropic.Anthropic(api_key=API_KEY)
                        processor = DocumentProcessor()
                        
                        # Load playbook
                        with open("playbook_dutch_law.json", 'r', encoding='utf-8') as f:
                            playbook = json.load(f)
                        
                        # Import analyze function
                        from analyze_new_documents import analyze_single_document
                        
                        all_results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Analyze each new document
                        for i, doc_file in enumerate(new_docs):
                            doc_path = os.path.join("documents", doc_file)
                            status_text.text(f"Analyzing {doc_file}...")
                            
                            results = analyze_single_document(client, processor, doc_path, playbook)
                            all_results.extend(results)
                            
                            progress_bar.progress((i + 1) / len(new_docs))
                        
                        # Save results
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_file = f"results/new_docs_analysis_{timestamp}.json"
                        
                        Path("results").mkdir(exist_ok=True)
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(all_results, f, indent=2, ensure_ascii=False)
                        
                        status_text.text("Analysis complete!")
                        st.success(f"✅ Successfully analyzed {len(new_docs)} documents!")
                        
                        # Display summary
                        risk_summary = {'ok': 0, 'risk': 0, 'missing': 0, 'error': 0}
                        for r in all_results:
                            risk_summary[r.get('risk_level', 'error')] += 1
                        
                        st.markdown("### Analysis Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("✅ OK", risk_summary['ok'])
                        with col2:
                            st.metric("🚨 Risk", risk_summary['risk'])
                        with col3:
                            st.metric("⚠️ Missing", risk_summary['missing'])
                        with col4:
                            st.metric("❌ Error", risk_summary['error'])
                        
                        st.info(f"Results saved to: {output_file}")
                        
                        # Clear cache to show new results
                        st.cache_data.clear()
                        
                        # Offer to refresh
                        if st.button("🔄 Refresh Page"):
                            st.rerun()
            else:
                st.success("✅ All documents have been analyzed! No new documents found.")
                st.info("To add more documents, simply place PDF or DOCX files in the 'documents' folder and click 'Update List'")

if __name__ == "__main__":
    main()