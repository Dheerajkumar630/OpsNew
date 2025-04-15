# Advanced Agentic Visualization System for JIRA Analytics

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import re
from pandas.api.types import is_numeric_dtype
from datetime import datetime, timedelta
import logging
import json
import psycopg2
import streamlit as st
import plotly.express as px
import anthropic
from datetime import datetime, timedelta, date
from sqlalchemy import create_engine
import os
import logging
import re
import io



def is_numeric_type(series):
    """Check if a pandas Series contains numeric data"""
    try:
        return is_numeric_dtype(series) or pd.to_numeric(series, errors='coerce').notna().any()
    except:
        return False
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelContextProtocol:
    """
    Implements the Model Context Protocol for structured AI interactions.
    Helps the model understand the context, constraints, and capabilities of the system.
    """
    
    def __init__(self, data_context=None, user_query=None, system_capabilities=None, analysis_history=None):
        self.data_context = data_context
        self.user_query = user_query
        self.system_capabilities = system_capabilities or self._get_default_capabilities()
        self.analysis_history = analysis_history or []
        
    def _get_default_capabilities(self):
        """Define the default capabilities of the JIRA analytics system."""
        return {
            "data_analysis": [
                "Perform statistical analysis on JIRA issue data",
                "Compare metrics across different dimensions (projects, developers, etc.)",
                "Calculate productivity and efficiency metrics",
                "Identify trends and patterns in issue resolution",
                "Analyze estimation accuracy"
            ],
            "visualization_types": [
                "Bar charts for categorical comparisons",
                "Pie charts for distributions",
                "Line charts for trends over time",
                "Scatter plots for correlation analysis",
                "Box plots for statistical distributions",
                "Heatmaps for two-dimensional analysis",
                "Timeline charts for temporal visualization"
            ],
            "available_filters": [
                "Date ranges for creation and resolution",
                "Project categories",
                "Issue types",
                "Platforms",
                "Products",
                "Fix versions",
                "Priority levels"
            ]
        }
        
    def format_for_claude(self):
        """Format the context information for Claude according to Model Context Protocol."""
        protocol_blocks = []
        
        # System Capabilities Block
        capabilities_block = """
<context>
<capabilities>
The JIRA Analytics System can:
"""
        for category, capabilities in self.system_capabilities.items():
            capabilities_block += f"\n{category.replace('_', ' ').title()}:\n"
            for capability in capabilities:
                capabilities_block += f"- {capability}\n"
        
        capabilities_block += """
</capabilities>
</context>
"""
        protocol_blocks.append(capabilities_block)
        
        # Data Context Block
        if self.data_context:
            data_block = """
<context>
<data_summary>
"""
            # Provide a summary of the data being analyzed
            if isinstance(self.data_context, dict):
                if 'summary_stats' in self.data_context:
                    stats = self.data_context['summary_stats']
                    data_block += f"Total Issues: {stats.get('total_issues', 'N/A')}\n"
                    
                    # Add key distributions if available
                    for field in ['priority', 'status', 'issue_type', 'project_category']:
                        if f"{field}_distribution" in stats:
                            data_block += f"\n{field.replace('_', ' ').title()} Distribution:\n"
                            for value, count in stats[f"{field}_distribution"].items():
                                data_block += f"- {value}: {count}\n"
                    
                    # Add date ranges
                    for field in ['created', 'resolved']:
                        if f"{field}_date_range" in stats:
                            date_range = stats[f"{field}_date_range"]
                            data_block += f"\n{field.capitalize()} Date Range: {date_range.get('earliest', 'N/A')} to {date_range.get('latest', 'N/A')}\n"
                
                if 'metadata' in self.data_context:
                    metadata = self.data_context['metadata']
                    data_block += "\nAvailable Fields:\n"
                    for category, fields in metadata.items():
                        if isinstance(fields, list) and fields:
                            data_block += f"- {category.replace('_', ' ').title()}: {', '.join(fields[:5])}"
                            if len(fields) > 5:
                                data_block += f" and {len(fields) - 5} more"
                            data_block += "\n"
            
            data_block += """
</data_summary>
</context>
"""
            protocol_blocks.append(data_block)
        
        # History and Context Block
        if self.analysis_history:
            history_block = """
<context>
<conversation_history>
"""
            for entry in self.analysis_history[-3:]:  # Show the last 3 entries for brevity
                history_block += f"Query: {entry.get('query', 'N/A')}\n"
                if 'key_findings' in entry:
                    history_block += "Key Findings:\n"
                    for finding in entry['key_findings'][:3]:  # Show top 3 findings
                        history_block += f"- {finding}\n"
                history_block += "\n"
                
            history_block += """
</conversation_history>
</context>
"""
            protocol_blocks.append(history_block)
        
        # Current Query Block
        if self.user_query:
            query_block = f"""
<context>
<current_query>
{self.user_query}
</current_query>
</context>
"""
            protocol_blocks.append(query_block)
        
        return "\n".join(protocol_blocks)

# Set page config
st.set_page_config(
    page_title="JIRA Analytics with Claude",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check for configuration
if not hasattr(st, "secrets") or not all(key in st.secrets for key in ["db_host", "db_name", "db_user", "db_password", "db_port", "anthropic_api_key"]):
    # For local development, you can use this section
    # In production, use st.secrets
    if not os.environ.get("STREAMLIT_SECRETS_LOADED"):
        st.error("Database credentials and API key not found. Please configure them in .streamlit/secrets.toml or environment variables.")
        
        # For demo purposes, show a sample format
        st.code("""
        # .streamlit/secrets.toml example:
        db_host = "your-jira-db-host"
        db_name = "jiradb"
        db_user = "your-username"
        db_password = "your-password"
        db_port = "5432"
        anthropic_api_key = "your-anthropic-api-key"
        """)
        st.stop()

# Database connection
def connect_to_jira_db():
    """Connect to JIRA PostgreSQL database using environment variables or secrets"""
    try:
        # Try to get credentials from Streamlit secrets
        conn_params = {
            "host": st.secrets.get("db_host"),
            "database": st.secrets.get("db_name"),
            "user": st.secrets.get("db_user"),
            "password": st.secrets.get("db_password"),
            "port": st.secrets.get("db_port")
        }
    except Exception:
        # Fallback to environment variables
        conn_params = {
            "host": os.environ.get("JIRA_DB_HOST"),
            "database": os.environ.get("JIRA_DB_NAME"),
            "user": os.environ.get("JIRA_DB_USER"),
            "password": os.environ.get("JIRA_DB_PASSWORD"),
            "port": os.environ.get("JIRA_DB_PORT")
        }
    
    try:
        conn = psycopg2.connect(**conn_params)
        logger.info("Successfully connected to JIRA database")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

# Data retrieval functions
@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_jira_data(query, params=None):
    """Execute SQL query and return results as DataFrame"""
    conn = connect_to_jira_db()
    if conn:
        try:
            df = pd.read_sql(query, conn, params=params)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            conn.close()
    return None
# Define enhanced_system_prompt with detailed field descriptions
enhanced_system_prompt = """
You are an expert JIRA analytics agent with deep knowledge of software development, project management, and agile methodologies. You have specialized knowledge about JIRA fields and their relationships.

## JIRA Field Knowledge:

### Issue Tracking Fields:
- ISSUEKEY: Unique identifier for JIRA issues, typically in format "PROJECT-123"
- PROJECT_KEY: Key identifier for the project
- SUMMARY: Brief description of the issue
- ISSUE_TYPE: Categorizes issues (Bug, Story, Epic, Task, etc.)
  * Bugs: Represent defects that need to be fixed
  * Stories: User-centric requirements that deliver value
  * Epics: Large bodies of work containing multiple stories
  * Tasks: General work items not tied directly to user value
- PRIORITY: Indicates importance (Blocker, Critical, Major, Minor, Trivial)
  * Priority affects scheduling, resource allocation, and release planning
- STATUS: Current workflow position (Open, In Progress, In Review, Done, etc.)
- STATUS_CATEGORY: Groups statuses (To Do, In Progress, Done)
- RESOLUTION: How an issue was resolved (Fixed, Won't Fix, Duplicate, etc.)
- PARENT: Parent issue key for hierarchical relationships
- EPIC_NAME: Name of the epic if the issue is an epic

### Time Tracking Fields:
- CREATED: When the issue was created
- UPDATED: When the issue was last modified
- RESOLVED: When the issue was marked as resolved
- DUE_DATE: Deadline for issue completion
- PLANNED_START: Scheduled start date
- PLANNED_END: Scheduled end date
- ORIGINAL_ESTIMATE: Initial time estimate in seconds (convert to hours by dividing by 3600)
- TIME_SPENT: Actual time logged in seconds (convert to hours by dividing by 3600)
- ROM_ESTIMATES_HOURS_: Rough Order of Magnitude estimate in hours
- STORY_POINTS: Effort estimation in story points
- T_SHIRT_SIZE: T-shirt size estimation (S, M, L, XL, etc.)

### People Fields:
- ASSIGNEE: Person currently responsible for the issue or who was assigned to that issue
- REPORTER: Person who created the issue
- ORIGINAL_DEVELOPER: Person who worked on the issue or wrote the code
- CODE_REVIEWED_BY: Person who reviewed the code
- ISSUE_CAUSED_BY: Developer who introduced the bug (for defects)

### Project Fields:
- PROJECT: The project the issue belongs to
- PROJECT_CATEGORY: Higher-level grouping of projects
- FIX_VERSIONS: Version(s) where this issue will be/was fixed or Release where the issue will be fixed
- RELEASE_STATUS: Whether the fix version has been released (true/false)
- RELEASE_DATE: When the fix version was/will be released date
- HOTFIX_RELEASE_CYCLE: Information about hotfix release cycles

### Technical Fields:
- PLATFORM_OS: Operating system platforms affected
- PRODUCT: Product affected by the issue
- CUSTOMER_TYPE: Type of customer impacted
- CUSTOMER: Specific customer affected by the issue
- FIGMA_LINK: Link to design specifications
- DUT_LINK: Link to developer unit testing documentation
- FIT_GAP_ANALYSIS: Assessment of how solution fits requirements
- BUG_TYPE: Category of bug 
- ROOT_CAUSE_ANALYSIS: Analysis of underlying issue causes
- FUNCTIONAL_SPECIFICATIONS: Details of functional specifications
- DESIGN_DOCUMENT_LINK: Link to design documentation
- MODULE_FEATURE: Module and feature information (parent-child hierarchy)

### Business & Process Fields:
- BUSINESS_REASON: Business justification for the issue
- ACCEPTANCE_CRITERIA: Criteria for accepting the solution
- LABEL: Labels attached to the issue
- VALUE_DRIVER: Business value driver category
- PHASE: Current development or project phase
- PHASE_BLOCKER: Issues blocking progress in the current phase

## Analysis Context:
Understanding relationships between fields is crucial:
- Resolution time patterns differ by priority, issue type, and project
- Estimation accuracy varies by developer, issue type, and complexity
- Bug patterns may correlate with platforms, products, or releases
- Developer productivity metrics should account for issue complexity and issue type
- Value drivers and business reasons provide insight into strategic alignment
- Module/feature data helps identify problem areas in the product architecture

## Agile Development Context:
- Sprint Cycles: Issues typically follow sprint patterns (usually 2-week cycles)
- Velocity: Team capacity measured by story points/issues completed per sprint
- Technical Debt: Bugs and maintenance tasks that accumulate over time
- Lead Time: Total time from issue creation to resolution
- Cycle Time: Time from work starting to completion

Your analysis should:
1. Focus only on what the user is asking for - don't provide tangential information and provide the relevant analysis.

## Model Context Protocol
Pay attention to the context blocks provided in the Model Context Protocol format.
These will give you information about system capabilities, data summary, and query history.
"""

# First, get lookup values for filters (cached function)
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_filter_options():
    """Fetch filter options from the database for dropdowns"""
    filter_options = {}
    
    # Get all project categories
    query = 'SELECT DISTINCT "NAME" FROM public."PROJECT_CATEGORIES" WHERE "NAME" IS NOT NULL ORDER BY "NAME"'
    df = get_jira_data(query)
    if df is not None and not df.empty:
        filter_options["project_categories"] = df["NAME"].tolist()
    else:
        filter_options["project_categories"] = []
    
    # Get all priorities
    query = 'SELECT DISTINCT "NAME" FROM public."PRIORITIES" WHERE "NAME" IS NOT NULL ORDER BY "NAME"'
    df = get_jira_data(query)
    if df is not None and not df.empty:
        filter_options["priorities"] = df["NAME"].tolist()
    else:
        filter_options["priorities"] = []
    
    # Get all issue types
    query = 'SELECT DISTINCT "NAME" FROM public."ISSUE_TYPES" WHERE "NAME" IS NOT NULL ORDER BY "NAME"'
    df = get_jira_data(query)
    if df is not None and not df.empty:
        filter_options["issue_types"] = df["NAME"].tolist()
    else:
        filter_options["issue_types"] = []
    
    # Get all platforms (this is trickier since it's aggregated in our query)
    query = '''
    SELECT DISTINCT "VALUE" FROM public."FIELD_OPTIONS" 
    WHERE "ID" IN (
        SELECT DISTINCT "FIELD_OPTION_ID" FROM public."ISSUE_PLATFORM_OS"
    )
    AND "VALUE" IS NOT NULL
    ORDER BY "VALUE"
    '''
    df = get_jira_data(query)
    if df is not None and not df.empty:
        filter_options["platforms"] = df["VALUE"].tolist()
    else:
        filter_options["platforms"] = []
    
    # Get all products
    query = '''
    SELECT DISTINCT "VALUE" FROM public."FIELD_OPTIONS" 
    WHERE "ID" IN (
        SELECT DISTINCT "FIELD_OPTION_ID" FROM public."ISSUE_PRODUCT"
    )
    AND "VALUE" IS NOT NULL
    ORDER BY "VALUE"
    '''
    df = get_jira_data(query)
    if df is not None and not df.empty:
        filter_options["products"] = df["VALUE"].tolist()
    else:
        filter_options["products"] = []
    
    # Get all projects
    query = 'SELECT DISTINCT "NAME" FROM public."PROJECTS" WHERE "NAME" IS NOT NULL ORDER BY "NAME"'
    df = get_jira_data(query)
    if df is not None and not df.empty:
        filter_options["projects"] = df["NAME"].tolist()
    else:
        filter_options["projects"] = []
    
    # Get all fix versions
    query = 'SELECT DISTINCT "NAME" FROM public."PROJECT_RELEASES" WHERE "NAME" IS NOT NULL ORDER BY "NAME"'
    df = get_jira_data(query)
    if df is not None and not df.empty:
        filter_options["fix_versions"] = df["NAME"].tolist()
    else:
        filter_options["fix_versions"] = []
    
    return filter_options

# Then, update the get_filtered_jira_data function to handle multi-select values
@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_filtered_jira_data_from_nl(query_params=None, limit=500):
    """
    Get JIRA data based on natural language generated filters, fetching only relevant fields
    
    Args:
        query_params: Dictionary containing filters and relevant fields
        limit: Maximum number of records to return
    
    Returns:
        DataFrame with JIRA issues containing only the relevant fields
    """
    if query_params is None:
        query_params = {
            "filters": {},
            "relevant_fields": ["ISSUEKEY", "SUMMARY", "ISSUE_TYPE", "PRIORITY", "STATUS", "CREATED", "RESOLVED"]
        }
    
    filters = query_params.get("filters", {})
    relevant_fields = query_params.get("relevant_fields", [])
    
    # Ensure we always have some basic fields for identification
    essential_fields = ["ISSUEKEY", "SUMMARY", "ISSUE_TYPE", "STATUS"]
    for field in essential_fields:
        if field not in relevant_fields:
            relevant_fields.append(field)
    
    # Get the database connection
    conn = connect_to_jira_db()
    if not conn:
        return None
        
    try:
        # Create a cursor for direct execution
        cursor = conn.cursor()
        
        # Map relevant fields to actual columns in the query
        # This is a simplified mapping - you would need to expand this based on your schema
        field_mapping = {
            "ISSUEKEY": "F.\"ISSUEKEY\"",
            "SUMMARY": "F.\"SUMMARY\"",
            "ISSUE_TYPE": "I.\"NAME\" as \"ISSUE_TYPE\"",
            "PRIORITY": "PR.\"NAME\" as \"PRIORITY\"",
            "STATUS": "S.\"NAME\" as \"STATUS\"",
            "STATUS_CATEGORY": "SE.\"NAME\" AS \"STATUS_CATEGORY\"",
            "CREATED": "F.\"CREATED\"",
            "UPDATED": "F.\"UPDATED\"",
            "RESOLVED": "F.\"RESOLVED\"",
            "DUE_DATE": "F.\"DUE_DATE\"",
            "PLANNED_START": "F.\"PLANNED_START\"",
            "PLANNED_END": "F.\"PLANNED_END\"",
            "ORIGINAL_ESTIMATE": "F.\"ORIGINAL_ESTIMATE\"",
            "TIME_SPENT": "F.\"TIME_SPENT\"",
            "ORIGINAL_DEVELOPER": "U.\"DISPLAY_NAME\" AS \"ORIGINAL_DEVELOPER\"",
            "ASSIGNEE": "US.\"DISPLAY_NAME\" as \"ASSIGNEE\"",
            "REPORTER": "USE.\"DISPLAY_NAME\" as \"REPORTER\"",
            "ISSUE_CAUSED_BY": "USES.\"DISPLAY_NAME\" as \"ISSUE_CAUSED_BY\"",
            "CODE_REVIEWED_BY": "USS.\"DISPLAY_NAME\" as \"CODE_REVIEWED_BY\"",
            "PROJECT": "P.\"NAME\" as \"PROJECT\"",
            "PROJECT_CATEGORY": "PC.\"NAME\" AS \"PROJECT_CATEGORY\"",
            "RESOLUTION": "R.\"NAME\" as \"RESOLUTION\"",
            "PLATFORM_OS": "PA.\"PLATFORM_OS\" as \"PLATFORM_OS\"",
            "PRODUCT": "PRO.\"PRODUCT\"",
            "FIX_VERSIONS": "FIX.\"FIX_VERSIONS\"",
            "RELEASE_STATUS": "FIX.\"RELEASE_STATUS\"",
            "RELEASE_DATE": "FIX.\"RELEASE_DATE\"",
            "CUSTOMER_TYPE": "FO.\"VALUE\" AS \"CUSTOMER_TYPE\"",
            "BUG_TYPE": "FOO.\"VALUE\" AS \"BUG_TYPE\"",
            "HOTFIX_RELEASE_CYCLE": "FIE.\"VALUE\" AS \"HOTFIX_RELEASE_CYCLE\"",
            "PARENT": "F.\"PARENT\"",
            "LABEL": "LA.\"LABEL\"",
            "VALUE_DRIVER": "VD.\"VALUE_DRIVER\"",
            "MODULE_FEATURE": "CONCAT(F.\"MODULE_FEATURE_PARENT\",' - ',F.\"MODULE_FEATURE_CHILD\") AS \"MODULE_FEATURE\"",
            "PHASE": "FIED.\"VALUE\" AS \"PHASE\"",
            "PHASE_BLOCKER": "PBL.\"PHASE_BLOCKER\"",
            "T_SHIRT_SIZE": "FIEOP.\"VALUE\" AS \"T_SHIRT_SIZE\"",
            "CUSTOMER": "CS.\"CUSTOMER\"",
            "STORY_POINTS": "F.\"STORY_POINTS\"",
            "ROM_ESTIMATES_HOURS_": "F.\"ROM_ESTIMATES_HOURS_\"",
            "ACCEPTANCE_CRITERIA": "F.\"ACCEPTANCE_CRITERIA\"",
            "BUSINESS_REASON": "F.\"BUSINESS_REASON\"",
            "FIGMA_LINK": "F.\"FIGMA_LINK\"",
            "FIT_GAP_ANALYSIS": "F.\"FIT_GAP_ANALYSIS\"",
            "EPIC_NAME": "F.\"EPIC_NAME\"",
            "ROOT_CAUSE_ANALYSIS": "F.\"ROOT_CAUSE_ANALYSIS\"",
            "FUNCTIONAL_SPECIFICATIONS": "F.\"FUNCTIONAL_SPECIFICATIONS\"",
            "DESIGN_DOCUMENT_LINK": "F.\"DESIGN_DOCUMENT_LINK\"",
            "DUT_LINK": "F.\"DUT_LINK\"",
            "JIRA_ISSUE_ID": "F.\"JIRA_ISSUE_ID\""
        }
        
        # Select only the mapped fields that are in relevant_fields
        selected_columns = []
        for field in relevant_fields:
            if field in field_mapping:
                selected_columns.append(field_mapping[field])
        
        # If no valid fields were found, use some essential ones
        if not selected_columns:
            selected_columns = [
                field_mapping["ISSUEKEY"], 
                field_mapping["SUMMARY"], 
                field_mapping["ISSUE_TYPE"],
                field_mapping["PRIORITY"],
                field_mapping["STATUS"]
            ]
        
        # Build the query with necessary CTEs
        # Only include CTEs that are needed for the selected fields
        cte_parts = []
        
        # Always need these base CTEs
        cte_parts.append("""
        PlatformAggregation AS (
            SELECT
                I."KEY",
                STRING_AGG(FO."VALUE", ', ') AS "PLATFORM_OS"
            FROM public."ISSUES" I
            LEFT JOIN public."ISSUE_PLATFORM_OS" PO ON I."ID" = PO."ISSUE_ID"
            LEFT JOIN public."FIELD_OPTIONS" FO ON PO."FIELD_OPTION_ID" = FO."ID"
            GROUP BY
                I."KEY"
        )""")
        
        # Only include other CTEs if their fields are needed
        if any(field in relevant_fields for field in ["FIX_VERSIONS", "RELEASE_STATUS", "RELEASE_DATE"]):
            cte_parts.append("""
            fixversion AS(
                SELECT 
                    ISS."KEY",
                    STRING_AGG(PR."NAME",',') AS "FIX_VERSIONS",
                    PR."RELEASED" AS "RELEASE_STATUS",
                    PR."USER_RELEASE_DATE" AS "RELEASE_DATE"
                FROM public."ISSUES" ISS
                LEFT JOIN public."ISSUE_FIX_VERSIONS" IFV ON ISS."ID"=IFV."ISSUE_ID"
                LEFT JOIN public."PROJECT_RELEASES" PR ON IFV."PROJECT_RELEASE_ID"=PR."ID"
                GROUP BY 
                    ISS."KEY",PR."RELEASED",PR."USER_RELEASE_DATE"
            )""")
        
        if "PRODUCT" in relevant_fields:
            cte_parts.append("""
            Product AS(
                SELECT 
                    ISSUE."KEY",
                    STRING_AGG(FO."VALUE", ', ') AS "PRODUCT"
                FROM 
                    public."ISSUES" ISSUE
                    LEFT JOIN public."ISSUE_PRODUCT" PO ON ISSUE."ID" = PO."ISSUE_ID"
                    LEFT JOIN public."FIELD_OPTIONS" FO ON PO."FIELD_OPTION_ID" = FO."ID"
                GROUP BY
                    ISSUE."KEY"
            )""")
        
        if "LABEL" in relevant_fields:
            cte_parts.append("""
            Labels AS (
                SELECT 
                    ISSU."KEY" as "KEY",
                    STRING_AGG(IL."VALUE",', ') AS "LABEL"
                FROM 
                    public."ISSUES" ISSU
                    LEFT JOIN public."ISSUE_LABELS" IL ON ISSU."ID" = IL."ISSUE_ID"
                GROUP BY
                    ISSU."KEY"
            )""")
        
        if "PHASE_BLOCKER" in relevant_fields:
            cte_parts.append("""
            PHASEBLOCKER AS (
                SELECT 
                    ISSSUE."KEY" AS "KEY",
                    STRING_AGG(FIEOP."VALUE",', ') AS "PHASE_BLOCKER"
                FROM 
                    public."ISSUES" ISSSUE
                    LEFT JOIN public."ISSUE_PHASE_BLOCKER" IPB ON ISSSUE."ID" = IPB."ISSUE_ID"
                    LEFT JOIN public."FIELD_OPTIONS" FIEOP ON IPB."FIELD_OPTION_ID" = FIEOP."ID"
                GROUP BY 
                    ISSSUE."KEY"
            )""")
        
        if "CUSTOMER" in relevant_fields:
            cte_parts.append("""
            Customer AS(
                SELECT 
                    ISSUE."KEY",
                    STRING_AGG(FO."VALUE", ', ') AS "CUSTOMER"
                FROM 
                    public."ISSUES" ISSUE
                    LEFT JOIN public."ISSUE_CUSTOMER" IC ON ISSUE."ID" = IC."ISSUE_ID"
                    LEFT JOIN public."FIELD_OPTIONS" FO ON IC."FIELD_OPTION_ID" = FO."ID"
                GROUP BY 
                    ISSUE."KEY"
            )""")
        
        if "VALUE_DRIVER" in relevant_fields:
            cte_parts.append("""
            ValueDriver AS (
                SELECT 
                    ISUE."KEY" AS "KEY",
                    STRING_AGG(FOS."VALUE",', ') AS "VALUE_DRIVER"
                FROM 
                    public."ISSUES" ISUE
                    LEFT JOIN public."ISSUE_VALUE_DRIVER" IVD ON ISUE."ID" = IVD."ISSUE_ID"
                    LEFT JOIN public."FIELD_OPTIONS" FOS ON IVD."FIELD_OPTION_ID" = FOS."ID"
                GROUP BY 
                    ISUE."KEY"
            )""")
        
        # Start building the main query
        query = "WITH " + ",\n".join(cte_parts) + "\n"
        query += "SELECT " + ",\n".join(selected_columns) + "\n"
        
        # Add the FROM clause with minimal necessary joins
        query += """
        FROM public."ISSUE_FIELD_VALUES" F
        JOIN public."ISSUE_TYPES" I ON F."ISSUE_TYPE_ID" = I."ID"
        """
        
        # Only add joins that are needed based on selected columns
        if "ORIGINAL_DEVELOPER" in relevant_fields:
            query += 'LEFT JOIN public."USERS" U on F."ORIGINAL_DEVELOPER_USER_ID"=U."ID"\n'
        if "ASSIGNEE" in relevant_fields:
            query += 'LEFT JOIN public."USERS" US on F."ASSIGNEE_USER_ID"=US."ID"\n'
        if "REPORTER" in relevant_fields:
            query += 'LEFT JOIN public."USERS" USE on F."REPORTER_USER_ID"=USE."ID"\n'
        if "ISSUE_CAUSED_BY" in relevant_fields:
            query += 'LEFT JOIN public."USERS" USES on F."ISSUE_CAUSED_BY_USER_ID" = USES."ID"\n'
        if "CODE_REVIEWED_BY" in relevant_fields:
            query += 'LEFT JOIN public."USERS" USS on F."CODE_REVIEWED_BY_USER_ID" = USS."ID"\n'
        if "STATUS" in relevant_fields:
            query += 'LEFT JOIN public."STATUSES" S on F."STATUS_ID" = S."ID"\n'
        if "STATUS_CATEGORY" in relevant_fields:
            query += 'LEFT JOIN public."STATUS_CATEGORIES" SE ON S."STATUS_CAT_ID"= SE."ID"\n'
        if "PROJECT" in relevant_fields or any(field in filters for field in ["projects", "project_categories"]):
            query += 'LEFT JOIN public."PROJECTS" P on F."PROJECT_ID" = P."ID"\n'
        if "PROJECT_CATEGORY" in relevant_fields or "project_categories" in filters:
            query += 'LEFT JOIN public."PROJECT_CATEGORIES" PC ON P."PROJ_CAT_ID" = PC."ID"\n'
        if "RESOLUTION" in relevant_fields:
            query += 'LEFT JOIN public."RESOLUTIONS" R on F."RESOLUTION_1_ID" = R."ID"\n'
        if "PLATFORM_OS" in relevant_fields or "platforms" in filters:
            query += 'LEFT JOIN PlatformAggregation PA ON F."ISSUEKEY" = PA."KEY"\n'
        if any(field in relevant_fields for field in ["FIX_VERSIONS", "RELEASE_STATUS", "RELEASE_DATE"]) or "fix_versions" in filters:
            query += 'LEFT JOIN fixversion FIX ON F."ISSUEKEY" = FIX."KEY"\n'
        if "PRIORITY" in relevant_fields or "priorities" in filters:
            query += 'LEFT JOIN public."PRIORITIES" PR ON F."PRIORITY_ID"=PR."ID"\n'
        if "CUSTOMER_TYPE" in relevant_fields:
            query += 'LEFT JOIN public."FIELD_OPTIONS" FO ON F."CUSTOMER_TYPE_FIELD_OPTION_ID" = FO."ID"\n'
        if "PRODUCT" in relevant_fields or "products" in filters:
            query += 'LEFT JOIN Product PRO ON F."ISSUEKEY"=PRO."KEY"\n'
        if "CUSTOMER" in relevant_fields or "customers" in filters:
            query += 'LEFT JOIN Customer CS ON F."ISSUEKEY" = CS."KEY"\n'
        if "BUG_TYPE" in relevant_fields or "bug_types" in filters:
            query += 'LEFT JOIN public."FIELD_OPTIONS" FOO ON F."BUG_TYPE_FIELD_OPTION_ID"= FOO."ID"\n'
        if "HOTFIX_RELEASE_CYCLE" in relevant_fields:
            query += 'LEFT JOIN public."FIELD_OPTIONS" FIE ON F."HOTFIX_RELEASE_CYCLE_FIELD_OPTION_ID" = FIE."ID"\n'
        if "LABEL" in relevant_fields or "labels" in filters:
            query += 'LEFT JOIN Labels LA ON F."ISSUEKEY" = LA."KEY"\n'
        if "VALUE_DRIVER" in relevant_fields or "value_drivers" in filters:
            query += 'LEFT JOIN ValueDriver VD ON F."ISSUEKEY" = VD."KEY"\n'
        if "PHASE" in relevant_fields or "phases" in filters:
            query += 'LEFT JOIN public."FIELD_OPTIONS" FIED ON F."PHASE_FIELD_OPTION_ID" = FIED."ID"\n'
        if "PHASE_BLOCKER" in relevant_fields:
            query += 'LEFT JOIN PHASEBLOCKER PBL ON F."ISSUEKEY" = PBL."KEY"\n'
        if "T_SHIRT_SIZE" in relevant_fields:
            query += 'LEFT JOIN public."FIELD_OPTIONS" FIEOP ON F."T_SHIRT_SIZE_FIELD_OPTION_ID" = FIEOP."ID"\n'
        
        # Add WHERE clause based on filters
        where_clauses = []
        params = {}
        
        if filters:
            # Date range filters
            if 'created_start_date' in filters and filters['created_start_date']:
                where_clauses.append('F."CREATED" >= %(created_start_date)s')
                params['created_start_date'] = filters['created_start_date']
            
            if 'created_end_date' in filters and filters['created_end_date']:
                where_clauses.append('F."CREATED" <= %(created_end_date)s')
                params['created_end_date'] = filters['created_end_date']
            
            if 'resolved_start_date' in filters and filters['resolved_start_date']:
                where_clauses.append('F."RESOLVED" >= %(resolved_start_date)s')
                params['resolved_start_date'] = filters['resolved_start_date']
            
            if 'resolved_end_date' in filters and filters['resolved_end_date']:
                where_clauses.append('F."RESOLVED" <= %(resolved_end_date)s')
                params['resolved_end_date'] = filters['resolved_end_date']
            
            # Project category filter
            if 'project_categories' in filters and filters['project_categories']:
                project_cat_placeholders = []
                for i, cat in enumerate(filters['project_categories']):
                    param_name = f'project_category_{i}'
                    project_cat_placeholders.append(f'PC."NAME" = %({param_name})s')
                    params[param_name] = cat
                
                if project_cat_placeholders:
                    where_clauses.append(f"({' OR '.join(project_cat_placeholders)})")
            
            # Priority filter
            if 'priorities' in filters and filters['priorities']:
                priority_placeholders = []
                for i, p in enumerate(filters['priorities']):
                    param_name = f'priority_{i}'
                    priority_placeholders.append(f'PR."NAME" = %({param_name})s')
                    params[param_name] = p
                
                if priority_placeholders:
                    where_clauses.append(f"({' OR '.join(priority_placeholders)})")
            
            # Platform filter
            if 'platforms' in filters and filters['platforms']:
                platform_conditions = []
                for i, platform in enumerate(filters['platforms']):
                    param_name = f'platform_{i}'
                    platform_conditions.append(f'PA."PLATFORM_OS" ILIKE %({param_name})s')
                    params[param_name] = f'%{platform}%'
                
                if platform_conditions:
                    where_clauses.append(f"({' OR '.join(platform_conditions)})")
            
            # Product filter
            if 'products' in filters and filters['products']:
                product_conditions = []
                for i, product in enumerate(filters['products']):
                    param_name = f'product_{i}'
                    product_conditions.append(f'PRO."PRODUCT" ILIKE %({param_name})s')
                    params[param_name] = f'%{product}%'
                
                if product_conditions:
                    where_clauses.append(f"({' OR '.join(product_conditions)})")
            
            # Issue type filter
            if 'issue_types' in filters and filters['issue_types']:
                issue_type_placeholders = []
                for i, it in enumerate(filters['issue_types']):
                    param_name = f'issue_type_{i}'
                    issue_type_placeholders.append(f'I."NAME" = %({param_name})s')
                    params[param_name] = it
                
                if issue_type_placeholders:
                    where_clauses.append(f"({' OR '.join(issue_type_placeholders)})")
                    
            # Project filter
            if 'projects' in filters and filters['projects']:
                project_placeholders = []
                for i, proj in enumerate(filters['projects']):
                    param_name = f'project_{i}'
                    project_placeholders.append(f'P."NAME" = %({param_name})s')
                    params[param_name] = proj
                
                if project_placeholders:
                    where_clauses.append(f"({' OR '.join(project_placeholders)})")
                    
            # Fix version filter
            if 'fix_versions' in filters and filters['fix_versions']:
                version_conditions = []
                for i, version in enumerate(filters['fix_versions']):
                    param_name = f'fix_version_{i}'
                    version_conditions.append(f'FIX."FIX_VERSIONS" ILIKE %({param_name})s')
                    params[param_name] = f'%{version}%'
                
                if version_conditions:
                    where_clauses.append(f"({' OR '.join(version_conditions)})")
                
            # Status filter
            if 'statuses' in filters and filters['statuses']:
                status_placeholders = []
                for i, status in enumerate(filters['statuses']):
                    param_name = f'status_{i}'
                    status_placeholders.append(f'S."NAME" = %({param_name})s')
                    params[param_name] = status
                
                if status_placeholders:
                    where_clauses.append(f"({' OR '.join(status_placeholders)})")
                    
            # Assignee filter
            if 'assignees' in filters and filters['assignees']:
                assignee_placeholders = []
                for i, assignee in enumerate(filters['assignees']):
                    param_name = f'assignee_{i}'
                    assignee_placeholders.append(f'US."DISPLAY_NAME" = %({param_name})s')
                    params[param_name] = assignee
                
                if assignee_placeholders:
                    where_clauses.append(f"({' OR '.join(assignee_placeholders)})")
                    
            # Reporter filter
            if 'reporters' in filters and filters['reporters']:
                reporter_placeholders = []
                for i, reporter in enumerate(filters['reporters']):
                    param_name = f'reporter_{i}'
                    reporter_placeholders.append(f'USE."DISPLAY_NAME" = %({param_name})s')
                    params[param_name] = reporter
                
                if reporter_placeholders:
                    where_clauses.append(f"({' OR '.join(reporter_placeholders)})")
                    
            # Phase filter
            if 'phases' in filters and filters['phases']:
                phase_placeholders = []
                for i, phase in enumerate(filters['phases']):
                    param_name = f'phase_{i}'
                    phase_placeholders.append(f'FIED."VALUE" = %({param_name})s')
                    params[param_name] = phase
                
                if phase_placeholders:
                    where_clauses.append(f"({' OR '.join(phase_placeholders)})")
                    
            # Bug type filter
            if 'bug_types' in filters and filters['bug_types']:
                bug_type_placeholders = []
                for i, bug_type in enumerate(filters['bug_types']):
                    param_name = f'bug_type_{i}'
                    bug_type_placeholders.append(f'FOO."VALUE" = %({param_name})s')
                    params[param_name] = bug_type
                
                if bug_type_placeholders:
                    where_clauses.append(f"({' OR '.join(bug_type_placeholders)})")
                    
            # Value driver filter
            if 'value_drivers' in filters and filters['value_drivers']:
                value_driver_conditions = []
                for i, value_driver in enumerate(filters['value_drivers']):
                    param_name = f'value_driver_{i}'
                    value_driver_conditions.append(f'VD."VALUE_DRIVER" ILIKE %({param_name})s')
                    params[param_name] = f'%{value_driver}%'
                
                if value_driver_conditions:
                    where_clauses.append(f"({' OR '.join(value_driver_conditions)})")
                    
            # Customer filter
            if 'customers' in filters and filters['customers']:
                customer_conditions = []
                for i, customer in enumerate(filters['customers']):
                    param_name = f'customer_{i}'
                    customer_conditions.append(f'CS."CUSTOMER" ILIKE %({param_name})s')
                    params[param_name] = f'%{customer}%'
                
                if customer_conditions:
                    where_clauses.append(f"({' OR '.join(customer_conditions)})")
                    
            # Label filter
            if 'labels' in filters and filters['labels']:
                label_conditions = []
                for i, label in enumerate(filters['labels']):
                    param_name = f'label_{i}'
                    label_conditions.append(f'LA."LABEL" ILIKE %({param_name})s')
                    params[param_name] = f'%{label}%'
                
                if label_conditions:
                    where_clauses.append(f"({' OR '.join(label_conditions)})")
        
        # Add WHERE clause if needed
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        # Add ORDER BY and LIMIT
        query += ' ORDER BY F."CREATED" DESC'
        
        # Apply limit - use either the one from filters or the default
        query_limit = filters.get('limit', limit) if filters else limit
        query += ' LIMIT %(limit)s'
        params['limit'] = query_limit
        
        # Execute the query
        cursor.execute(query, params)
        
        # Fetch all rows
        rows = cursor.fetchall()
        
        # Get column names
        col_names = [desc[0] for desc in cursor.description]
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=col_names)
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        # Calculate resolution days if we have the necessary data
        if "CREATED" in df.columns and "RESOLVED" in df.columns:
            df["CREATED_DT"] = pd.to_datetime(df["CREATED"], errors='coerce')
            df["RESOLVED_DT"] = pd.to_datetime(df["RESOLVED"], errors='coerce')
            
            # Calculate resolution days for resolved issues
            mask = df["RESOLVED_DT"].notna() & df["CREATED_DT"].notna()
            df.loc[mask, "RESOLUTION_DAYS"] = (
                df.loc[mask, "RESOLVED_DT"] - df.loc[mask, "CREATED_DT"]
            ).dt.total_seconds() / (24*60*60)
            
        # Convert time fields to hours
        for field in ["TIME_SPENT", "ORIGINAL_ESTIMATE"]:
            if field in df.columns:
                # Convert from seconds to hours
                df[f"{field}_HOURS"] = pd.to_numeric(df[field], errors='coerce') / 3600
                
        return df
        
    except Exception as e:
        logger.error(f"Error getting JIRA data: {e}")
        if conn:
            conn.close()
        return None

# Schema information for the filter generator
JIRA_SCHEMA = """
ISSUE TRACKING FIELDS:
- ISSUEKEY: Unique identifier for JIRA issues (e.g., "PROJECT-123")
- PROJECT_KEY: Key identifier for the project (e.g., "PROJ")
- SUMMARY: Brief description of the issue
- ISSUE_TYPE: Type of issue (Bug, Story, Epic, Task, etc.)
- PRIORITY: Importance level (Blocker, Critical, Major, Minor, Trivial)
- STATUS: Current workflow position (Open, In Progress, Done, etc.)
- STATUS_CATEGORY: Status grouping (To Do, In Progress, Done)
- RESOLUTION: How the issue was resolved (Fixed, Won't Fix, Duplicate, etc.)
- PARENT: Parent issue key for hierarchical relationships
- EPIC_NAME: Name of the epic if the issue is an epic

TIME TRACKING FIELDS:
- CREATED: When the issue was created
- UPDATED: When the issue was last modified
- RESOLVED: When the issue was marked as resolved
- DUE_DATE: Deadline for issue completion
- PLANNED_START: Scheduled start date
- PLANNED_END: Scheduled end date
- ORIGINAL_ESTIMATE: Initial time estimate in seconds
- TIME_SPENT: Actual time logged in seconds
- ROM_ESTIMATES_HOURS_: Rough Order of Magnitude estimate in hours
- STORY_POINTS: Effort estimation in story points
- T_SHIRT_SIZE: T-shirt size estimation (S, M, L, XL, etc.)

PEOPLE FIELDS:
- ASSIGNEE: Person currently responsible for the issue
- REPORTER: Person who created the issue
- ORIGINAL_DEVELOPER: Person who worked on the issue
- CODE_REVIEWED_BY: Person who reviewed the code
- ISSUE_CAUSED_BY: Developer who introduced the bug (for defects)

PROJECT FIELDS:
- PROJECT: The project the issue belongs to
- PROJECT_CATEGORY: Higher-level grouping of projects
- FIX_VERSIONS: Version(s) where the issue will be/was fixed
- RELEASE_STATUS: Whether the fix version has been released (true/false)
- RELEASE_DATE: When the fix version was/will be released
- HOTFIX_RELEASE_CYCLE: Information about hotfix release cycles

TECHNICAL FIELDS:
- PLATFORM_OS: Operating system platforms affected
- PRODUCT: Product affected by the issue
- CUSTOMER_TYPE: Type of customer impacted
- CUSTOMER: Specific customer affected by the issue
- FIGMA_LINK: Link to design specifications
- DUT_LINK: Link to developer unit testing documentation
- FIT_GAP_ANALYSIS: Assessment of how solution fits requirements
- BUG_TYPE: Category of bug
- ROOT_CAUSE_ANALYSIS: Analysis of underlying issue causes
- FUNCTIONAL_SPECIFICATIONS: Details of functional specifications
- DESIGN_DOCUMENT_LINK: Link to design documentation
- MODULE_FEATURE: Module and feature information

BUSINESS & PROCESS FIELDS:
- BUSINESS_REASON: Business justification for the issue
- ACCEPTANCE_CRITERIA: Criteria for accepting the solution
- LABEL: Labels attached to the issue
- VALUE_DRIVER: Business value driver category
- PHASE: Current development or project phase
- PHASE_BLOCKER: Issues blocking progress in the current phase
"""

# Data preparation functions
def extract_basic_stats(jira_data):
    """Extract comprehensive summary statistics from JIRA data"""
    if jira_data is None or jira_data.empty:
        return {}
    
    basic_stats = {
        "total_issues": int(len(jira_data)),
        "issue_count": int(len(jira_data))
    }
    
    # Process all categorical fields to ensure comprehensive statistics
    categorical_fields = [
        "PRIORITY", "STATUS", "ISSUE_TYPE", "PROJECT", "PROJECT_CATEGORY",
        "ORIGINAL_DEVELOPER", "ASSIGNEE", "REPORTER", "PLATFORM_OS", "PRODUCT"
    ]
    
    # Get counts and distributions for each field
    for field in categorical_fields:
        if field in jira_data.columns:
            # Count non-null values
            valid_count = jira_data[field].notna().sum()
            field_key = field.lower()
            
            # Skip if no valid values
            if valid_count == 0:
                continue
                
            # Add count and percentage
            basic_stats[f"{field_key}_count"] = int(valid_count)
            basic_stats[f"{field_key}_pct"] = float((valid_count / len(jira_data)) * 100)
            
            # Add value distribution (limit to top 30 for large datasets)
            value_counts = jira_data[field].value_counts().head(30).to_dict()
            basic_stats[f"{field_key}_distribution"] = {
                str(k) if pd.notna(k) else "None": int(v) 
                for k, v in value_counts.items() if pd.notna(k)
            }
            
            # Create pivot tables for key relationships
            # For each categorical field, create pivot with ISSUE_TYPE if available
            if "ISSUE_TYPE" in jira_data.columns and field != "ISSUE_TYPE":
                try:
                    pivot_data = pd.pivot_table(
                        jira_data.dropna(subset=[field]),
                        index=field,
                        columns="ISSUE_TYPE",
                        values="ISSUEKEY", 
                        aggfunc="count",
                        fill_value=0
                    )
                    
                    # Add total column
                    pivot_data["Total"] = pivot_data.sum(axis=1)
                    
                    # Sort by total and limit to top 20 rows
                    pivot_data = pivot_data.sort_values("Total", ascending=False).head(20)
                    
                    # Convert to dictionary format for JSON
                    pivot_dict = {
                        "columns": [field] + list(pivot_data.columns),
                        "data": []
                    }
                    
                    for idx, row in pivot_data.iterrows():
                        row_data = [str(idx) if pd.notna(idx) else "None"]
                        for col in pivot_data.columns:
                            row_data.append(int(row[col]))
                        pivot_dict["data"].append(row_data)
                    
                    # Add to stats with naming convention field_by_issue_type
                    basic_stats[f"{field_key}_by_issue_type"] = pivot_dict
                except Exception as e:
                    # Skip on error
                    logger.error(f"Error creating pivot for {field}: {e}")
                    pass
    
    # Add date range information
    date_fields = ["CREATED", "RESOLVED", "PLANNED_START", "PLANNED_END"]
    for field in date_fields:
        if field in jira_data.columns:
            jira_data[f"{field}_DT"] = pd.to_datetime(jira_data[field], errors='coerce')
            valid_dates = jira_data[f"{field}_DT"].dropna()
            
            if not valid_dates.empty:
                min_date = valid_dates.min()
                max_date = valid_dates.max()
                basic_stats[f"{field.lower()}_date_range"] = {
                    "earliest": str(min_date),
                    "latest": str(max_date)
                }
    
    # Add resolved percentage
    if "RESOLVED" in jira_data.columns:
        resolved_count = jira_data["RESOLVED"].notna().sum()
        resolved_pct = (resolved_count / len(jira_data)) * 100
        basic_stats["resolved_count"] = int(resolved_count)
        basic_stats["resolved_percentage"] = float(resolved_pct)
    
    # Add time statistics
    if "TIME_SPENT" in jira_data.columns:
        # Convert to hours
        time_spent_hours = pd.to_numeric(jira_data["TIME_SPENT"], errors='coerce') / 3600
        valid_times = time_spent_hours.dropna()
        
        if not valid_times.empty:
            basic_stats["time_spent_stats"] = {
                "mean_hours": float(valid_times.mean()),
                "median_hours": float(valid_times.median()),
                "total_hours": float(valid_times.sum()),
                "issues_with_time": int(len(valid_times))
            }
    
    return basic_stats
def create_simple_visualization(jira_data, field, chart_type="bar", title=None, secondary_field=None):
    """Create simple visualizations based on chart type and fields"""
    try:
        if field not in jira_data.columns:
            return None
            
        if chart_type == "bar":
            # Create bar chart for categorical data
            value_counts = jira_data[field].value_counts().reset_index()
            value_counts.columns = [field, "COUNT"]
            
            # Limit to top 15 values if there are more
            if len(value_counts) > 15:
                value_counts = value_counts.sort_values("COUNT", ascending=False).head(15)
                auto_title = f"Top 15 Issues by {field.replace('_', ' ').title()}"
            else:
                auto_title = f"Issues by {field.replace('_', ' ').title()}"
                
            fig = px.bar(
                value_counts,
                x=field,
                y="COUNT",
                title=title or auto_title,
                labels={field: field.replace("_", " ").title(), "COUNT": "Number of Issues"}
            )
            
            # Improve readability for text-heavy categories
            if len(value_counts) > 5:
                fig.update_layout(xaxis_tickangle=45)
                
            return fig
            
        elif chart_type == "pie":
            # Create pie chart for distribution
            value_counts = jira_data[field].value_counts().reset_index()
            value_counts.columns = [field, "COUNT"]
            
            # Limit to top 7 values plus "Other" if there are more
            if len(value_counts) > 8:
                top_values = value_counts.head(7)
                other_count = value_counts.iloc[7:]["COUNT"].sum()
                other_row = pd.DataFrame({field: ["Other"], "COUNT": [other_count]})
                value_counts = pd.concat([top_values, other_row])
                
            auto_title = f"{field.replace('_', ' ').title()} Distribution"
                
            fig = px.pie(
                value_counts,
                values="COUNT",
                names=field,
                title=title or auto_title,
            )
            
            # Improve layout
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            return fig
            
        elif chart_type == "line" and field in ["CREATED", "RESOLVED", "UPDATED", "PLANNED_START", "PLANNED_END", "RELEASE_DATE"]:
            # Create line chart for time-based data
            # Convert to datetime
            jira_data[f"{field}_DT"] = pd.to_datetime(jira_data[field], errors='coerce')
            
            # Group by month
            jira_data["MONTH"] = jira_data[f"{field}_DT"].dt.strftime("%Y-%m")
            
            if secondary_field and secondary_field in jira_data.columns:
                # Group by month and secondary field (e.g., PRIORITY, STATUS)
                monthly_counts = jira_data.groupby(["MONTH", secondary_field]).size().reset_index()
                monthly_counts.columns = ["MONTH", secondary_field, "COUNT"]
                
                # Add a date column for sorting
                monthly_counts["MONTH_DT"] = pd.to_datetime(monthly_counts["MONTH"] + "-01")
                monthly_counts = monthly_counts.sort_values("MONTH_DT")
                
                auto_title = f"Monthly {field.replace('_', ' ').title()} Trend by {secondary_field.replace('_', ' ').title()}"
                
                fig = px.line(
                    monthly_counts,
                    x="MONTH",
                    y="COUNT",
                    color=secondary_field,
                    title=title or auto_title,
                    labels={"MONTH": "Month", "COUNT": "Number of Issues"}
                )
            else:
                # Simple trend by month
                monthly_counts = jira_data.groupby("MONTH").size().reset_index()
                monthly_counts.columns = ["MONTH", "COUNT"]
                
                # Add a date column for sorting
                monthly_counts["MONTH_DT"] = pd.to_datetime(monthly_counts["MONTH"] + "-01")
                monthly_counts = monthly_counts.sort_values("MONTH_DT")
                
                auto_title = f"Monthly {field.replace('_', ' ').title()} Trend"
                
                fig = px.line(
                    monthly_counts,
                    x="MONTH",
                    y="COUNT",
                    title=title or auto_title,
                    labels={"MONTH": "Month", "COUNT": "Number of Issues"}
                )
            
            # Improve readability
            fig.update_layout(xaxis_tickangle=45)
            
            return fig
            
        elif chart_type == "table":
            # Create a table view of specified fields
            if secondary_field and secondary_field in jira_data.columns:
                # Create a pivot table
                pivot_data = pd.pivot_table(
                    jira_data,
                    index=field,
                    columns=secondary_field if secondary_field else None,
                    values="ISSUEKEY",
                    aggfunc="count",
                    fill_value=0
                )
                
                # Convert to Plotly table
                fig = go.Figure(data=[go.Table(
                    header=dict(
                        values=[field] + list(pivot_data.columns),
                        fill_color='paleturquoise',
                        align='left'
                    ),
                    cells=dict(
                        values=[pivot_data.index] + [pivot_data[col] for col in pivot_data.columns],
                        fill_color='lavender',
                        align='left'
                    )
                )])
                
                auto_title = f"{field.replace('_', ' ').title()} by {secondary_field.replace('_', ' ').title()}"
                fig.update_layout(title=title or auto_title)
                
                return fig
            else:
                # Create frequency table
                value_counts = jira_data[field].value_counts().reset_index()
                value_counts.columns = [field, "COUNT"]
                
                fig = go.Figure(data=[go.Table(
                    header=dict(
                        values=[field, "Count"],
                        fill_color='paleturquoise',
                        align='left'
                    ),
                    cells=dict(
                        values=[value_counts[field], value_counts["COUNT"]],
                        fill_color='lavender',
                        align='left'
                    )
                )])
                
                auto_title = f"{field.replace('_', ' ').title()} Distribution"
                fig.update_layout(title=title or auto_title)
                
                return fig
                
        elif chart_type == "pivot":
            # Create a pivot chart if secondary field is provided
            if not secondary_field or secondary_field not in jira_data.columns:
                return None
                
            # Create pivot data
            pivot_data = pd.pivot_table(
                jira_data,
                index=field,
                columns=secondary_field,
                values="ISSUEKEY",
                aggfunc="count",
                fill_value=0
            )
            
            # Convert to heatmap for visualization
            auto_title = f"{field.replace('_', ' ').title()} vs {secondary_field.replace('_', ' ').title()}"
            
            fig = px.imshow(
                pivot_data,
                labels=dict(x=secondary_field.replace("_", " ").title(), 
                           y=field.replace("_", " ").title(), 
                           color="Count"),
                title=title or auto_title,
                text_auto=True,
                aspect="auto"
            )
            
            return fig
            
        return None
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        return None
    
def extract_advanced_metrics(jira_data):
    """
    Extract advanced metrics from JIRA data for deeper Claude analysis
    
    Args:
        jira_data: DataFrame with JIRA data
    
    Returns:
        Dictionary with advanced metrics
    """
    if jira_data is None or jira_data.empty:
        return {}
        
    metrics = {}
    
    # Convert relevant columns to appropriate types
    if "CREATED" in jira_data.columns:
        jira_data["CREATED"] = pd.to_datetime(jira_data["CREATED"], errors='coerce')
    
    if "RESOLVED" in jira_data.columns:
        jira_data["RESOLVED"] = pd.to_datetime(jira_data["RESOLVED"], errors='coerce')
    
    if "PLANNED_START" in jira_data.columns:
        jira_data["PLANNED_START"] = pd.to_datetime(jira_data["PLANNED_START"], errors='coerce')
    
    if "PLANNED_END" in jira_data.columns:
        jira_data["PLANNED_END"] = pd.to_datetime(jira_data["PLANNED_END"], errors='coerce')
    
    if "RELEASE_DATE" in jira_data.columns:
        jira_data["RELEASE_DATE"] = pd.to_datetime(jira_data["RELEASE_DATE"], errors='coerce')
    
    # 1. Timeline adherence metrics
    if all(col in jira_data.columns for col in ["PLANNED_START", "PLANNED_END", "CREATED", "RESOLVED"]):
        planned_issues = jira_data.dropna(subset=["PLANNED_START", "PLANNED_END"]).copy()
        resolved_planned_issues = planned_issues.dropna(subset=["RESOLVED"]).copy()
        
        if not resolved_planned_issues.empty:
            # Calculate if issues were resolved before planned end date
            resolved_planned_issues["on_time"] = resolved_planned_issues["RESOLVED"] <= resolved_planned_issues["PLANNED_END"]
            metrics["timeline_adherence"] = {
                "issues_with_planned_dates": int(len(planned_issues)),
                "resolved_issues_with_planned_dates": int(len(resolved_planned_issues)),
                "on_time_percentage": float((resolved_planned_issues["on_time"].sum() / len(resolved_planned_issues)) * 100),
                "average_days_off_target": float((resolved_planned_issues["RESOLVED"] - 
                                          resolved_planned_issues["PLANNED_END"]).dt.total_seconds().mean() / (24*60*60))
            }
    
    # 2. Release adherence metrics
    if all(col in jira_data.columns for col in ["RELEASE_DATE", "RESOLVED", "FIX_VERSIONS"]):
        issues_with_release = jira_data.dropna(subset=["RELEASE_DATE", "FIX_VERSIONS"]).copy()
        resolved_with_release = issues_with_release.dropna(subset=["RESOLVED"]).copy()
        
        if not resolved_with_release.empty:
            # Calculate if issues were resolved before release date
            resolved_with_release["before_release"] = resolved_with_release["RESOLVED"] <= resolved_with_release["RELEASE_DATE"]
            metrics["release_adherence"] = {
                "issues_with_release_dates": int(len(issues_with_release)),
                "resolved_before_release_percentage": float((resolved_with_release["before_release"].sum() / len(resolved_with_release)) * 100),
                "average_days_to_release": float((resolved_with_release["RELEASE_DATE"] - 
                                           resolved_with_release["RESOLVED"]).dt.total_seconds().mean() / (24*60*60))
            }
    
    # 3. Time estimation accuracy
    if all(col in jira_data.columns for col in ["ORIGINAL_ESTIMATE", "TIME_SPENT"]):
        # Convert to numeric and seconds to hours
        jira_data["ORIGINAL_ESTIMATE_HOURS"] = pd.to_numeric(jira_data["ORIGINAL_ESTIMATE"], errors='coerce') / 3600
        jira_data["TIME_SPENT_HOURS"] = pd.to_numeric(jira_data["TIME_SPENT"], errors='coerce') / 3600
        
        valid_estimates = jira_data.dropna(subset=["ORIGINAL_ESTIMATE_HOURS", "TIME_SPENT_HOURS"]).copy()
        
        if not valid_estimates.empty:
            # Calculate estimation accuracy
            valid_estimates["estimate_accuracy"] = valid_estimates["TIME_SPENT_HOURS"] / valid_estimates["ORIGINAL_ESTIMATE_HOURS"]
            
            metrics["estimation_accuracy"] = {
                "issues_with_estimates": int(len(valid_estimates)),
                "average_accuracy_ratio": float(valid_estimates["estimate_accuracy"].mean()),
                "median_accuracy_ratio": float(valid_estimates["estimate_accuracy"].median()),
                "over_estimated_percentage": float((valid_estimates["estimate_accuracy"] < 1).mean() * 100),
                "under_estimated_percentage": float((valid_estimates["estimate_accuracy"] > 1).mean() * 100)
            }
    
    # 4. ROM estimation accuracy
    if all(col in jira_data.columns for col in ["ROM_ESTIMATES_HOURS_", "TIME_SPENT"]):
        # Convert to numeric
        jira_data["ROM_HOURS"] = pd.to_numeric(jira_data["ROM_ESTIMATES_HOURS_"], errors='coerce')
        jira_data["TIME_SPENT_HOURS"] = pd.to_numeric(jira_data["TIME_SPENT"], errors='coerce') / 3600
        
        valid_rom_data = jira_data.dropna(subset=["ROM_HOURS", "TIME_SPENT_HOURS"]).copy()
        
        if not valid_rom_data.empty:
            # Calculate estimation accuracy
            valid_rom_data["rom_accuracy"] = valid_rom_data["TIME_SPENT_HOURS"] / valid_rom_data["ROM_HOURS"]
            
            metrics["rom_estimation_accuracy"] = {
                "issues_with_valid_data": int(len(valid_rom_data)),
                "average_accuracy_ratio": float(valid_rom_data["rom_accuracy"].mean()),
                "median_accuracy_ratio": float(valid_rom_data["rom_accuracy"].median()),
                "over_estimated_percentage": float((valid_rom_data["rom_accuracy"] < 1).mean() * 100),
                "under_estimated_percentage": float((valid_rom_data["rom_accuracy"] > 1).mean() * 100)
            }
    
    # 5. Requirement quality metrics (based on acceptance criteria)
    if "ACCEPTANCE_CRITERIA" in jira_data.columns:
        # Measure acceptance criteria by length as proxy for detail
        jira_data["ac_length"] = jira_data["ACCEPTANCE_CRITERIA"].astype(str).apply(len)
        jira_data["has_detailed_ac"] = jira_data["ac_length"] > 100  # Arbitrary threshold
        
        if "RESOLVED" in jira_data.columns and "CREATED" in jira_data.columns:
            resolved_issues = jira_data.dropna(subset=["RESOLVED"]).copy()
            
            if not resolved_issues.empty:
                resolved_issues["resolution_days"] = (resolved_issues["RESOLVED"] - 
                                                    resolved_issues["CREATED"]).dt.total_seconds() / (24*60*60)
                
                # Group by AC quality
                with_detailed_ac = resolved_issues[resolved_issues["has_detailed_ac"]]
                without_detailed_ac = resolved_issues[~resolved_issues["has_detailed_ac"]]
                
                metrics["requirements_quality"] = {
                    "issues_with_detailed_ac_percentage": float((resolved_issues["has_detailed_ac"].sum() / len(resolved_issues)) * 100),
                    "avg_ac_length": float(resolved_issues["ac_length"].mean()),
                    "resolution_time_with_detailed_ac": float(with_detailed_ac["resolution_days"].mean()) if not with_detailed_ac.empty else None,
                    "resolution_time_without_detailed_ac": float(without_detailed_ac["resolution_days"].mean()) if not without_detailed_ac.empty else None
                }
    
    # 6. Resolution time by priority
    if all(col in jira_data.columns for col in ["PRIORITY", "CREATED", "RESOLVED"]):
        resolved = jira_data.dropna(subset=["RESOLVED", "PRIORITY"]).copy()
        
        if not resolved.empty:
            resolved["resolution_days"] = (resolved["RESOLVED"] - resolved["CREATED"]).dt.total_seconds() / (24*60*60)
            
            # Group by priority
            priority_metrics = {}
            for priority in resolved["PRIORITY"].unique():
                if pd.notna(priority):
                    priority_data = resolved[resolved["PRIORITY"] == priority]
                    if not priority_data.empty:
                        priority_metrics[priority] = {
                            "issue_count": int(len(priority_data)),
                            "avg_resolution_days": float(priority_data["resolution_days"].mean()),
                            "median_resolution_days": float(priority_data["resolution_days"].median())
                        }
            
            if priority_metrics:
                metrics["priority_resolution_time"] = priority_metrics
    
    # 7. Resolution time by platform
    if all(col in jira_data.columns for col in ["PLATFORM_OS", "CREATED", "RESOLVED"]):
        resolved_platform = jira_data.dropna(subset=["RESOLVED", "PLATFORM_OS"]).copy()
        
        if not resolved_platform.empty:
            resolved_platform["resolution_days"] = (resolved_platform["RESOLVED"] - 
                                                 resolved_platform["CREATED"]).dt.total_seconds() / (24*60*60)
            
            # Extract individual platforms
            platform_metrics = {}
            
            # Process each issue, handling multi-platform entries
            for _, row in resolved_platform.iterrows():
                platforms = [p.strip() for p in str(row["PLATFORM_OS"]).split(',') if p.strip()]
                
                for platform in platforms:
                    if platform not in platform_metrics:
                        platform_metrics[platform] = {
                            "issue_count": 0,
                            "total_days": 0
                        }
                    
                    platform_metrics[platform]["issue_count"] += 1
                    platform_metrics[platform]["total_days"] += row["resolution_days"]
            
            # Calculate averages
            for platform in platform_metrics:
                if platform_metrics[platform]["issue_count"] > 0:
                    platform_metrics[platform]["avg_resolution_days"] = platform_metrics[platform]["total_days"] / platform_metrics[platform]["issue_count"]
                    del platform_metrics[platform]["total_days"]
                    platform_metrics[platform]["issue_count"] = int(platform_metrics[platform]["issue_count"])
                    platform_metrics[platform]["avg_resolution_days"] = float(platform_metrics[platform]["avg_resolution_days"])
            
            if platform_metrics:
                metrics["platform_resolution_time"] = platform_metrics
    
    return metrics

def extract_sample_issues(jira_data, sample_size=300):
    """Extract a representative sample of issues for detailed analysis"""
    if jira_data is None or jira_data.empty:
        return []
    
    # Cap the sample size to prevent excessive data
    max_sample_size = min(1000, len(jira_data))
    target_sample_size = min(sample_size, max_sample_size)
    
    # List of important categorical fields to ensure representation
    important_fields = [
        "ORIGINAL_DEVELOPER", 
        "ISSUE_TYPE", 
        "PRIORITY", 
        "PROJECT_CATEGORY",
        "PLATFORM_OS", 
        "PRODUCT"
    ]
    
    # Filter to fields that actually exist in the data
    available_fields = [field for field in important_fields if field in jira_data.columns]
    
    # If no important fields are available, just take a random sample
    if not available_fields:
        if len(jira_data) > target_sample_size:
            sample_df = jira_data.sample(target_sample_size)
        else:
            sample_df = jira_data
    else:
        # Choose the field with the most non-null values for stratification
        field_counts = {field: jira_data[field].notna().sum() for field in available_fields}
        if max(field_counts.values()) > 0:
            stratify_field = max(field_counts.items(), key=lambda x: x[1])[0]
            
            # Get unique values for the chosen field
            unique_values = jira_data[stratify_field].dropna().unique()
            
            if len(unique_values) > 0:
                # Calculate samples per value
                samples_per_value = max(5, target_sample_size // len(unique_values))
                
                # Collect stratified samples
                strata_samples = []
                for value in unique_values:
                    value_data = jira_data[jira_data[stratify_field] == value]
                    if len(value_data) > samples_per_value:
                        strata_samples.append(value_data.sample(samples_per_value))
                    else:
                        strata_samples.append(value_data)
                
                # Combine and limit to target size
                sample_df = pd.concat(strata_samples)
                if len(sample_df) > target_sample_size:
                    # Take a random subset to hit the target size
                    sample_df = sample_df.sample(target_sample_size)
            else:
                # Fallback to random sampling
                sample_df = jira_data.sample(min(target_sample_size, len(jira_data)))
        else:
            # Fallback to random sampling
            sample_df = jira_data.sample(min(target_sample_size, len(jira_data)))
    
    # Convert sample to list of dictionaries
    sample_issues = []
    
    for _, row in sample_df.iterrows():
        issue_data = {}
        
        # Process all columns in the dataframe, ensuring we don't miss anything
        for column in sample_df.columns:
            # Convert to appropriate type based on column name
            if column in ["TIME_SPENT", "ORIGINAL_ESTIMATE"]:
                # Convert to hours
                if pd.notna(row[column]):
                    try:
                        issue_data[column.lower()] = float(row[column]) / 3600  # Convert seconds to hours
                    except (ValueError, TypeError):
                        issue_data[column.lower()] = str(row[column])
                else:
                    issue_data[column.lower()] = None
            elif column in ["ACCEPTANCE_CRITERIA", "FIT_GAP_ANALYSIS"]:
                # Truncate long text fields
                if pd.notna(row[column]):
                    text = str(row[column])
                    issue_data[column.lower()] = (text[:500] + "...") if len(text) > 500 else text
                else:
                    issue_data[column.lower()] = None
            elif column in ["CREATED", "UPDATED", "RESOLVED", "PLANNED_START", "PLANNED_END", "RELEASE_DATE"]:
                # Handle date fields
                issue_data[column.lower()] = str(row[column]) if pd.notna(row[column]) else None
            else:
                # Handle all other fields
                issue_data[column.lower()] = str(row[column]) if pd.notna(row[column]) else None
        
        sample_issues.append(issue_data)
    
    return sample_issues

def prepare_data_for_claude(jira_data):
    """Main function to prepare condensed data for Claude"""
    # Handle empty dataframe
    if jira_data is None or jira_data.empty:
        return json.dumps({"error": "No data available"})
    
    # Limit rows for analysis - keep only up to 300 rows
    MAX_ROWS = 300
    analysis_data = jira_data
    if len(jira_data) > MAX_ROWS:
        # Get a representative sample
        analysis_data = jira_data.sample(MAX_ROWS)
    
    # Pre-process data for analysis - calculate resolution days if needed
    if "CREATED" in analysis_data.columns and "RESOLVED" in analysis_data.columns:
        analysis_data["CREATED_DT"] = pd.to_datetime(analysis_data["CREATED"], errors='coerce')
        analysis_data["RESOLVED_DT"] = pd.to_datetime(analysis_data["RESOLVED"], errors='coerce')
        
        # Calculate resolution days for resolved issues
        mask = analysis_data["RESOLVED_DT"].notna() & analysis_data["CREATED_DT"].notna()
        analysis_data.loc[mask, "RESOLUTION_DAYS"] = (
            analysis_data.loc[mask, "RESOLVED_DT"] - analysis_data.loc[mask, "CREATED_DT"]
        ).dt.total_seconds() / (24*60*60)
    
    # Get comprehensive statistics
    basic_stats = extract_basic_stats(analysis_data)
    advanced_metrics = extract_advanced_metrics(analysis_data)
    
    # Add metadata about the dataset to help Claude understand what's available
    fields_metadata = {
        "available_fields": list(analysis_data.columns),
        "categorical_fields": [col for col in analysis_data.columns if analysis_data[col].dtype == 'object'][:10],  # Limit to 10
        "date_fields": [col for col in analysis_data.columns if 'date' in col.lower() or col in 
                      ["CREATED", "RESOLVED", "PLANNED_START", "PLANNED_END"]][:5],  # Limit to 5
        "numeric_fields": [col for col in analysis_data.columns if is_numeric_dtype(analysis_data[col])][:10]  # Limit to 10
    }
    
    # Create specific pivots for important relationships - limit to 5 key pivots
    pivots = {}
    
    # Key pivot pairs that are informative for any analysis - select based on available columns
    possible_pivot_pairs = [
        ("ORIGINAL_DEVELOPER", "ISSUE_TYPE"),
        ("PROJECT_CATEGORY", "ISSUE_TYPE"),
        ("PRIORITY", "STATUS"),
        ("PLATFORM_OS", "ISSUE_TYPE"),
        ("PRODUCT", "ISSUE_TYPE")
    ]
    
    # Filter to pivot pairs where both fields exist in the data
    valid_pivot_pairs = [
        pair for pair in possible_pivot_pairs 
        if pair[0] in analysis_data.columns and pair[1] in analysis_data.columns
    ]
    
    # Limit to 3 pivots to reduce size
    for field1, field2 in valid_pivot_pairs[:3]:
        try:
            valid_data = analysis_data.dropna(subset=[field1])
            if len(valid_data) > 0:
                # Create pivot
                pivot = pd.pivot_table(
                    valid_data,
                    index=field1,
                    columns=field2,
                    values="ISSUEKEY",
                    aggfunc="count",
                    fill_value=0
                )
                
                # Add total column
                pivot["Total"] = pivot.sum(axis=1)
                
                # Sort by total and get top rows
                pivot = pivot.sort_values("Total", ascending=False).head(10)  # Limit to 10 rows
                
                # Format for JSON
                pivot_dict = {
                    "columns": [field1] + list(pivot.columns),
                    "data": []
                }
                
                for idx, row in pivot.iterrows():
                    row_data = [str(idx) if pd.notna(idx) else "None"]
                    for col in pivot.columns:
                        row_data.append(int(row[col]))
                    pivot_dict["data"].append(row_data)
                
                # Add to pivots
                pivot_key = f"{field1.lower()}_by_{field2.lower()}"
                pivots[pivot_key] = pivot_dict
        except Exception as e:
            logger.error(f"Error creating pivot for {field1} by {field2}: {e}")
    
    # Combine all data
    data_context = {
        "metadata": fields_metadata,
        "summary_stats": basic_stats,
        "pivots": pivots,
        # Skip detailed_issues to reduce size
    }
    data_context.update(advanced_metrics)
    
    # Use standard json.dumps with custom serializer
    try:
        return json.dumps(data_context, indent=2, cls=NumpyEncoder)
    except TypeError as e:
        # Fallback if the custom encoder doesn't handle something
        logger.error(f"Error during JSON serialization: {e}")
        # Convert all data to basic Python types
        simplified_data = convert_to_serializable(data_context)
        return json.dumps(simplified_data, indent=2)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime, date)):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

def convert_to_serializable(obj):
    """Recursively convert all values to JSON-serializable types"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_serializable(obj.tolist())
    elif isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    else:
        # Try to convert to string as a last resort
        try:
            return str(obj)
        except:
            return "UNCONVERTIBLE_VALUE"
        

# Integration function to use the AgenticVisualizer in the main analytics system
def generate_visualizations(jira_data, claude_response, user_prompt):
    """
    Generate simpler visualizations based on Claude's response
    
    Args:
        jira_data: DataFrame with JIRA data
        claude_response: Text response from Claude
        user_prompt: Original user query
        
    Returns:
        List of (viz_id, figure) tuples
    """
    if jira_data is None or jira_data.empty:
        return []
    
    # Extract visualization recommendations from Claude's response
    recommendations = extract_visualization_recommendations(claude_response)
    
    # Generate visualizations
    visualizations = []
    
    # Process each recommendation
    for i, rec in enumerate(recommendations):
        chart_type = rec.get("chart_type", "bar")
        fields = rec.get("fields", [])
        title = rec.get("title", "")
        
        # Skip if no fields specified
        if not fields:
            continue
        
        # Find fields that exist in the dataframe
        valid_fields = [field for field in fields if field in jira_data.columns]
        
        # Skip if no valid fields
        if not valid_fields:
            continue
        
        # Create visualization based on chart type
        if chart_type == "bar":
            fig = create_simple_visualization(jira_data, valid_fields[0], "bar", title)
            if fig:
                visualizations.append((f"bar_{i}", fig))
                
        elif chart_type == "pie":
            fig = create_simple_visualization(jira_data, valid_fields[0], "pie", title)
            if fig:
                visualizations.append((f"pie_{i}", fig))
                
        elif chart_type == "line":
            # Find date field for line chart
            date_field = None
            for field in valid_fields:
                if field in ["CREATED", "RESOLVED", "UPDATED", "PLANNED_START", "PLANNED_END", "RELEASE_DATE"]:
                    date_field = field
                    break
            
            if date_field:
                # Check if we have a secondary field for coloring
                secondary_field = None
                for field in valid_fields:
                    if field != date_field:
                        secondary_field = field
                        break
                
                fig = create_simple_visualization(jira_data, date_field, "line", title, secondary_field)
                if fig:
                    visualizations.append((f"line_{i}", fig))
                    
        elif chart_type == "table" or chart_type == "pivot":
            # Use first field as primary and second (if available) as secondary
            primary_field = valid_fields[0]
            secondary_field = valid_fields[1] if len(valid_fields) > 1 else None
            
            fig = create_simple_visualization(jira_data, primary_field, chart_type, title, secondary_field)
            if fig:
                visualizations.append((f"{chart_type}_{i}", fig))
    
    # If no visualizations were created, add some defaults based on the query
    if not visualizations:
        # Determine relevant fields based on query
        query_lower = user_prompt.lower()
        
        if "status" in query_lower:
            fig = create_simple_visualization(jira_data, "STATUS", "pie", "Status Distribution")
            if fig:
                visualizations.append(("default_status", fig))
                
        if "priority" in query_lower:
            fig = create_simple_visualization(jira_data, "PRIORITY", "bar", "Issues by Priority")
            if fig:
                visualizations.append(("default_priority", fig))
                
        if "time" in query_lower or "trend" in query_lower:
            fig = create_simple_visualization(jira_data, "CREATED", "line", "Issue Creation Trend")
            if fig:
                visualizations.append(("default_trend", fig))
                
        if "developer" in query_lower or "assignee" in query_lower:
            field = "ORIGINAL_DEVELOPER" if "ORIGINAL_DEVELOPER" in jira_data.columns else "ASSIGNEE"
            fig = create_simple_visualization(jira_data, field, "bar", f"Issues by {field.replace('_', ' ').title()}")
            if fig:
                visualizations.append((f"default_{field.lower()}", fig))
                
        # Add at least one visualization if none were created
        if not visualizations:
            # Try common fields
            for field in ["STATUS", "PRIORITY", "ISSUE_TYPE", "PROJECT"]:
                if field in jira_data.columns:
                    fig = create_simple_visualization(jira_data, field, "bar")
                    if fig:
                        visualizations.append((f"default_{field.lower()}", fig))
                        break
    
    return visualizations

def create_default_chart(jira_data, title="Data Analysis"):
    """Create a default chart when recommendations fail"""
    # First try to create a priority distribution
    if "PRIORITY" in jira_data.columns:
        value_counts = jira_data["PRIORITY"].value_counts().reset_index()
        value_counts.columns = ["PRIORITY", "COUNT"]
        
        fig = px.bar(
            value_counts,
            x="PRIORITY",
            y="COUNT",
            title=title,
            labels={
                "PRIORITY": "Priority",
                "COUNT": "Number of Issues"
            }
        )
        
        return fig
    
    # If no priority column, try issue type
    elif "ISSUE_TYPE" in jira_data.columns:
        value_counts = jira_data["ISSUE_TYPE"].value_counts().reset_index()
        value_counts.columns = ["ISSUE_TYPE", "COUNT"]
        
        fig = px.bar(
            value_counts,
            x="ISSUE_TYPE",
            y="COUNT",
            title=title,
            labels={
                "ISSUE_TYPE": "Issue Type",
                "COUNT": "Number of Issues"
            }
        )
        
        return fig
    
    # Last resort - use the first categorical column
    else:
        for col in jira_data.columns:
            if not is_numeric_type(jira_data[col]):
                value_counts = jira_data[col].value_counts().reset_index()
                value_counts.columns = [col, "COUNT"]
                
                fig = px.bar(
                    value_counts,
                    x=col,
                    y="COUNT",
                    title=title,
                    labels={
                        col: col.replace("_", " ").title(),
                        "COUNT": "Number of Issues"
                    }
                )
                
                return fig
    
    return None
    

def create_simple_heatmap(jira_data, fields, title, recommendation):
    """Create a simple heatmap based on fields"""
    if len(fields) < 2:
        return None
    
    # Need two categorical fields
    cat_fields = []
    for field in fields:
        if not is_numeric_type(jira_data[field]):
            cat_fields.append(field)
            if len(cat_fields) >= 2:
                break
    
    if len(cat_fields) < 2:
        return None
    
    x_field = cat_fields[0]
    y_field = cat_fields[1]
    
    if not title or title.lower() in ["heatmap", "heat map", "chart", "visualization"]:
        title = f"{y_field.replace('_', ' ').title()} vs {x_field.replace('_', ' ').title()} Distribution"
    
    # Create crosstab
    heatmap_data = pd.crosstab(jira_data[y_field], jira_data[x_field])
    
    # Create heatmap
    fig = px.imshow(
        heatmap_data,
        title=title,
        labels={
            "x": x_field.replace("_", " ").title(),
            "y": y_field.replace("_", " ").title(),
            "color": "Count"
        }
    )
    
    return fig
    
def create_simple_box_plot(jira_data, fields, title, recommendation):
    """Create a simple box plot based on fields"""
    if len(fields) < 2:
        return None
    
    # Need one categorical and one numeric field
    cat_field = None
    num_field = None
    
    for field in fields:
        if not cat_field and not is_numeric_type(jira_data[field]):
            cat_field = field
        elif not num_field and is_numeric_type(jira_data[field]):
            num_field = field
    
    if not cat_field or not num_field:
        return None
    
    if not title or title.lower() in ["box plot", "chart", "visualization"]:
        title = f"{num_field.replace('_', ' ').title()} by {cat_field.replace('_', ' ').title()}"
    
    # Create box plot
    fig = px.box(
        jira_data,
        x=cat_field,
        y=num_field,
        title=title,
        labels={
            cat_field: cat_field.replace("_", " ").title(),
            num_field: num_field.replace("_", " ").title()
        }
    )
    
    return fig

def create_simple_scatter_plot(jira_data, fields, title, recommendation):
    """Create a simple scatter plot based on fields"""
    if len(fields) < 2:
        return None
    
    # Identify numeric fields
    numeric_fields = []
    for field in fields:
        if is_numeric_type(jira_data[field]):
            numeric_fields.append(field)
    
    if len(numeric_fields) < 2:
        # Try to find numeric fields in the data
        for field in jira_data.columns:
            if is_numeric_type(jira_data[field]) and field not in numeric_fields:
                numeric_fields.append(field)
                if len(numeric_fields) >= 2:
                    break
    
    if len(numeric_fields) < 2:
        return None
    
    x_field = numeric_fields[0]
    y_field = numeric_fields[1]
    
    if not title or title.lower() in ["scatter plot", "chart", "visualization"]:
        title = f"{y_field.replace('_', ' ').title()} vs {x_field.replace('_', ' ').title()}"
    
    # Create scatter plot
    fig = px.scatter(
        jira_data,
        x=x_field,
        y=y_field,
        title=title,
        labels={
            x_field: x_field.replace("_", " ").title(),
            y_field: y_field.replace("_", " ").title()
        },
        hover_data=["ISSUEKEY", "SUMMARY"] if "ISSUEKEY" in jira_data.columns and "SUMMARY" in jira_data.columns else None
    )
    
    return fig

def create_simple_bar_chart(jira_data, fields, title, recommendation):
    """Create a simple bar chart based on fields"""
    if not fields:
        return None
    
    categorical_field = fields[0]
    
    if not title or title.lower() in ["bar chart", "chart", "visualization"]:
        title = f"Distribution by {categorical_field.replace('_', ' ').title()}"
    
    # Create count-based bar chart
    value_counts = jira_data[categorical_field].value_counts().reset_index()
    value_counts.columns = [categorical_field, "COUNT"]
    
    # Limit to top 10 values if there are more
    if len(value_counts) > 10:
        value_counts = value_counts.head(10)
        #title = f"Top 10 {title}"
    
    fig = px.bar(
        value_counts,
        x=categorical_field,
        y="COUNT",
        title=title,
        labels={
            categorical_field: categorical_field.replace("_", " ").title(),
            "COUNT": "Number of Issues"
        }
    )
    
    return fig

def create_simple_pie_chart(jira_data, fields, title, recommendation):
    """Create a simple pie chart based on fields"""
    if not fields:
        return None
    
    categorical_field = fields[0]
    
    if not title or title.lower() in ["pie chart", "chart", "visualization"]:
        title = f"Distribution by {categorical_field.replace('_', ' ').title()}"
    
    # Create count-based pie chart
    value_counts = jira_data[categorical_field].value_counts().reset_index()
    value_counts.columns = [categorical_field, "COUNT"]
    
    # Limit to top 8 values if there are more
    if len(value_counts) > 8:
        top_values = value_counts.head(7)
        other_count = value_counts.iloc[7:]["COUNT"].sum()
        other_row = pd.DataFrame({categorical_field: ["Other"], "COUNT": [other_count]})
        value_counts = pd.concat([top_values, other_row])
    
    fig = px.pie(
        value_counts,
        values="COUNT",
        names=categorical_field,
        title=title
    )
    
    # Improve layout
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig

def create_simple_line_chart(jira_data, fields, title, recommendation):
    """Create a simple line chart based on fields"""
    # Find a date field
    date_field = None
    for field in fields:
        if field in ["CREATED", "RESOLVED", "UPDATED"]:
            date_field = field
            break
    
    if not date_field:
        for field in ["CREATED", "RESOLVED", "UPDATED"]:
            if field in jira_data.columns:
                date_field = field
                break
    
    if not date_field:
        return None
    
    if not title or title.lower() in ["line chart", "chart", "visualization"]:
        title = f"Issues Over Time"
    
    # Convert to datetime
    jira_data[date_field] = pd.to_datetime(jira_data[date_field], errors='coerce')
    
    # Group by month
    jira_data["MONTH"] = jira_data[date_field].dt.strftime("%Y-%m")
    
    # Check if we have a color/grouping field
    group_field = None
    for field in fields:
        if field != date_field and field in jira_data.columns:
            group_field = field
            break
    
    if group_field:
        # Count issues by month and group
        counts = jira_data.groupby(["MONTH", group_field]).size().reset_index()
        counts.columns = ["MONTH", group_field, "COUNT"]
        
        # Convert to datetime for proper sorting
        counts["MONTH_SORT"] = pd.to_datetime(counts["MONTH"] + "-01")
        counts = counts.sort_values("MONTH_SORT")
        
        fig = px.line(
            counts,
            x="MONTH",
            y="COUNT",
            color=group_field,
            title=title,
            labels={
                "MONTH": "Month",
                "COUNT": "Number of Issues",
                group_field: group_field.replace("_", " ").title()
            }
        )
    else:
        # Count issues by month
        counts = jira_data.groupby("MONTH").size().reset_index()
        counts.columns = ["MONTH", "COUNT"]
        
        # Convert to datetime for proper sorting
        counts["MONTH_SORT"] = pd.to_datetime(counts["MONTH"] + "-01")
        counts = counts.sort_values("MONTH_SORT")
        
        fig = px.line(
            counts,
            x="MONTH",
            y="COUNT",
            title=title,
            labels={
                "MONTH": "Month",
                "COUNT": "Number of Issues"
            }
        )
    
    return fig


def generate_claude_recommended_visualizations(jira_data, recommendations):
    """Generate visualizations based on Claude's recommendations"""
    charts = []
    
    for i, rec in enumerate(recommendations):
        chart_type = rec["chart_type"].lower()
        title = rec["title"]
        print(f"Processing recommendation {i+1}: {title}, type: {chart_type}")
        
        # Extract field names from data_fields
        data_fields = rec["data_fields"].upper()
        fields = []
        
        # First, try to find exact field matches
        for col in jira_data.columns:
            if col in data_fields:
                fields.append(col)
        
        # If we don't have enough fields, try to extract from the title
        if len(fields) < 2:
            for col in jira_data.columns:
                if col not in fields and col in title.upper():
                    fields.append(col)
        
        # Handle specific chart types
        try:
            if "bar" in chart_type:
                # Select appropriate fields for bar chart
                if not fields:
                    # Default fields if none found
                    for field in ["PRIORITY", "STATUS", "ISSUE_TYPE", "PROJECT"]:
                        if field in jira_data.columns:
                            fields = [field]
                            break
                
                fig = create_simple_bar_chart(jira_data, fields, title, rec)
                if fig:
                    charts.append((f"bar_{i}", fig))
            
            elif "pie" in chart_type:
                # Select appropriate field for pie chart
                if not fields:
                    # Default fields if none found
                    for field in ["PRIORITY", "STATUS", "ISSUE_TYPE"]:
                        if field in jira_data.columns:
                            fields = [field]
                            break
                
                fig = create_simple_pie_chart(jira_data, fields, title, rec)
                if fig:
                    charts.append((f"pie_{i}", fig))
            
            elif "line" in chart_type or "trend" in chart_type:
                # Ensure we have a date field for line chart
                date_field = None
                for field in ["CREATED", "RESOLVED", "UPDATED"]:
                    if field in jira_data.columns:
                        date_field = field
                        if field not in fields:
                            fields.append(field)
                        break
                
                if date_field:
                    fig = create_simple_line_chart(jira_data, fields, title, rec)
                    if fig:
                        charts.append((f"line_{i}", fig))
            
            elif "scatter" in chart_type:
                # Need two numeric fields for scatter plot
                numeric_fields = []
                for field in jira_data.columns:
                    if is_numeric_type(jira_data[field]) and field not in numeric_fields:
                        numeric_fields.append(field)
                        if len(numeric_fields) >= 2:
                            break
                
                if len(numeric_fields) >= 2:
                    if not fields:
                        fields = numeric_fields
                    
                    fig = create_simple_scatter_plot(jira_data, fields, title, rec)
                    if fig:
                        charts.append((f"scatter_{i}", fig))
            
            elif "box" in chart_type:
                # Need one categorical and one numeric field
                cat_field = None
                num_field = None
                
                for field in jira_data.columns:
                    if not cat_field and not is_numeric_type(jira_data[field]):
                        cat_field = field
                    elif not num_field and is_numeric_type(jira_data[field]):
                        num_field = field
                    
                    if cat_field and num_field:
                        break
                
                if cat_field and num_field:
                    if not fields:
                        fields = [cat_field, num_field]
                    
                    fig = create_simple_box_plot(jira_data, fields, title, rec)
                    if fig:
                        charts.append((f"box_{i}", fig))
            
            elif "heatmap" in chart_type:
                # Need two categorical fields
                cat_fields = []
                for field in jira_data.columns:
                    if not is_numeric_type(jira_data[field]) and field not in cat_fields:
                        cat_fields.append(field)
                        if len(cat_fields) >= 2:
                            break
                
                if len(cat_fields) >= 2:
                    if not fields:
                        fields = cat_fields
                    
                    fig = create_simple_heatmap(jira_data, fields, title, rec)
                    if fig:
                        charts.append((f"heatmap_{i}", fig))
            
            # If we couldn't create a chart, make a default one
            if len(charts) <= i:
                default_fig = create_default_chart(jira_data, title)
                if default_fig:
                    charts.append((f"default_{i}", default_fig))
                    
        except Exception as e:
            print(f"Error creating visualization for recommendation {i+1}: {e}")
            # Create a fallback visualization
            try:
                fallback_fig = create_default_chart(jira_data, f"Data Analysis {i+1}")
                if fallback_fig:
                    charts.append((f"fallback_{i}", fallback_fig))
            except:
                pass
    
    print(f"Successfully generated {len(charts)} visualizations")
    return charts
    
def perform_agentic_analysis_with_filters(user_prompt, api_key):
    """
    Perform analysis that automatically extracts filters from the user query
    and fetches only relevant data
    """
    try:
        # 1. First, extract filters and relevant fields from the query
        with st.spinner("Analyzing your question to find relevant data..."):
            query_params = generate_query_filters(user_prompt, JIRA_SCHEMA, api_key)
            
            # Log the extracted filters for debugging
            logger.info(f"Extracted query params: {query_params}")
            
            # Get relevant data based on extracted filters and fields
            jira_data = get_filtered_jira_data_from_nl(query_params)
        
        # Display what filters were applied (informational only)
        if "filters" in query_params and query_params["filters"]:
            filters = query_params["filters"]
            applied_filters = []
            for key, value in filters.items():
                if key != 'limit' and value:  # Skip limit and empty values
                    if isinstance(value, list):
                        applied_filters.append(f"{key.replace('_', ' ').title()}: {', '.join(value)}")
                    else:
                        applied_filters.append(f"{key.replace('_', ' ').title()}: {value}")
            
            if applied_filters:
                print(f"Automatically applied filters: {'; '.join(applied_filters)}")
        
        if jira_data is None or jira_data.empty:
            return "I couldn't find any data matching your query. Please try a different question or broaden your criteria.", None
        
        # 2. Prepare data for Claude
        with st.spinner("Preparing data for analysis..."):
            data_context = prepare_data_for_claude(jira_data)
        
        # 3. Get analysis history from session state
        analysis_history = []
        if "analysis_context" in st.session_state:
            if "previous_questions" in st.session_state.analysis_context:
                for q in st.session_state.analysis_context["previous_questions"]:
                    if "question" in q and "key_findings" in q:
                        analysis_history.append({
                            "query": q["question"],
                            "key_findings": q["key_findings"]
                        })
        
        # 4. Create Model Context Protocol instance
        data_context_dict = None
        if isinstance(data_context, str):
            try:
                data_context_dict = json.loads(data_context)
            except:
                data_context_dict = {"error": "Could not parse data context"}
        else:
            data_context_dict = data_context
            
        mcp = ModelContextProtocol(
            data_context=data_context_dict,
            user_query=user_prompt,
            analysis_history=analysis_history
        )
        
        # Generate the MCP-formatted context
        mcp_context = mcp.format_for_claude()

        # Get current date for proper date context
        current_date = datetime.now().strftime('%Y-%m-%d')

        # 5. Create a focused analysis prompt
        analysis_prompt = f"""
        {mcp_context}
        
        I need you to analyze JIRA data to answer this specific question:
        
        {user_prompt}
        
        Today's date is {current_date} (YYYY-MM-DD). Use this as the reference date for any relative date calculations (like "last week", "this month", etc.).
        
        Focus only on what the user is asking for - don't provide tangential information.
        Provide a direct, concise analysis with:
        
        1. Focus only on what the user is asking for and provide the relevant analysis
        
        Here's a summary of the JIRA data available for analysis:
        ```json
        {data_context}
        ```
        """
        
        # 6. Get Claude's analysis
        client = anthropic.Anthropic(api_key=api_key)
        with st.spinner("Performing analysis with Claude AI..."):
            response = client.messages.create(
                model="claude-3-5-haiku-20241022", 
                max_tokens=3000,
                temperature=0.0,
                system=enhanced_system_prompt,  # Use the enhanced system prompt with all field definitions
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            
            response_text = ""
            if hasattr(response, 'content') and isinstance(response.content, list):
                response_text = "".join(block.text for block in response.content if hasattr(block, 'text'))
            elif hasattr(response, 'content') and isinstance(response.content, str):
                response_text = response.content
            else:
                response_text = str(response)
        
        # 7. Extract key findings to update analysis history
        key_findings = extract_insights_from_analysis(response_text)
        
        # 8. Update analysis history in session state
        if "analysis_context" in st.session_state:
            if "previous_questions" not in st.session_state.analysis_context:
                st.session_state.analysis_context["previous_questions"] = []
                
            st.session_state.analysis_context["previous_questions"].append({
                "question": user_prompt,
                "key_findings": key_findings
            })
            
            # Keep only the last 5 questions for context
            if len(st.session_state.analysis_context["previous_questions"]) > 5:
                st.session_state.analysis_context["previous_questions"] = st.session_state.analysis_context["previous_questions"][-5:]
                
            # Store the current data for visualization
            st.session_state.analysis_context["current_data"] = jira_data
        
        return response_text, jira_data
            
    except Exception as e:
        logger.error(f"Error in analysis with filters: {e}")
        return f"Error performing analysis: {str(e)}", None
    
# Visualization functions
def extract_topics(user_prompt, claude_response):
    """Extract topics from user prompt and Claude's response for visualization suggestions"""
    topics = []
    
    # Check for topics in the prompt
    prompt_lower = user_prompt.lower()
    if "platform" in prompt_lower or "os" in prompt_lower:
        topics.append("platform")
    if "priority" in prompt_lower:
        topics.append("priority")
    if "timeline" in prompt_lower or "resolution time" in prompt_lower or "resolve" in prompt_lower:
        topics.append("resolution_time")
    if "product" in prompt_lower:
        topics.append("product")
    if "estimate" in prompt_lower or "rom" in prompt_lower:
        topics.append("estimation")
    if "release" in prompt_lower or "version" in prompt_lower:
        topics.append("release")
    if "project" in prompt_lower or "category" in prompt_lower:
        topics.append("project")
    if "assignee" in prompt_lower or "team" in prompt_lower:
        topics.append("personnel")
    if "developer" in prompt_lower or "resource" in prompt_lower or "productivity" in prompt_lower:
        topics.append("developer_productivity")
    
    # Also check Claude's response for topics it highlighted
    response_lower = claude_response.lower()
    
    # Identify chart suggestions in Claude's response
    chart_indicators = [
        "chart would show", "visualization would show", "graph showing", 
        "recommend visualizing", "could be visualized", "suggest a"
    ]
    
    for indicator in chart_indicators:
        if indicator in response_lower:
            # Find nearby topics
            if "platform" in response_lower.split(indicator)[1][:100]:
                topics.append("platform")
            if "priority" in response_lower.split(indicator)[1][:100]:
                topics.append("priority")
            if "resolution time" in response_lower.split(indicator)[1][:100] or "timeline" in response_lower.split(indicator)[1][:100]:
                topics.append("resolution_time")
            if "product" in response_lower.split(indicator)[1][:100]:
                topics.append("product")
            if "estimate" in response_lower.split(indicator)[1][:100]:
                topics.append("estimation")
    
    # Remove duplicates
    return list(set(topics))
# Enhanced visualization system

def analyze_followup_question(data_context, followup_context, api_key):
    """
    Analyze a follow-up question with awareness of previous context using Model Context Protocol
    
    Args:
        data_context: The JSON data context
        followup_context: Dict with previous question, analysis, and follow-up question
        api_key: Anthropic API key
        
    Returns:
        Analysis of the follow-up question
    """
    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        logger.error(f"Error initializing Anthropic client: {e}")
        return f"Error connecting to Claude API: {str(e)}"
    
    # Parse the data context from JSON string to dict if needed
    data_context_dict = None
    if isinstance(data_context, str):
        try:
            data_context_dict = json.loads(data_context)
        except:
            data_context_dict = {"error": "Could not parse data context"}
    else:
        data_context_dict = data_context
    
    # Get analysis history from session state
    analysis_history = []
    if "analysis_context" in st.session_state:
        if "previous_questions" in st.session_state.analysis_context:
            for q in st.session_state.analysis_context["previous_questions"]:
                if "question" in q and "key_findings" in q:
                    analysis_history.append({
                        "query": q["question"],
                        "key_findings": q["key_findings"]
                    })
    
    # Add the previous question specifically mentioned in followup_context
    if 'previous_question' in followup_context and 'previous_analysis' in followup_context:
        # Extract key findings from previous analysis
        key_findings = extract_insights_from_analysis(followup_context['previous_analysis'])
        analysis_history.append({
            "query": followup_context['previous_question'],
            "key_findings": key_findings
        })
    
    # Create Model Context Protocol instance
    mcp = ModelContextProtocol(
        data_context=data_context_dict,
        user_query=followup_context['follow_up_question'],
        analysis_history=analysis_history
    )
    
    # Generate the MCP-formatted context
    mcp_context = mcp.format_for_claude()
    
    # Create specialized system prompt for follow-up
    followup_system_prompt = """
    You are an expert JIRA analytics agent who excels at answering follow-up questions about data analysis.

    When answering follow-up questions:
    1. Build upon the previous analysis rather than repeating it
    2. Focus specifically on answering the new question
    3. Reference relevant points from the previous analysis if applicable
    4. Provide new insights that directly address the follow-up question

    Use the same structured format for your response, with ANALYSIS.
    
    Make your answers concise and focused on what's new or different from the previous analysis.
    
    ## Model Context Protocol
    Pay attention to the context blocks provided in the Model Context Protocol format.
    These will give you information about system capabilities, data summary, conversation history, and the current query.
    """
    
    followup_prompt = f"""
    {mcp_context}
    
    You previously analyzed some JIRA data with this question:
    
    {followup_context['previous_question']}
    
    Your previous analysis included these findings:
    
    {followup_context['previous_analysis'][:1000]}... (truncated for brevity)
    
    Now I have a follow-up question:
    
    {followup_context['follow_up_question']}
    
    Please analyze the same JIRA data to answer this follow-up question.
    Build upon your previous analysis rather than repeating it.
    
    Here's the JIRA data context to analyze:
    ```json
    {data_context}
    ```
    """
    
    try:
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=3000,
            temperature=0.0,
            system=followup_system_prompt,
            messages=[{"role": "user", "content": followup_prompt}]
        )
        
        response_text = ""
        if hasattr(message, 'content') and isinstance(message.content, list):
            response_text = "".join(block.text for block in message.content if hasattr(block, 'text'))
        elif hasattr(message, 'content') and isinstance(message.content, str):
            response_text = message.content
        else:
            response_text = str(message)
        
        # Extract key findings to update analysis history
        key_findings = extract_insights_from_analysis(response_text)
        
        # Update analysis history in session state
        if "analysis_context" in st.session_state:
            if "previous_questions" not in st.session_state.analysis_context:
                st.session_state.analysis_context["previous_questions"] = []
                
            st.session_state.analysis_context["previous_questions"].append({
                "question": followup_context['follow_up_question'],
                "key_findings": key_findings
            })
            
            # Keep only the last 5 questions for context
            if len(st.session_state.analysis_context["previous_questions"]) > 5:
                st.session_state.analysis_context["previous_questions"] = st.session_state.analysis_context["previous_questions"][-5:]
        
        return response_text
            
    except Exception as e:
        logger.error(f"Error analyzing follow-up question: {e}")
        return f"Error analyzing follow-up: {str(e)}"
    
def extract_insights_from_analysis(analysis_text):
    """Extract key insights from Claude's analysis text with improved pattern matching"""
    insights = []
    
    # Look for key findings section
    findings_pattern = r'(?:##|###)\s*KEY FINDINGS.*?(?=##|###|$)'
    findings_match = re.search(findings_pattern, analysis_text, re.DOTALL | re.IGNORECASE)
    
    if findings_match:
        findings_text = findings_match.group(0)
        
        # Extract numbered or bulleted points
        point_patterns = [
            r'(?:^\d+\.|\*)\s*(.*?)(?=^\d+\.|\*|$)',  # Numbered or bulleted lists
            r'(?<=\n)-\s*(.*?)(?=\n|$)',  # Dash bullet points
            r'(?<=\n)•\s*(.*?)(?=\n|$)'   # Bullet symbol points
        ]
        
        for pattern in point_patterns:
            point_matches = re.findall(pattern, findings_text, re.MULTILINE | re.DOTALL)
            
            for point in point_matches:
                # Clean up and add to insights
                clean_point = re.sub(r'\s+', ' ', point).strip()
                if clean_point and len(clean_point) > 10:  # Min length to filter out noise
                    insights.append(clean_point)
    
    # If no key findings section or no points found, look for strong statements in the text
    if not insights:
        # Look for sentences that often indicate insights
        insight_patterns = [
            r'(?:significantly|notably|importantly|interestingly|clearly|evidently)[^.!?]*[.!?]',
            r'(?:analysis shows|data indicates|we found|results reveal)[^.!?]*[.!?]',
            r'(?:key takeaway|critical finding|major insight)[^.!?]*[.!?]'
        ]
        
        for pattern in insight_patterns:
            matches = re.findall(pattern, analysis_text, re.IGNORECASE)
            for match in matches:
                clean_match = re.sub(r'\s+', ' ', match).strip()
                if clean_match and len(clean_match) > 15:  # Slightly longer min length
                    insights.append(clean_match)
    
    return insights

def extract_visualization_recommendations(claude_response):
    """
    Extract visualization recommendations from Claude's response
    with improved NLP capabilities and consistency checking.
    
    Args:
        claude_response: The text response from Claude
        
    Returns:
        List of recommendation dictionaries
    """
    recommendations = []
    viz_section = claude_response
    
    
    # Find sentences or paragraphs suggesting visualizations
    suggestion_patterns = [
        # Sentences mentioning visualizations
        r'(?:could be|can be|should be|would be|might be) (?:visualized|illustrated|shown|displayed|represented) (?:with|using|by|in|as) a ([a-z\s]+chart|[a-z\s]+plot|[a-z\s]+graph|[a-z\s]+map)[^.]*\.',
        # Sentences recommending specific chart types
        r'(?:recommend|suggest|create|generate|use)[^.]*?(bar chart|pie chart|line chart|scatter plot|heatmap|box plot)[^.]*\.',
        # Sentences describing what a visualization would show
        r'[A|a] ([a-z\s]+chart|[a-z\s]+plot|[a-z\s]+graph|[a-z\s]+map)[^.]*?would (?:show|display|illustrate|visualize|reveal)[^.]*\.',
        # Field-based visualization suggestions
        r'(?:visualiz|display|show|plot)(?:ing|e)? (?:the|this) ([A-Z][A-Z_]+)[^.]*?(?:against|versus|vs|with|by|and) ([A-Z][A-Z_]+)[^.]*?(?:in|using|with) a ([a-z\s]+chart|[a-z\s]+plot|[a-z\s]+graph|[a-z\s]+map)[^.]*\.'
    ]
    
    viz_items = []
    for pattern in suggestion_patterns:
        items = re.findall(pattern, viz_section, re.DOTALL | re.IGNORECASE)
        if items:
            # Find the complete sentences containing these matches
            for match in items:
                # Get sentences containing the match
                match_str = match[0] if isinstance(match, tuple) else match
                sentence_pattern = r'[^.!?]*(?:' + re.escape(match_str) + r')[^.!?]*[.!?]'
                sentences = re.findall(sentence_pattern, viz_section, re.IGNORECASE)
                viz_items.extend(sentences)
    
    # Process each visualization recommendation
    for item in viz_items:
        # Skip very short or empty items
        if not item or len(item) < 15:
            continue
        
        # Map visualization types to standardized types
        viz_type_map = {
            'bar chart': 'bar', 
            'bar graph': 'bar',
            'column chart': 'bar',
            'pie chart': 'pie', 
            'donut chart': 'pie',
            'line chart': 'line',
            'line graph': 'line',
            'trend chart': 'line',
            'scatter plot': 'scatter',
            'scatter chart': 'scatter',
            'correlation chart': 'scatter',
            'heatmap': 'heatmap',
            'heat map': 'heatmap',
            'box plot': 'box',
            'boxplot': 'box',
            'box chart': 'box',
            'timeline chart': 'timeline',
            'gantt chart': 'timeline'
        }
        
        # Default chart type
        chart_type = 'bar'
        
        # Detect chart type
        for viz_name, viz_type in viz_type_map.items():
            if viz_name in item.lower():
                chart_type = viz_type
                break
        
        # Extract title using different patterns
        title_patterns = [
            r'"([^"]+)"',  # Quoted text
            r'titled "([^"]+)"', 
            r'title[d:]? "([^"]+)"',
            r'title[d:]? \'([^\']+)\'',
            r'showing ([^\.]+)',
            r'displays? ([^\.]+)',
            r'visualiz(?:e|ing) ([^\.]+)'
        ]
        
        title = ""
        for pattern in title_patterns:
            title_match = re.search(pattern, item, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()
                # If title contains chart type, remove it for cleaner titles
                for viz_name in viz_type_map.keys():
                    title = re.sub(f"{viz_name}", "", title, flags=re.IGNORECASE).strip()
                break
        
        # If no title found, create one from the beginning of the item
        if not title:
            # Use first sentence or part up to 50 chars
            first_part = re.split(r'[.!?]', item)[0].strip()
            title = (first_part[:50] + '...') if len(first_part) > 50 else first_part
            
            # Remove chart type references from title for clarity
            for viz_name in viz_type_map.keys():
                title = re.sub(f"{viz_name}", "", title, flags=re.IGNORECASE).strip()
        
        # Extract fields - look for JIRA field names in the item
        field_patterns = [
            # Field vs field pattern
            r'(?:of|between|comparing|relates|map|show)\s+([A-Z][A-Z_]+)[^A-Z_]*(?:and|vs\.?|versus|to|with|by)\s+([A-Z][A-Z_]+)',
            # Single field pattern
            r'(?:of|by|for|using|with)\s+([A-Z][A-Z_]+)\b',
            # Any capitalized field-like name
            r'\b([A-Z][A-Z_]+)\b'
        ]
        
        fields = []
        for pattern in field_patterns:
            field_matches = re.findall(pattern, item)
            if field_matches:
                for match in field_matches:
                    if isinstance(match, tuple):
                        for field in match:
                            if field and field.upper() not in fields:
                                fields.append(field.upper())
                    elif match and match.upper() not in fields:
                        fields.append(match.upper())
                        
                # Once we've found fields with a pattern, stop looking with other patterns
                if fields:
                    break
        
        # Extract purpose
        purpose = ""
        purpose_patterns = [
            r'to (show|display|visualize|analyze|examine|illustrate|highlight) ([^\.]+)',
            r'(?:helps|helping) (?:to )?(understand|analyze|visualize|see|identify) ([^\.]+)',
            r'useful for (understanding|analyzing|comparing|identifying|tracking) ([^\.]+)',
            r'purpose[d:]? (?:is )?to (show|display|visualize|analyze|examine) ([^\.]+)'
        ]
        
        for pattern in purpose_patterns:
            purpose_match = re.search(pattern, item, re.IGNORECASE)
            if purpose_match:
                purpose = f"{purpose_match.group(1)} {purpose_match.group(2)}".strip()
                break
        
        # Add inferred purpose based on chart type if none is explicitly found
        if not purpose:
            if chart_type == "pie":
                # Infer purpose for pie charts based on content
                if re.search(r'(status|distribution|percentage|breakdown|proportion)', item, re.IGNORECASE):
                    purpose = "Show the distribution of issues across categories"
                else:
                    purpose = "Visualize data distribution"
            elif chart_type == "bar":
                if re.search(r'(comparison|compare|across|between)', item, re.IGNORECASE):
                    purpose = "Compare values across different categories"
                else:
                    purpose = "Show values by category"
            elif chart_type == "line":
                purpose = "Track changes over time or trend analysis"
            elif chart_type == "scatter":
                purpose = "Examine relationship or correlation between variables"
            elif chart_type == "box":
                purpose = "Compare distributions and identify outliers"
            elif chart_type == "heatmap":
                purpose = "Visualize patterns or correlations between two categorical variables"
            elif chart_type == "timeline":
                purpose = "Show events or activities over time"
        
        # Generate more specific title based on fields and chart type if title is generic
        if not title or title.lower() in ["chart", "graph", "plot", "visualization"]:
            if chart_type == "pie" and "STATUS" in fields:
                title = "Issue Status Distribution"
            elif chart_type == "pie" and "PRIORITY" in fields:
                title = "Issue Priority Distribution"
            elif chart_type == "pie" and "ISSUE_TYPE" in fields:
                title = "Issue Type Distribution"
            elif chart_type == "bar" and "PRIORITY" in fields:
                title = "Issues by Priority"
            elif chart_type == "bar" and "STATUS" in fields:
                title = "Issues by Status"
            elif chart_type == "bar" and "PLATFORM_OS" in fields:
                title = "Issues by Platform"
            elif chart_type == "line" and "CREATED" in fields:
                title = "Issue Creation Trend"
            elif chart_type == "scatter" and "ORIGINAL_ESTIMATE" in fields and "TIME_SPENT" in fields:
                title = "Estimation Accuracy: Estimate vs Actual"
        
        # Ensure data_fields is properly formatted as a string representing field list
        data_fields = ", ".join(fields) if fields else ""
        
        recommendation = {
            "title": title,
            "chart_type": chart_type,
            "fields": fields,
            "data_fields": data_fields,
            "purpose": purpose,
            "original_text": item
        }
        
        recommendations.append(recommendation)
                
    # If we couldn't extract any recommendations, create intelligent defaults
    if not recommendations:
        # Default status distribution
        recommendations.append({
            "title": "Issue Status Distribution",
            "chart_type": "pie",
            "fields": ["STATUS"],
            "data_fields": "STATUS",
            "purpose": "Show the distribution of issues across different status categories",
            "original_text": ""
        })
        
        # Default priority distribution
        recommendations.append({
            "title": "Issues by Priority",
            "chart_type": "bar",
            "fields": ["PRIORITY"],
            "data_fields": "PRIORITY",
            "purpose": "Show the distribution of issues across different priority levels",
            "original_text": ""
        })
        
        # Default time trend if CREATED exists
        recommendations.append({
            "title": "Issue Creation Trend",
            "chart_type": "line",
            "fields": ["CREATED"],
            "data_fields": "CREATED",
            "purpose": "Show how issue creation has changed over time",
            "original_text": ""
        })
    
    return recommendations   

def create_default_visualizations(jira_data):
    """Create standard default visualizations"""
    charts = []
    
    # 1. Issue Status Distribution 
    if "STATUS" in jira_data.columns:
        status_counts = jira_data["STATUS"].value_counts().reset_index()
        status_counts.columns = ["STATUS", "COUNT"]
        
        fig = px.pie(
            status_counts,
            values="COUNT",
            names="STATUS",
            title="Issue Status Distribution"
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        charts.append(("status_distribution", fig))
    
    # 2. Priority Distribution
    if "PRIORITY" in jira_data.columns:
        priority_counts = jira_data["PRIORITY"].value_counts().reset_index()
        priority_counts.columns = ["PRIORITY", "COUNT"]
        
        fig = px.bar(
            priority_counts,
            x="PRIORITY",
            y="COUNT",
            title="Issues by Priority Level",
            labels={"PRIORITY": "Priority", "COUNT": "Number of Issues"},
            color="PRIORITY"
        )
        
        charts.append(("priority_distribution", fig))
    
    # 3. Issue Type Distribution
    if "ISSUE_TYPE" in jira_data.columns:
        type_counts = jira_data["ISSUE_TYPE"].value_counts().reset_index()
        type_counts.columns = ["ISSUE_TYPE", "COUNT"]
        
        fig = px.bar(
            type_counts,
            x="ISSUE_TYPE",
            y="COUNT",
            title="Distribution of Issue Types",
            labels={"ISSUE_TYPE": "Issue Type", "COUNT": "Number of Issues"},
            color="ISSUE_TYPE"
        )
        
        charts.append(("issue_type_distribution", fig))
    
    # 4. Time Trend (if date available)
    if "CREATED" in jira_data.columns:
        jira_data["CREATED_DT"] = pd.to_datetime(jira_data["CREATED"], errors='coerce')
        
        # Group by month
        jira_data["MONTH"] = jira_data["CREATED_DT"].dt.strftime("%Y-%m")
        
        # Count issues by month
        monthly_counts = jira_data.groupby("MONTH").size().reset_index()
        monthly_counts.columns = ["MONTH", "COUNT"]
        
        # Sort by month
        monthly_counts["MONTH_DT"] = pd.to_datetime(monthly_counts["MONTH"] + "-01")
        monthly_counts = monthly_counts.sort_values("MONTH_DT")
        
        fig = px.line(
            monthly_counts,
            x="MONTH",
            y="COUNT",
            title="Monthly Issue Creation Trend",
            labels={"MONTH": "Month", "COUNT": "Number of Issues"},
            markers=True
        )
        
        charts.append(("time_trend", fig))
    
    # 5. Developer productivity (if available)
    if "ORIGINAL_DEVELOPER" in jira_data.columns:
        dev_counts = jira_data["ORIGINAL_DEVELOPER"].value_counts().reset_index().head(10)
        dev_counts.columns = ["ORIGINAL_DEVELOPER", "COUNT"]
        
        fig = px.bar(
            dev_counts,
            y="ORIGINAL_DEVELOPER",
            x="COUNT",
            title="Top 10 Developers by Issue Count",
            labels={"ORIGINAL_DEVELOPER": "Developer", "COUNT": "Number of Issues"},
            orientation='h'
        )
        
        charts.append(("developer_productivity", fig))
    
    # 6. Resolution time by priority (if available)
    if "RESOLUTION_DAYS" in jira_data.columns and "PRIORITY" in jira_data.columns:
        valid_data = jira_data.dropna(subset=["RESOLUTION_DAYS", "PRIORITY"]).copy()
        
        if not valid_data.empty:
            res_by_priority = valid_data.groupby("PRIORITY")["RESOLUTION_DAYS"].mean().reset_index()
            
            fig = px.bar(
                res_by_priority,
                x="PRIORITY",
                y="RESOLUTION_DAYS",
                title="Average Resolution Time by Priority",
                labels={"PRIORITY": "Priority", "RESOLUTION_DAYS": "Average Resolution Time (Days)"},
                color="PRIORITY"
            )
            
            charts.append(("resolution_by_priority", fig))
    
    return charts

def generate_query_filters(user_prompt, jira_data_schema, api_key):
    """
    Use Claude to automatically generate SQL filters and relevant fields from a natural language query
    
    Args:
        user_prompt: User's question in natural language
        jira_data_schema: Schema information about available fields
        api_key: Anthropic API key
        
    Returns:
        Dictionary of filters and relevant fields to apply to the query
    """
    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        logger.error(f"Error initializing Anthropic client: {e}")
        return {}
    
    # Get the current date for proper date range calculations
    current_date = datetime.now()
    
    # Define relative date periods
    last_week_start = (current_date - timedelta(days=current_date.weekday() + 7)).strftime('%Y-%m-%d')
    last_week_end = (current_date - timedelta(days=current_date.weekday() + 1)).strftime('%Y-%m-%d')
    last_month_start = (current_date.replace(day=1) - timedelta(days=1)).replace(day=1).strftime('%Y-%m-%d')
    last_month_end = (current_date.replace(day=1) - timedelta(days=1)).strftime('%Y-%m-%d')
    last_quarter_start = (current_date - timedelta(days=90)).strftime('%Y-%m-%d')
    last_quarter_end = current_date.strftime('%Y-%m-%d')
    
    # Create a prompt for Claude to identify necessary filters and fields
    filter_prompt = f"""
    You're an expert at extracting database query parameters and relevant fields from natural language questions.
    
    I have a JIRA database with the following fields:
    ```
    {jira_data_schema}
    ```
    
    The user has asked this question:
    "{user_prompt}"
    
    Today's date is {current_date.strftime('%Y-%m-%d')} (YYYY-MM-DD).
    
    For reference, here are some date ranges:
    - Last week: {last_week_start} to {last_week_end}
    - Last month: {last_month_start} to {last_month_end}
    - Last quarter: {last_quarter_start} to {last_quarter_end}
    
    Based on this question, identify:
    1. Which data filters should be applied to the database query
    2. Which specific fields are most relevant to answering this question (to minimize data size)
    
    Return ONLY a JSON object with filter parameters and relevant fields. Use the following format:
    
    {{
        "filters": {{
            "issue_types": ["Bug", "Task"],           // Filter to specific issue types (optional)
            "projects": ["Project1", "Project2"],     // Filter to specific projects (optional)
            "project_categories": ["Category1"],      // Filter to specific project categories (optional)
            "priorities": ["High", "Critical"],       // Filter to specific priorities (optional)
            "created_start_date": "2023-01-01",       // Filter by creation date range (optional)
            "created_end_date": "2023-12-31",         // Filter by creation date range (optional)
            "resolved_start_date": "2023-01-01",      // Filter by resolution date range (optional)
            "resolved_end_date": "2023-12-31",        // Filter by resolution date range (optional)
            "platforms": ["Windows", "iOS"],          // Filter to specific platforms (optional)
            "products": ["Product1", "Product2"],     // Filter to specific products (optional)
            "fix_versions": ["2.0", "3.1"],           // Filter to specific fix versions (optional)
            "statuses": ["Open", "In Progress"],      // Filter to specific statuses (optional)
            "assignees": ["Person1", "Person2"],      // Filter to specific assignees (optional)
            "reporters": ["Person3", "Person4"],      // Filter to specific reporters (optional)
            "phases": ["Development", "Testing"],     // Filter to specific phases (optional)
            "bug_types": ["UI", "Backend"],           // Filter to specific bug types (optional)
            "value_drivers": ["Cost Reduction"],      // Filter to specific value drivers (optional)
            "customers": ["Customer1", "Customer2"],  // Filter to specific customers (optional)
            "labels": ["label1", "label2"],           // Filter to specific labels (optional)
            "limit": 500                              // Limit results (default to a reasonable number)
        }},
        "relevant_fields": ["ISSUEKEY", "SUMMARY", "ISSUE_TYPE", "PRIORITY", "STATUS", "CREATED", "RESOLVED"]  // Only include fields relevant to the question
    }}
    
    Only include filters that are clearly implied by the user's question. 
    For the relevant_fields, only include fields that are directly needed to answer the query.
    Always include basic identifier fields like ISSUEKEY.
    """
    
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",  # Using a smaller model for extraction
            max_tokens=1000,
            temperature=0.0,
            system="You extract database query parameters and relevant fields from natural language. Respond ONLY with the requested JSON object, nothing else.",
            messages=[{"role": "user", "content": filter_prompt}]
        )
        
        response_text = ""
        if hasattr(response, 'content') and isinstance(response.content, list):
            response_text = "".join(block.text for block in response.content if hasattr(block, 'text'))
        elif hasattr(response, 'content') and isinstance(response.content, str):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Extract the JSON object from the response
        json_match = re.search(r'({.*})', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            try:
                result = json.loads(json_str)
                # If the result has filters and relevant_fields keys
                if isinstance(result, dict) and "filters" in result and "relevant_fields" in result:
                    return result
                # If the result only has filters (old format)
                elif isinstance(result, dict) and not "filters" in result:
                    # Convert to new format
                    return {
                        "filters": result,
                        "relevant_fields": ["ISSUEKEY", "SUMMARY", "ISSUE_TYPE", "PRIORITY", "STATUS", "CREATED", "RESOLVED"]
                    }
            except json.JSONDecodeError:
                logger.error(f"Error parsing JSON from Claude's response: {json_str}")
                return {
                    "filters": {},
                    "relevant_fields": ["ISSUEKEY", "SUMMARY", "ISSUE_TYPE", "PRIORITY", "STATUS", "CREATED", "RESOLVED"]
                }
        else:
            logger.error(f"No JSON found in Claude's response: {response_text}")
            return {
                "filters": {},
                "relevant_fields": ["ISSUEKEY", "SUMMARY", "ISSUE_TYPE", "PRIORITY", "STATUS", "CREATED", "RESOLVED"]
            }
            
    except Exception as e:
        logger.error(f"Error extracting filters: {e}")
        return {
            "filters": {},
            "relevant_fields": ["ISSUEKEY", "SUMMARY", "ISSUE_TYPE", "PRIORITY", "STATUS", "CREATED", "RESOLVED"]
        }
    
def new_main():
    # Streamlined header with just the logo and custom text below it
    st.image("OpsGenie-Logo-3.png", width=220)
    st.markdown("<p style='text-align: left; color: #12c0fc; margin-left: 25px; margin-top: 0rem; margin-bottom: 25px; font-size: 1.1rem;'>Turning Operational Data into Strategic Intelligence</p>", unsafe_allow_html=True)
    
    # Add custom CSS styling
    st.markdown("""
    <style>
    /* Button styling */
    .stButton button {
        background-color: #4981F5;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #2563EB;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    /* Chat message styling */
    .user-message {
        background-color: #E6F7FF;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1890FF;
    }
    .assistant-message {
        background-color: #F0F5FF;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #722ED1;
    }
    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 5px;
        border: 1px solid #E2E8F0;
        padding: 0.75rem;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        border-radius: 5px 5px 0 0;
        padding: 0 1rem;
    }
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #4B5563;
        background-color: #F9FAFB;
        border-radius: 5px;
    }
    /* Reset button styling */
    .reset-button-container {
        display: flex;
        justify-content: center;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    /* Center logo */
    [data-testid="stImage"] {
        display: block;
        margin-left: 5px;
        margin-right: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Handle state reset via query params
    if "reset_ui" in st.query_params:
        # Reset all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Remove the query parameter
        st.query_params.clear()
        st.rerun()
    
    # Initialize state
    if "analysis_context" not in st.session_state:
        st.session_state.analysis_context = {
            "previous_questions": [],
            "current_data": None,
            "insights": []
        }
    

    # Chat interface for analysis
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm your JIRA analysis assistant. Ask me any question about your JIRA data to get started."}
        ]

    # Display chat messages with improved styling
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)

    # Get prompt from chat input - with placeholder text
    prompt = st.chat_input("Ask a question like: 'What are the most common issue types in the last quarter?'")

    # Process user input
    if prompt:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)
        
        # Get API key
        try:
            api_key = st.secrets.get("anthropic_api_key")
        except:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if not api_key:
            st.error("Anthropic API key not found. Please configure it in .streamlit/secrets.toml or as an environment variable.")
            st.stop()
        
        # Show analysis in progress indicator
        with st.status("Analyzing your question...", expanded=True) as status:
            st.write("🔍 Finding relevant JIRA data...")
            
            # Perform analysis with automatic filter extraction
            analysis, jira_data = perform_agentic_analysis_with_filters(prompt, api_key)
            
            st.write("📊 Generating insights and visualizations...")
            status.update(label="Analysis complete!", state="complete", expanded=False)
        
        if isinstance(analysis, str) and jira_data is None:
            # Error or no data found
            st.session_state.messages.append({"role": "assistant", "content": analysis})
            
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"<div class='assistant-message'>{analysis}</div>", unsafe_allow_html=True)
        else:
            # Add Claude's response to chat
            st.session_state.messages.append({"role": "assistant", "content": analysis})
            
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"<div class='assistant-message'>{analysis}</div>", unsafe_allow_html=True)

            # Generate visualizations based on Claude's recommendations
            st.markdown("<h3 style='color:#1E3A8A;margin-top:30px;'>Data Visualizations</h3>", unsafe_allow_html=True)

            # Extract Claude's visualization recommendations
            recommendations = extract_visualization_recommendations(analysis)

            # Generate visualizations based on recommendations
            claude_visualizations = []
            with st.spinner("Generating visualizations..."):
                # Use the simpler visualization system
                claude_visualizations = generate_visualizations(jira_data, analysis, prompt)

            # Display the visualizations
            if claude_visualizations:
                # Create tabs for visualizations with proper titles
                viz_titles = []
                
                for i, (viz_id, fig) in enumerate(claude_visualizations):
                    # Get the visualization's current title from the figure
                    current_title = "Visualization"
                    if hasattr(fig.layout, 'title') and hasattr(fig.layout.title, 'text'):
                        current_title = fig.layout.title.text
                    
                    # Create a descriptive tab title
                    viz_titles.append(f"Chart {i+1}: {current_title}")
                
                # Create tabs with proper titles
                tabs = st.tabs(viz_titles)
                
                # Display each visualization in its tab
                for i, (viz_id, fig) in enumerate(claude_visualizations):
                    if i < len(tabs):
                        with tabs[i]:
                            # Update figure layout for better appearance
                            fig.update_layout(
                                template="plotly_white",
                                margin=dict(l=20, r=20, t=40, b=20),
                                title_font=dict(size=20, color="#1E3A8A"),
                                legend_title_font=dict(size=14),
                                legend_font=dict(size=12),
                                colorway=["#4981F5", "#38B09D", "#F7B32B", "#F26419", "#2E294E", "#8661C1"],
                            )
                            # Display the visualization
                            st.plotly_chart(fig, use_container_width=True, key=f"claude_viz_{i}")
            else:
                st.info("No specific visualizations were generated for this query.")

            # Add "Explore This Data Further" section
            explore_further_section(jira_data, analysis, prompt, api_key)
            
            # Show raw data option with improved styling
            with st.expander("View Raw Data"):
                # Add some controls for viewing the data
                st.markdown("<h4 style='color:#1E3A8A;'>Raw Data Preview</h4>", unsafe_allow_html=True)
                
                # Select columns to display
                all_columns = jira_data.columns.tolist()
                default_columns = ["ISSUEKEY", "SUMMARY", "ISSUE_TYPE", "PRIORITY", "STATUS", 
                                "ORIGINAL_DEVELOPER", "CREATED", "RESOLVED"]
                
                # Only include default columns that actually exist in the data
                default_columns = [col for col in default_columns if col in all_columns]
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    selected_columns = st.multiselect(
                        "Select columns to display",
                        options=all_columns,
                        default=default_columns,
                        key="raw_data_columns"
                    )
                
                with col2:
                    # Add download option
                    csv = jira_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Data as CSV",
                        data=csv,
                        file_name='jira_data.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                
                # Display the data with selected columns
                if selected_columns:
                    st.dataframe(jira_data[selected_columns], height=400, use_container_width=True)
                else:
                    st.dataframe(jira_data, height=400, use_container_width=True)
    
    # Add Reset button at the bottom with improved styling - centered
    st.markdown("<div class='reset-button-container'>", unsafe_allow_html=True)
    reset_button = st.button("🔄 Reset All Analysis", use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if reset_button:
        # Use query parameters to trigger state reset on next load
        st.query_params["reset_ui"] = "true"
        st.rerun()

def explore_further_section(jira_data, analysis, prompt, api_key):
    """Create the 'Explore This Data Further' section with follow-up question capabilities"""
    with st.expander("Explore This Data Further", expanded=True):
        st.markdown("<h4 style='color:#1E3A8A;'>Ask Follow-up Questions</h4>", unsafe_allow_html=True)
        
        # Display previous insights with improved styling
        if st.session_state.analysis_context["insights"]:
            st.markdown("<h5 style='color:#1E3A8A;margin-top:20px;'>Key Insights</h5>", unsafe_allow_html=True)
            
            # Create a clean list of insights with nicer styling
            insights_html = "<div class='insights-container' style='background-color:#F7FAFC;padding:15px;border-radius:8px;margin-bottom:20px;'>"
            for i, insight in enumerate(st.session_state.analysis_context["insights"]):
                insights_html += f"<div style='padding:8px 0;border-bottom:1px solid #E2E8F0;'>{i+1}. {insight}</div>"
            insights_html += "</div>"
            
            st.markdown(insights_html, unsafe_allow_html=True)
        
        # Extract new insights from the current analysis
        new_insights = extract_insights_from_analysis(analysis)
        if new_insights:
            # Update insights in session state
            for insight in new_insights:
                if insight not in st.session_state.analysis_context["insights"]:
                    st.session_state.analysis_context["insights"].append(insight)
        
        # Let user ask a follow-up question - using a form with better styling
        st.markdown("<div style='background-color:#F0F5FF;padding:15px;border-radius:8px;margin:15px 0;'>", unsafe_allow_html=True)
        with st.form(key="followup_form"):
            st.markdown("<h5 style='color:#1E3A8A;margin:0;'>Dig Deeper</h5>", unsafe_allow_html=True)
            st.markdown("<p style='color:#475569;font-size:0.9rem;margin-bottom:15px;'>Ask a follow-up question to explore further or dive deeper into specific aspects</p>", unsafe_allow_html=True)
            
            followup_prompt = st.text_input(
                "Your follow-up question:",
                placeholder="Example: How does this compare to last month's data?",
                key="followup_question"
            )
            
            col1, col2 = st.columns([4, 1])
            with col2:
                submit_button = st.form_submit_button("Analyze", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Process follow-up question if submitted
        if submit_button and followup_prompt:
            with st.spinner("Analyzing your follow-up question..."):
                # Prepare data context if needed
                data_context = prepare_data_for_claude(jira_data)
                
                # Create a special prompt that includes previous context
                followup_context = {
                    "previous_question": prompt,
                    "previous_analysis": analysis,
                    "follow_up_question": followup_prompt
                }
                
                # Analyze the follow-up question in context
                followup_analysis = analyze_followup_question(data_context, followup_context, api_key)
                
                # Display results with improved styling
                st.markdown("<h5 style='color:#1E3A8A;margin-top:25px;'>Follow-up Analysis</h5>", unsafe_allow_html=True)
                st.markdown(f"<div style='background-color:#F0F5FF;padding:15px;border-radius:8px;margin:10px 0;'>{followup_analysis}</div>", unsafe_allow_html=True)
                
                # Extract and add new insights
                more_insights = extract_insights_from_analysis(followup_analysis)
                if more_insights:
                    st.markdown("<h5 style='color:#1E3A8A;margin-top:20px;'>New Insights Identified</h5>", unsafe_allow_html=True)
                    insights_html = "<div style='background-color:#F7FAFC;padding:15px;border-radius:8px;'>"
                    for insight in more_insights:
                        if insight not in st.session_state.analysis_context["insights"]:
                            st.session_state.analysis_context["insights"].append(insight)
                            insights_html += f"<div style='padding:5px 0;'><span style='color:#2563EB;'>•</span> {insight}</div>"
                    insights_html += "</div>"
                    st.markdown(insights_html, unsafe_allow_html=True)
                
                # Extract and display any new visualizations
                followup_recommendations = extract_visualization_recommendations(followup_analysis)
                if followup_recommendations:
                    followup_visualizations = generate_claude_recommended_visualizations(
                        jira_data, followup_recommendations
                    )
                    
                    if followup_visualizations:
                        st.markdown("<h5 style='color:#1E3A8A;margin-top:25px;'>Additional Visualizations</h5>", unsafe_allow_html=True)
                        for i, (viz_id, fig) in enumerate(followup_visualizations):
                            # Improve visualization appearance
                            fig.update_layout(
                                template="plotly_white",
                                margin=dict(l=20, r=20, t=40, b=20),
                                title_font=dict(size=18, color="#1E3A8A"),
                                legend_title_font=dict(size=14),
                                legend_font=dict(size=12),
                                colorway=["#4981F5", "#38B09D", "#F7B32B", "#F26419", "#2E294E", "#8661C1"],
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"followup_viz_{i}")
                
if __name__ == "__main__":
    new_main()