"""
Natural Language Processing agent for understanding user queries and extracting analysis intent.
"""
import openai
import re
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from config import Config

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Types of analysis operations."""
    SUM = "sum"
    AVERAGE = "average"
    COUNT = "count"
    GROUP_BY = "group_by"
    TREND = "trend"
    SORT = "sort"
    FILTER = "filter"
    CORRELATION = "correlation"
    STATISTICAL = "statistical"
    VISUALIZATION = "visualization"
    COMPARISON = "comparison"

@dataclass
class AnalysisIntent:
    """Structured representation of analysis intent."""
    analysis_type: AnalysisType
    target_columns: List[str]
    group_by_columns: List[str]
    filter_conditions: List[Dict[str, Any]]
    sort_columns: List[str]
    sort_ascending: bool
    chart_type: Optional[str]
    time_column: Optional[str]
    description: str
    confidence: float
    openai_error: Optional[str] = None

class NLPAgent:
    """Natural Language Processing agent for Excel analysis."""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        self.analysis_patterns = self._load_analysis_patterns()
    
    def _load_analysis_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load common analysis patterns and keywords."""
        return {
            "sum": {
                "keywords": ["sum", "total", "add up", "aggregate", "总计", "求和", "汇总"],
                "analysis_type": AnalysisType.SUM
            },
            "average": {
                "keywords": ["average", "mean", "avg", "average", "平均", "均值"],
                "analysis_type": AnalysisType.AVERAGE
            },
            "count": {
                "keywords": ["count", "number of", "how many", "计数", "数量"],
                "analysis_type": AnalysisType.COUNT
            },
            "group_by": {
                "keywords": ["group by", "by region", "by category", "by", "按", "分组"],
                "analysis_type": AnalysisType.GROUP_BY
            },
            "trend": {
                "keywords": ["trend", "over time", "change", "趋势", "变化", "时间"],
                "analysis_type": AnalysisType.TREND
            },
            "sort": {
                "keywords": ["sort", "order", "rank", "排序", "排列"],
                "analysis_type": AnalysisType.SORT
            },
            "filter": {
                "keywords": ["filter", "where", "condition", "筛选", "条件"],
                "analysis_type": AnalysisType.FILTER
            },
            "correlation": {
                "keywords": ["correlation", "relationship", "correlate", "相关", "关系"],
                "analysis_type": AnalysisType.CORRELATION
            },
            "visualization": {
                "keywords": ["chart", "graph", "plot", "visualize", "图表", "可视化"],
                "analysis_type": AnalysisType.VISUALIZATION
            }
        }
    
    def parse_query(self, query: str, available_columns: List[str] = None) -> AnalysisIntent:
        """
        Parse natural language query and extract analysis intent.
        
        Args:
            query: Natural language query
            available_columns: List of available columns in the dataset
            
        Returns:
            AnalysisIntent object containing structured analysis requirements
        """
        try:
            # First, try to extract intent using pattern matching
            intent = self._extract_intent_patterns(query, available_columns)
            
            # Then enhance with OpenAI for complex queries
            enhanced_intent = self._enhance_with_openai(query, intent, available_columns)
            
            return enhanced_intent
            
        except Exception as e:
            logger.error(f"Error parsing query: {str(e)}")
            # Fallback to basic pattern matching
            return self._extract_intent_patterns(query, available_columns)
    
    def _extract_intent_patterns(self, query: str, available_columns: List[str] = None) -> AnalysisIntent:
        """Extract analysis intent using pattern matching."""
        query_lower = query.lower()
        
        # Determine analysis type with priority
        analysis_type = AnalysisType.SUM  # Default
        confidence = 0.5
        
        # Priority order for analysis types (higher priority first)
        priority_order = [
            AnalysisType.GROUP_BY,
            AnalysisType.TREND,
            AnalysisType.CORRELATION,
            AnalysisType.SUM,
            AnalysisType.AVERAGE,
            AnalysisType.COUNT,
            AnalysisType.SORT,
            AnalysisType.FILTER
        ]
        
        matched_types = []
        for pattern_name, pattern_info in self.analysis_patterns.items():
            for keyword in pattern_info["keywords"]:
                if keyword in query_lower:
                    matched_types.append(pattern_info["analysis_type"])
                    break
        
        # Choose the highest priority matched type
        for priority_type in priority_order:
            if priority_type in matched_types:
                analysis_type = priority_type
                confidence = 0.8
                break
        
        # Extract group by columns first
        group_by_columns = self._extract_group_by_columns(query, available_columns)
        
        # Extract target columns (excluding group by columns)
        all_mentioned_columns = self._extract_column_mentions(query, available_columns)
        target_columns = [col for col in all_mentioned_columns if col not in group_by_columns]
        
        # Extract filter conditions
        filter_conditions = self._extract_filter_conditions(query)
        
        # Extract sort requirements
        sort_columns, sort_ascending = self._extract_sort_requirements(query, available_columns)
        
        # Determine chart type
        chart_type = self._extract_chart_type(query)
        
        # Extract time column for trend analysis
        time_column = self._extract_time_column(query, available_columns)
        
        return AnalysisIntent(
            analysis_type=analysis_type,
            target_columns=target_columns,
            group_by_columns=group_by_columns,
            filter_conditions=filter_conditions,
            sort_columns=sort_columns,
            sort_ascending=sort_ascending,
            chart_type=chart_type,
            time_column=time_column,
            description=query,
            confidence=confidence
        )
    
    def _enhance_with_openai(self, query: str, base_intent: AnalysisIntent, available_columns: List[str] = None) -> AnalysisIntent:
        """Enhance analysis intent using OpenAI for complex queries."""
        try:
            prompt = self._create_analysis_prompt(query, available_columns)
            
            response = self.client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert data analyst. Parse the user's natural language query and extract structured analysis requirements. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Get the response content
            response_content = response.choices[0].message.content.strip()
            
            # Debug logging
            logger.info(f"OpenAI response: {response_content}")
            
            # Try to parse JSON with better error handling
            try:
                result = json.loads(response_content)
            except json.JSONDecodeError as json_err:
                # Try to extract JSON from the response if it's wrapped in text
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        raise json_err
                else:
                    raise json_err
            
            # Update base intent with OpenAI insights
            if result.get("analysis_type"):
                try:
                    base_intent.analysis_type = AnalysisType(result["analysis_type"])
                except ValueError:
                    pass
            
            if result.get("target_columns"):
                base_intent.target_columns = result["target_columns"]
            
            if result.get("group_by_columns"):
                base_intent.group_by_columns = result["group_by_columns"]
            
            if result.get("filter_conditions"):
                base_intent.filter_conditions = result["filter_conditions"]
            
            if result.get("chart_type"):
                base_intent.chart_type = result["chart_type"]
            
            if result.get("confidence"):
                base_intent.confidence = result["confidence"]
            
            return base_intent
            
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "insufficient" in error_msg.lower():
                logger.warning(f"OpenAI quota exceeded, using pattern matching only: {error_msg}")
                base_intent.openai_error = f"OpenAI quota exceeded: {error_msg}"
            elif "json" in error_msg.lower() or "expecting value" in error_msg.lower():
                logger.warning(f"OpenAI JSON parsing failed, using pattern matching: {error_msg}")
                base_intent.openai_error = f"OpenAI response parsing failed: {error_msg}"
            else:
                logger.warning(f"OpenAI enhancement failed: {error_msg}")
                base_intent.openai_error = f"OpenAI API error: {error_msg}"
            return base_intent
    
    def _create_analysis_prompt(self, query: str, available_columns: List[str] = None) -> str:
        """Create prompt for OpenAI analysis."""
        columns_info = ""
        if available_columns:
            columns_info = f"Available columns: {', '.join(available_columns)}"
        
        prompt = f"""
        Analyze this natural language query and extract structured analysis requirements:
        
        Query: "{query}"
        {columns_info}
        
        IMPORTANT: Respond with ONLY a valid JSON object, no other text.
        
        Return a JSON object with the following structure:
        {{
            "analysis_type": "sum|average|count|group_by|trend|sort|filter|correlation|visualization|comparison",
            "target_columns": ["column1", "column2"],
            "group_by_columns": ["region", "category"],
            "filter_conditions": [{{"column": "sales", "operator": ">", "value": 1000}}],
            "sort_columns": ["sales"],
            "sort_ascending": true,
            "chart_type": "line|bar|scatter|pie",
            "time_column": "date",
            "confidence": 0.9
        }}
        
        Examples:
        - "total sales by region" → {{"analysis_type": "group_by", "target_columns": ["Sales"], "group_by_columns": ["Region"], "confidence": 0.9}}
        - "sum of all sales" → {{"analysis_type": "sum", "target_columns": ["Sales"], "confidence": 0.8}}
        - "average price" → {{"analysis_type": "average", "target_columns": ["Price"], "confidence": 0.8}}
        
        Rules:
        - Use "group_by" when the query mentions grouping by a category (e.g., "by region", "by category")
        - Use "sum" only when asking for a total without grouping
        - "total X by Y" should always be "group_by", not "sum"
        
        Focus on extracting the core analysis intent and relevant columns.
        """
        return prompt
    
    def _extract_column_mentions(self, query: str, available_columns: List[str] = None) -> List[str]:
        """Extract column mentions from the query."""
        if not available_columns:
            return []
        
        mentioned_columns = []
        query_lower = query.lower()
        
        for column in available_columns:
            column_lower = column.lower()
            if column_lower in query_lower or column in query:
                mentioned_columns.append(column)
        
        return mentioned_columns
    
    def _extract_group_by_columns(self, query: str, available_columns: List[str] = None) -> List[str]:
        """Extract group by columns from the query."""
        group_patterns = [
            r"by\s+(\w+)",
            r"group\s+by\s+(\w+)",
            r"按\s*(\w+)\s*分组",
            r"按\s*(\w+)\s*统计"
        ]
        
        group_columns = []
        for pattern in group_patterns:
            matches = re.findall(pattern, query.lower())
            for match in matches:
                if available_columns and match in [col.lower() for col in available_columns]:
                    # Find the original case column name
                    for col in available_columns:
                        if col.lower() == match:
                            group_columns.append(col)
                            break
        
        return group_columns
    
    def _extract_filter_conditions(self, query: str) -> List[Dict[str, Any]]:
        """Extract filter conditions from the query."""
        conditions = []
        
        # Simple pattern matching for common filter expressions
        filter_patterns = [
            r"(\w+)\s*>\s*(\d+)",
            r"(\w+)\s*<\s*(\d+)",
            r"(\w+)\s*=\s*(\w+)",
            r"(\w+)\s*大于\s*(\d+)",
            r"(\w+)\s*小于\s*(\d+)",
            r"(\w+)\s*等于\s*(\w+)"
        ]
        
        for pattern in filter_patterns:
            matches = re.findall(pattern, query.lower())
            for match in matches:
                column, value = match
                operator = ">" if ">" in query else "<" if "<" in query else "="
                conditions.append({
                    "column": column,
                    "operator": operator,
                    "value": value
                })
        
        return conditions
    
    def _extract_sort_requirements(self, query: str, available_columns: List[str] = None) -> Tuple[List[str], bool]:
        """Extract sort requirements from the query."""
        sort_columns = []
        ascending = True
        
        # Look for sort keywords
        if "descending" in query.lower() or "desc" in query.lower() or "降序" in query:
            ascending = False
        elif "ascending" in query.lower() or "asc" in query.lower() or "升序" in query:
            ascending = True
        
        # Extract column names after sort keywords
        sort_patterns = [
            r"sort\s+by\s+(\w+)",
            r"order\s+by\s+(\w+)",
            r"按\s*(\w+)\s*排序"
        ]
        
        for pattern in sort_patterns:
            matches = re.findall(pattern, query.lower())
            for match in matches:
                if available_columns and match in [col.lower() for col in available_columns]:
                    for col in available_columns:
                        if col.lower() == match:
                            sort_columns.append(col)
                            break
        
        return sort_columns, ascending
    
    def _extract_chart_type(self, query: str) -> Optional[str]:
        """Extract chart type from the query."""
        query_lower = query.lower()
        
        if "line" in query_lower or "trend" in query_lower or "线" in query:
            return "line"
        elif "bar" in query_lower or "column" in query_lower or "柱" in query:
            return "bar"
        elif "pie" in query_lower or "饼" in query:
            return "pie"
        elif "scatter" in query_lower or "散点" in query:
            return "scatter"
        
        return None
    
    def _extract_time_column(self, query: str, available_columns: List[str] = None) -> Optional[str]:
        """Extract time column from the query."""
        if not available_columns:
            return None
        
        time_keywords = ["date", "time", "month", "year", "day", "日期", "时间", "月份", "年份"]
        
        for column in available_columns:
            column_lower = column.lower()
            if any(keyword in column_lower for keyword in time_keywords):
                return column
        
        return None
    
    def validate_intent(self, intent: AnalysisIntent, available_columns: List[str]) -> bool:
        """Validate that the analysis intent is feasible with available columns."""
        # Check if target columns exist
        for col in intent.target_columns:
            if col not in available_columns:
                return False
        
        # Check if group by columns exist
        for col in intent.group_by_columns:
            if col not in available_columns:
                return False
        
        # Check if sort columns exist
        for col in intent.sort_columns:
            if col not in available_columns:
                return False
        
        return True


