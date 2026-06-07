import json
from typing import Optional, List

def find_json_blocks(text: str) -> List[str]:
    """Find all top-level JSON-like object blocks in text by counting matching braces,
    taking into account string literals and escape characters.
    """
    blocks = []
    start = -1
    brace_count = 0
    in_string = False
    escape = False
    
    for i, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == '\\':
                escape = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == '{':
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif char == '}':
                if brace_count > 0:
                    brace_count -= 1
                    if brace_count == 0 and start != -1:
                        blocks.append(text[start:i+1])
                        start = -1
    return blocks

def parse_capability_call(text: str) -> Optional[dict]:
    """Parse assistant text to find a structured capability json_call block.
    
    Supports:
    - Fenced code blocks of type 'json_call' or 'json'
    - Raw JSON blocks anywhere in the text containing "capability" and "action".
    """
    # 1. First look for fenced code blocks
    import re
    pattern = r"```(?:json_call|json)?\s*(\{\s*\"capability\".*?\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            val = json.loads(match.group(1))
            if "capability" in val and "action" in val:
                return val
        except Exception:
            pass

    # 2. Extract and parse matching-braces JSON blocks
    blocks = find_json_blocks(text)
    for block in blocks:
        try:
            val = json.loads(block)
            if isinstance(val, dict) and "capability" in val and "action" in val:
                return val
        except Exception:
            pass
            
    return None
