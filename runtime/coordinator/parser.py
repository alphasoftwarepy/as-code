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

KNOWN_CAPABILITY_IDS: frozenset = frozenset({"documents", "rag", "git", "terminal"})

def _validate_call_envelope(val: dict) -> Optional[dict]:
    """Validate schema and whitelist for capability call envelope."""
    if not isinstance(val, dict):
        return None
    cap = val.get("capability")
    act = val.get("action")
    if not isinstance(cap, str) or not cap.strip():
        return None
    if not isinstance(act, str) or not act.strip():
        return None
    if cap not in KNOWN_CAPABILITY_IDS:
        return None
        
    params = val.get("params", {})
    if params is None:
        params = {}
    elif not isinstance(params, dict):
        return None
        
    return {
        "capability": cap.strip(),
        "action": act.strip(),
        "params": params
    }

def parse_capability_call(text: str) -> Optional[dict]:
    """Parse assistant text to find a structured capability json_call block.
    
    Supports:
    - Fenced code blocks of type 'json_call' or 'json'
    - Raw JSON blocks anywhere in the text containing "capability" and "action".
    """
    if not text or not isinstance(text, str):
        return None

    # 1. First look for fenced code blocks
    import re
    pattern = r"```(?:json_call|json)?\s*(\{\s*\"capability\".*?\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            val = json.loads(match.group(1))
            validated = _validate_call_envelope(val)
            if validated:
                return validated
        except Exception:
            pass

    # 2. Extract and parse matching-braces JSON blocks
    blocks = find_json_blocks(text)
    for block in blocks:
        try:
            val = json.loads(block)
            validated = _validate_call_envelope(val)
            if validated:
                return validated
        except Exception:
            pass
            
    return None
