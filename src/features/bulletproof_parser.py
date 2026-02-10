import json
import re
import logging

logger = logging.getLogger("BulletproofParser")

def parse_json_safely(text: str, default_fallback: any = None) -> any:
    """
    Robustly extracts JSON from a string, handling markdown backticks, 
    conversational prefix/suffix, and common formatting errors.
    """
    if not text:
        return default_fallback

    # 1. Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Try cleaning markdown blocks
    # Look for ```json ... ``` or just ``` ... ```
    md_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except json.JSONDecodeError:
            text = md_match.group(1) # Continue cleaning the inner content

    # 3. Try finding the first '[' or '{' and last ']' or '}'
    start_bracket = text.find('[')
    start_brace = text.find('{')
    
    if start_bracket == -1 and start_brace == -1:
        logger.warning("No JSON structures found in text.")
        return default_fallback

    # Determine which starts first
    if start_bracket != -1 and (start_brace == -1 or start_bracket < start_brace):
        start_idx = start_bracket
        end_idx = text.rfind(']') + 1
    else:
        start_idx = start_brace
        end_idx = text.rfind('}') + 1

    if start_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(text[start_idx:end_idx])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extracted JSON block: {e}")
            
    # 4. Final attempt: basic string cleanup (replacing common illegal chars)
    # This is a bit aggressive, but helpful for LLM oddities
    try:
        cleaned = text.strip().replace("'", '"')
        return json.loads(cleaned)
    except Exception:
        pass

    return default_fallback
