"""
Labeling function for ticket priority (weak supervision).

This module defines the heuristic rules used to assign 'urgent' vs 'normal'
labels to customer support tweets. The rules are intentionally simple and
documented – they represent weak supervision, not ground truth.

Rules applied in order:
1. Presence of urgency keywords (case-insensitive)
2. Three or more exclamation marks
3. More than 30% of words in ALL CAPS

Returns:
    str: 'urgent' if any rule matches, otherwise 'normal'
"""

def create_priority_labels(text: str) -> str:
    """
    Assign a priority label to a tweet based on heuristic rules.

    Args:
        text (str): The tweet text.

    Returns:
        str: 'urgent' or 'normal'
    """
    if not isinstance(text, str):
        return 'normal'

    text_lower = text.lower()
    urgent_keywords = [
        'refund', 'cancel', 'broken', 'help',
        'urgent', 'asap', 'stuck', 'issue', 'error', 'fail'
    ]

    # Rule 1: Urgency keywords
    if any(keyword in text_lower for keyword in urgent_keywords):
        return 'urgent'

    # Rule 2: Exclamation marks
    if text.count('!') >= 3:
        return 'urgent'

    # Rule 3: ALL CAPS ratio
    words = text.split()
    if words:
        caps_words = sum(1 for w in words if w.isupper())
        caps_ratio = caps_words / len(words)
        if caps_ratio > 0.3:
            return 'urgent'

    return 'normal'