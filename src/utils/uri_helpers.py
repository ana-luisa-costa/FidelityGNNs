
import re
from typing import Optional

CASE_RE = re.compile(r"caseID\.org/(\d+)$")
EVENT_RE = re.compile(r"eventID\.org/(\d+)_(\d+)$")  # group1=event#, group2=caseID


def get_case_of_entity(uri: str) -> Optional[str]:
    m = EVENT_RE.search(uri)
    if m:
        return m.group(2)  # Return caseID (group 2)
    m = CASE_RE.search(uri)
    if m:
        return m.group(1)
    return None
