import re
from datetime import datetime, timezone
from typing import Dict, Hashable, List, Optional, Set, Tuple

from rdflib import URIRef

CASE_RE = re.compile(r"caseID\.org/(\d+)$")
EVENT_RE = re.compile(r"eventID\.org/(\d+)_(\d+)$")  # group1=event#, group2=caseID

CASE_URI = "http://caseID.org/{}"  # case node URI, e.g. CASE_URI.format("1147898401")

P_TIME = "http://example.org/hasevent_timestamp_time_timestamp"

# predicates that only identify an event, so they cannot tell duplicates apart
IDENTIFIER_PREDICATE_PARTS = ("rdf-schema", "#type", "/type", "contains",
                              "hascaseID", "OtherC_id", "EventIDOther")


def get_case_of_entity(uri: str) -> Optional[str]:
    m = EVENT_RE.search(uri)
    if m:
        return m.group(2)  # Return caseID (group 2)
    m = CASE_RE.search(uri)
    if m:
        return m.group(1)
    return None


def parse_ts(ts: str) -> Optional[float]:
    ts = str(ts).strip()
    if len(ts) < 14 or not ts[:14].isdigit():
        return None
    try:
        dt = datetime.strptime(ts[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        frac = ts[14:]
        frac_sec = (int(frac) / (10 ** len(frac))) if (frac and frac.isdigit()) else 0.0
        return dt.timestamp() + frac_sec
    except Exception:
        return None


def order_case_events(events: Dict[int, Tuple[Hashable, Optional[float]]]) -> List[Hashable]:
    """Order one case's events by timestamp, falling back to event# for ties.

    `events` maps event# → (item, timestamp). The event# follows CSV row order,
    which is not always temporal, so it only breaks ties. Events without a
    timestamp go last.
    """
    def key(num):
        ts = events[num][1]
        return (ts is None, ts if ts is not None else 0.0, num)
    return [events[num][0] for num in sorted(events, key=key)]


def event_signature(g, event_uri: str) -> frozenset:
    """Everything that describes an event (timestamp included) except identifiers."""
    return frozenset((str(p), str(o)) for p, o in g.predicate_objects(URIRef(event_uri))
                     if not any(part in str(p) for part in IDENTIFIER_PREDICATE_PARTS))


def event_timestamps(g) -> Dict[str, Optional[float]]:
    return {str(s): parse_ts(str(o)) for s, o in g.subject_objects(URIRef(P_TIME))}


def usable_event_uris(g) -> Set[str]:
    """Events every pipeline stage may use, shared so all stages agree.

    An event is dropped if it has no timestamp or if an earlier event of its
    case has the same signature (a duplicate). Both decisions only look at the
    event and earlier events, so a prefix never depends on later events.
    """
    events_by_case: Dict[str, Dict[int, Tuple[Tuple[str, Optional[float]], Optional[float]]]] = {}
    for s, o in g.subject_objects(URIRef(P_TIME)):
        match = EVENT_RE.search(str(s))
        if match:
            ts = parse_ts(str(o))
            events_by_case.setdefault(match.group(2), {})[int(match.group(1))] = ((str(s), ts), ts)

    usable = set()
    for case_id, events in events_by_case.items():
        seen = set()
        for uri, ts in order_case_events(events):
            if ts is None:
                continue
            signature = event_signature(g, uri)
            if signature not in seen:
                seen.add(signature)
                usable.add(uri)
    return usable
