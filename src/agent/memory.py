"""Working memory for agent context and intermediate results."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MemoryEntry:
    """A single entry in working memory."""

    key: str
    value: Any
    entry_type: str  # fact, observation, intermediate_result, tool_output
    source: str  # tool_name, reasoning, user_input
    timestamp: datetime = field(default_factory=datetime.utcnow)
    relevance_score: float = 1.0
    loop_number: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize entry to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "entry_type": self.entry_type,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "relevance_score": self.relevance_score,
            "loop_number": self.loop_number,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """Deserialize entry from dictionary."""
        return cls(
            key=data["key"],
            value=data["value"],
            entry_type=data["entry_type"],
            source=data["source"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            relevance_score=data.get("relevance_score", 1.0),
            loop_number=data.get("loop_number", 0),
        )


class WorkingMemory:
    """Working memory for storing agent context during execution.

    Supports:
    - Key-value storage with metadata
    - Recency-based retrieval
    - Type-based filtering
    - Relevance scoring
    - Snapshot/restore for checkpointing
    """

    def __init__(self, max_entries: int = 100) -> None:
        self.max_entries = max_entries
        self._entries: dict[str, MemoryEntry] = {}
        self._history: deque[str] = deque(maxlen=max_entries)

    def store(
        self,
        key: str,
        value: Any,
        entry_type: str = "fact",
        source: str = "unknown",
        relevance_score: float = 1.0,
        loop_number: int = 0,
    ) -> None:
        """Store a value in working memory."""
        entry = MemoryEntry(
            key=key,
            value=value,
            entry_type=entry_type,
            source=source,
            relevance_score=relevance_score,
            loop_number=loop_number,
        )
        self._entries[key] = entry
        self._history.append(key)

        # Prune old entries if exceeding limit
        self._prune_if_needed()

    def retrieve(self, key: str) -> Any | None:
        """Retrieve a value by key."""
        entry = self._entries.get(key)
        return entry.value if entry else None

    def get_entry(self, key: str) -> MemoryEntry | None:
        """Retrieve full entry by key."""
        return self._entries.get(key)

    def remove(self, key: str) -> bool:
        """Remove an entry from memory."""
        if key in self._entries:
            del self._entries[key]
            return True
        return False

    def get_by_type(self, entry_type: str) -> list[MemoryEntry]:
        """Get all entries of a specific type."""
        return [e for e in self._entries.values() if e.entry_type == entry_type]

    def get_by_source(self, source: str) -> list[MemoryEntry]:
        """Get all entries from a specific source."""
        return [e for e in self._entries.values() if e.source == source]

    def get_recent(self, n: int = 10) -> list[MemoryEntry]:
        """Get N most recent entries."""
        recent_keys = list(self._history)[-n:]
        return [self._entries[k] for k in recent_keys if k in self._entries]

    def get_relevant(self, threshold: float = 0.5) -> list[MemoryEntry]:
        """Get entries above relevance threshold."""
        return [e for e in self._entries.values() if e.relevance_score >= threshold]

    def get_from_loop(self, loop_number: int) -> list[MemoryEntry]:
        """Get all entries from a specific loop iteration."""
        return [e for e in self._entries.values() if e.loop_number == loop_number]

    def update_relevance(self, key: str, new_score: float) -> bool:
        """Update relevance score for an entry."""
        if key in self._entries:
            self._entries[key].relevance_score = new_score
            return True
        return False

    def decay_relevance(self, decay_factor: float = 0.9) -> None:
        """Decay relevance scores of all entries."""
        for entry in self._entries.values():
            entry.relevance_score *= decay_factor

    def search(self, query: str, case_sensitive: bool = False) -> list[MemoryEntry]:
        """Search entries by key or value content."""
        results = []
        search_query = query if case_sensitive else query.lower()

        for entry in self._entries.values():
            key_match = (
                search_query in entry.key
                if case_sensitive
                else search_query in entry.key.lower()
            )
            value_str = str(entry.value)
            value_match = (
                search_query in value_str
                if case_sensitive
                else search_query in value_str.lower()
            )

            if key_match or value_match:
                results.append(entry)

        return results

    def to_context_string(self, max_entries: int = 20) -> str:
        """Convert memory to a context string for LLM."""
        recent = self.get_recent(max_entries)
        if not recent:
            return "Working memory is empty."

        lines = ["Current working memory:"]
        for entry in recent:
            value_str = str(entry.value)
            if len(value_str) > 200:
                value_str = value_str[:200] + "..."
            lines.append(f"- [{entry.entry_type}] {entry.key}: {value_str}")

        return "\n".join(lines)

    def snapshot(self) -> dict[str, Any]:
        """Create a snapshot of current memory state."""
        return {
            "entries": {k: v.to_dict() for k, v in self._entries.items()},
            "history": list(self._history),
            "max_entries": self.max_entries,
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore memory from a snapshot."""
        self._entries = {
            k: MemoryEntry.from_dict(v) for k, v in snapshot.get("entries", {}).items()
        }
        self._history = deque(snapshot.get("history", []), maxlen=self.max_entries)

    def clear(self) -> None:
        """Clear all memory entries."""
        self._entries.clear()
        self._history.clear()

    def _prune_if_needed(self) -> None:
        """Remove oldest entries if exceeding max capacity."""
        while len(self._entries) > self.max_entries:
            # Remove oldest entry not in recent history
            oldest_key = None
            oldest_time = datetime.max

            for key, entry in self._entries.items():
                if entry.timestamp < oldest_time:
                    oldest_time = entry.timestamp
                    oldest_key = key

            if oldest_key:
                del self._entries[oldest_key]

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __iter__(self):
        return iter(self._entries.values())

    @property
    def keys(self) -> list[str]:
        """Get all keys in memory."""
        return list(self._entries.keys())

    @property
    def size(self) -> int:
        """Get number of entries in memory."""
        return len(self._entries)
