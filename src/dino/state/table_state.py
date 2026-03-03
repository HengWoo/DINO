from enum import Enum


class TableState(Enum):
    EMPTY = "empty"
    OCCUPIED = "occupied"
    SERVED = "served"
    CLEARING = "clearing"


VALID_TRANSITIONS: dict[TableState, set[TableState]] = {
    TableState.EMPTY: {TableState.OCCUPIED},
    TableState.OCCUPIED: {TableState.SERVED, TableState.EMPTY},
    TableState.SERVED: {TableState.CLEARING},
    TableState.CLEARING: {TableState.EMPTY},
}


class TableStateMachine:
    def __init__(self, table_id: str, hysteresis: int = 3):
        self.table_id = table_id
        self._state = TableState.EMPTY
        self._hysteresis = hysteresis
        self._pending_state: TableState | None = None
        self._pending_count: int = 0
        self._history: list[dict] = [{"state": TableState.EMPTY, "frame_number": 0}]

    @property
    def state(self) -> TableState:
        return self._state

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def update(self, observed_state: TableState, frame_number: int = 0) -> TableState:
        if observed_state == self._state:
            self._pending_state = None
            self._pending_count = 0
            return self._state

        if observed_state not in VALID_TRANSITIONS.get(self._state, set()):
            self._pending_state = None
            self._pending_count = 0
            return self._state

        if observed_state == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = observed_state
            self._pending_count = 1

        if self._pending_count >= self._hysteresis:
            self._state = observed_state
            self._history.append({"state": observed_state, "frame_number": frame_number})
            self._pending_state = None
            self._pending_count = 0

        return self._state
