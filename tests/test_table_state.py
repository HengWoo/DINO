import json

from dino.state.table_state import TableState, TableStateMachine


class TestTableStateMachine:
    def test_initial_state_is_empty(self):
        fsm = TableStateMachine(table_id="T1")
        assert fsm.state == TableState.EMPTY

    def test_valid_transition_empty_to_occupied(self):
        fsm = TableStateMachine(table_id="T1", hysteresis=1)
        fsm.update(TableState.OCCUPIED)
        assert fsm.state == TableState.OCCUPIED

    def test_valid_transition_occupied_to_served(self):
        fsm = TableStateMachine(table_id="T1", hysteresis=1)
        fsm.update(TableState.OCCUPIED)
        fsm.update(TableState.SERVED)
        assert fsm.state == TableState.SERVED

    def test_valid_transition_served_to_clearing(self):
        fsm = TableStateMachine(table_id="T1", hysteresis=1)
        fsm.update(TableState.OCCUPIED)
        fsm.update(TableState.SERVED)
        fsm.update(TableState.CLEARING)
        assert fsm.state == TableState.CLEARING

    def test_valid_transition_clearing_to_empty(self):
        fsm = TableStateMachine(table_id="T1", hysteresis=1)
        fsm.update(TableState.OCCUPIED)
        fsm.update(TableState.SERVED)
        fsm.update(TableState.CLEARING)
        fsm.update(TableState.EMPTY)
        assert fsm.state == TableState.EMPTY

    def test_invalid_transition_empty_to_served(self):
        fsm = TableStateMachine(table_id="T1", hysteresis=1)
        fsm.update(TableState.SERVED)
        assert fsm.state == TableState.EMPTY

    def test_invalid_transition_empty_to_clearing(self):
        fsm = TableStateMachine(table_id="T1", hysteresis=1)
        fsm.update(TableState.CLEARING)
        assert fsm.state == TableState.EMPTY

    def test_hysteresis_requires_consecutive_frames(self):
        fsm = TableStateMachine(table_id="T1", hysteresis=3)
        fsm.update(TableState.OCCUPIED)
        assert fsm.state == TableState.EMPTY
        fsm.update(TableState.OCCUPIED)
        assert fsm.state == TableState.EMPTY
        fsm.update(TableState.OCCUPIED)
        assert fsm.state == TableState.OCCUPIED

    def test_hysteresis_resets_on_same_state(self):
        fsm = TableStateMachine(table_id="T1", hysteresis=3)
        fsm.update(TableState.OCCUPIED)
        fsm.update(TableState.OCCUPIED)
        fsm.update(TableState.EMPTY)  # same as current state, resets counter
        fsm.update(TableState.OCCUPIED)
        assert fsm.state == TableState.EMPTY

    def test_invalid_transition_does_not_reset_pending(self):
        """An invalid observation should not disrupt a valid pending sequence."""
        fsm = TableStateMachine(table_id="T1", hysteresis=3)
        fsm.update(TableState.OCCUPIED)   # pending=OCC, count=1
        fsm.update(TableState.OCCUPIED)   # pending=OCC, count=2
        fsm.update(TableState.SERVED)     # invalid from EMPTY, ignored — pending preserved
        fsm.update(TableState.OCCUPIED)   # pending=OCC, count=3 → transition
        assert fsm.state == TableState.OCCUPIED

    def test_history_recording(self):
        fsm = TableStateMachine(table_id="T1", hysteresis=1)
        fsm.update(TableState.OCCUPIED)
        fsm.update(TableState.SERVED)
        history = fsm.history
        assert len(history) == 3
        assert history[0]["state"] == "empty"
        assert history[1]["state"] == "occupied"
        assert history[2]["state"] == "served"

    def test_history_has_frame_numbers(self):
        fsm = TableStateMachine(table_id="T1", hysteresis=1)
        fsm.update(TableState.OCCUPIED, frame_number=10)
        history = fsm.history
        assert history[1]["frame_number"] == 10

    def test_table_id(self):
        fsm = TableStateMachine(table_id="T42")
        assert fsm.table_id == "T42"

    def test_same_state_update_does_not_add_history(self):
        fsm = TableStateMachine(table_id="T1", hysteresis=1)
        fsm.update(TableState.EMPTY)
        assert len(fsm.history) == 1

    def test_occupied_can_go_back_to_empty(self):
        fsm = TableStateMachine(table_id="T1", hysteresis=1)
        fsm.update(TableState.OCCUPIED)
        fsm.update(TableState.EMPTY)
        assert fsm.state == TableState.EMPTY

    def test_history_is_json_serializable(self):
        fsm = TableStateMachine(table_id="T1", hysteresis=1)
        fsm.update(TableState.OCCUPIED)
        fsm.update(TableState.SERVED)
        # Should not raise
        json_str = json.dumps(fsm.history)
        parsed = json.loads(json_str)
        assert parsed[0]["state"] == "empty"
        assert parsed[1]["state"] == "occupied"
        assert parsed[2]["state"] == "served"
