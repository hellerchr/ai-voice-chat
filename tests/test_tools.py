from datetime import date, timedelta

import assistant
import tools


def test_declarations_come_from_signatures_and_docstrings():
    weather = tools.TOOLS["get_weather"].declaration["function"]
    assert weather["description"] == "Weather forecast for a place and day."
    assert weather["parameters"]["properties"]["city"] == {"description": "Name of the city or place.", "type": "string"}
    assert weather["parameters"]["required"] == []
    calendar = tools.TOOLS["get_calendar"].declaration["function"]
    assert calendar["parameters"]["properties"]["days"]["type"] == "integer"


def test_run_reports_errors_instead_of_raising():
    assert tools.run("missing", {}) == {"error": "Unknown function missing"}
    assert "Unknown day" in tools.run("get_weather", {"day": "someday"})["error"]


def test_demo_functions_understand_days():
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    assert assistant.get_weather("Hamburg", "morgen")["day"] == tomorrow
    saturday = date.fromisoformat(assistant.get_weather("Hamburg", "this weekend")["day"])
    assert saturday.weekday() == 5 and 0 <= (saturday - date.today()).days < 7
    assert assistant.get_weather("Hamburg")["max_celsius"] == assistant.get_weather("hamburg")["max_celsius"]
    week = assistant.get_calendar("today", 3)["events"]
    assert [e["title"] for e in week][:2] == ["Team-Meeting", "Mittagessen mit Anna"] and len(week) == 6
