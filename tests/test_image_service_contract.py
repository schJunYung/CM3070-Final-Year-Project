from backend.image_service import build_scene_prompt, get_model_spec


def test_image_prompt_is_compact_and_contains_scene_context():
    prompt = build_scene_prompt(
        title="Tower stair",
        story_text=(
            "The party climbs the tower. A stone guardian blocks the doorway. "
            "The upper stair glows with blue light."
        ),
        genre="Fantasy",
        tone="Ominous",
        memory_snapshot={
            "location": "Ancient tower",
            "active_characters": ["Arin", "Mira"],
            "threat": "High",
        },
    )
    assert "Ancient tower" in prompt
    assert "Tower stair" in prompt
    assert len(prompt.split()) <= 72


def test_model_operating_points_are_distinct():
    turbo = get_model_spec("sd-turbo")
    baseline = get_model_spec("sd15-int8")

    assert turbo.guidance_scale == 0.0
    assert turbo.steps <= 4
    assert baseline.steps > turbo.steps
    assert baseline.supports_negative_prompt
