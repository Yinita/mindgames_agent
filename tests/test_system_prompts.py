from src.agents.agent import _detect_game_from_observation
from src.utils.prompts import load_system_prompts, get_game_prompt


def test_load_prompts_optional_file():
    data = load_system_prompts()
    assert isinstance(data, dict)
    assert data == {}


def test_get_game_prompt_keys():
    assert get_game_prompt("colonel_blotto")
    assert get_game_prompt("three_player_ipd")
    assert get_game_prompt("secret_mafia")
    assert get_game_prompt("codenames")


def test_strong_variant_provides_augmented_prompt():
    prompt = get_game_prompt("colonel_blotto", variant="strong")
    assert prompt is not None
    assert "<think>" in prompt or "<think></think>" in prompt
    assert "Colonel Blotto" in prompt
    assert "Opponent Analysis" in prompt or "Data Extraction" in prompt


def test_detect_game_from_observation_samples():
    mafia_obs = "[GAME] Welcome to Secret Mafia! Night phase..."
    blotto_obs = "[GAME] Colonel Blotto round 1. Commander Alpha..."
    ipd_obs = "[GAME] Iterated Prisoner's Dilemma. You can converse freely."
    codenames_obs = "[GAME] Codenames Words: RED, BLUE. Spymaster briefing."
    unknown_obs = "Some totally unrelated text."

    assert _detect_game_from_observation(mafia_obs) == "secret_mafia"
    assert _detect_game_from_observation(blotto_obs) == "colonel_blotto"
    assert _detect_game_from_observation(ipd_obs) == "three_player_ipd"
    assert _detect_game_from_observation(codenames_obs) == "codenames"
    assert _detect_game_from_observation(unknown_obs) is None
