import pytest

from herbenzo_agent.intake.red_flags import load_lexicon, merge_model_codes, needs_model_check, screen_text

POSITIVE = {
    "RF_CHEST_PAIN": [
        "I have chest pain right now",
        "there's a pressure going down to my left arm",
        "crushing feeling in my chest",
        "I think it's a heart attack",
        "No, but I have chest tightness",
    ],
    "RF_BREATHING": [
        "I can't breathe properly",
        "it's hard to breathe",
        "sudden shortness of breath",
        "my lips are turning blue",
        "I keep gasping",
    ],
    "RF_STROKE": [
        "her face is drooping on one side",
        "my speech is slurred",
        "sudden numbness on one side",
        "I think he had a stroke",
        "slurred speech since an hour",
    ],
    "RF_SEVERE_BLEEDING": [
        "I've been vomiting blood",
        "black tarry stools",
        "bleeding that won't stop",
        "there is blood in my stool",
        "heavy bleeding from a cut",
    ],
    "RF_SELF_HARM": [
        "I want to kill myself",
        "I've been feeling suicidal",
        "I want to end my life",
        "I don't want to live anymore",
        "I'm not suicidal but I think about self harm",
    ],
    "RF_ANAPHYLAXIS": [
        "my throat is swelling",
        "tongue swollen after eating nuts",
        "I had anaphylaxis before and it's happening",
        "lips swelling up",
        "my throat is closing",
    ],
    "RF_UNCONSCIOUS_SEIZURE": [
        "he passed out",
        "I fainted twice today",
        "she had a seizure",
        "having fits since morning",
        "he collapsed",
    ],
    "RF_SEVERE_ABDOMINAL": [
        "worst stomach pain of my life",
        "unbearable abdominal pain",
        "excruciating belly pain",
        "my abdomen is rigid",
        "belly is hard as a board",
    ],
    "RF_HIGH_FEVER_STIFF_NECK": [
        "stiff neck and a high fever",
        "fever since yesterday and now a stiff neck",
        "I have a fever with stiff neck",
        "stiff neck, fever",
        "high fever plus a stiff neck",
    ],
    "RF_PREGNANCY_BLEEDING": [
        "I'm pregnant and bleeding",
        "bleeding while pregnant",
        "severe pain during pregnancy",
        "pregnant, 12 weeks, and there's bleeding",
        "I am pregnant and have severe pain",
    ],
    "RF_POISONING_OVERDOSE": [
        "I took an overdose",
        "I think I was poisoned",
        "took too many pills",
        "she overdosed",
        "my child swallowed poison",
    ],
}

NEGATIVE = [
    "no chest pain at all",
    "I don't have chest pain",
    "never had a seizure",
    "my pain restarts every morning",
    "burning acidity after spicy food",
    "I have a mild headache",
    "I can breathe fine, just a cough",
    "my stools are a bit loose",
]


@pytest.mark.parametrize("code,text", [(c, t) for c, texts in POSITIVE.items() for t in texts])
def test_positive_phrasings_flag(code, text):
    flags = screen_text(text, "t:1")
    assert code in {f.code for f in flags}, f"{code} not detected in: {text!r}"
    flag = next(f for f in flags if f.code == code)
    assert flag.source == "rule" and flag.evidence_turn_id == "t:1" and flag.matched_text


@pytest.mark.parametrize("text", NEGATIVE)
def test_negative_phrasings_do_not_flag(text):
    assert screen_text(text, "t:1") == []


def test_every_lexicon_code_has_tests():
    assert load_lexicon().codes == set(POSITIVE)


def test_one_flag_per_code_per_turn():
    flags = screen_text("chest pain, chest pressure and pain in my left arm", "t:1")
    assert [f.code for f in flags] == ["RF_CHEST_PAIN"]


def test_model_trigger_keywords():
    assert needs_model_check("I feel dizzy and weak")
    assert not needs_model_check("I'm 34 and vegetarian")


def test_model_codes_restricted_to_lexicon():
    merged = merge_model_codes([], ["rf_stroke", "RF_MADE_UP", "RF_STROKE"], "feels odd", "t:2")
    assert [(f.code, f.source) for f in merged] == [("RF_STROKE", "model")]


def test_lexicon_hash_is_stable():
    assert len(load_lexicon().sha256) == 64
