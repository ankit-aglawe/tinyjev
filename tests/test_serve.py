from tinyjev.families.pointer import option_text, render


def test_kev_render_flattens_objects_with_labels():
    text = render({"subject": "Charged twice", "tags": ["billing", "refund"], "n": 2})
    assert "subject: Charged twice" in text and "- billing" in text and "n: 2" in text


def test_kev_option_text_uses_name_when_description_missing():
    assert option_text("calm", None) == "calm"
    assert option_text("calm", "") == "calm"
    assert option_text("billing", "payments") == "billing: payments"
