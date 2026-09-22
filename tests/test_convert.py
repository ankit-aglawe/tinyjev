from tinyjev.convert import normalize_tokenizer_config


def test_list_extra_special_tokens_become_mapping():
    cfg, changed = normalize_tokenizer_config({"extra_special_tokens": ["<a>", "<b>"],
                                               "tokenizer_class": "TokenizersBackend"})
    assert changed
    assert cfg["extra_special_tokens"] == {"extra_0": "<a>", "extra_1": "<b>"}
    assert cfg["tokenizer_class"] == "PreTrainedTokenizerFast"


def test_clean_config_untouched():
    cfg, changed = normalize_tokenizer_config({"tokenizer_class": "PreTrainedTokenizerFast"})
    assert not changed
