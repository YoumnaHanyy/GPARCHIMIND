from presentation.routes.srs_routes import check_srs_sections
# TC-D2: Empty SRS content
# =========================================
def test_tc_d2_empty_srs():
    text = ""

    has_fr, has_nfr = check_srs_sections(text)

    assert has_fr is False
    assert has_nfr is False


# =========================================
# TC-D2: Random (invalid) SRS content
# =========================================
def test_tc_d2_random_text():
    text = "This is just a random document without requirements"

    has_fr, has_nfr = check_srs_sections(text)

    assert has_fr is False
    assert has_nfr is False