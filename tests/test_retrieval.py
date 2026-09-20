"""Check that offline retrieval searches original article bodies."""

from easy_tech_help.retrieval import load_documents, search_documents


def test_popup_query_returns_tech_support_source():
    hits = search_documents(
        "fake virus popup says call support number", category="popup"
    )

    assert hits[0].document.id == "popup_tech_support"
    assert hits[0].document.source_urls[0].startswith("https://consumer.ftc.gov/")


def test_public_wifi_query_returns_original_ftc_page():
    hits = search_documents("public Wi-Fi hotspot security", category="wifi")

    assert hits[0].document.id == "wifi_public_safety"
    assert "Public Wi-Fi networks" in hits[0].document.text


def test_catalog_distinguishes_originals_and_summaries():
    documents = load_documents()

    assert len(documents) == 10
    assert all(
        document.source_urls[0].startswith("https://consumer.ftc.gov/")
        for document in documents
        if document.source_type == "original_html"
    )
    assert sum(document.source_type == "original_html" for document in documents) == 6
    assert (
        sum(document.source_type == "authored_summary" for document in documents) == 4
    )
    assert all(len(document.text) > 100 for document in documents)
    assert "Scammers are spoofing car dealership websites" not in documents[0].text


def test_unrelated_query_has_no_source():
    assert search_documents("astronomy telescope") == []


def test_iphone_wifi_troubleshooting_prefers_apple_based_summary():
    hits = search_documents("iPhone Wi-Fi connected but no internet", category="wifi")

    assert hits[0].document.id == "wifi_iphone"
    assert hits[0].document.source_type == "authored_summary"


def test_safari_popup_prefers_iphone_summary():
    hits = search_documents("Safari fake close button on popup", category="popup")

    assert hits[0].document.id == "popup_safari"
    assert hits[0].document.source_urls == ("https://support.apple.com/en-us/102524",)


def test_ordinary_safari_popup_finds_non_scam_guidance():
    hits = search_documents("Safari ordinary pop-up ad", category="popup")

    assert hits[0].document.id == "popup_safari"


def test_wifi_off_finds_iphone_steps_without_unrelated_security_sources():
    hits = search_documents("iPhone Wi-Fi off", category="wifi")

    assert [hit.document.id for hit in hits] == ["wifi_iphone"]
