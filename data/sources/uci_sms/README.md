# UCI SMS source and attribution

Source: Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset].
UCI Machine Learning Repository. <https://doi.org/10.24432/C5CC84>.
The [UCI dataset page](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
lists [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The original author notice is preserved in `SOURCE_NOTICE.txt`.
The authors and UCI do not endorse EasyTechHelp.

The downloaded ZIP contains 5,574 SMS (4,827 ham, 747 spam). It combines multiple
historical collections; see the source notice for their origins. We selected 40
ham and 40 spam messages, read their contents, redacted contact information and
authored EasyTechHelp observation labels. Original `spam`/`ham` remains source
metadata and never becomes a model output or a verified fraud verdict.

The unmodified archive is stored locally at
`artifacts/sources/uci_sms_spam_collection.zip` (Git ignored). The repository
contains only the selected redacted messages in `data/text/*.jsonl`, their source
row numbers and raw-text hashes, this attribution and `manifest.json`.

Re-download and verify from the repository root:

```bash
mkdir -p artifacts/sources
curl -fL 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip' -o artifacts/sources/uci_sms_spam_collection.zip
python -m easy_tech_help.sms_source
```

Verification checks the archive hash, record counts, source labels, selected-row
hashes, and exact reproduction of every redacted text. No message links are
opened. Only the curated subset was reviewed for remaining identifiers; the
redaction function is not suitable for automatically publishing arbitrary inputs.

An initial seed-42 candidate shuffle was inspected to choose short English
messages suitable for this scope. The subset excludes explicit/adult content,
identifying names, malformed controls and redundant templates. It is not a
representative sample of all SMS. Two variants of one prize-draw campaign share
one scenario group inside train. A stricter 0.75 lexical similarity threshold
guards public SMS crossing splits. Cases were selected and labeled before model
evaluation, not from model errors.

Ordinary SMS are necessary controls against false alarms. Historical spam also
provides controls: URGENT prize language is not an urgent device security threat;
a displayed promotional code is not a request for a login verification code;
advertising alone does not prove fraud. Conversely, explicit premium-rate call
or text subscription invitations can contain a payment request.

The corpus does not supply contemporary phishing coverage or iPhone Wi-Fi and
popup ground truth. Public UCI messages might already be in a base model's
pretraining data, so our held-out split means unused for this project's updates
and tuning, not guaranteed unseen during pretraining.
