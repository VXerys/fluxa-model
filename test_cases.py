"""Quick manual test for the problematic cases from the logs."""
from datetime import date
from src.text_normalizer import normalize_text
from src.date_parser import parse_date
from src.amount_parser import parse_amount
from src.title_extractor import extract_title

BASE = date(2026, 7, 5)

CASES = [
    # From the screenshot
    "tanggal 4 juni mesen gojek 18 500",
    # Other absolute date variants
    "tanggal 4 juni beli kopi 15rb",
    "4 juni naik gojek 18500",
    "tgl 15 bayar listrik 200000",
    "beli nasi padang 4 juni 25000",
    # STT typos
    "bayar wifi isokan 300rb untuk bulan ini",
    "beli kopi kenangan kemaren 25rb buat lembur",
    # Day-only tanggal
    "tanggal 3 bayar kos 800rb",
]

for text in CASES:
    print(f"\n{'='*60}")
    print(f"INPUT   : {text}")
    normalized = normalize_text(text)
    print(f"NORMAL  : {normalized}")
    date_str, after_date = parse_date(normalized, BASE)
    print(f"DATE    : {date_str}  |  after: {after_date}")
    amount, after_amount = parse_amount(after_date)
    print(f"AMOUNT  : {amount}  |  after: {after_amount}")
    title = extract_title(after_amount)
    print(f"TITLE   : {title}")
