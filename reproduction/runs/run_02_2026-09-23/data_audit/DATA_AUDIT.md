# Data audit

All author files are read-only. checksums.txt covers every non-Git author file.

- Database/Oringinal_data.xlsx: n=420, positive=210, negative=210, sequence duplicates=0, conflicts=0, invalid=0, lengths=2–80.
- Database/X_320_clean_LR.csv + y_after_clean.csv: n=339, positive=161, negative=178, sequence duplicates=0, conflicts=0, invalid=0, lengths=2–80.

Paper claims raw 420 (210/210), cleaned 339 (161/178). See calculated CSV; blank trailing Excel formatting rows are not observations. Raw duplicate full rows include the unique No column; sequence/label duplicate counts are reported separately.

Cleaned CSV No indices join to raw No with exact label agreement; the current cleaned CSV contains 339 rows, whereas stored training notebook outputs show 343 rows and different feature values. Thus saved notebook outputs are not results of the current published input files.
