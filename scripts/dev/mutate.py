"""Swap one exact string in a source file, preserving CRLF.

Used to check that a test actually catches the bug it describes: break the
implementation, watch the test fail, put it back. sed cannot be used for this
on Windows -- the file is CRLF, sed's pattern silently fails to match, and a
mutation that was never applied is indistinguishable from a test too weak to
catch it.
"""

import pathlib
import sys

path, old, new = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
text = path.read_text(encoding="utf-8")  # newline translation: \r\n -> \n
count = text.count(old)
assert count == 1, f"site is not unique ({count} occurrences): {old!r}"
path.write_text(text.replace(old, new), encoding="utf-8", newline="\r\n")
print(f"mutated {path}")
