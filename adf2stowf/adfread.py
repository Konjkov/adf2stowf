#!/usr/bin/env python3

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

# Constants for formatting
INT_FIELD_WIDTH = 12
FLOAT_FIELD_WIDTH = 28
STRING_BLOCK_SIZE = 160


class AdfParser:
    """Parser for ADF TAPE21.asc files."""

    def __init__(self, filename: Union[str, Path]) -> None:
        self.filename = Path(filename)
        self.lines: List[str] = []
        self.data: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _float_x(x: str) -> float:
        """Convert ADF float string to Python float."""
        x = re.sub(r'E*-', 'E-', x)
        if x.startswith('E'):
            x = x[1:]
        return float(x)

    @staticmethod
    def _split_n(s: str, n: int) -> List[str]:
        """Split string `s` into fixed-length chunks of size `n`."""
        return [s[i : i + n].strip() for i in range(0, len(s), n)]

    @staticmethod
    def _int_x(s: str) -> int:
        """Convert ADF int string to Python int, handling overflow marker."""
        return -(2**31) if s == '**********' else int(s)

    def _parse_integers(self, start: int, count: int) -> (np.ndarray, int):
        values: List[int] = []
        i = start
        while len(values) < count:
            values.extend(self._int_x(s) for s in self._split_n(self.lines[i], INT_FIELD_WIDTH))
            i += 1
        return np.array(values, int), i

    def _parse_floats(self, start: int, count: int) -> (np.ndarray, int):
        values: List[float] = []
        i = start
        while len(values) < count:
            values.extend(self._float_x(s) for s in self._split_n(self.lines[i], FLOAT_FIELD_WIDTH))
            i += 1
        return np.array(values, float), i

    def _parse_strings(self, start: int, count: int) -> (List[str], int):
        raw = ''
        i = start
        while len(raw) < count:
            raw += self.lines[i]
            i += 1
        values = [
            raw[STRING_BLOCK_SIZE * j : STRING_BLOCK_SIZE * (j + 1)].strip() for j in range((len(raw) + STRING_BLOCK_SIZE - 1) // STRING_BLOCK_SIZE)
        ]
        return values, i

    def _parse_bools(self, start: int, count: int) -> (List[bool], int):
        values: List[bool] = []
        i = start
        while len(values) < count:
            values.extend({'T': True, 'F': False}[c] for c in self.lines[i])
            i += 1
        return values, i

    def _parse_value(self, typ: int, start: int, count: int) -> (Any, int):
        if typ == 1:
            return self._parse_integers(start, count)
        elif typ == 2:
            return self._parse_floats(start, count)
        elif typ == 3:
            return self._parse_strings(start, count)
        elif typ == 4:
            return self._parse_bools(start, count)
        else:
            raise ValueError(f'Unexpected type {typ}, expected 1..4')

    def load(self) -> None:
        """Read file contents into memory."""
        try:
            with open(self.filename, encoding='latin-1') as f:
                self.lines = f.read().splitlines()
        except (FileNotFoundError, IsADirectoryError):
            print(f'File {self.filename} could not be read.')
            sys.exit(1)

    def parse(self) -> Dict[str, Dict[str, Any]]:
        """Parse the loaded file into a nested dictionary."""
        if not self.lines:
            self.load()

        i = 0
        self.data = {}

        try:
            while i < len(self.lines):
                group = self.lines[i].strip()
                i += 1
                key = self.lines[i].strip()
                i += 1

                len1, len2, typ = map(int, self.lines[i].split())
                i += 1

                value, i = self._parse_value(typ, i, len2)

                if len1 == 0:
                    i += 1

                self.data.setdefault(group, {})[key] = value

        except Exception:
            print('Parsing error around:')
            for x in range(max(0, i - 3), min(i + 4, len(self.lines))):
                marker = '>>>>' if x == i else '   |'
                print(f'{marker}{self.lines[x]}')
            raise

        return self.data

    def write_dump(self, outfile: Union[str, Path]) -> None:
        """Write human-readable dump of parsed data."""
        if not self.data:
            self.parse()

        with open(outfile, 'w', encoding='latin-1') as f:
            last_group = ''
            for group, keys in self.data.items():
                if group != last_group:
                    f.write(f'\n{group}\n')
                    last_group = group
                for key, value in keys.items():
                    tab = '\t'
                    newline = '\n'
                    valstr = str(value)
                    if newline in valstr:
                        f.write(f'  {key} = {{{len(value)}}}\n' + tab + f'{valstr.replace(chr(10), chr(10) + tab)}\n')
                    else:
                        f.write(f'  {key} = {valstr}\n')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: adfread.py <file.asc>')
        sys.exit(1)

    ascfname = Path(sys.argv[1])
    if ascfname.suffix != '.asc':
        print('Error: input file must end with .asc')
        sys.exit(1)

    parser = AdfParser(ascfname)
    parser.parse()
    parser.write_dump(ascfname.with_suffix('.txt'))
