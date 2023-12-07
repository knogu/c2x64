#!/bin/bash

cargo build --color=always --package c2x64 --bin c2x64

assert() {
  expected="$1"
  input="$2"

  echo "$input" | ~/c2x64/target/debug/c2x64 > tmp.s
  docker run --rm --platform linux/amd64 -v ~/c2x64:/c2x64 -w /c2x64 c2x64 gcc -static -o ./tmp ./tmp.s
  docker run --rm --platform linux/amd64 -v ~/c2x64:/c2x64 -w /c2x64 c2x64 chmod +x ./tmp
  docker run --rm --platform linux/amd64 -v ~/c2x64:/c2x64 -w /c2x64 c2x64 ./tmp
  actual="$?"

  if [ "$actual" = "$expected" ]; then
    echo "$input => $actual"
  else
    echo "$input => $expected expected, but got $actual"
    exit 1
  fi
}

assert 0 0
assert 42 42
assert 21 '5+20-4'
assert 41 ' 12 + 34 - 5 '
assert 47 '5+6*7'
assert 15 '5*(9-6)'
assert 4 '(3+5)/2'

echo OK
