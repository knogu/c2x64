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

assert 42 42
assert 0 0

echo OK
