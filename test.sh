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

assert 0 'return 0;'
assert 42 'return 42;'
assert 21 'return 5+20-4;'
assert 41 'return  12 + 34 - 5 ;'
assert 47 'return 5+6*7;'
assert 15 'return 5*(9-6);'
assert 4 'return (3+5)/2;'
assert 10 'return -10+20;'
assert 10 'return -(-10);'
assert 10 'return -(-(+10));'

assert 0 'return 0==1;'
assert 1 'return 42==42;'
assert 1 'return 0!=1;'
assert 0 'return 42!=42;'

assert 1 'return 0<1;'
assert 0 'return 1<1;'
assert 0 'return 2<1;'
assert 1 'return 0<=1;'
assert 1 'return 1<=1;'
assert 0 'return 2<=1;'

assert 1 'return 1>0;'
assert 0 'return 1>1;'
assert 0 'return 1>2;'
assert 1 'return 1>=0;'
assert 1 'return 1>=1;'
assert 0 'return 1>=2;'

assert 1 'return 1; 2; 3;'
assert 2 '1; return 2; 3;'
assert 3 '1; 2; return 3;'

echo OK
