#!/bin/sh

if [ -z "$1" ]
then
      DLFOLDER="."
else
      DLFOLDER="$1"
fi

wget --continue -P $DLFOLDER http://www.openslr.org/resources/17/musan.tar.gz
tar xzf $DLFOLDER/musan.tar.gz -C $DLFOLDER
